# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Agent - Economic Analysis Module

This module implements economic analysis for heat exchangers including:
- Energy cost calculations
- Cleaning vs. replacement analysis
- Total cost of ownership (TCO)
- NPV and ROI calculations
- Lifecycle cost analysis

Economic Models:
    - Energy loss costing
    - Fouling cost impact
    - Optimal replacement timing
    - Maintenance scheduling economics

References:
    - Sinnott & Towler, "Chemical Engineering Design" (6th Ed.)
    - Peters, Timmerhaus & West, "Plant Design and Economics"
    - HTRI Economic Guidelines

Example:
    >>> from greenlang.agents.process_heat.gl_014_heat_exchanger.economics import (
    ...     EconomicAnalyzer
    ... )
    >>> analyzer = EconomicAnalyzer(config.economics)
    >>> result = analyzer.analyze_economics(
    ...     current_u=400, clean_u=500, area=100, lmtd=30
    ... )
    >>> print(f"Annual energy loss: ${result.energy_cost_usd_per_year:,.0f}")
"""

import math
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_014_heat_exchanger.config import (
    EconomicsConfig,
)
from greenlang.agents.process_heat.gl_014_heat_exchanger.schemas import (
    EconomicAnalysisResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Hours per year for continuous operation
HOURS_PER_YEAR = 8760

# Typical operating hours per year (allowing for turnaround)
TYPICAL_OPERATING_HOURS = 8000

# Depreciation period (years) for heat exchangers
TYPICAL_DEPRECIATION_YEARS = 15

# Maintenance cost as percentage of capital
TYPICAL_MAINTENANCE_PERCENT = 0.03

# Energy conversion factors
KWH_PER_MMBTU = 293.07
KW_PER_HP = 0.7457


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EnergyCostBreakdown:
    """Breakdown of energy costs."""

    heat_loss_kw: float
    heat_loss_cost_usd_per_hour: float
    pumping_cost_usd_per_hour: float
    total_cost_usd_per_hour: float
    annual_cost_usd: float


@dataclass
class ReplacementAnalysis:
    """Replacement vs. maintenance analysis."""

    years_to_replacement: float
    npv_maintain_usd: float
    npv_replace_usd: float
    recommendation: str  # "maintain", "replace_now", "replace_planned"
    annual_savings_if_replace_usd: float
    payback_period_years: float
    irr_percent: Optional[float]


@dataclass
class LifecycleCost:
    """Total lifecycle cost breakdown."""

    capital_cost_usd: float
    installation_cost_usd: float
    operating_cost_usd: float
    maintenance_cost_usd: float
    energy_cost_usd: float
    downtime_cost_usd: float
    disposal_cost_usd: float
    total_lifecycle_cost_usd: float
    annualized_cost_usd: float


@dataclass
class OptimizationOpportunity:
    """Identified optimization opportunity."""

    opportunity_type: str
    description: str
    annual_savings_usd: float
    implementation_cost_usd: float
    payback_years: float
    priority: str  # "high", "medium", "low"


# =============================================================================
# ECONOMIC ANALYZER
# =============================================================================

class EconomicAnalyzer:
    """
    Economic analysis for heat exchangers.

    This class provides comprehensive economic analysis including energy
    costs, maintenance optimization, and replacement timing. All calculations
    are deterministic with zero hallucination guarantee.

    Economic Methods:
        - Time value of money (NPV, IRR)
        - Energy loss costing
        - Optimal cleaning frequency
        - Replacement timing analysis
        - Total cost of ownership

    Attributes:
        config: Economics configuration
        discount_rate: Annual discount rate

    Example:
        >>> analyzer = EconomicAnalyzer(config)
        >>> result = analyzer.analyze_economics(u_current=400, u_clean=500, ...)
    """

    def __init__(
        self,
        config: EconomicsConfig,
    ) -> None:
        """
        Initialize the economic analyzer.

        Args:
            config: Economics configuration
        """
        self.config = config
        self._calculation_count = 0

        logger.info(
            f"EconomicAnalyzer initialized, "
            f"discount_rate={config.discount_rate:.1%}"
        )

    def analyze_economics(
        self,
        u_current_w_m2k: float,
        u_clean_w_m2k: float,
        heat_transfer_area_m2: float,
        lmtd_c: float,
        operating_hours_per_year: float = TYPICAL_OPERATING_HOURS,
        fouling_rate_m2kw_per_day: float = 0.0,
        cleaning_cost_usd: float = 5000.0,
        remaining_life_years: float = 10.0,
    ) -> EconomicAnalysisResult:
        """
        Perform comprehensive economic analysis.

        Args:
            u_current_w_m2k: Current U value
            u_clean_w_m2k: Clean U value
            heat_transfer_area_m2: Heat transfer area
            lmtd_c: Log mean temperature difference
            operating_hours_per_year: Annual operating hours
            fouling_rate_m2kw_per_day: Fouling rate
            cleaning_cost_usd: Cleaning cost
            remaining_life_years: Estimated remaining equipment life

        Returns:
            EconomicAnalysisResult with complete analysis
        """
        self._calculation_count += 1

        # Calculate energy loss
        energy_loss = self._calculate_energy_loss(
            u_current_w_m2k,
            u_clean_w_m2k,
            heat_transfer_area_m2,
            lmtd_c,
        )

        # Energy costs
        energy_cost_daily = (
            energy_loss * 24 * self.config.energy_cost_usd_per_kwh
        )
        energy_cost_monthly = energy_cost_daily * 30
        energy_cost_annual = (
            energy_loss * operating_hours_per_year *
            self.config.energy_cost_usd_per_kwh
        )

        # Cleaning economics
        optimal_cleaning_freq, cleaning_roi = self._optimize_cleaning_frequency(
            energy_cost_annual,
            cleaning_cost_usd,
            fouling_rate_m2kw_per_day,
            u_clean_w_m2k,
            heat_transfer_area_m2,
            lmtd_c,
            operating_hours_per_year,
        )

        # Payback period for cleaning
        if cleaning_roi > 0:
            payback_days = (
                cleaning_cost_usd / (energy_cost_daily * 0.8)
            )  # Assume 80% recovery
        else:
            payback_days = None

        # Annual cleaning cost
        cleanings_per_year = 365 / optimal_cleaning_freq
        annual_cleaning_cost = cleaning_cost_usd * cleanings_per_year

        # Replacement analysis
        replacement = self._analyze_replacement(
            energy_cost_annual,
            annual_cleaning_cost,
            remaining_life_years,
        )

        # Total cost of ownership
        annual_tco = (
            energy_cost_annual +
            annual_cleaning_cost +
            self.config.replacement_cost_usd * TYPICAL_MAINTENANCE_PERCENT
        )

        # Lifecycle cost
        lifecycle_cost = (
            annual_tco * remaining_life_years
        )

        # Optimization opportunities
        opportunities = self._identify_optimization_opportunities(
            energy_cost_annual,
            u_current_w_m2k,
            u_clean_w_m2k,
            fouling_rate_m2kw_per_day,
        )

        return EconomicAnalysisResult(
            energy_loss_kw=energy_loss,
            energy_cost_usd_per_day=energy_cost_daily,
            energy_cost_usd_per_month=energy_cost_monthly,
            energy_cost_usd_per_year=energy_cost_annual,
            cleaning_roi_percent=cleaning_roi,
            payback_period_days=payback_days,
            optimal_cleaning_frequency_days=optimal_cleaning_freq,
            annual_cleaning_cost_usd=annual_cleaning_cost,
            remaining_value_usd=replacement.npv_maintain_usd,
            replacement_timing_years=(
                replacement.years_to_replacement
                if replacement.recommendation != "replace_now" else 0
            ),
            replace_vs_maintain_npv_usd=(
                replacement.npv_replace_usd - replacement.npv_maintain_usd
            ),
            annual_tco_usd=annual_tco,
            lifecycle_cost_usd=lifecycle_cost,
            optimization_savings_usd_per_year=sum(
                o.annual_savings_usd for o in opportunities
            ),
            optimization_recommendations=[
                o.description for o in opportunities[:5]
            ],
        )

    def calculate_energy_loss_cost(
        self,
        u_current_w_m2k: float,
        u_clean_w_m2k: float,
        heat_transfer_area_m2: float,
        lmtd_c: float,
        operating_hours_per_year: float = TYPICAL_OPERATING_HOURS,
    ) -> EnergyCostBreakdown:
        """
        Calculate energy loss cost due to fouling.

        Args:
            u_current_w_m2k: Current U value
            u_clean_w_m2k: Clean U value
            heat_transfer_area_m2: Heat transfer area
            lmtd_c: Log mean temperature difference
            operating_hours_per_year: Annual operating hours

        Returns:
            EnergyCostBreakdown with detailed costs
        """
        self._calculation_count += 1

        # Heat loss (kW)
        heat_loss = self._calculate_energy_loss(
            u_current_w_m2k,
            u_clean_w_m2k,
            heat_transfer_area_m2,
            lmtd_c,
        )

        # Heat loss cost per hour
        heat_cost_per_hour = heat_loss * self.config.energy_cost_usd_per_kwh

        # Pumping cost increase (estimated from pressure drop increase)
        # Fouling increases DP, thus pumping power
        fouling_ratio = u_clean_w_m2k / u_current_w_m2k - 1
        pumping_increase = fouling_ratio * 0.1  # 10% DP increase per U decrease

        # Estimate base pumping power (kW per m2 of exchanger)
        base_pumping_kw = heat_transfer_area_m2 * 0.01  # 10 W/m2 typical
        pumping_increase_kw = base_pumping_kw * pumping_increase
        pumping_cost_per_hour = (
            pumping_increase_kw * self.config.energy_cost_usd_per_kwh
        )

        # Total
        total_cost_per_hour = heat_cost_per_hour + pumping_cost_per_hour
        annual_cost = total_cost_per_hour * operating_hours_per_year

        return EnergyCostBreakdown(
            heat_loss_kw=heat_loss,
            heat_loss_cost_usd_per_hour=heat_cost_per_hour,
            pumping_cost_usd_per_hour=pumping_cost_per_hour,
            total_cost_usd_per_hour=total_cost_per_hour,
            annual_cost_usd=annual_cost,
        )

    def analyze_replacement_economics(
        self,
        annual_operating_cost_usd: float,
        annual_maintenance_cost_usd: float,
        remaining_life_years: float,
        new_equipment_cost_usd: Optional[float] = None,
        new_equipment_life_years: float = 20.0,
        new_operating_cost_factor: float = 0.7,
    ) -> ReplacementAnalysis:
        """
        Analyze replacement vs. continued operation economics.

        Args:
            annual_operating_cost_usd: Current annual operating cost
            annual_maintenance_cost_usd: Current annual maintenance cost
            remaining_life_years: Remaining equipment life
            new_equipment_cost_usd: New equipment cost (uses config if None)
            new_equipment_life_years: Expected life of new equipment
            new_operating_cost_factor: Operating cost as fraction of current

        Returns:
            ReplacementAnalysis with detailed comparison
        """
        self._calculation_count += 1

        if new_equipment_cost_usd is None:
            new_equipment_cost_usd = self.config.replacement_cost_usd

        # NPV of maintaining current equipment
        npv_maintain = self._calculate_npv(
            annual_cost=annual_operating_cost_usd + annual_maintenance_cost_usd,
            years=remaining_life_years,
            discount_rate=self.config.discount_rate,
        )

        # NPV of replacing now
        new_annual_cost = (
            annual_operating_cost_usd * new_operating_cost_factor +
            new_equipment_cost_usd * TYPICAL_MAINTENANCE_PERCENT
        )
        npv_replace = new_equipment_cost_usd + self._calculate_npv(
            annual_cost=new_annual_cost,
            years=new_equipment_life_years,
            discount_rate=self.config.discount_rate,
        )

        # Annual savings if replace
        annual_savings = (
            (annual_operating_cost_usd + annual_maintenance_cost_usd) -
            new_annual_cost
        )

        # Payback period
        if annual_savings > 0:
            payback = new_equipment_cost_usd / annual_savings
        else:
            payback = float('inf')

        # IRR calculation
        irr = self._calculate_irr(
            initial_investment=new_equipment_cost_usd,
            annual_savings=annual_savings,
            years=min(remaining_life_years, new_equipment_life_years),
        )

        # Recommendation
        if npv_replace < npv_maintain * 0.8:
            recommendation = "replace_now"
            years_to_replacement = 0
        elif remaining_life_years < 3:
            recommendation = "replace_planned"
            years_to_replacement = remaining_life_years
        else:
            recommendation = "maintain"
            years_to_replacement = remaining_life_years

        return ReplacementAnalysis(
            years_to_replacement=years_to_replacement,
            npv_maintain_usd=npv_maintain,
            npv_replace_usd=npv_replace,
            recommendation=recommendation,
            annual_savings_if_replace_usd=annual_savings,
            payback_period_years=payback,
            irr_percent=irr * 100 if irr else None,
        )

    def calculate_lifecycle_cost(
        self,
        capital_cost_usd: Optional[float] = None,
        installation_factor: float = 1.3,
        annual_energy_cost_usd: float = 0.0,
        annual_maintenance_cost_usd: float = 0.0,
        annual_downtime_hours: float = 24.0,
        equipment_life_years: float = 20.0,
        disposal_cost_factor: float = 0.05,
    ) -> LifecycleCost:
        """
        Calculate total lifecycle cost.

        Args:
            capital_cost_usd: Capital cost (uses config if None)
            installation_factor: Installation cost as multiple of capital
            annual_energy_cost_usd: Annual energy cost
            annual_maintenance_cost_usd: Annual maintenance cost
            annual_downtime_hours: Annual downtime hours
            equipment_life_years: Equipment life
            disposal_cost_factor: Disposal cost as fraction of capital

        Returns:
            LifecycleCost with complete breakdown
        """
        self._calculation_count += 1

        if capital_cost_usd is None:
            capital_cost_usd = self.config.replacement_cost_usd

        # Installation cost
        installation_cost = capital_cost_usd * (installation_factor - 1)

        # Operating cost (energy over life)
        energy_cost_total = self._calculate_npv(
            annual_energy_cost_usd,
            equipment_life_years,
            self.config.discount_rate,
        )

        # Maintenance cost over life
        maintenance_cost_total = self._calculate_npv(
            annual_maintenance_cost_usd,
            equipment_life_years,
            self.config.discount_rate,
        )

        # Downtime cost over life
        annual_downtime_cost = (
            annual_downtime_hours *
            self.config.production_value_usd_per_hour
        )
        downtime_cost_total = self._calculate_npv(
            annual_downtime_cost,
            equipment_life_years,
            self.config.discount_rate,
        )

        # Disposal cost
        disposal_cost = capital_cost_usd * disposal_cost_factor

        # Total lifecycle cost
        total_lifecycle = (
            capital_cost_usd +
            installation_cost +
            energy_cost_total +
            maintenance_cost_total +
            downtime_cost_total +
            disposal_cost
        )

        # Annualized cost
        annualized = self._calculate_annualized_cost(
            total_lifecycle,
            equipment_life_years,
            self.config.discount_rate,
        )

        return LifecycleCost(
            capital_cost_usd=capital_cost_usd,
            installation_cost_usd=installation_cost,
            operating_cost_usd=energy_cost_total,
            maintenance_cost_usd=maintenance_cost_total,
            energy_cost_usd=energy_cost_total,
            downtime_cost_usd=downtime_cost_total,
            disposal_cost_usd=disposal_cost,
            total_lifecycle_cost_usd=total_lifecycle,
            annualized_cost_usd=annualized,
        )

    def identify_optimization_opportunities(
        self,
        u_current_w_m2k: float,
        u_clean_w_m2k: float,
        u_design_w_m2k: float,
        fouling_rate_m2kw_per_day: float,
        current_cleaning_frequency_days: float,
        heat_transfer_area_m2: float,
        lmtd_c: float,
        operating_hours_per_year: float = TYPICAL_OPERATING_HOURS,
    ) -> List[OptimizationOpportunity]:
        """
        Identify optimization opportunities.

        Args:
            u_current_w_m2k: Current U value
            u_clean_w_m2k: Clean U value
            u_design_w_m2k: Design U value
            fouling_rate_m2kw_per_day: Fouling rate
            current_cleaning_frequency_days: Current cleaning interval
            heat_transfer_area_m2: Heat transfer area
            lmtd_c: Log mean temperature difference
            operating_hours_per_year: Annual operating hours

        Returns:
            List of OptimizationOpportunity objects
        """
        self._calculation_count += 1

        opportunities: List[OptimizationOpportunity] = []

        # 1. Cleaning frequency optimization
        optimal_freq, _ = self._optimize_cleaning_frequency(
            self._calculate_energy_loss(
                u_current_w_m2k, u_clean_w_m2k,
                heat_transfer_area_m2, lmtd_c
            ) * operating_hours_per_year * self.config.energy_cost_usd_per_kwh,
            5000,  # Typical cleaning cost
            fouling_rate_m2kw_per_day,
            u_clean_w_m2k,
            heat_transfer_area_m2,
            lmtd_c,
            operating_hours_per_year,
        )

        if abs(optimal_freq - current_cleaning_frequency_days) > 30:
            # Calculate savings
            current_cleanings = 365 / current_cleaning_frequency_days
            optimal_cleanings = 365 / optimal_freq
            cleaning_cost_diff = (
                (current_cleanings - optimal_cleanings) * 5000
            )

            opportunities.append(OptimizationOpportunity(
                opportunity_type="cleaning_schedule",
                description=(
                    f"Adjust cleaning frequency from {current_cleaning_frequency_days:.0f} "
                    f"to {optimal_freq:.0f} days"
                ),
                annual_savings_usd=abs(cleaning_cost_diff),
                implementation_cost_usd=0,
                payback_years=0,
                priority="high" if abs(cleaning_cost_diff) > 10000 else "medium",
            ))

        # 2. Enhanced cleaning method
        u_recovery_potential = u_clean_w_m2k - u_current_w_m2k
        if u_recovery_potential > u_clean_w_m2k * 0.15:
            energy_savings = (
                u_recovery_potential / u_clean_w_m2k *
                heat_transfer_area_m2 * lmtd_c *
                operating_hours_per_year *
                self.config.energy_cost_usd_per_kwh / 1000
            )

            opportunities.append(OptimizationOpportunity(
                opportunity_type="enhanced_cleaning",
                description=(
                    "Implement enhanced cleaning (chemical + mechanical) "
                    "to recover full heat transfer capacity"
                ),
                annual_savings_usd=energy_savings,
                implementation_cost_usd=15000,
                payback_years=15000 / energy_savings if energy_savings > 0 else 10,
                priority="high" if energy_savings > 20000 else "medium",
            ))

        # 3. Online cleaning system
        if fouling_rate_m2kw_per_day > 0.000002:
            # High fouling rate - consider online cleaning
            annual_fouling_cost = (
                fouling_rate_m2kw_per_day * 365 *
                u_clean_w_m2k ** 2 *
                heat_transfer_area_m2 * lmtd_c *
                self.config.energy_cost_usd_per_kwh / 1000
            )

            opportunities.append(OptimizationOpportunity(
                opportunity_type="online_cleaning",
                description=(
                    "Install online tube cleaning system (e.g., ATCS) "
                    "to reduce fouling rate"
                ),
                annual_savings_usd=annual_fouling_cost * 0.6,  # 60% reduction
                implementation_cost_usd=50000,
                payback_years=50000 / (annual_fouling_cost * 0.6) if annual_fouling_cost > 0 else 10,
                priority="high" if annual_fouling_cost > 50000 else "low",
            ))

        # 4. Water treatment optimization (for cooling water service)
        if fouling_rate_m2kw_per_day > 0.0000005:
            opportunities.append(OptimizationOpportunity(
                opportunity_type="water_treatment",
                description=(
                    "Optimize water treatment program to reduce fouling rate"
                ),
                annual_savings_usd=fouling_rate_m2kw_per_day * 365 * 1e6 * 100,
                implementation_cost_usd=10000,
                payback_years=1.5,
                priority="medium",
            ))

        # 5. Performance monitoring
        opportunities.append(OptimizationOpportunity(
            opportunity_type="monitoring",
            description=(
                "Implement continuous performance monitoring with "
                "automatic alerts for degradation"
            ),
            annual_savings_usd=5000,  # Estimated from early detection
            implementation_cost_usd=15000,
            payback_years=3,
            priority="medium",
        ))

        # Sort by priority and payback
        priority_order = {"high": 0, "medium": 1, "low": 2}
        opportunities.sort(
            key=lambda x: (priority_order[x.priority], x.payback_years)
        )

        return opportunities

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _calculate_energy_loss(
        self,
        u_current: float,
        u_clean: float,
        area_m2: float,
        lmtd_c: float,
    ) -> float:
        """Calculate energy loss due to fouling (kW)."""
        # Q = U * A * LMTD
        # Loss = Q_clean - Q_current
        q_clean = u_clean * area_m2 * lmtd_c / 1000  # kW
        q_current = u_current * area_m2 * lmtd_c / 1000  # kW

        return max(0, q_clean - q_current)

    def _calculate_npv(
        self,
        annual_cost: float,
        years: float,
        discount_rate: float,
    ) -> float:
        """Calculate NPV of annual costs."""
        if discount_rate < 0.001:
            return annual_cost * years

        # Present value of annuity
        factor = (1 - (1 + discount_rate) ** (-years)) / discount_rate
        return annual_cost * factor

    def _calculate_annualized_cost(
        self,
        npv: float,
        years: float,
        discount_rate: float,
    ) -> float:
        """Calculate equivalent annualized cost from NPV."""
        if discount_rate < 0.001 or years < 0.1:
            return npv / max(1, years)

        factor = (1 - (1 + discount_rate) ** (-years)) / discount_rate
        return npv / factor if factor > 0 else npv

    def _calculate_irr(
        self,
        initial_investment: float,
        annual_savings: float,
        years: float,
    ) -> Optional[float]:
        """Calculate Internal Rate of Return using Newton-Raphson."""
        if annual_savings <= 0:
            return None

        # Initial guess
        irr = 0.1

        for _ in range(50):  # Max iterations
            # NPV at current IRR
            npv = -initial_investment
            for year in range(1, int(years) + 1):
                npv += annual_savings / ((1 + irr) ** year)

            # Derivative
            d_npv = 0
            for year in range(1, int(years) + 1):
                d_npv -= year * annual_savings / ((1 + irr) ** (year + 1))

            if abs(d_npv) < 1e-10:
                break

            # Update
            irr_new = irr - npv / d_npv

            if abs(irr_new - irr) < 1e-6:
                return max(0, irr_new)

            irr = irr_new

            # Bounds
            irr = max(-0.5, min(2.0, irr))

        return irr if irr > -0.5 else None

    def _optimize_cleaning_frequency(
        self,
        annual_energy_cost: float,
        cleaning_cost: float,
        fouling_rate: float,
        u_clean: float,
        area_m2: float,
        lmtd_c: float,
        operating_hours: float,
    ) -> Tuple[float, float]:
        """Optimize cleaning frequency to minimize total cost."""
        # Simple optimization by evaluating costs at different intervals
        min_interval = 30
        max_interval = 365
        step = 7

        best_interval = 180
        best_total_cost = float('inf')

        for interval in range(min_interval, max_interval + 1, step):
            # Number of cleanings per year
            cleanings = 365 / interval

            # Annual cleaning cost
            annual_cleaning = cleanings * cleaning_cost

            # Average fouling over interval
            avg_fouling = fouling_rate * interval / 2

            # Average U during interval
            u_avg = 1 / ((1 / u_clean) + avg_fouling)

            # Energy loss cost
            energy_loss = (
                (u_clean - u_avg) * area_m2 * lmtd_c *
                operating_hours * self.config.energy_cost_usd_per_kwh / 1000
            )

            # Total cost
            total = annual_cleaning + energy_loss

            if total < best_total_cost:
                best_total_cost = total
                best_interval = interval

        # Calculate ROI
        if annual_energy_cost > 0:
            roi = (annual_energy_cost - best_total_cost) / cleaning_cost * 100
        else:
            roi = 0

        return best_interval, roi

    def _analyze_replacement(
        self,
        annual_energy_cost: float,
        annual_maintenance_cost: float,
        remaining_life: float,
    ) -> ReplacementAnalysis:
        """Simplified replacement analysis."""
        return self.analyze_replacement_economics(
            annual_operating_cost_usd=annual_energy_cost,
            annual_maintenance_cost_usd=annual_maintenance_cost,
            remaining_life_years=remaining_life,
        )

    def _identify_optimization_opportunities(
        self,
        annual_energy_cost: float,
        u_current: float,
        u_clean: float,
        fouling_rate: float,
    ) -> List[OptimizationOpportunity]:
        """Identify basic optimization opportunities."""
        opportunities = []

        # Cleaning opportunity
        u_loss_percent = (u_clean - u_current) / u_clean * 100
        if u_loss_percent > 10:
            savings = annual_energy_cost * u_loss_percent / 100
            opportunities.append(OptimizationOpportunity(
                opportunity_type="cleaning",
                description=f"Clean exchanger to recover {u_loss_percent:.0f}% capacity",
                annual_savings_usd=savings,
                implementation_cost_usd=5000,
                payback_years=5000 / savings if savings > 0 else 10,
                priority="high" if u_loss_percent > 20 else "medium",
            ))

        return opportunities

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count
