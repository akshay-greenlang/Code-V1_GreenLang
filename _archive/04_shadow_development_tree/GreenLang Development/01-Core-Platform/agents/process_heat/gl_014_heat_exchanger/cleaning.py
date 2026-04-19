# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Agent - Cleaning Schedule Optimization

This module implements cleaning schedule optimization for heat exchangers,
balancing fouling losses against cleaning costs and production impacts.

Optimization Approaches:
    - Economic optimization (minimize total cost)
    - Availability optimization (maximize uptime)
    - Performance optimization (maintain effectiveness threshold)
    - Hybrid optimization (weighted objectives)

References:
    - Smaili et al., "Optimal Cleaning Schedule of Heat Exchangers"
    - Ishiyama et al., "Targeting Fouling Mitigation"
    - HTRI Guidelines for Exchanger Cleaning

Example:
    >>> from greenlang.agents.process_heat.gl_014_heat_exchanger.cleaning import (
    ...     CleaningScheduleOptimizer
    ... )
    >>> optimizer = CleaningScheduleOptimizer(config.cleaning)
    >>> schedule = optimizer.optimize_schedule(fouling_data, economics)
    >>> print(f"Optimal cleaning interval: {schedule.optimal_interval_days} days")
"""

import math
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_014_heat_exchanger.config import (
    CleaningConfig,
    CleaningMethod,
    EconomicsConfig,
    FoulingConfig,
)
from greenlang.agents.process_heat.gl_014_heat_exchanger.schemas import (
    AlertSeverity,
    CleaningRecommendation,
    CleaningRecord,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Cleaning method characteristics
CLEANING_METHODS = {
    CleaningMethod.MECHANICAL_BRUSHING: {
        "effectiveness": 0.85,
        "duration_hours": 8,
        "cost_multiplier": 1.0,
        "tube_safe": True,
        "shell_applicable": False,
        "fouling_types": ["particulate", "biological"],
    },
    CleaningMethod.MECHANICAL_RODDING: {
        "effectiveness": 0.90,
        "duration_hours": 12,
        "cost_multiplier": 1.2,
        "tube_safe": True,
        "shell_applicable": False,
        "fouling_types": ["particulate", "crystallization"],
    },
    CleaningMethod.HIGH_PRESSURE_WATER: {
        "effectiveness": 0.95,
        "duration_hours": 6,
        "cost_multiplier": 0.8,
        "tube_safe": True,
        "shell_applicable": True,
        "fouling_types": ["particulate", "biological", "light_scale"],
    },
    CleaningMethod.CHEMICAL_ACID: {
        "effectiveness": 0.95,
        "duration_hours": 16,
        "cost_multiplier": 2.0,
        "tube_safe": True,
        "shell_applicable": True,
        "fouling_types": ["crystallization", "scale", "rust"],
    },
    CleaningMethod.CHEMICAL_ALKALINE: {
        "effectiveness": 0.90,
        "duration_hours": 12,
        "cost_multiplier": 1.8,
        "tube_safe": True,
        "shell_applicable": True,
        "fouling_types": ["biological", "organic", "oil"],
    },
    CleaningMethod.CHEMICAL_SOLVENT: {
        "effectiveness": 0.92,
        "duration_hours": 10,
        "cost_multiplier": 2.5,
        "tube_safe": True,
        "shell_applicable": True,
        "fouling_types": ["coking", "polymer", "heavy_organic"],
    },
    CleaningMethod.PIGGING: {
        "effectiveness": 0.85,
        "duration_hours": 4,
        "cost_multiplier": 0.6,
        "tube_safe": True,
        "shell_applicable": False,
        "fouling_types": ["particulate", "soft_deposit"],
    },
    CleaningMethod.STEAM_BLOWING: {
        "effectiveness": 0.75,
        "duration_hours": 4,
        "cost_multiplier": 0.5,
        "tube_safe": True,
        "shell_applicable": True,
        "fouling_types": ["light_organic", "volatile"],
    },
    CleaningMethod.ULTRASONIC: {
        "effectiveness": 0.88,
        "duration_hours": 6,
        "cost_multiplier": 1.5,
        "tube_safe": False,
        "shell_applicable": True,
        "fouling_types": ["particulate", "light_scale"],
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CleaningCostBreakdown:
    """Breakdown of cleaning costs."""

    direct_cleaning_cost_usd: float
    production_loss_cost_usd: float
    labor_cost_usd: float
    materials_cost_usd: float
    waste_disposal_cost_usd: float
    total_cost_usd: float


@dataclass
class OptimalScheduleResult:
    """Result of cleaning schedule optimization."""

    optimal_interval_days: float
    minimum_annual_cost_usd: float
    next_cleaning_date: datetime
    cleaning_method: CleaningMethod
    expected_fouling_at_cleaning: float
    expected_effectiveness_at_cleaning: float
    confidence: float
    sensitivity_analysis: Dict[str, float]


@dataclass
class CleaningScenario:
    """A single cleaning scenario for analysis."""

    interval_days: int
    annual_cleanings: float
    annual_cleaning_cost_usd: float
    annual_fouling_loss_usd: float
    annual_production_loss_usd: float
    total_annual_cost_usd: float
    average_effectiveness: float


# =============================================================================
# CLEANING SCHEDULE OPTIMIZER
# =============================================================================

class CleaningScheduleOptimizer:
    """
    Cleaning schedule optimizer for heat exchangers.

    This class optimizes cleaning schedules by balancing:
    - Fouling-induced energy losses
    - Cleaning costs (direct and indirect)
    - Production losses during cleaning
    - Equipment degradation

    The optimization uses deterministic economic models with
    zero hallucination guarantee.

    Attributes:
        config: Cleaning configuration
        economics: Economics configuration
        fouling_config: Fouling configuration

    Example:
        >>> optimizer = CleaningScheduleOptimizer(
        ...     cleaning_config, economics_config, fouling_config
        ... )
        >>> result = optimizer.optimize_schedule(
        ...     current_fouling=0.0003,
        ...     fouling_rate=0.000001,
        ...     clean_u=500,
        ... )
    """

    def __init__(
        self,
        cleaning_config: CleaningConfig,
        economics_config: Optional[EconomicsConfig] = None,
        fouling_config: Optional[FoulingConfig] = None,
    ) -> None:
        """
        Initialize the cleaning schedule optimizer.

        Args:
            cleaning_config: Cleaning configuration
            economics_config: Economics configuration
            fouling_config: Fouling configuration
        """
        self.config = cleaning_config
        self.economics = economics_config or EconomicsConfig()
        self.fouling_config = fouling_config or FoulingConfig()
        self._calculation_count = 0

        logger.info("CleaningScheduleOptimizer initialized")

    def optimize_schedule(
        self,
        current_fouling_m2kw: float,
        fouling_rate_m2kw_per_day: float,
        clean_u_w_m2k: float,
        area_m2: float,
        operating_hours_per_year: float = 8000,
        days_since_last_cleaning: Optional[float] = None,
    ) -> OptimalScheduleResult:
        """
        Optimize the cleaning schedule.

        Uses economic optimization to find the interval that minimizes
        total annual cost (cleaning + fouling losses + production loss).

        Args:
            current_fouling_m2kw: Current fouling resistance
            fouling_rate_m2kw_per_day: Fouling rate
            clean_u_w_m2k: Clean U value
            area_m2: Heat transfer area
            operating_hours_per_year: Annual operating hours
            days_since_last_cleaning: Days since last cleaning

        Returns:
            OptimalScheduleResult with optimal schedule parameters
        """
        self._calculation_count += 1

        # Evaluate scenarios across range of intervals
        scenarios: List[CleaningScenario] = []

        min_interval = self.config.minimum_interval_days
        max_interval = self.config.maximum_interval_days
        step = max(1, (max_interval - min_interval) // 50)

        for interval in range(min_interval, max_interval + 1, step):
            scenario = self._evaluate_scenario(
                interval_days=interval,
                fouling_rate=fouling_rate_m2kw_per_day,
                clean_u=clean_u_w_m2k,
                area_m2=area_m2,
                operating_hours=operating_hours_per_year,
            )
            scenarios.append(scenario)

        # Find optimal scenario
        optimal = min(scenarios, key=lambda s: s.total_annual_cost_usd)

        # Calculate next cleaning date
        if days_since_last_cleaning is not None:
            days_until_cleaning = max(
                0,
                optimal.interval_days - days_since_last_cleaning
            )
        else:
            # Estimate based on current fouling
            if fouling_rate_m2kw_per_day > 0:
                threshold = self.config.fouling_threshold_m2kw
                days_until_cleaning = max(
                    0,
                    (threshold - current_fouling_m2kw) / fouling_rate_m2kw_per_day
                )
            else:
                days_until_cleaning = optimal.interval_days

        next_cleaning = datetime.now(timezone.utc) + timedelta(days=days_until_cleaning)

        # Select best cleaning method
        best_method = self._select_cleaning_method()

        # Expected fouling at cleaning time
        expected_fouling = current_fouling_m2kw + (
            fouling_rate_m2kw_per_day * days_until_cleaning
        )

        # Expected effectiveness at cleaning
        u_at_cleaning = 1.0 / ((1.0 / clean_u_w_m2k) + expected_fouling)
        expected_effectiveness = u_at_cleaning / clean_u_w_m2k

        # Sensitivity analysis
        sensitivity = self._sensitivity_analysis(
            optimal.interval_days,
            fouling_rate_m2kw_per_day,
            clean_u_w_m2k,
            area_m2,
            operating_hours_per_year,
        )

        return OptimalScheduleResult(
            optimal_interval_days=optimal.interval_days,
            minimum_annual_cost_usd=optimal.total_annual_cost_usd,
            next_cleaning_date=next_cleaning,
            cleaning_method=best_method,
            expected_fouling_at_cleaning=expected_fouling,
            expected_effectiveness_at_cleaning=expected_effectiveness,
            confidence=0.85,
            sensitivity_analysis=sensitivity,
        )

    def generate_recommendation(
        self,
        current_fouling_m2kw: float,
        fouling_rate_m2kw_per_day: float,
        current_effectiveness: float,
        clean_u_w_m2k: float,
        area_m2: float,
        days_since_last_cleaning: Optional[float] = None,
    ) -> CleaningRecommendation:
        """
        Generate a cleaning recommendation.

        Args:
            current_fouling_m2kw: Current fouling resistance
            fouling_rate_m2kw_per_day: Fouling rate
            current_effectiveness: Current thermal effectiveness
            clean_u_w_m2k: Clean U value
            area_m2: Heat transfer area
            days_since_last_cleaning: Days since last cleaning

        Returns:
            CleaningRecommendation with complete recommendation
        """
        self._calculation_count += 1

        # Check immediate need based on thresholds
        fouling_exceeded = current_fouling_m2kw > self.config.fouling_threshold_m2kw
        effectiveness_low = current_effectiveness < self.config.effectiveness_threshold

        # Determine urgency
        if fouling_exceeded or effectiveness_low:
            if current_effectiveness < 0.6:
                urgency = AlertSeverity.CRITICAL
            elif current_effectiveness < 0.7:
                urgency = AlertSeverity.ALARM
            else:
                urgency = AlertSeverity.WARNING
            recommended = True
        else:
            urgency = AlertSeverity.INFO
            recommended = False

        # Optimize schedule for timing
        schedule = self.optimize_schedule(
            current_fouling_m2kw=current_fouling_m2kw,
            fouling_rate_m2kw_per_day=fouling_rate_m2kw_per_day,
            clean_u_w_m2k=clean_u_w_m2k,
            area_m2=area_m2,
            days_since_last_cleaning=days_since_last_cleaning,
        )

        # Select cleaning method
        recommended_method = self._select_cleaning_method()
        method_info = CLEANING_METHODS[recommended_method]

        # Calculate costs
        cleaning_cost = self.config.cleaning_cost_usd * method_info["cost_multiplier"]
        production_loss = (
            self.config.production_loss_usd_per_hour *
            method_info["duration_hours"]
        )

        # Calculate expected results
        expected_u_after = clean_u_w_m2k * method_info["effectiveness"]
        expected_effectiveness_after = method_info["effectiveness"]

        # Calculate energy savings
        energy_loss_kw = self._calculate_energy_loss(
            clean_u_w_m2k,
            current_fouling_m2kw,
            area_m2,
            30.0,  # Assumed LMTD
        )
        energy_savings_monthly = (
            energy_loss_kw * 24 * 30 *
            self.economics.energy_cost_usd_per_kwh
        )

        # NPV of cleaning
        npv = self._calculate_cleaning_npv(
            cleaning_cost=cleaning_cost + production_loss,
            monthly_savings=energy_savings_monthly,
            months_ahead=12,
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            current_fouling_m2kw,
            current_effectiveness,
            fouling_rate_m2kw_per_day,
            schedule,
        )

        # Days until recommended cleaning
        days_until = (
            schedule.next_cleaning_date - datetime.now(timezone.utc)
        ).total_seconds() / 86400

        # Latest acceptable date (before critical threshold)
        critical_threshold = self.config.fouling_threshold_m2kw * 2.0
        if fouling_rate_m2kw_per_day > 0:
            days_to_critical = (
                (critical_threshold - current_fouling_m2kw) /
                fouling_rate_m2kw_per_day
            )
            latest_date = datetime.now(timezone.utc) + timedelta(
                days=max(0, days_to_critical)
            )
        else:
            latest_date = None

        return CleaningRecommendation(
            recommended=recommended or days_until < 30,
            urgency=urgency,
            recommended_method=recommended_method,
            alternative_methods=self._get_alternative_methods(recommended_method),
            optimal_cleaning_date=schedule.next_cleaning_date,
            latest_cleaning_date=latest_date,
            days_until_recommended=max(0, days_until),
            estimated_cleaning_cost_usd=cleaning_cost,
            production_loss_usd=production_loss,
            energy_savings_potential_usd_per_month=energy_savings_monthly,
            npv_of_cleaning_usd=npv,
            expected_u_after_w_m2k=expected_u_after,
            expected_effectiveness_after=expected_effectiveness_after,
            expected_fouling_removal_percent=method_info["effectiveness"] * 100,
            reasoning=reasoning,
        )

    def analyze_cleaning_history(
        self,
        cleaning_records: List[CleaningRecord],
    ) -> Dict[str, Any]:
        """
        Analyze historical cleaning records.

        Args:
            cleaning_records: List of historical cleaning records

        Returns:
            Analysis results dictionary
        """
        if not cleaning_records:
            return {"error": "No cleaning records provided"}

        self._calculation_count += 1

        # Sort by date
        records = sorted(cleaning_records, key=lambda r: r.cleaning_date)

        # Calculate intervals
        intervals = []
        for i in range(1, len(records)):
            interval = (
                records[i].cleaning_date - records[i-1].cleaning_date
            ).days
            intervals.append(interval)

        # Calculate effectiveness recovery
        recoveries = []
        for record in records:
            recovery = (
                (record.effectiveness_after - record.effectiveness_before) /
                (1.0 - record.effectiveness_before)
                if record.effectiveness_before < 1.0 else 0.0
            )
            recoveries.append(recovery)

        # U value improvement
        u_improvements = [r.u_improvement_percent for r in records]

        # Costs
        total_cost = sum(r.cost_usd for r in records)
        avg_cost = total_cost / len(records) if records else 0

        # Method effectiveness
        method_performance: Dict[str, List[float]] = {}
        for record in records:
            method = record.cleaning_method.value
            if method not in method_performance:
                method_performance[method] = []
            method_performance[method].append(record.u_improvement_percent)

        method_avg = {
            m: sum(v) / len(v) if v else 0
            for m, v in method_performance.items()
        }

        return {
            "total_cleanings": len(records),
            "average_interval_days": (
                sum(intervals) / len(intervals) if intervals else 0
            ),
            "min_interval_days": min(intervals) if intervals else 0,
            "max_interval_days": max(intervals) if intervals else 0,
            "average_effectiveness_recovery": (
                sum(recoveries) / len(recoveries) if recoveries else 0
            ),
            "average_u_improvement_percent": (
                sum(u_improvements) / len(u_improvements)
                if u_improvements else 0
            ),
            "total_cleaning_cost_usd": total_cost,
            "average_cleaning_cost_usd": avg_cost,
            "method_performance": method_avg,
            "most_effective_method": (
                max(method_avg.items(), key=lambda x: x[1])[0]
                if method_avg else None
            ),
        }

    def calculate_cleaning_cost(
        self,
        method: CleaningMethod,
        duration_hours: Optional[float] = None,
    ) -> CleaningCostBreakdown:
        """
        Calculate detailed cleaning cost breakdown.

        Args:
            method: Cleaning method
            duration_hours: Override duration (uses default if None)

        Returns:
            CleaningCostBreakdown with detailed costs
        """
        method_info = CLEANING_METHODS.get(
            method,
            CLEANING_METHODS[CleaningMethod.HIGH_PRESSURE_WATER]
        )

        if duration_hours is None:
            duration_hours = method_info["duration_hours"]

        # Base cleaning cost
        base_cost = self.config.cleaning_cost_usd
        direct_cost = base_cost * method_info["cost_multiplier"]

        # Production loss
        production_loss = (
            self.config.production_loss_usd_per_hour * duration_hours
        )

        # Labor cost (estimated as 30% of direct cost)
        labor_cost = direct_cost * 0.30

        # Materials cost (estimated as 40% of direct cost)
        materials_cost = direct_cost * 0.40

        # Waste disposal (estimated as 10% of direct cost for chemical methods)
        if method in [
            CleaningMethod.CHEMICAL_ACID,
            CleaningMethod.CHEMICAL_ALKALINE,
            CleaningMethod.CHEMICAL_SOLVENT,
        ]:
            waste_disposal = direct_cost * 0.10
        else:
            waste_disposal = direct_cost * 0.02

        total = (
            direct_cost + production_loss + labor_cost +
            materials_cost + waste_disposal
        )

        return CleaningCostBreakdown(
            direct_cleaning_cost_usd=direct_cost,
            production_loss_cost_usd=production_loss,
            labor_cost_usd=labor_cost,
            materials_cost_usd=materials_cost,
            waste_disposal_cost_usd=waste_disposal,
            total_cost_usd=total,
        )

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _evaluate_scenario(
        self,
        interval_days: int,
        fouling_rate: float,
        clean_u: float,
        area_m2: float,
        operating_hours: float,
    ) -> CleaningScenario:
        """Evaluate a cleaning scenario."""
        # Number of cleanings per year
        annual_cleanings = 365.0 / interval_days

        # Cleaning cost per cleaning
        method = self._select_cleaning_method()
        method_info = CLEANING_METHODS[method]

        cleaning_cost_per_event = (
            self.config.cleaning_cost_usd * method_info["cost_multiplier"] +
            self.config.production_loss_usd_per_hour * method_info["duration_hours"]
        )
        annual_cleaning_cost = cleaning_cost_per_event * annual_cleanings

        # Average fouling over interval
        max_fouling = fouling_rate * interval_days
        avg_fouling = max_fouling / 2  # Linear assumption

        # Average effectiveness
        u_avg = 1.0 / ((1.0 / clean_u) + avg_fouling)
        avg_effectiveness = u_avg / clean_u

        # Fouling energy loss (annual)
        energy_loss_kw = self._calculate_energy_loss(
            clean_u, avg_fouling, area_m2, 30.0
        )
        hours_per_day = operating_hours / 365
        annual_fouling_loss = (
            energy_loss_kw * operating_hours *
            self.economics.energy_cost_usd_per_kwh
        )

        # Production loss from cleaning downtime
        cleaning_hours = method_info["duration_hours"] * annual_cleanings
        annual_production_loss = (
            self.economics.production_value_usd_per_hour * cleaning_hours
        )

        # Total annual cost
        total_annual_cost = (
            annual_cleaning_cost +
            annual_fouling_loss +
            annual_production_loss
        )

        return CleaningScenario(
            interval_days=interval_days,
            annual_cleanings=annual_cleanings,
            annual_cleaning_cost_usd=annual_cleaning_cost,
            annual_fouling_loss_usd=annual_fouling_loss,
            annual_production_loss_usd=annual_production_loss,
            total_annual_cost_usd=total_annual_cost,
            average_effectiveness=avg_effectiveness,
        )

    def _select_cleaning_method(self) -> CleaningMethod:
        """Select the best cleaning method based on configuration."""
        if self.config.preferred_methods:
            return self.config.preferred_methods[0]
        return CleaningMethod.HIGH_PRESSURE_WATER

    def _get_alternative_methods(
        self,
        primary: CleaningMethod,
    ) -> List[CleaningMethod]:
        """Get alternative cleaning methods."""
        alternatives = []

        for method in self.config.preferred_methods[1:]:
            if method != primary:
                alternatives.append(method)

        if not alternatives:
            # Default alternatives
            if primary != CleaningMethod.HIGH_PRESSURE_WATER:
                alternatives.append(CleaningMethod.HIGH_PRESSURE_WATER)
            if primary != CleaningMethod.CHEMICAL_ACID:
                alternatives.append(CleaningMethod.CHEMICAL_ACID)

        return alternatives[:3]

    def _calculate_energy_loss(
        self,
        clean_u: float,
        fouling: float,
        area_m2: float,
        lmtd_c: float,
    ) -> float:
        """Calculate energy loss due to fouling (kW)."""
        # U with fouling
        u_fouled = 1.0 / ((1.0 / clean_u) + fouling)

        # Duty loss
        duty_clean = clean_u * area_m2 * lmtd_c / 1000  # kW
        duty_fouled = u_fouled * area_m2 * lmtd_c / 1000  # kW

        return duty_clean - duty_fouled

    def _calculate_cleaning_npv(
        self,
        cleaning_cost: float,
        monthly_savings: float,
        months_ahead: int,
    ) -> float:
        """Calculate NPV of cleaning."""
        discount_rate_monthly = self.economics.discount_rate / 12

        npv = -cleaning_cost

        for month in range(1, months_ahead + 1):
            discount_factor = (1 + discount_rate_monthly) ** month
            npv += monthly_savings / discount_factor

        return npv

    def _sensitivity_analysis(
        self,
        optimal_interval: int,
        fouling_rate: float,
        clean_u: float,
        area_m2: float,
        operating_hours: float,
    ) -> Dict[str, float]:
        """Perform sensitivity analysis on key parameters."""
        sensitivity = {}

        # Baseline cost
        baseline = self._evaluate_scenario(
            optimal_interval, fouling_rate, clean_u, area_m2, operating_hours
        )

        # Fouling rate +10%
        scenario_fr_up = self._evaluate_scenario(
            optimal_interval, fouling_rate * 1.1, clean_u, area_m2, operating_hours
        )
        sensitivity["fouling_rate_+10%"] = (
            (scenario_fr_up.total_annual_cost_usd - baseline.total_annual_cost_usd)
            / baseline.total_annual_cost_usd * 100
        )

        # Cleaning cost +10%
        original_cost = self.config.cleaning_cost_usd
        self.config.cleaning_cost_usd = original_cost * 1.1
        scenario_cc_up = self._evaluate_scenario(
            optimal_interval, fouling_rate, clean_u, area_m2, operating_hours
        )
        self.config.cleaning_cost_usd = original_cost
        sensitivity["cleaning_cost_+10%"] = (
            (scenario_cc_up.total_annual_cost_usd - baseline.total_annual_cost_usd)
            / baseline.total_annual_cost_usd * 100
        )

        # Energy price +10%
        original_energy = self.economics.energy_cost_usd_per_kwh
        self.economics.energy_cost_usd_per_kwh = original_energy * 1.1
        scenario_ep_up = self._evaluate_scenario(
            optimal_interval, fouling_rate, clean_u, area_m2, operating_hours
        )
        self.economics.energy_cost_usd_per_kwh = original_energy
        sensitivity["energy_price_+10%"] = (
            (scenario_ep_up.total_annual_cost_usd - baseline.total_annual_cost_usd)
            / baseline.total_annual_cost_usd * 100
        )

        return sensitivity

    def _generate_reasoning(
        self,
        current_fouling: float,
        current_effectiveness: float,
        fouling_rate: float,
        schedule: OptimalScheduleResult,
    ) -> str:
        """Generate human-readable reasoning for recommendation."""
        parts = []

        # Current state
        parts.append(
            f"Current fouling resistance is {current_fouling*1e6:.1f} m2K/MW "
            f"with thermal effectiveness at {current_effectiveness:.1%}."
        )

        # Fouling rate
        if fouling_rate > 0:
            days_to_threshold = (
                (self.config.fouling_threshold_m2kw - current_fouling) /
                fouling_rate
            )
            parts.append(
                f"At the current fouling rate of {fouling_rate*1e6:.2f} m2K/MW per day, "
                f"the cleaning threshold will be reached in approximately {days_to_threshold:.0f} days."
            )
        else:
            parts.append("Fouling rate is stable or decreasing.")

        # Optimal interval
        parts.append(
            f"Economic optimization suggests a cleaning interval of "
            f"{schedule.optimal_interval_days:.0f} days to minimize total costs."
        )

        # Next cleaning
        days_until = (
            schedule.next_cleaning_date - datetime.now(timezone.utc)
        ).days
        if days_until <= 0:
            parts.append("Cleaning is due immediately.")
        elif days_until < 30:
            parts.append(f"Next cleaning recommended within {days_until} days.")
        else:
            parts.append(
                f"Next cleaning scheduled for "
                f"{schedule.next_cleaning_date.strftime('%Y-%m-%d')}."
            )

        return " ".join(parts)

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count
