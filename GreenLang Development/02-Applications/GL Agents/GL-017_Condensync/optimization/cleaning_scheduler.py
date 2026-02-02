# -*- coding: utf-8 -*-
"""
Cleaning Schedule Optimizer for GL-017 CONDENSYNC

Predictive cleaning schedule optimization for condenser tube maintenance.
Uses cleanliness factor degradation modeling to determine optimal cleaning timing.

Key Features:
    - Predict CF(t) over 7-30 days with uncertainty quantification
    - Estimate energy/fuel loss from CF degradation
    - Calculate cleaning ROI: savings - (cleaning cost + downtime)
    - Select optimal cleaning date (maximize NPV)
    - Generate CMMS recommendation package

Zero-Hallucination Guarantee:
    - Deterministic degradation models (exponential, linear)
    - Physics-based energy loss calculations
    - No AI/ML inference in financial calculations
    - Complete audit trail with SHA-256 provenance

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Default cleaning parameters
DEFAULT_CLEANING_DURATION_HOURS = 8.0
DEFAULT_CLEANING_COST_USD = 15000.0
DEFAULT_CF_AFTER_CLEANING = 0.95
DEFAULT_DISCOUNT_RATE = 0.08  # 8% annual discount rate

# Default degradation parameters
DEFAULT_FOULING_RATE_PER_DAY = 0.002  # 0.2% CF loss per day


# ============================================================================
# ENUMERATIONS
# ============================================================================

class DegradationModel(Enum):
    """Cleanliness factor degradation model types."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    ASYMPTOTIC = "asymptotic"
    CUSTOM = "custom"


class CleaningMethod(Enum):
    """Condenser cleaning methods."""
    BALL_SYSTEM = "ball_system"  # Online ball cleaning
    BACKWASH = "backwash"  # Backwash flush
    CHEMICAL = "chemical"  # Chemical cleaning
    MECHANICAL = "mechanical"  # Mechanical brush/scrape
    HYDROBLAST = "hydroblast"  # High-pressure water
    OFFLINE_FULL = "offline_full"  # Full offline cleaning


class RecommendationUrgency(Enum):
    """Urgency level for cleaning recommendation."""
    IMMEDIATE = "immediate"  # Clean within 24 hours
    HIGH = "high"  # Clean within 3 days
    MEDIUM = "medium"  # Clean within 7 days
    LOW = "low"  # Clean within 14 days
    SCHEDULED = "scheduled"  # Normal scheduled maintenance


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CleaningCosts:
    """Costs associated with condenser cleaning."""
    labor_cost_usd: float = 5000.0
    materials_cost_usd: float = 2000.0
    chemical_cost_usd: float = 0.0  # If chemical cleaning
    contractor_cost_usd: float = 0.0  # If outsourced
    equipment_rental_usd: float = 0.0
    disposal_cost_usd: float = 500.0

    @property
    def total_direct_cost_usd(self) -> float:
        """Calculate total direct cleaning cost."""
        return (
            self.labor_cost_usd +
            self.materials_cost_usd +
            self.chemical_cost_usd +
            self.contractor_cost_usd +
            self.equipment_rental_usd +
            self.disposal_cost_usd
        )


@dataclass
class DowntimeCosts:
    """Costs associated with cleaning downtime."""
    duration_hours: float = DEFAULT_CLEANING_DURATION_HOURS
    generation_capacity_mw: float = 500.0
    capacity_factor_during_cleaning: float = 0.0  # Assume offline
    power_price_usd_mwh: float = 40.0
    fixed_om_cost_usd_hr: float = 100.0

    @property
    def lost_generation_mwh(self) -> float:
        """Calculate lost generation during cleaning."""
        return self.generation_capacity_mw * self.duration_hours * (
            1 - self.capacity_factor_during_cleaning
        )

    @property
    def lost_revenue_usd(self) -> float:
        """Calculate lost revenue during cleaning."""
        return self.lost_generation_mwh * self.power_price_usd_mwh

    @property
    def total_downtime_cost_usd(self) -> float:
        """Calculate total downtime cost."""
        return self.lost_revenue_usd + (self.fixed_om_cost_usd_hr * self.duration_hours)


@dataclass
class CleaningParameters:
    """Parameters for cleaning optimization."""
    cleaning_method: CleaningMethod = CleaningMethod.OFFLINE_FULL
    cleaning_costs: CleaningCosts = field(default_factory=CleaningCosts)
    downtime_costs: DowntimeCosts = field(default_factory=DowntimeCosts)
    cf_after_cleaning: float = DEFAULT_CF_AFTER_CLEANING
    min_cf_threshold: float = 0.70  # Mandatory cleaning below this CF
    target_cf: float = 0.85  # Target CF for normal operation


@dataclass
class DegradationParameters:
    """Parameters for CF degradation modeling."""
    model_type: DegradationModel = DegradationModel.EXPONENTIAL
    fouling_rate_per_day: float = DEFAULT_FOULING_RATE_PER_DAY
    asymptotic_cf_floor: float = 0.50  # Minimum CF for asymptotic model
    seasonality_factor: float = 0.0  # Additional summer fouling
    uncertainty_std_per_day: float = 0.001  # Uncertainty growth


@dataclass
class EnergyParameters:
    """Parameters for energy loss calculations."""
    base_heat_rate_btu_kwh: float = 9500.0
    heat_rate_sensitivity_pct_per_inhga: float = 1.5  # % increase per inHgA
    fuel_cost_per_mmbtu: float = 3.50
    base_generation_mw: float = 500.0
    capacity_factor: float = 0.85


@dataclass
class SchedulerConfig:
    """Configuration for cleaning scheduler."""
    planning_horizon_days: int = 30
    min_days_to_clean: int = 1
    max_days_to_clean: int = 30
    discount_rate_annual: float = DEFAULT_DISCOUNT_RATE
    num_monte_carlo_samples: int = 1000
    confidence_level: float = 0.90


@dataclass
class CFPrediction:
    """Cleanliness factor prediction for a specific date."""
    date: datetime
    days_from_now: int
    predicted_cf: float
    cf_lower_bound: float  # Lower confidence bound
    cf_upper_bound: float  # Upper confidence bound
    uncertainty_std: float


@dataclass
class EnergyLossEstimate:
    """Estimated energy/fuel loss from CF degradation."""
    date: datetime
    cf_value: float
    backpressure_increase_inhga: float
    heat_rate_increase_pct: float
    fuel_penalty_usd_hr: float
    cumulative_loss_usd: float


@dataclass
class CleaningROI:
    """ROI calculation for cleaning on a specific date."""
    cleaning_date: datetime
    days_until_cleaning: int

    # Costs
    direct_cleaning_cost_usd: float
    downtime_cost_usd: float
    total_cleaning_cost_usd: float

    # Benefits
    cumulative_energy_loss_if_not_cleaned_usd: float
    avoided_energy_loss_usd: float
    backpressure_improvement_inhga: float

    # ROI metrics
    net_savings_usd: float
    roi_pct: float
    npv_usd: float
    payback_days: float


@dataclass
class CleaningRecommendation:
    """Complete cleaning recommendation package."""
    recommendation_id: str
    timestamp: datetime

    # Optimal timing
    optimal_cleaning_date: datetime
    days_until_optimal: int
    urgency: RecommendationUrgency

    # ROI at optimal date
    optimal_roi: CleaningROI

    # Alternative dates
    alternative_dates: List[CleaningROI]

    # Current state
    current_cf: float
    predicted_cf_at_cleaning: float

    # CMMS integration data
    cmms_work_order_data: Dict[str, Any]

    # Provenance
    provenance_hash: str
    processing_time_ms: float


# ============================================================================
# DEGRADATION MODEL CLASS
# ============================================================================

class CFDegradationModel:
    """
    Cleanliness Factor degradation prediction model.

    Supports multiple degradation model types:
    - Linear: CF(t) = CF0 - rate * t
    - Exponential: CF(t) = CF0 * exp(-rate * t)
    - Asymptotic: CF(t) = CF_floor + (CF0 - CF_floor) * exp(-rate * t)

    Zero-Hallucination Guarantee:
        All predictions use deterministic mathematical formulas.
        No AI/ML models used for prediction.
    """

    def __init__(self, params: DegradationParameters):
        """
        Initialize degradation model.

        Args:
            params: Degradation model parameters
        """
        self.params = params

    def predict_cf(
        self,
        current_cf: float,
        days_ahead: int,
        current_date: Optional[datetime] = None
    ) -> CFPrediction:
        """
        Predict CF at a future date.

        Args:
            current_cf: Current cleanliness factor
            days_ahead: Number of days to predict ahead
            current_date: Current date (default: now)

        Returns:
            CFPrediction with point estimate and bounds
        """
        if current_date is None:
            current_date = datetime.now(timezone.utc)

        target_date = current_date + timedelta(days=days_ahead)

        # Apply seasonality adjustment
        seasonality_adjustment = self._calculate_seasonality(current_date, days_ahead)

        # Calculate predicted CF based on model type
        if self.params.model_type == DegradationModel.LINEAR:
            rate = self.params.fouling_rate_per_day * (1 + seasonality_adjustment)
            predicted_cf = current_cf - rate * days_ahead

        elif self.params.model_type == DegradationModel.EXPONENTIAL:
            rate = self.params.fouling_rate_per_day * (1 + seasonality_adjustment)
            predicted_cf = current_cf * np.exp(-rate * days_ahead)

        elif self.params.model_type == DegradationModel.ASYMPTOTIC:
            rate = self.params.fouling_rate_per_day * (1 + seasonality_adjustment)
            cf_floor = self.params.asymptotic_cf_floor
            predicted_cf = cf_floor + (current_cf - cf_floor) * np.exp(-rate * days_ahead)

        else:
            # Default to linear
            rate = self.params.fouling_rate_per_day
            predicted_cf = current_cf - rate * days_ahead

        # Ensure CF stays in valid range
        predicted_cf = max(0.0, min(1.0, predicted_cf))

        # Calculate uncertainty bounds
        uncertainty_std = self.params.uncertainty_std_per_day * np.sqrt(days_ahead)

        # 90% confidence interval (1.645 standard deviations)
        z_score = 1.645
        cf_lower = max(0.0, predicted_cf - z_score * uncertainty_std)
        cf_upper = min(1.0, predicted_cf + z_score * uncertainty_std)

        return CFPrediction(
            date=target_date,
            days_from_now=days_ahead,
            predicted_cf=predicted_cf,
            cf_lower_bound=cf_lower,
            cf_upper_bound=cf_upper,
            uncertainty_std=uncertainty_std
        )

    def predict_cf_trajectory(
        self,
        current_cf: float,
        horizon_days: int,
        current_date: Optional[datetime] = None
    ) -> List[CFPrediction]:
        """
        Predict CF trajectory over planning horizon.

        Args:
            current_cf: Current cleanliness factor
            horizon_days: Planning horizon in days
            current_date: Current date (default: now)

        Returns:
            List of CFPrediction for each day
        """
        if current_date is None:
            current_date = datetime.now(timezone.utc)

        predictions = []
        for day in range(1, horizon_days + 1):
            pred = self.predict_cf(current_cf, day, current_date)
            predictions.append(pred)

        return predictions

    def _calculate_seasonality(
        self,
        start_date: datetime,
        days_ahead: int
    ) -> float:
        """
        Calculate seasonality adjustment for fouling rate.

        Higher fouling in summer months due to biological growth.

        Args:
            start_date: Starting date
            days_ahead: Days to project

        Returns:
            Seasonality multiplier (0 to seasonality_factor)
        """
        if self.params.seasonality_factor == 0:
            return 0.0

        # Calculate average month over the period
        mid_date = start_date + timedelta(days=days_ahead / 2)
        month = mid_date.month

        # Peak fouling in July-August (months 7-8)
        # Use sinusoidal pattern
        seasonality = self.params.seasonality_factor * (
            0.5 + 0.5 * np.sin(2 * np.pi * (month - 4) / 12)
        )

        return max(0.0, seasonality)

    def estimate_days_to_threshold(
        self,
        current_cf: float,
        threshold_cf: float
    ) -> int:
        """
        Estimate days until CF reaches threshold.

        Args:
            current_cf: Current CF value
            threshold_cf: Target CF threshold

        Returns:
            Estimated days to reach threshold
        """
        if current_cf <= threshold_cf:
            return 0

        if self.params.model_type == DegradationModel.LINEAR:
            days = (current_cf - threshold_cf) / self.params.fouling_rate_per_day

        elif self.params.model_type == DegradationModel.EXPONENTIAL:
            if threshold_cf <= 0:
                return 365  # Cap at 1 year
            days = -np.log(threshold_cf / current_cf) / self.params.fouling_rate_per_day

        elif self.params.model_type == DegradationModel.ASYMPTOTIC:
            cf_floor = self.params.asymptotic_cf_floor
            if threshold_cf <= cf_floor:
                return 365  # Never reaches if below floor
            days = -np.log(
                (threshold_cf - cf_floor) / (current_cf - cf_floor)
            ) / self.params.fouling_rate_per_day

        else:
            days = (current_cf - threshold_cf) / self.params.fouling_rate_per_day

        return max(0, int(np.ceil(days)))


# ============================================================================
# ENERGY LOSS CALCULATOR
# ============================================================================

class EnergyLossCalculator:
    """
    Calculate energy and fuel losses from CF degradation.

    Uses physics-based relationships:
    - CF degradation -> backpressure increase
    - Backpressure increase -> heat rate penalty
    - Heat rate penalty -> fuel cost increase

    Zero-Hallucination Guarantee:
        All calculations use deterministic thermodynamic relationships.
    """

    def __init__(self, params: EnergyParameters):
        """
        Initialize energy loss calculator.

        Args:
            params: Energy calculation parameters
        """
        self.params = params

    def calculate_backpressure_increase(
        self,
        current_cf: float,
        degraded_cf: float,
        base_backpressure_inhga: float = 2.0
    ) -> float:
        """
        Calculate backpressure increase from CF degradation.

        Uses empirical relationship:
        delta_BP = base_BP * (1/CF_degraded - 1/CF_current)

        Args:
            current_cf: Current cleanliness factor
            degraded_cf: Degraded cleanliness factor
            base_backpressure_inhga: Base backpressure at CF=1.0

        Returns:
            Backpressure increase in inHgA
        """
        if degraded_cf <= 0 or current_cf <= 0:
            return 5.0  # Cap at maximum reasonable increase

        # Backpressure scales inversely with CF
        # BP = BP_base / CF
        bp_current = base_backpressure_inhga / current_cf
        bp_degraded = base_backpressure_inhga / degraded_cf

        bp_increase = bp_degraded - bp_current
        return max(0.0, bp_increase)

    def calculate_heat_rate_penalty(
        self,
        backpressure_increase_inhga: float
    ) -> float:
        """
        Calculate heat rate penalty from backpressure increase.

        Args:
            backpressure_increase_inhga: Backpressure increase

        Returns:
            Heat rate penalty as percentage
        """
        penalty_pct = (
            backpressure_increase_inhga *
            self.params.heat_rate_sensitivity_pct_per_inhga
        )
        return penalty_pct

    def calculate_fuel_penalty(
        self,
        heat_rate_penalty_pct: float,
        generation_mw: Optional[float] = None
    ) -> float:
        """
        Calculate hourly fuel cost penalty.

        Args:
            heat_rate_penalty_pct: Heat rate penalty percentage
            generation_mw: Current generation (default: base)

        Returns:
            Fuel penalty in USD/hr
        """
        if generation_mw is None:
            generation_mw = self.params.base_generation_mw * self.params.capacity_factor

        # Additional fuel consumption
        base_heat_rate = self.params.base_heat_rate_btu_kwh
        heat_rate_increase = base_heat_rate * (heat_rate_penalty_pct / 100)

        # Fuel consumption increase (MMBTU/hr)
        fuel_increase_mmbtu_hr = (generation_mw * 1000 * heat_rate_increase) / 1e6

        # Cost
        fuel_penalty_usd_hr = fuel_increase_mmbtu_hr * self.params.fuel_cost_per_mmbtu

        return fuel_penalty_usd_hr

    def calculate_energy_loss_trajectory(
        self,
        cf_predictions: List[CFPrediction],
        current_cf: float,
        base_backpressure_inhga: float = 2.0
    ) -> List[EnergyLossEstimate]:
        """
        Calculate energy loss trajectory over prediction horizon.

        Args:
            cf_predictions: List of CF predictions
            current_cf: Current CF value
            base_backpressure_inhga: Base backpressure

        Returns:
            List of energy loss estimates
        """
        losses = []
        cumulative_loss = 0.0

        for pred in cf_predictions:
            bp_increase = self.calculate_backpressure_increase(
                current_cf, pred.predicted_cf, base_backpressure_inhga
            )

            hr_penalty = self.calculate_heat_rate_penalty(bp_increase)
            fuel_penalty = self.calculate_fuel_penalty(hr_penalty)

            # Accumulate daily loss (24 hours)
            daily_loss = fuel_penalty * 24
            cumulative_loss += daily_loss

            losses.append(EnergyLossEstimate(
                date=pred.date,
                cf_value=pred.predicted_cf,
                backpressure_increase_inhga=bp_increase,
                heat_rate_increase_pct=hr_penalty,
                fuel_penalty_usd_hr=fuel_penalty,
                cumulative_loss_usd=cumulative_loss
            ))

        return losses


# ============================================================================
# CLEANING SCHEDULER CLASS
# ============================================================================

class CleaningScheduler:
    """
    Cleaning schedule optimizer for condenser maintenance.

    Determines optimal cleaning timing by maximizing NPV of cleaning:
    NPV = PV(avoided energy losses) - PV(cleaning costs)

    Zero-Hallucination Guarantee:
        - All financial calculations use deterministic formulas
        - NPV uses standard discounting methodology
        - No AI/ML models for decision making

    Example:
        >>> scheduler = CleaningScheduler(cleaning_params, degradation_params, energy_params)
        >>> recommendation = scheduler.optimize_cleaning_schedule(current_cf=0.82)
        >>> print(f"Optimal cleaning date: {recommendation.optimal_cleaning_date}")
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        cleaning_params: CleaningParameters,
        degradation_params: DegradationParameters,
        energy_params: EnergyParameters,
        config: Optional[SchedulerConfig] = None
    ):
        """
        Initialize cleaning scheduler.

        Args:
            cleaning_params: Cleaning cost and performance parameters
            degradation_params: CF degradation model parameters
            energy_params: Energy loss calculation parameters
            config: Scheduler configuration
        """
        self.cleaning_params = cleaning_params
        self.degradation_params = degradation_params
        self.energy_params = energy_params
        self.config = config or SchedulerConfig()

        self.degradation_model = CFDegradationModel(degradation_params)
        self.energy_calculator = EnergyLossCalculator(energy_params)

        logger.info(
            f"CleaningScheduler initialized: model={degradation_params.model_type.value}, "
            f"horizon={self.config.planning_horizon_days} days"
        )

    # =========================================================================
    # ROI CALCULATION
    # =========================================================================

    def calculate_cleaning_roi(
        self,
        current_cf: float,
        cleaning_day: int,
        current_date: Optional[datetime] = None,
        base_backpressure_inhga: float = 2.0
    ) -> CleaningROI:
        """
        Calculate ROI for cleaning on a specific day.

        Args:
            current_cf: Current cleanliness factor
            cleaning_day: Days from now to clean
            current_date: Current date
            base_backpressure_inhga: Base backpressure

        Returns:
            CleaningROI for the specified cleaning date
        """
        if current_date is None:
            current_date = datetime.now(timezone.utc)

        cleaning_date = current_date + timedelta(days=cleaning_day)

        # Calculate costs
        direct_cost = self.cleaning_params.cleaning_costs.total_direct_cost_usd
        downtime_cost = self.cleaning_params.downtime_costs.total_downtime_cost_usd
        total_cost = direct_cost + downtime_cost

        # Predict CF trajectory if NOT cleaned
        cf_trajectory_no_clean = self.degradation_model.predict_cf_trajectory(
            current_cf,
            self.config.planning_horizon_days,
            current_date
        )

        # Calculate cumulative energy loss if NOT cleaned
        energy_losses_no_clean = self.energy_calculator.calculate_energy_loss_trajectory(
            cf_trajectory_no_clean,
            current_cf,
            base_backpressure_inhga
        )

        cumulative_loss_no_clean = (
            energy_losses_no_clean[-1].cumulative_loss_usd
            if energy_losses_no_clean else 0.0
        )

        # Predict CF trajectory if cleaned on cleaning_day
        # Before cleaning: degradation continues
        # After cleaning: CF resets to cf_after_cleaning
        cf_at_cleaning = self.degradation_model.predict_cf(
            current_cf, cleaning_day, current_date
        ).predicted_cf

        cf_after_cleaning = self.cleaning_params.cf_after_cleaning

        # Calculate losses before cleaning
        losses_before = 0.0
        if cleaning_day > 0:
            cf_before_cleaning = self.degradation_model.predict_cf_trajectory(
                current_cf, cleaning_day, current_date
            )
            losses_before_list = self.energy_calculator.calculate_energy_loss_trajectory(
                cf_before_cleaning, current_cf, base_backpressure_inhga
            )
            losses_before = losses_before_list[-1].cumulative_loss_usd if losses_before_list else 0.0

        # Calculate losses after cleaning
        days_after = self.config.planning_horizon_days - cleaning_day
        losses_after = 0.0
        if days_after > 0:
            cf_after_trajectory = self.degradation_model.predict_cf_trajectory(
                cf_after_cleaning, days_after, cleaning_date
            )
            losses_after_list = self.energy_calculator.calculate_energy_loss_trajectory(
                cf_after_trajectory, cf_after_cleaning, base_backpressure_inhga
            )
            losses_after = losses_after_list[-1].cumulative_loss_usd if losses_after_list else 0.0

        cumulative_loss_with_clean = losses_before + losses_after

        # Avoided energy loss
        avoided_loss = cumulative_loss_no_clean - cumulative_loss_with_clean

        # Net savings
        net_savings = avoided_loss - total_cost

        # ROI percentage
        roi_pct = (net_savings / total_cost * 100) if total_cost > 0 else 0.0

        # NPV calculation
        daily_discount_rate = self.config.discount_rate_annual / 365
        discount_factor = 1 / (1 + daily_discount_rate) ** cleaning_day
        npv = net_savings * discount_factor

        # Payback calculation
        if avoided_loss > 0:
            daily_savings = avoided_loss / self.config.planning_horizon_days
            payback_days = total_cost / daily_savings if daily_savings > 0 else 999
        else:
            payback_days = 999

        # Backpressure improvement
        bp_before = self.energy_calculator.calculate_backpressure_increase(
            current_cf, cf_at_cleaning, base_backpressure_inhga
        )
        bp_after = self.energy_calculator.calculate_backpressure_increase(
            current_cf, cf_after_cleaning, base_backpressure_inhga
        )
        bp_improvement = bp_before - bp_after

        return CleaningROI(
            cleaning_date=cleaning_date,
            days_until_cleaning=cleaning_day,
            direct_cleaning_cost_usd=direct_cost,
            downtime_cost_usd=downtime_cost,
            total_cleaning_cost_usd=total_cost,
            cumulative_energy_loss_if_not_cleaned_usd=cumulative_loss_no_clean,
            avoided_energy_loss_usd=avoided_loss,
            backpressure_improvement_inhga=bp_improvement,
            net_savings_usd=net_savings,
            roi_pct=roi_pct,
            npv_usd=npv,
            payback_days=payback_days
        )

    def find_optimal_cleaning_date(
        self,
        current_cf: float,
        current_date: Optional[datetime] = None,
        base_backpressure_inhga: float = 2.0
    ) -> Tuple[int, List[CleaningROI]]:
        """
        Find optimal cleaning date by evaluating all candidates.

        Args:
            current_cf: Current cleanliness factor
            current_date: Current date
            base_backpressure_inhga: Base backpressure

        Returns:
            Tuple of (optimal_day, list of all ROI calculations)
        """
        if current_date is None:
            current_date = datetime.now(timezone.utc)

        roi_results = []

        for day in range(
            self.config.min_days_to_clean,
            min(self.config.max_days_to_clean, self.config.planning_horizon_days) + 1
        ):
            roi = self.calculate_cleaning_roi(
                current_cf, day, current_date, base_backpressure_inhga
            )
            roi_results.append(roi)

        # Find optimal by maximum NPV
        if not roi_results:
            return 1, []

        optimal_idx = max(range(len(roi_results)), key=lambda i: roi_results[i].npv_usd)
        optimal_day = roi_results[optimal_idx].days_until_cleaning

        # Sort by NPV for alternatives
        roi_results.sort(key=lambda x: x.npv_usd, reverse=True)

        return optimal_day, roi_results

    # =========================================================================
    # URGENCY DETERMINATION
    # =========================================================================

    def determine_urgency(
        self,
        current_cf: float,
        optimal_day: int
    ) -> RecommendationUrgency:
        """
        Determine urgency level for cleaning recommendation.

        Args:
            current_cf: Current cleanliness factor
            optimal_day: Optimal days until cleaning

        Returns:
            RecommendationUrgency level
        """
        # Check if CF is below mandatory threshold
        if current_cf < self.cleaning_params.min_cf_threshold:
            return RecommendationUrgency.IMMEDIATE

        # Days to mandatory threshold
        days_to_mandatory = self.degradation_model.estimate_days_to_threshold(
            current_cf,
            self.cleaning_params.min_cf_threshold
        )

        if days_to_mandatory <= 1:
            return RecommendationUrgency.IMMEDIATE
        elif days_to_mandatory <= 3:
            return RecommendationUrgency.HIGH
        elif days_to_mandatory <= 7:
            return RecommendationUrgency.MEDIUM
        elif days_to_mandatory <= 14:
            return RecommendationUrgency.LOW
        else:
            return RecommendationUrgency.SCHEDULED

    # =========================================================================
    # CMMS INTEGRATION
    # =========================================================================

    def generate_cmms_work_order(
        self,
        recommendation_id: str,
        cleaning_date: datetime,
        roi: CleaningROI,
        current_cf: float
    ) -> Dict[str, Any]:
        """
        Generate CMMS work order data package.

        Args:
            recommendation_id: Unique recommendation ID
            cleaning_date: Scheduled cleaning date
            roi: ROI calculation
            current_cf: Current CF value

        Returns:
            Dict with CMMS work order fields
        """
        return {
            "work_order_type": "PM",  # Preventive Maintenance
            "priority": self._urgency_to_cmms_priority(
                self.determine_urgency(current_cf, roi.days_until_cleaning)
            ),
            "equipment_id": "CONDENSER-001",
            "equipment_description": "Main Condenser",
            "work_description": (
                f"Condenser tube cleaning - {self.cleaning_params.cleaning_method.value}\n"
                f"Current CF: {current_cf:.2%}\n"
                f"Expected CF after: {self.cleaning_params.cf_after_cleaning:.2%}\n"
                f"Estimated savings: ${roi.net_savings_usd:,.0f}"
            ),
            "scheduled_start": cleaning_date.isoformat(),
            "estimated_duration_hours": self.cleaning_params.downtime_costs.duration_hours,
            "estimated_labor_cost": self.cleaning_params.cleaning_costs.labor_cost_usd,
            "estimated_material_cost": self.cleaning_params.cleaning_costs.materials_cost_usd,
            "justification": f"ROI: {roi.roi_pct:.1f}%, NPV: ${roi.npv_usd:,.0f}",
            "reference_document": recommendation_id,
            "safety_permits_required": [
                "LOTO",  # Lock-Out Tag-Out
                "Confined Space" if self.cleaning_params.cleaning_method == CleaningMethod.MECHANICAL else None,
                "Hot Work" if self.cleaning_params.cleaning_method == CleaningMethod.HYDROBLAST else None
            ],
            "parts_required": self._get_parts_list(),
            "special_instructions": self._get_special_instructions()
        }

    def _urgency_to_cmms_priority(self, urgency: RecommendationUrgency) -> int:
        """Convert urgency to CMMS priority code."""
        priority_map = {
            RecommendationUrgency.IMMEDIATE: 1,
            RecommendationUrgency.HIGH: 2,
            RecommendationUrgency.MEDIUM: 3,
            RecommendationUrgency.LOW: 4,
            RecommendationUrgency.SCHEDULED: 5
        }
        return priority_map.get(urgency, 5)

    def _get_parts_list(self) -> List[Dict[str, Any]]:
        """Get parts list for cleaning method."""
        if self.cleaning_params.cleaning_method == CleaningMethod.BALL_SYSTEM:
            return [
                {"part_number": "BALL-001", "description": "Cleaning balls", "quantity": 500},
                {"part_number": "SEAL-001", "description": "Ball strainer seals", "quantity": 4}
            ]
        elif self.cleaning_params.cleaning_method == CleaningMethod.CHEMICAL:
            return [
                {"part_number": "CHEM-001", "description": "Cleaning chemical", "quantity": 200},
                {"part_number": "NEUT-001", "description": "Neutralizer", "quantity": 50}
            ]
        else:
            return [
                {"part_number": "BRUSH-001", "description": "Tube brushes", "quantity": 100},
                {"part_number": "LANCE-001", "description": "Cleaning lances", "quantity": 4}
            ]

    def _get_special_instructions(self) -> str:
        """Get special instructions for cleaning method."""
        method = self.cleaning_params.cleaning_method

        instructions = {
            CleaningMethod.BALL_SYSTEM: "Online cleaning - no outage required. Monitor ball recovery count.",
            CleaningMethod.BACKWASH: "Perform during low load. Monitor discharge water quality.",
            CleaningMethod.CHEMICAL: "Full PPE required. Dispose chemicals per environmental permit.",
            CleaningMethod.MECHANICAL: "Tube entry required. Ensure proper ventilation.",
            CleaningMethod.HYDROBLAST: "High pressure hazard. Maintain safe standoff distance.",
            CleaningMethod.OFFLINE_FULL: "Full unit outage required. Coordinate with dispatch."
        }

        return instructions.get(method, "Follow standard cleaning procedure.")

    # =========================================================================
    # MAIN OPTIMIZATION ENTRY POINT
    # =========================================================================

    def optimize_cleaning_schedule(
        self,
        current_cf: float,
        current_date: Optional[datetime] = None,
        base_backpressure_inhga: float = 2.0
    ) -> CleaningRecommendation:
        """
        Generate optimal cleaning schedule recommendation.

        Args:
            current_cf: Current cleanliness factor
            current_date: Current date (default: now)
            base_backpressure_inhga: Base backpressure at CF=1.0

        Returns:
            Complete CleaningRecommendation package
        """
        start_time = datetime.now(timezone.utc)

        if current_date is None:
            current_date = datetime.now(timezone.utc)

        logger.info(f"Starting cleaning schedule optimization, current CF={current_cf:.3f}")

        # Find optimal cleaning date
        optimal_day, all_roi = self.find_optimal_cleaning_date(
            current_cf, current_date, base_backpressure_inhga
        )

        optimal_roi = all_roi[0] if all_roi else None

        # Determine urgency
        urgency = self.determine_urgency(current_cf, optimal_day)

        # Get predicted CF at optimal cleaning date
        predicted_cf_at_cleaning = self.degradation_model.predict_cf(
            current_cf, optimal_day, current_date
        ).predicted_cf

        # Generate recommendation ID
        recommendation_id = f"CLEAN-{current_date.strftime('%Y%m%d%H%M%S')}"

        # Generate CMMS data
        cleaning_date = current_date + timedelta(days=optimal_day)
        cmms_data = self.generate_cmms_work_order(
            recommendation_id,
            cleaning_date,
            optimal_roi,
            current_cf
        ) if optimal_roi else {}

        # Calculate provenance hash
        provenance_data = {
            "version": self.VERSION,
            "current_cf": round(current_cf, 4),
            "optimal_day": optimal_day,
            "npv": round(optimal_roi.npv_usd, 2) if optimal_roi else 0,
            "model_type": self.degradation_params.model_type.value
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        logger.info(
            f"Optimization complete in {processing_time:.1f}ms. "
            f"Optimal: day {optimal_day}, NPV=${optimal_roi.npv_usd:,.0f}" if optimal_roi else ""
        )

        return CleaningRecommendation(
            recommendation_id=recommendation_id,
            timestamp=current_date,
            optimal_cleaning_date=cleaning_date,
            days_until_optimal=optimal_day,
            urgency=urgency,
            optimal_roi=optimal_roi,
            alternative_dates=all_roi[1:4] if len(all_roi) > 1 else [],
            current_cf=current_cf,
            predicted_cf_at_cleaning=predicted_cf_at_cleaning,
            cmms_work_order_data=cmms_data,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time
        )

    # =========================================================================
    # REPORTING
    # =========================================================================

    def generate_recommendation_report(
        self,
        recommendation: CleaningRecommendation
    ) -> str:
        """
        Generate human-readable recommendation report.

        Args:
            recommendation: Cleaning recommendation

        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            "      CONDENSER CLEANING SCHEDULE RECOMMENDATION",
            "=" * 70,
            f"Recommendation ID: {recommendation.recommendation_id}",
            f"Generated: {recommendation.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "CURRENT STATUS:",
            f"  Cleanliness Factor: {recommendation.current_cf:.2%}",
            f"  Status: {'BELOW THRESHOLD' if recommendation.current_cf < self.cleaning_params.min_cf_threshold else 'NORMAL'}",
            "",
            "RECOMMENDATION:",
            f"  Urgency: {recommendation.urgency.value.upper()}",
            f"  Optimal Cleaning Date: {recommendation.optimal_cleaning_date.strftime('%Y-%m-%d')}",
            f"  Days Until Cleaning: {recommendation.days_until_optimal}",
            f"  Predicted CF at Cleaning: {recommendation.predicted_cf_at_cleaning:.2%}",
            "",
        ]

        if recommendation.optimal_roi:
            roi = recommendation.optimal_roi
            lines.extend([
                "FINANCIAL ANALYSIS:",
                f"  Total Cleaning Cost: ${roi.total_cleaning_cost_usd:,.0f}",
                f"    - Direct Cost: ${roi.direct_cleaning_cost_usd:,.0f}",
                f"    - Downtime Cost: ${roi.downtime_cost_usd:,.0f}",
                "",
                f"  Avoided Energy Loss: ${roi.avoided_energy_loss_usd:,.0f}",
                f"  Net Savings: ${roi.net_savings_usd:,.0f}",
                f"  ROI: {roi.roi_pct:.1f}%",
                f"  NPV: ${roi.npv_usd:,.0f}",
                f"  Payback: {roi.payback_days:.0f} days",
                "",
            ])

        if recommendation.alternative_dates:
            lines.append("ALTERNATIVE DATES:")
            for alt in recommendation.alternative_dates[:3]:
                lines.append(
                    f"  {alt.cleaning_date.strftime('%Y-%m-%d')}: "
                    f"NPV=${alt.npv_usd:,.0f}, ROI={alt.roi_pct:.1f}%"
                )
            lines.append("")

        lines.extend([
            f"Provenance: {recommendation.provenance_hash}",
            f"Processing Time: {recommendation.processing_time_ms:.1f}ms",
            "=" * 70
        ])

        return "\n".join(lines)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_default_scheduler() -> CleaningScheduler:
    """
    Create scheduler with default parameters.

    Returns:
        Configured CleaningScheduler instance
    """
    cleaning_params = CleaningParameters()
    degradation_params = DegradationParameters()
    energy_params = EnergyParameters()
    config = SchedulerConfig()

    return CleaningScheduler(
        cleaning_params,
        degradation_params,
        energy_params,
        config
    )
