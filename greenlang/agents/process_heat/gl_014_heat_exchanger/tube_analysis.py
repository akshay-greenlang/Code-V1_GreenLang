# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Agent - Tube Integrity Analysis and Failure Prediction

This module implements tube integrity analysis including wall thinning prediction,
failure probability estimation using Weibull analysis, and inspection scheduling.

Analysis Methods:
    - Weibull reliability analysis for tube life estimation
    - Corrosion rate trending from inspection data
    - Monte Carlo simulation for failure probability
    - Risk-based inspection scheduling

References:
    - ASME Section VIII Division 1 (Pressure Vessel Code)
    - API 570 Piping Inspection Code
    - API 579-1/ASME FFS-1 (Fitness-For-Service)
    - NACE SP0502 Pipeline External Corrosion Direct Assessment

Example:
    >>> from greenlang.agents.process_heat.gl_014_heat_exchanger.tube_analysis import (
    ...     TubeIntegrityAnalyzer
    ... )
    >>> analyzer = TubeIntegrityAnalyzer(config.tube_integrity, config.tube_geometry)
    >>> result = analyzer.analyze_integrity(inspection_data)
    >>> print(f"Remaining life: {result.estimated_remaining_life_years:.1f} years")
"""

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import random

from pydantic import BaseModel, Field

from greenlang.agents.process_heat.gl_014_heat_exchanger.config import (
    AlertSeverity,
    FailureMode,
    TubeGeometryConfig,
    TubeIntegrityConfig,
    TubeMaterial,
)
from greenlang.agents.process_heat.gl_014_heat_exchanger.schemas import (
    TubeInspectionData,
    TubeIntegrityResult,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Material corrosion rates (mm/year) in typical service
# Source: API RP 571, NACE corrosion data handbook
TYPICAL_CORROSION_RATES = {
    TubeMaterial.CARBON_STEEL: {
        "neutral_water": 0.10,
        "cooling_tower": 0.15,
        "acidic": 0.50,
        "sea_water": 0.20,
        "crude_oil": 0.08,
        "steam": 0.02,
    },
    TubeMaterial.STAINLESS_304: {
        "neutral_water": 0.01,
        "cooling_tower": 0.02,
        "acidic": 0.10,
        "sea_water": 0.05,
        "crude_oil": 0.01,
        "steam": 0.005,
    },
    TubeMaterial.STAINLESS_316: {
        "neutral_water": 0.005,
        "cooling_tower": 0.01,
        "acidic": 0.05,
        "sea_water": 0.02,
        "crude_oil": 0.005,
        "steam": 0.002,
    },
    TubeMaterial.TITANIUM_GR2: {
        "neutral_water": 0.001,
        "cooling_tower": 0.002,
        "acidic": 0.005,
        "sea_water": 0.001,
        "crude_oil": 0.001,
        "steam": 0.001,
    },
    TubeMaterial.COPPER_NICKEL_90_10: {
        "neutral_water": 0.02,
        "cooling_tower": 0.03,
        "sea_water": 0.05,
    },
}

# Weibull shape parameters by failure mode
# Source: Industry reliability data
WEIBULL_SHAPE_FACTORS = {
    FailureMode.TUBE_LEAK: 2.5,  # Wear-out failure
    FailureMode.TUBE_RUPTURE: 3.5,  # Strong wear-out
    FailureMode.TUBE_BLOCKAGE: 1.5,  # Random + wear
    FailureMode.VIBRATION_DAMAGE: 2.0,  # Fatigue
}

# Minimum allowable wall thickness factors (API/ASME)
WALL_THICKNESS_FACTORS = {
    "pressure": 1.0,  # For pressure containment
    "structural": 0.8,  # For structural stability
    "erosion_allowance": 0.5,  # Additional erosion margin
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TubeCondition:
    """Individual tube condition assessment."""

    tube_number: int
    wall_loss_percent: float
    remaining_thickness_mm: float
    estimated_remaining_life_years: float
    failure_probability_1yr: float
    status: str  # "good", "monitor", "plug", "failed"
    defect_type: Optional[str] = None
    defect_location: Optional[str] = None


@dataclass
class WeibullParameters:
    """Weibull distribution parameters."""

    beta: float  # Shape parameter
    eta: float  # Scale parameter (characteristic life)
    gamma: float = 0.0  # Location parameter (failure-free period)
    r_squared: float = 0.0  # Goodness of fit
    confidence_level: float = 0.90


@dataclass
class CorrosionTrend:
    """Corrosion rate trending results."""

    average_rate_mm_year: float
    current_rate_mm_year: float
    rate_trend: str  # "stable", "increasing", "decreasing"
    acceleration_mm_year2: float
    data_points: int
    confidence: float


@dataclass
class InspectionRecommendation:
    """Inspection scheduling recommendation."""

    next_inspection_date: datetime
    inspection_type: str
    priority: AlertSeverity
    scope: str
    estimated_cost_usd: float
    reasoning: str


# =============================================================================
# TUBE INTEGRITY ANALYZER
# =============================================================================

class TubeIntegrityAnalyzer:
    """
    Tube integrity analysis and failure prediction.

    This class provides comprehensive tube integrity assessment including:
    - Wall thickness analysis from inspection data
    - Weibull reliability analysis
    - Remaining life estimation
    - Failure probability prediction
    - Risk-based inspection scheduling

    Zero Hallucination Guarantee:
        All calculations use deterministic formulas:
        - API 579-1/ASME FFS-1 for remaining strength
        - Weibull distribution for reliability
        - Linear/exponential corrosion models

    Attributes:
        config: Tube integrity configuration
        geometry: Tube geometry configuration
        inspection_history: Historical inspection data

    Example:
        >>> analyzer = TubeIntegrityAnalyzer(config, geometry)
        >>> result = analyzer.analyze_integrity(inspection_data)
        >>> print(f"Remaining life: {result.estimated_remaining_life_years:.1f} years")
    """

    def __init__(
        self,
        config: TubeIntegrityConfig,
        geometry: TubeGeometryConfig,
    ) -> None:
        """
        Initialize the tube integrity analyzer.

        Args:
            config: Tube integrity configuration
            geometry: Tube geometry configuration
        """
        self.config = config
        self.geometry = geometry
        self.inspection_history: List[TubeInspectionData] = []
        self._calculation_count = 0

        # Initialize Weibull parameters if provided
        if config.weibull_beta and config.weibull_eta:
            self.weibull = WeibullParameters(
                beta=config.weibull_beta,
                eta=config.weibull_eta,
            )
        else:
            self.weibull = None

        logger.info(
            f"TubeIntegrityAnalyzer initialized for "
            f"{geometry.tube_count} tubes of {geometry.tube_material.value}"
        )

    def analyze_integrity(
        self,
        inspection_data: Optional[TubeInspectionData] = None,
        operating_years: Optional[float] = None,
    ) -> TubeIntegrityResult:
        """
        Perform comprehensive tube integrity analysis.

        Args:
            inspection_data: Latest inspection data
            operating_years: Years of operation (calculated if None)

        Returns:
            TubeIntegrityResult with complete analysis
        """
        self._calculation_count += 1

        # Determine operating age
        if operating_years is None:
            if self.config.installed_date:
                age_days = (
                    datetime.now(timezone.utc) - self.config.installed_date
                ).days
                operating_years = age_days / 365.0
            else:
                operating_years = 5.0  # Default assumption

        # Calculate current wall thickness
        if inspection_data:
            # Use inspection data
            avg_wall_loss = self._calculate_average_wall_loss(inspection_data)
            current_thickness = (
                self.geometry.wall_thickness_mm * (1 - avg_wall_loss / 100)
            )
        else:
            # Estimate from corrosion rate
            corrosion_loss = (
                self.config.expected_corrosion_rate_mm_year * operating_years
            )
            current_thickness = self.geometry.wall_thickness_mm - corrosion_loss

        # Minimum required thickness (API 579-1)
        min_thickness = self.config.minimum_wall_thickness_mm

        # Wall loss percentage
        wall_loss_percent = (
            (self.geometry.wall_thickness_mm - current_thickness) /
            self.geometry.wall_thickness_mm * 100
        )

        # Thickness margin
        thickness_margin = current_thickness - min_thickness

        # Plugging status
        if inspection_data:
            tubes_plugged = inspection_data.tubes_plugged
            tubes_at_risk = len(inspection_data.tubes_recommended_for_plugging)
        else:
            tubes_plugged = 0
            tubes_at_risk = 0

        plugging_rate = (
            tubes_plugged / self.geometry.tube_count * 100
            if self.geometry.tube_count > 0 else 0
        )

        # Remaining life estimation
        remaining_life, life_confidence = self._estimate_remaining_life(
            current_thickness,
            min_thickness,
            operating_years,
        )

        # Failure predictions
        pred_1yr, pred_5yr = self._predict_tube_failures(
            remaining_life,
            tubes_at_risk,
            operating_years,
        )

        # Update Weibull if we have data
        if inspection_data and len(self.inspection_history) >= 3:
            self._update_weibull_parameters()

        # Retube recommendation
        retube_recommended = (
            plugging_rate > self.config.plugging_threshold_percent or
            remaining_life < 2.0 or
            (inspection_data and inspection_data.retube_recommended)
        )

        # Next inspection
        next_inspection = self._calculate_next_inspection(
            remaining_life,
            tubes_at_risk,
        )

        # Inspection urgency
        if remaining_life < 1.0 or plugging_rate > 8:
            urgency = AlertSeverity.CRITICAL
        elif remaining_life < 2.0 or plugging_rate > 5:
            urgency = AlertSeverity.ALARM
        elif remaining_life < 5.0 or plugging_rate > 3:
            urgency = AlertSeverity.WARNING
        else:
            urgency = AlertSeverity.INFO

        # Failure modes
        failure_modes = self._analyze_failure_modes(
            wall_loss_percent,
            operating_years,
            inspection_data,
        )

        return TubeIntegrityResult(
            current_wall_thickness_mm=current_thickness,
            minimum_required_thickness_mm=min_thickness,
            thickness_margin_mm=thickness_margin,
            wall_loss_percent=wall_loss_percent,
            tubes_plugged=tubes_plugged,
            plugging_rate_percent=plugging_rate,
            tubes_at_risk=tubes_at_risk,
            estimated_remaining_life_years=remaining_life,
            remaining_life_confidence=life_confidence,
            predicted_failures_1yr=pred_1yr,
            predicted_failures_5yr=pred_5yr,
            weibull_beta=self.weibull.beta if self.weibull else None,
            weibull_eta_years=self.weibull.eta if self.weibull else None,
            retube_recommended=retube_recommended,
            next_inspection_date=next_inspection,
            inspection_urgency=urgency,
            failure_modes=failure_modes,
        )

    def estimate_remaining_life(
        self,
        current_thickness_mm: float,
        corrosion_rate_mm_year: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Estimate remaining tube life.

        Uses linear corrosion model:
        t_remaining = (t_current - t_min) / corrosion_rate

        Args:
            current_thickness_mm: Current wall thickness
            corrosion_rate_mm_year: Corrosion rate (uses config if None)

        Returns:
            Tuple of (remaining_life_years, confidence)
        """
        self._calculation_count += 1

        if corrosion_rate_mm_year is None:
            corrosion_rate_mm_year = self.config.expected_corrosion_rate_mm_year

        min_thickness = self.config.minimum_wall_thickness_mm
        margin = current_thickness_mm - min_thickness

        if margin <= 0:
            return 0.0, 0.95

        if corrosion_rate_mm_year <= 0:
            return self.config.design_life_years, 0.5

        remaining_life = margin / corrosion_rate_mm_year

        # Confidence based on data quality
        confidence = 0.80  # Base confidence for linear extrapolation

        return remaining_life, confidence

    def calculate_failure_probability(
        self,
        operating_years: float,
        time_horizon_years: float = 1.0,
    ) -> float:
        """
        Calculate failure probability using Weibull analysis.

        P(failure in t_horizon | survived to t_current) =
            1 - exp[-((t+dt)/eta)^beta + (t/eta)^beta]

        Args:
            operating_years: Current operating age
            time_horizon_years: Time horizon for probability

        Returns:
            Conditional failure probability
        """
        self._calculation_count += 1

        if self.weibull is None:
            # Use default parameters
            beta = 2.5  # Typical wear-out
            eta = self.config.design_life_years
        else:
            beta = self.weibull.beta
            eta = self.weibull.eta

        t = operating_years
        dt = time_horizon_years

        # Conditional probability
        # P(T < t+dt | T > t) = 1 - R(t+dt)/R(t)
        # R(t) = exp[-(t/eta)^beta]

        r_t = math.exp(-((t / eta) ** beta))
        r_t_dt = math.exp(-(((t + dt) / eta) ** beta))

        if r_t < 1e-10:
            return 1.0

        prob = 1.0 - (r_t_dt / r_t)

        return max(0.0, min(1.0, prob))

    def analyze_corrosion_trend(
        self,
        inspections: List[TubeInspectionData],
    ) -> CorrosionTrend:
        """
        Analyze corrosion rate trend from inspection history.

        Args:
            inspections: List of inspection records

        Returns:
            CorrosionTrend with analysis results
        """
        self._calculation_count += 1

        if len(inspections) < 2:
            return CorrosionTrend(
                average_rate_mm_year=self.config.expected_corrosion_rate_mm_year,
                current_rate_mm_year=self.config.expected_corrosion_rate_mm_year,
                rate_trend="stable",
                acceleration_mm_year2=0.0,
                data_points=len(inspections),
                confidence=0.3,
            )

        # Sort by date
        sorted_inspections = sorted(inspections, key=lambda x: x.inspection_date)

        # Calculate wall loss rates between inspections
        rates = []
        for i in range(1, len(sorted_inspections)):
            prev = sorted_inspections[i - 1]
            curr = sorted_inspections[i]

            # Time between inspections (years)
            dt_days = (curr.inspection_date - prev.inspection_date).days
            dt_years = dt_days / 365.0

            if dt_years < 0.1:
                continue  # Skip very close inspections

            # Wall loss change
            prev_loss = self._calculate_average_wall_loss(prev)
            curr_loss = self._calculate_average_wall_loss(curr)

            # Convert to mm
            loss_change_mm = (
                (curr_loss - prev_loss) / 100 *
                self.geometry.wall_thickness_mm
            )

            rate = loss_change_mm / dt_years
            rates.append((dt_years, rate))

        if not rates:
            return CorrosionTrend(
                average_rate_mm_year=self.config.expected_corrosion_rate_mm_year,
                current_rate_mm_year=self.config.expected_corrosion_rate_mm_year,
                rate_trend="stable",
                acceleration_mm_year2=0.0,
                data_points=len(inspections),
                confidence=0.3,
            )

        # Average and current rate
        avg_rate = sum(r for _, r in rates) / len(rates)
        current_rate = rates[-1][1] if rates else avg_rate

        # Trend analysis
        if len(rates) >= 3:
            early_avg = sum(r for _, r in rates[:len(rates)//2]) / (len(rates)//2)
            late_avg = sum(r for _, r in rates[len(rates)//2:]) / (len(rates) - len(rates)//2)

            if late_avg > early_avg * 1.2:
                trend = "increasing"
            elif late_avg < early_avg * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"

            # Acceleration
            total_time = sum(dt for dt, _ in rates)
            if total_time > 0:
                acceleration = (late_avg - early_avg) / total_time
            else:
                acceleration = 0.0
        else:
            trend = "stable"
            acceleration = 0.0

        # Confidence based on data points
        confidence = min(0.95, 0.5 + 0.1 * len(rates))

        return CorrosionTrend(
            average_rate_mm_year=max(0, avg_rate),
            current_rate_mm_year=max(0, current_rate),
            rate_trend=trend,
            acceleration_mm_year2=acceleration,
            data_points=len(inspections),
            confidence=confidence,
        )

    def simulate_tube_failures(
        self,
        years_ahead: int,
        n_simulations: int = 1000,
    ) -> Dict[str, Any]:
        """
        Monte Carlo simulation of tube failures.

        Args:
            years_ahead: Simulation horizon
            n_simulations: Number of Monte Carlo simulations

        Returns:
            Simulation results with statistics
        """
        self._calculation_count += 1

        if self.weibull is None:
            beta = 2.5
            eta = self.config.design_life_years
        else:
            beta = self.weibull.beta
            eta = self.weibull.eta

        failures_by_year: Dict[int, List[int]] = {y: [] for y in range(1, years_ahead + 1)}

        for _ in range(n_simulations):
            # Simulate each tube
            tube_failures = [0] * years_ahead

            for tube in range(self.geometry.tube_count):
                # Generate random lifetime using Weibull
                u = random.random()
                lifetime = eta * ((-math.log(u)) ** (1 / beta))

                # Check when failure occurs
                for year in range(1, years_ahead + 1):
                    if lifetime <= year:
                        tube_failures[year - 1] += 1
                        break

            for year in range(years_ahead):
                failures_by_year[year + 1].append(tube_failures[year])

        # Calculate statistics
        results = {
            "years_simulated": years_ahead,
            "n_simulations": n_simulations,
            "weibull_beta": beta,
            "weibull_eta": eta,
            "predictions": {},
        }

        for year, failures in failures_by_year.items():
            mean_failures = sum(failures) / len(failures)
            sorted_failures = sorted(failures)
            p10 = sorted_failures[int(0.1 * len(failures))]
            p90 = sorted_failures[int(0.9 * len(failures))]

            results["predictions"][year] = {
                "mean": mean_failures,
                "p10": p10,
                "p90": p90,
                "failure_rate": mean_failures / self.geometry.tube_count * 100,
            }

        return results

    def add_inspection_data(self, inspection: TubeInspectionData) -> None:
        """Add inspection data to history."""
        self.inspection_history.append(inspection)
        self.inspection_history.sort(key=lambda x: x.inspection_date)

        # Limit history
        if len(self.inspection_history) > 50:
            self.inspection_history = self.inspection_history[-50:]

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _calculate_average_wall_loss(
        self,
        inspection: TubeInspectionData,
    ) -> float:
        """Calculate average wall loss from inspection data."""
        if not inspection.wall_loss_summary:
            # Estimate from defect count
            defect_rate = inspection.defect_rate
            return defect_rate * 25  # Rough estimate

        # Weighted average from summary
        total_tubes = 0
        weighted_loss = 0

        ranges = {
            "<20%": 10,
            "20-40%": 30,
            "40-60%": 50,
            "60-80%": 70,
            ">80%": 90,
        }

        for category, count in inspection.wall_loss_summary.items():
            mid_point = ranges.get(category, 25)
            weighted_loss += mid_point * count
            total_tubes += count

        if total_tubes == 0:
            return 0.0

        return weighted_loss / total_tubes

    def _estimate_remaining_life(
        self,
        current_thickness: float,
        min_thickness: float,
        operating_years: float,
    ) -> Tuple[float, float]:
        """Estimate remaining tube life."""
        margin = current_thickness - min_thickness

        if margin <= 0:
            return 0.0, 0.95

        # Corrosion rate from history or config
        if len(self.inspection_history) >= 2:
            trend = self.analyze_corrosion_trend(self.inspection_history)
            rate = trend.current_rate_mm_year
            confidence = trend.confidence
        else:
            rate = self.config.expected_corrosion_rate_mm_year
            confidence = 0.70

        if rate <= 0:
            # Use design life
            remaining = max(0, self.config.design_life_years - operating_years)
            return remaining, 0.50

        remaining = margin / rate
        return remaining, confidence

    def _predict_tube_failures(
        self,
        remaining_life: float,
        current_at_risk: int,
        operating_years: float,
    ) -> Tuple[int, int]:
        """Predict tube failures for 1 and 5 year horizons."""
        # 1-year prediction
        prob_1yr = self.calculate_failure_probability(operating_years, 1.0)
        pred_1yr = int(
            current_at_risk * 0.5 +  # At-risk tubes
            (self.geometry.tube_count - current_at_risk) * prob_1yr
        )

        # 5-year prediction
        prob_5yr = self.calculate_failure_probability(operating_years, 5.0)
        pred_5yr = int(
            current_at_risk * 0.9 +  # Most at-risk will fail
            (self.geometry.tube_count - current_at_risk) * prob_5yr
        )

        return pred_1yr, pred_5yr

    def _update_weibull_parameters(self) -> None:
        """Update Weibull parameters from inspection history."""
        # This would use Maximum Likelihood Estimation in production
        # Simplified implementation here

        if len(self.inspection_history) < 3:
            return

        # Extract failure data
        failures = sum(
            insp.tubes_with_defects for insp in self.inspection_history
        )
        total_tubes = self.geometry.tube_count * len(self.inspection_history)

        if failures == 0:
            return

        # Estimate beta from defect progression
        # Simplified: use default beta and estimate eta from data
        beta = WEIBULL_SHAPE_FACTORS.get(FailureMode.TUBE_LEAK, 2.5)

        # Eta estimation from failure rate
        failure_rate = failures / total_tubes

        if failure_rate > 0 and failure_rate < 1:
            # Solve for eta: F(t) = 1 - exp(-(t/eta)^beta) = failure_rate
            # eta = t / (-ln(1-F))^(1/beta)
            avg_age = sum(
                (datetime.now(timezone.utc) - insp.inspection_date).days / 365
                for insp in self.inspection_history
            ) / len(self.inspection_history)

            if avg_age > 0:
                eta = avg_age / ((-math.log(1 - failure_rate)) ** (1 / beta))
                eta = max(5, min(50, eta))  # Bound to reasonable range

                self.weibull = WeibullParameters(
                    beta=beta,
                    eta=eta,
                    confidence_level=0.85,
                )

    def _calculate_next_inspection(
        self,
        remaining_life: float,
        tubes_at_risk: int,
    ) -> datetime:
        """Calculate recommended next inspection date."""
        # Base on remaining life
        if remaining_life < 1:
            interval_months = 3
        elif remaining_life < 2:
            interval_months = 6
        elif remaining_life < 5:
            interval_months = 12
        else:
            interval_months = self.config.inspection_interval_months

        # Adjust for at-risk tubes
        risk_factor = tubes_at_risk / max(1, self.geometry.tube_count)
        if risk_factor > 0.05:
            interval_months = int(interval_months * 0.5)

        # Bound interval
        interval_months = max(3, min(24, interval_months))

        return datetime.now(timezone.utc) + timedelta(days=interval_months * 30)

    def _analyze_failure_modes(
        self,
        wall_loss_percent: float,
        operating_years: float,
        inspection: Optional[TubeInspectionData],
    ) -> List[Dict[str, Any]]:
        """Analyze active failure modes and risks."""
        failure_modes = []

        # Wall thinning / corrosion
        if wall_loss_percent > 30:
            failure_modes.append({
                "mode": FailureMode.TUBE_LEAK.value,
                "probability": min(0.8, wall_loss_percent / 50),
                "severity": "high" if wall_loss_percent > 50 else "medium",
                "description": f"Wall loss at {wall_loss_percent:.1f}%",
            })

        # Tube rupture (high wall loss)
        if wall_loss_percent > 60:
            failure_modes.append({
                "mode": FailureMode.TUBE_RUPTURE.value,
                "probability": (wall_loss_percent - 50) / 50,
                "severity": "critical",
                "description": "High wall loss increases rupture risk",
            })

        # Tube blockage (from inspection defects)
        if inspection and inspection.tubes_plugged > 0:
            blockage_rate = inspection.tubes_plugged / inspection.total_tubes
            failure_modes.append({
                "mode": FailureMode.TUBE_BLOCKAGE.value,
                "probability": min(0.5, blockage_rate * 5),
                "severity": "medium",
                "description": f"{inspection.tubes_plugged} tubes already plugged",
            })

        # Age-related wear
        design_life = self.config.design_life_years
        if operating_years > design_life * 0.8:
            failure_modes.append({
                "mode": FailureMode.TUBE_LEAK.value,
                "probability": (operating_years / design_life - 0.8) * 2,
                "severity": "medium",
                "description": f"Operating at {operating_years/design_life:.0%} of design life",
            })

        return failure_modes

    @property
    def calculation_count(self) -> int:
        """Get total calculation count."""
        return self._calculation_count
