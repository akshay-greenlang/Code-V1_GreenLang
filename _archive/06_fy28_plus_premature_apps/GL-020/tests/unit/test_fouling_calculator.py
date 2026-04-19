"""
GL-020 ECONOPULSE - Fouling Calculator Unit Tests

Comprehensive unit tests for FoulingCalculator with 95%+ coverage target.
Tests fouling factor calculation, fouling rate trending, cleaning time prediction,
cleanliness factor, efficiency loss, and fuel penalty calculations.

Target Coverage: 95%+
Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import os
import math
import hashlib
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test fixtures from conftest
from tests.conftest import (
    EconomizerConfig, FoulingData, FoulingLevel, PerformanceBaseline
)


# =============================================================================
# MOCK CALCULATOR CLASS FOR TESTING
# =============================================================================

@dataclass
class FoulingResult:
    """Result of fouling calculation."""
    fouling_factor_m2k_w: float
    cleanliness_factor: float
    fouling_level: FoulingLevel
    efficiency_loss_pct: float
    fuel_penalty_pct: float
    days_since_cleaning: int
    fouling_rate_per_day: float
    estimated_days_to_cleaning: int
    cleaning_recommended: bool
    provenance_hash: str
    calculation_timestamp: datetime


@dataclass
class FoulingTrendResult:
    """Result of fouling trend analysis."""
    current_fouling_factor: float
    trend_slope: float  # m2K/W per day
    trend_r_squared: float
    days_to_threshold: int
    projected_cleaning_date: date
    confidence_level: float


class FoulingCalculator:
    """
    Fouling calculator for economizer performance monitoring.

    Calculates:
    - Fouling factor (Rf)
    - Cleanliness factor (CF)
    - Fouling rate and trends
    - Cleaning time prediction
    - Efficiency loss from fouling
    - Fuel penalty from fouling
    """

    VERSION = "1.0.0"
    NAME = "FoulingCalculator"
    AGENT_ID = "GL-020"

    # Fouling thresholds (m2.K/W)
    FOULING_THRESHOLDS = {
        FoulingLevel.CLEAN: 0.0002,
        FoulingLevel.LIGHT: 0.0004,
        FoulingLevel.MODERATE: 0.0007,
        FoulingLevel.HEAVY: 0.001,
        FoulingLevel.SEVERE: float('inf')
    }

    # Default cleaning threshold
    DEFAULT_CLEANING_THRESHOLD = 0.0008  # m2.K/W

    def __init__(self, cleaning_threshold: float = None):
        self.cleaning_threshold = cleaning_threshold or self.DEFAULT_CLEANING_THRESHOLD

    def calculate_fouling_factor(
        self,
        clean_u_value: float,
        current_u_value: float
    ) -> float:
        """
        Calculate fouling factor from U-values.

        Rf = (1/U_fouled) - (1/U_clean)

        Args:
            clean_u_value: Clean U-value (W/m2.K)
            current_u_value: Current (fouled) U-value (W/m2.K)

        Returns:
            Fouling factor in m2.K/W

        Raises:
            ValueError: If U-values are invalid
        """
        if clean_u_value <= 0:
            raise ValueError("Clean U-value must be positive")

        if current_u_value <= 0:
            raise ValueError("Current U-value must be positive")

        if current_u_value > clean_u_value:
            # Current is better than clean (could be measurement noise)
            return 0.0

        # Rf = 1/U_current - 1/U_clean
        fouling_factor = (1 / current_u_value) - (1 / clean_u_value)

        return max(0.0, fouling_factor)

    def calculate_cleanliness_factor(
        self,
        clean_u_value: float,
        current_u_value: float
    ) -> float:
        """
        Calculate cleanliness factor (CF).

        CF = U_current / U_clean

        Args:
            clean_u_value: Clean U-value (W/m2.K)
            current_u_value: Current U-value (W/m2.K)

        Returns:
            Cleanliness factor (0 to 1, where 1 is perfectly clean)
        """
        if clean_u_value <= 0:
            raise ValueError("Clean U-value must be positive")

        if current_u_value <= 0:
            raise ValueError("Current U-value must be positive")

        cf = current_u_value / clean_u_value

        # Clamp to valid range
        return max(0.0, min(1.0, cf))

    def determine_fouling_level(
        self,
        fouling_factor: float
    ) -> FoulingLevel:
        """
        Determine fouling severity level from fouling factor.

        Args:
            fouling_factor: Fouling factor (m2.K/W)

        Returns:
            FoulingLevel enum value
        """
        if fouling_factor < self.FOULING_THRESHOLDS[FoulingLevel.CLEAN]:
            return FoulingLevel.CLEAN
        elif fouling_factor < self.FOULING_THRESHOLDS[FoulingLevel.LIGHT]:
            return FoulingLevel.LIGHT
        elif fouling_factor < self.FOULING_THRESHOLDS[FoulingLevel.MODERATE]:
            return FoulingLevel.MODERATE
        elif fouling_factor < self.FOULING_THRESHOLDS[FoulingLevel.HEAVY]:
            return FoulingLevel.HEAVY
        else:
            return FoulingLevel.SEVERE

    def calculate_efficiency_loss(
        self,
        clean_u_value: float,
        current_u_value: float,
        design_effectiveness: float = 0.75
    ) -> float:
        """
        Calculate efficiency loss percentage due to fouling.

        Args:
            clean_u_value: Clean U-value (W/m2.K)
            current_u_value: Current U-value (W/m2.K)
            design_effectiveness: Design heat exchanger effectiveness

        Returns:
            Efficiency loss percentage
        """
        if clean_u_value <= 0 or current_u_value <= 0:
            raise ValueError("U-values must be positive")

        cleanliness_factor = current_u_value / clean_u_value

        if cleanliness_factor >= 1.0:
            return 0.0

        # Simplified efficiency loss model
        # Assumes linear relationship between U-value reduction and efficiency loss
        u_value_reduction = (clean_u_value - current_u_value) / clean_u_value
        efficiency_loss = u_value_reduction * 100 * design_effectiveness

        return max(0.0, efficiency_loss)

    def calculate_fuel_penalty(
        self,
        efficiency_loss_pct: float,
        boiler_efficiency_base: float = 0.85
    ) -> float:
        """
        Calculate fuel penalty percentage due to fouling.

        Args:
            efficiency_loss_pct: Efficiency loss percentage
            boiler_efficiency_base: Base boiler efficiency

        Returns:
            Additional fuel consumption percentage
        """
        if efficiency_loss_pct < 0:
            return 0.0

        # Fuel penalty approximation:
        # For every 1% efficiency loss, roughly 0.2% more fuel needed
        # (depends on economizer contribution to overall efficiency)
        fuel_penalty = efficiency_loss_pct * 0.2

        return fuel_penalty

    def calculate_fouling_rate(
        self,
        fouling_history: List[FoulingData],
        min_data_points: int = 5
    ) -> Tuple[float, float]:
        """
        Calculate fouling rate from historical data.

        Args:
            fouling_history: List of historical fouling data points
            min_data_points: Minimum points required for calculation

        Returns:
            Tuple of (fouling_rate_per_day, r_squared)
        """
        if len(fouling_history) < min_data_points:
            return 0.0, 0.0

        # Sort by timestamp
        sorted_data = sorted(fouling_history, key=lambda x: x.timestamp)

        # Extract days since cleaning and fouling factors
        x_values = [d.days_since_cleaning for d in sorted_data]
        y_values = [d.fouling_factor_m2k_w for d in sorted_data]

        # Linear regression: y = mx + b
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0, 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        intercept = (sum_y - slope * sum_x) / n

        # Calculate R-squared
        y_mean = sum_y / n
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return slope, r_squared

    def predict_cleaning_date(
        self,
        current_fouling_factor: float,
        fouling_rate_per_day: float,
        threshold: float = None,
        current_date: date = None
    ) -> Tuple[int, date]:
        """
        Predict days until cleaning is needed.

        Args:
            current_fouling_factor: Current fouling factor (m2.K/W)
            fouling_rate_per_day: Rate of fouling increase per day
            threshold: Cleaning threshold (defaults to class threshold)
            current_date: Current date (defaults to today)

        Returns:
            Tuple of (days_to_cleaning, projected_cleaning_date)
        """
        threshold = threshold or self.cleaning_threshold
        current_date = current_date or date.today()

        if fouling_rate_per_day <= 0:
            # No fouling progression
            return 365 * 10, current_date + timedelta(days=365 * 10)

        if current_fouling_factor >= threshold:
            # Already at or above threshold
            return 0, current_date

        remaining_fouling = threshold - current_fouling_factor
        days_to_threshold = int(remaining_fouling / fouling_rate_per_day)

        projected_date = current_date + timedelta(days=days_to_threshold)

        return days_to_threshold, projected_date

    def analyze_trend(
        self,
        fouling_history: List[FoulingData],
        threshold: float = None
    ) -> FoulingTrendResult:
        """
        Analyze fouling trend and project cleaning date.

        Args:
            fouling_history: List of historical fouling data
            threshold: Cleaning threshold

        Returns:
            FoulingTrendResult with trend analysis
        """
        threshold = threshold or self.cleaning_threshold

        if not fouling_history:
            raise ValueError("Fouling history cannot be empty")

        # Get most recent fouling factor
        sorted_history = sorted(fouling_history, key=lambda x: x.timestamp)
        current_data = sorted_history[-1]

        # Calculate trend
        slope, r_squared = self.calculate_fouling_rate(fouling_history)

        # Predict cleaning date
        days_to_threshold, projected_date = self.predict_cleaning_date(
            current_fouling_factor=current_data.fouling_factor_m2k_w,
            fouling_rate_per_day=slope,
            threshold=threshold,
            current_date=current_data.timestamp.date()
        )

        # Confidence level based on R-squared and data points
        data_points_factor = min(len(fouling_history) / 30, 1.0)  # More points = more confidence
        confidence = r_squared * data_points_factor

        return FoulingTrendResult(
            current_fouling_factor=current_data.fouling_factor_m2k_w,
            trend_slope=slope,
            trend_r_squared=r_squared,
            days_to_threshold=days_to_threshold,
            projected_cleaning_date=projected_date,
            confidence_level=confidence
        )

    def calculate_all(
        self,
        clean_u_value: float,
        current_u_value: float,
        days_since_cleaning: int,
        fouling_rate_per_day: float = None,
        design_effectiveness: float = 0.75
    ) -> FoulingResult:
        """
        Perform complete fouling calculation.

        Args:
            clean_u_value: Clean U-value (W/m2.K)
            current_u_value: Current U-value (W/m2.K)
            days_since_cleaning: Days since last cleaning
            fouling_rate_per_day: Optional fouling rate
            design_effectiveness: Design heat exchanger effectiveness

        Returns:
            FoulingResult with all calculated values
        """
        # Calculate fouling factor
        fouling_factor = self.calculate_fouling_factor(clean_u_value, current_u_value)

        # Calculate cleanliness factor
        cleanliness_factor = self.calculate_cleanliness_factor(clean_u_value, current_u_value)

        # Determine fouling level
        fouling_level = self.determine_fouling_level(fouling_factor)

        # Calculate efficiency loss
        efficiency_loss = self.calculate_efficiency_loss(
            clean_u_value, current_u_value, design_effectiveness
        )

        # Calculate fuel penalty
        fuel_penalty = self.calculate_fuel_penalty(efficiency_loss)

        # Calculate or estimate fouling rate
        if fouling_rate_per_day is None and days_since_cleaning > 0:
            fouling_rate_per_day = fouling_factor / days_since_cleaning
        elif fouling_rate_per_day is None:
            fouling_rate_per_day = 0.0

        # Predict cleaning date
        days_to_cleaning, _ = self.predict_cleaning_date(
            current_fouling_factor=fouling_factor,
            fouling_rate_per_day=fouling_rate_per_day
        )

        # Determine if cleaning is recommended
        cleaning_recommended = (
            fouling_factor >= self.cleaning_threshold or
            fouling_level in [FoulingLevel.HEAVY, FoulingLevel.SEVERE]
        )

        # Generate provenance hash
        provenance_data = f"{clean_u_value},{current_u_value},{days_since_cleaning}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        return FoulingResult(
            fouling_factor_m2k_w=fouling_factor,
            cleanliness_factor=cleanliness_factor,
            fouling_level=fouling_level,
            efficiency_loss_pct=efficiency_loss,
            fuel_penalty_pct=fuel_penalty,
            days_since_cleaning=days_since_cleaning,
            fouling_rate_per_day=fouling_rate_per_day,
            estimated_days_to_cleaning=days_to_cleaning,
            cleaning_recommended=cleaning_recommended,
            provenance_hash=provenance_hash,
            calculation_timestamp=datetime.now(timezone.utc)
        )


# =============================================================================
# UNIT TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.calculator
@pytest.mark.fouling
@pytest.mark.critical
class TestFoulingCalculator:
    """Comprehensive test suite for FoulingCalculator."""

    # =========================================================================
    # INITIALIZATION TESTS
    # =========================================================================

    def test_initialization_default(self):
        """Test FoulingCalculator initializes with default threshold."""
        calculator = FoulingCalculator()

        assert calculator.VERSION == "1.0.0"
        assert calculator.NAME == "FoulingCalculator"
        assert calculator.cleaning_threshold == 0.0008

    def test_initialization_custom_threshold(self):
        """Test FoulingCalculator initializes with custom threshold."""
        calculator = FoulingCalculator(cleaning_threshold=0.001)

        assert calculator.cleaning_threshold == 0.001

    # =========================================================================
    # FOULING FACTOR CALCULATION TESTS
    # =========================================================================

    def test_fouling_factor_no_fouling(self):
        """Test fouling factor is zero when U-values match."""
        calculator = FoulingCalculator()

        rf = calculator.calculate_fouling_factor(
            clean_u_value=45.0,
            current_u_value=45.0
        )

        assert rf == 0.0

    def test_fouling_factor_light_fouling(self):
        """Test fouling factor for light fouling."""
        calculator = FoulingCalculator()

        rf = calculator.calculate_fouling_factor(
            clean_u_value=45.0,
            current_u_value=40.0
        )

        # Rf = 1/40 - 1/45 = 0.025 - 0.0222 = 0.00278 m2.K/W
        assert rf == pytest.approx(0.00278, rel=0.01)

    def test_fouling_factor_moderate_fouling(self):
        """Test fouling factor for moderate fouling."""
        calculator = FoulingCalculator()

        rf = calculator.calculate_fouling_factor(
            clean_u_value=45.0,
            current_u_value=35.0
        )

        # Rf = 1/35 - 1/45 = 0.02857 - 0.0222 = 0.00635 m2.K/W
        assert rf == pytest.approx(0.00635, rel=0.01)

    def test_fouling_factor_severe_fouling(self):
        """Test fouling factor for severe fouling."""
        calculator = FoulingCalculator()

        rf = calculator.calculate_fouling_factor(
            clean_u_value=45.0,
            current_u_value=25.0
        )

        # Rf = 1/25 - 1/45 = 0.04 - 0.0222 = 0.01778 m2.K/W
        assert rf == pytest.approx(0.01778, rel=0.01)

    @pytest.mark.parametrize("clean_u,fouled_u,expected_rf", [
        (45.0, 45.0, 0.0),
        (45.0, 40.0, 0.00278),
        (45.0, 35.0, 0.00635),
        (45.0, 30.0, 0.01111),
        (45.0, 25.0, 0.01778),
        (50.0, 45.0, 0.00222),
        (55.0, 40.0, 0.00682),
    ])
    def test_fouling_factor_parametrized(self, clean_u, fouled_u, expected_rf):
        """Test fouling factor with parametrized inputs."""
        calculator = FoulingCalculator()

        rf = calculator.calculate_fouling_factor(
            clean_u_value=clean_u,
            current_u_value=fouled_u
        )

        assert rf == pytest.approx(expected_rf, abs=0.0002)

    def test_fouling_factor_current_better_than_clean(self):
        """Test fouling factor returns zero when current > clean."""
        calculator = FoulingCalculator()

        rf = calculator.calculate_fouling_factor(
            clean_u_value=45.0,
            current_u_value=48.0  # Higher than clean
        )

        assert rf == 0.0

    def test_fouling_factor_zero_clean_u_raises(self):
        """Test fouling factor raises error for zero clean U-value."""
        calculator = FoulingCalculator()

        with pytest.raises(ValueError, match="must be positive"):
            calculator.calculate_fouling_factor(
                clean_u_value=0.0,
                current_u_value=40.0
            )

    def test_fouling_factor_negative_clean_u_raises(self):
        """Test fouling factor raises error for negative clean U-value."""
        calculator = FoulingCalculator()

        with pytest.raises(ValueError, match="must be positive"):
            calculator.calculate_fouling_factor(
                clean_u_value=-45.0,
                current_u_value=40.0
            )

    def test_fouling_factor_zero_current_u_raises(self):
        """Test fouling factor raises error for zero current U-value."""
        calculator = FoulingCalculator()

        with pytest.raises(ValueError, match="must be positive"):
            calculator.calculate_fouling_factor(
                clean_u_value=45.0,
                current_u_value=0.0
            )

    # =========================================================================
    # CLEANLINESS FACTOR TESTS
    # =========================================================================

    def test_cleanliness_factor_perfectly_clean(self):
        """Test cleanliness factor is 1.0 when perfectly clean."""
        calculator = FoulingCalculator()

        cf = calculator.calculate_cleanliness_factor(
            clean_u_value=45.0,
            current_u_value=45.0
        )

        assert cf == 1.0

    def test_cleanliness_factor_light_fouling(self):
        """Test cleanliness factor for light fouling."""
        calculator = FoulingCalculator()

        cf = calculator.calculate_cleanliness_factor(
            clean_u_value=45.0,
            current_u_value=40.0
        )

        # CF = 40/45 = 0.889
        assert cf == pytest.approx(0.889, rel=0.01)

    def test_cleanliness_factor_severe_fouling(self):
        """Test cleanliness factor for severe fouling."""
        calculator = FoulingCalculator()

        cf = calculator.calculate_cleanliness_factor(
            clean_u_value=45.0,
            current_u_value=25.0
        )

        # CF = 25/45 = 0.556
        assert cf == pytest.approx(0.556, rel=0.01)

    def test_cleanliness_factor_clamped_to_one(self):
        """Test cleanliness factor is clamped to 1.0 max."""
        calculator = FoulingCalculator()

        cf = calculator.calculate_cleanliness_factor(
            clean_u_value=45.0,
            current_u_value=50.0  # Better than clean
        )

        assert cf == 1.0

    def test_cleanliness_factor_clamped_to_zero(self):
        """Test cleanliness factor is clamped to 0.0 min."""
        calculator = FoulingCalculator()

        cf = calculator.calculate_cleanliness_factor(
            clean_u_value=45.0,
            current_u_value=0.001  # Almost zero
        )

        assert cf >= 0.0
        assert cf <= 1.0

    # =========================================================================
    # FOULING LEVEL TESTS
    # =========================================================================

    def test_determine_fouling_level_clean(self):
        """Test fouling level determination for clean."""
        calculator = FoulingCalculator()

        level = calculator.determine_fouling_level(0.0001)

        assert level == FoulingLevel.CLEAN

    def test_determine_fouling_level_light(self):
        """Test fouling level determination for light."""
        calculator = FoulingCalculator()

        level = calculator.determine_fouling_level(0.0003)

        assert level == FoulingLevel.LIGHT

    def test_determine_fouling_level_moderate(self):
        """Test fouling level determination for moderate."""
        calculator = FoulingCalculator()

        level = calculator.determine_fouling_level(0.0005)

        assert level == FoulingLevel.MODERATE

    def test_determine_fouling_level_heavy(self):
        """Test fouling level determination for heavy."""
        calculator = FoulingCalculator()

        level = calculator.determine_fouling_level(0.0008)

        assert level == FoulingLevel.HEAVY

    def test_determine_fouling_level_severe(self):
        """Test fouling level determination for severe."""
        calculator = FoulingCalculator()

        level = calculator.determine_fouling_level(0.0015)

        assert level == FoulingLevel.SEVERE

    @pytest.mark.parametrize("fouling_factor,expected_level", [
        (0.00005, FoulingLevel.CLEAN),
        (0.0001, FoulingLevel.CLEAN),
        (0.00015, FoulingLevel.CLEAN),
        (0.0002, FoulingLevel.LIGHT),
        (0.0003, FoulingLevel.LIGHT),
        (0.0004, FoulingLevel.MODERATE),
        (0.0005, FoulingLevel.MODERATE),
        (0.0006, FoulingLevel.MODERATE),
        (0.0007, FoulingLevel.HEAVY),
        (0.0009, FoulingLevel.HEAVY),
        (0.001, FoulingLevel.SEVERE),
        (0.002, FoulingLevel.SEVERE),
    ])
    def test_fouling_level_parametrized(self, fouling_factor, expected_level):
        """Test fouling level determination with parametrized inputs."""
        calculator = FoulingCalculator()

        level = calculator.determine_fouling_level(fouling_factor)

        assert level == expected_level

    # =========================================================================
    # EFFICIENCY LOSS TESTS
    # =========================================================================

    def test_efficiency_loss_no_fouling(self):
        """Test efficiency loss is zero with no fouling."""
        calculator = FoulingCalculator()

        loss = calculator.calculate_efficiency_loss(
            clean_u_value=45.0,
            current_u_value=45.0
        )

        assert loss == 0.0

    def test_efficiency_loss_moderate_fouling(self):
        """Test efficiency loss for moderate fouling."""
        calculator = FoulingCalculator()

        loss = calculator.calculate_efficiency_loss(
            clean_u_value=45.0,
            current_u_value=35.0,
            design_effectiveness=0.75
        )

        # U reduction = (45-35)/45 = 0.222 = 22.2%
        # Efficiency loss = 22.2% * 100 * 0.75 = 16.67%
        assert loss == pytest.approx(16.67, rel=0.05)

    def test_efficiency_loss_severe_fouling(self):
        """Test efficiency loss for severe fouling."""
        calculator = FoulingCalculator()

        loss = calculator.calculate_efficiency_loss(
            clean_u_value=45.0,
            current_u_value=25.0,
            design_effectiveness=0.75
        )

        # U reduction = (45-25)/45 = 0.444 = 44.4%
        # Efficiency loss = 44.4% * 100 * 0.75 = 33.3%
        assert loss == pytest.approx(33.33, rel=0.05)

    def test_efficiency_loss_invalid_u_values_raises(self):
        """Test efficiency loss raises error for invalid U-values."""
        calculator = FoulingCalculator()

        with pytest.raises(ValueError, match="must be positive"):
            calculator.calculate_efficiency_loss(
                clean_u_value=0.0,
                current_u_value=35.0
            )

    # =========================================================================
    # FUEL PENALTY TESTS
    # =========================================================================

    def test_fuel_penalty_no_loss(self):
        """Test fuel penalty is zero with no efficiency loss."""
        calculator = FoulingCalculator()

        penalty = calculator.calculate_fuel_penalty(0.0)

        assert penalty == 0.0

    def test_fuel_penalty_moderate_loss(self):
        """Test fuel penalty for moderate efficiency loss."""
        calculator = FoulingCalculator()

        penalty = calculator.calculate_fuel_penalty(10.0)  # 10% efficiency loss

        # Fuel penalty = 10 * 0.2 = 2%
        assert penalty == pytest.approx(2.0, rel=0.01)

    def test_fuel_penalty_negative_loss_returns_zero(self):
        """Test fuel penalty returns zero for negative loss."""
        calculator = FoulingCalculator()

        penalty = calculator.calculate_fuel_penalty(-5.0)

        assert penalty == 0.0

    @pytest.mark.parametrize("efficiency_loss,expected_penalty", [
        (0.0, 0.0),
        (5.0, 1.0),
        (10.0, 2.0),
        (15.0, 3.0),
        (20.0, 4.0),
        (25.0, 5.0),
    ])
    def test_fuel_penalty_parametrized(self, efficiency_loss, expected_penalty):
        """Test fuel penalty with parametrized inputs."""
        calculator = FoulingCalculator()

        penalty = calculator.calculate_fuel_penalty(efficiency_loss)

        assert penalty == pytest.approx(expected_penalty, rel=0.01)

    # =========================================================================
    # FOULING RATE TESTS
    # =========================================================================

    def test_fouling_rate_insufficient_data(self):
        """Test fouling rate returns zero with insufficient data."""
        calculator = FoulingCalculator()

        history = [
            FoulingData(
                economizer_id="ECON-001",
                timestamp=datetime.now(timezone.utc),
                fouling_factor_m2k_w=0.0002,
                fouling_level=FoulingLevel.CLEAN,
                efficiency_loss_pct=1.0,
                estimated_fuel_penalty_pct=0.2,
                days_since_cleaning=10,
                cleaning_recommended=False,
                estimated_days_to_cleaning=100
            )
        ]

        slope, r_squared = calculator.calculate_fouling_rate(history, min_data_points=5)

        assert slope == 0.0
        assert r_squared == 0.0

    def test_fouling_rate_linear_trend(self, fouling_trend_data):
        """Test fouling rate calculation with linear trend data."""
        calculator = FoulingCalculator()

        slope, r_squared = calculator.calculate_fouling_rate(fouling_trend_data)

        # Should detect positive slope (fouling increasing)
        assert slope > 0
        # Linear trend should have high R-squared
        assert r_squared > 0.5

    def test_fouling_rate_constant_fouling(self):
        """Test fouling rate for constant fouling (no progression)."""
        calculator = FoulingCalculator()

        base_time = datetime.now(timezone.utc)
        history = [
            FoulingData(
                economizer_id="ECON-001",
                timestamp=base_time - timedelta(days=i * 7),
                fouling_factor_m2k_w=0.0003,  # Constant
                fouling_level=FoulingLevel.LIGHT,
                efficiency_loss_pct=3.0,
                estimated_fuel_penalty_pct=0.6,
                days_since_cleaning=i * 7,
                cleaning_recommended=False,
                estimated_days_to_cleaning=100
            )
            for i in range(10)
        ]

        slope, r_squared = calculator.calculate_fouling_rate(history)

        # Slope should be near zero for constant fouling
        assert abs(slope) < 0.00001

    # =========================================================================
    # CLEANING PREDICTION TESTS
    # =========================================================================

    def test_predict_cleaning_date_normal(self):
        """Test cleaning date prediction with normal fouling rate."""
        calculator = FoulingCalculator()

        days, projected_date = calculator.predict_cleaning_date(
            current_fouling_factor=0.0004,
            fouling_rate_per_day=0.000005,
            threshold=0.0008,
            current_date=date(2025, 1, 1)
        )

        # Remaining = 0.0008 - 0.0004 = 0.0004
        # Days = 0.0004 / 0.000005 = 80 days
        assert days == 80
        assert projected_date == date(2025, 3, 22)

    def test_predict_cleaning_date_already_exceeded(self):
        """Test cleaning prediction when already above threshold."""
        calculator = FoulingCalculator()

        days, projected_date = calculator.predict_cleaning_date(
            current_fouling_factor=0.001,  # Above threshold
            fouling_rate_per_day=0.000005,
            threshold=0.0008,
            current_date=date(2025, 1, 1)
        )

        assert days == 0
        assert projected_date == date(2025, 1, 1)

    def test_predict_cleaning_date_zero_rate(self):
        """Test cleaning prediction with zero fouling rate."""
        calculator = FoulingCalculator()

        days, projected_date = calculator.predict_cleaning_date(
            current_fouling_factor=0.0004,
            fouling_rate_per_day=0.0,  # No fouling
            threshold=0.0008,
            current_date=date(2025, 1, 1)
        )

        # Should return very far future
        assert days >= 3650  # At least 10 years

    def test_predict_cleaning_date_negative_rate(self):
        """Test cleaning prediction with negative fouling rate (improving)."""
        calculator = FoulingCalculator()

        days, projected_date = calculator.predict_cleaning_date(
            current_fouling_factor=0.0004,
            fouling_rate_per_day=-0.000001,  # Improving
            threshold=0.0008,
            current_date=date(2025, 1, 1)
        )

        # Should return very far future
        assert days >= 3650

    # =========================================================================
    # TREND ANALYSIS TESTS
    # =========================================================================

    def test_analyze_trend_success(self, fouling_trend_data):
        """Test trend analysis with valid data."""
        calculator = FoulingCalculator()

        result = calculator.analyze_trend(fouling_trend_data)

        assert result.current_fouling_factor > 0
        assert result.trend_slope >= 0
        assert 0 <= result.trend_r_squared <= 1
        assert result.days_to_threshold >= 0
        assert result.confidence_level >= 0

    def test_analyze_trend_empty_history_raises(self):
        """Test trend analysis raises error for empty history."""
        calculator = FoulingCalculator()

        with pytest.raises(ValueError, match="cannot be empty"):
            calculator.analyze_trend([])

    def test_analyze_trend_single_point(self):
        """Test trend analysis with single data point."""
        calculator = FoulingCalculator()

        history = [
            FoulingData(
                economizer_id="ECON-001",
                timestamp=datetime.now(timezone.utc),
                fouling_factor_m2k_w=0.0003,
                fouling_level=FoulingLevel.LIGHT,
                efficiency_loss_pct=3.0,
                estimated_fuel_penalty_pct=0.6,
                days_since_cleaning=30,
                cleaning_recommended=False,
                estimated_days_to_cleaning=100
            )
        ]

        result = calculator.analyze_trend(history)

        # Should return valid result even with single point
        assert result.current_fouling_factor == 0.0003

    # =========================================================================
    # COMPLETE CALCULATION TESTS
    # =========================================================================

    def test_calculate_all_clean(self, clean_baseline_performance):
        """Test complete calculation for clean economizer."""
        calculator = FoulingCalculator()

        result = calculator.calculate_all(
            clean_u_value=45.0,
            current_u_value=44.5,  # Slight reduction
            days_since_cleaning=10,
            design_effectiveness=0.75
        )

        assert result.fouling_factor_m2k_w < 0.0002
        assert result.fouling_level == FoulingLevel.CLEAN
        assert result.cleanliness_factor > 0.98
        assert result.efficiency_loss_pct < 2.0
        assert not result.cleaning_recommended

    def test_calculate_all_moderately_fouled(self):
        """Test complete calculation for moderately fouled economizer."""
        calculator = FoulingCalculator()

        result = calculator.calculate_all(
            clean_u_value=45.0,
            current_u_value=35.0,
            days_since_cleaning=120,
            design_effectiveness=0.75
        )

        assert 0.0004 < result.fouling_factor_m2k_w < 0.001
        assert result.fouling_level in [FoulingLevel.MODERATE, FoulingLevel.HEAVY]
        assert result.cleanliness_factor < 0.85
        assert result.efficiency_loss_pct > 10.0
        assert result.fuel_penalty_pct > 2.0

    def test_calculate_all_severely_fouled(self):
        """Test complete calculation for severely fouled economizer."""
        calculator = FoulingCalculator()

        result = calculator.calculate_all(
            clean_u_value=45.0,
            current_u_value=25.0,
            days_since_cleaning=365,
            design_effectiveness=0.75
        )

        assert result.fouling_factor_m2k_w > 0.01
        assert result.fouling_level == FoulingLevel.SEVERE
        assert result.cleanliness_factor < 0.6
        assert result.efficiency_loss_pct > 25.0
        assert result.cleaning_recommended

    def test_calculate_all_provenance_hash(self):
        """Test provenance hash is generated correctly."""
        calculator = FoulingCalculator()

        result = calculator.calculate_all(
            clean_u_value=45.0,
            current_u_value=35.0,
            days_since_cleaning=100
        )

        assert len(result.provenance_hash) == 64

    def test_calculate_all_reproducibility(self):
        """Test calculation reproducibility."""
        calculator = FoulingCalculator()

        results = []
        for _ in range(5):
            result = calculator.calculate_all(
                clean_u_value=45.0,
                current_u_value=35.0,
                days_since_cleaning=100
            )
            results.append(result)

        # All provenance hashes should match
        first_hash = results[0].provenance_hash
        for result in results[1:]:
            assert result.provenance_hash == first_hash

    # =========================================================================
    # PERFORMANCE TESTS
    # =========================================================================

    @pytest.mark.performance
    def test_fouling_calculation_speed(self, benchmark):
        """Test fouling calculation meets performance target."""
        calculator = FoulingCalculator()

        def run_calculation():
            return calculator.calculate_all(
                clean_u_value=45.0,
                current_u_value=35.0,
                days_since_cleaning=100
            )

        result = benchmark(run_calculation)
        assert result.fouling_factor_m2k_w > 0

    @pytest.mark.performance
    def test_batch_fouling_throughput(self):
        """Test batch fouling calculation throughput."""
        calculator = FoulingCalculator()
        import time

        num_calculations = 10000
        start = time.time()

        for i in range(num_calculations):
            current_u = 45.0 - (i % 20)  # Vary U-value
            if current_u <= 0:
                current_u = 25.0
            calculator.calculate_all(
                clean_u_value=45.0,
                current_u_value=current_u,
                days_since_cleaning=i % 365
            )

        duration = time.time() - start
        throughput = num_calculations / duration

        assert throughput > 50000  # >50,000 calculations per second


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

@pytest.mark.unit
class TestFoulingEdgeCases:
    """Edge case tests for fouling calculations."""

    def test_very_small_u_value_difference(self):
        """Test with very small U-value difference."""
        calculator = FoulingCalculator()

        rf = calculator.calculate_fouling_factor(
            clean_u_value=45.0,
            current_u_value=44.999
        )

        assert rf >= 0
        assert rf < 0.00001

    def test_very_large_u_value_drop(self):
        """Test with very large U-value drop."""
        calculator = FoulingCalculator()

        rf = calculator.calculate_fouling_factor(
            clean_u_value=45.0,
            current_u_value=10.0
        )

        assert rf > 0.05  # Very high fouling factor

    def test_very_high_u_values(self):
        """Test with very high U-values."""
        calculator = FoulingCalculator()

        rf = calculator.calculate_fouling_factor(
            clean_u_value=500.0,
            current_u_value=400.0
        )

        assert rf == pytest.approx(0.0005, rel=0.01)

    def test_very_low_u_values(self):
        """Test with very low U-values."""
        calculator = FoulingCalculator()

        rf = calculator.calculate_fouling_factor(
            clean_u_value=5.0,
            current_u_value=4.0
        )

        assert rf == pytest.approx(0.05, rel=0.01)

    def test_fouling_at_threshold_boundary(self):
        """Test fouling level at threshold boundaries."""
        calculator = FoulingCalculator()

        # Test at each boundary
        for threshold, expected_level in [
            (0.0002, FoulingLevel.LIGHT),
            (0.0004, FoulingLevel.MODERATE),
            (0.0007, FoulingLevel.HEAVY),
            (0.001, FoulingLevel.SEVERE),
        ]:
            level = calculator.determine_fouling_level(threshold)
            assert level == expected_level, f"Boundary {threshold} failed"


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

@pytest.mark.unit
class TestFoulingIntegration:
    """Integration tests for fouling calculations with real-world scenarios."""

    def test_seasonal_fouling_pattern(self, fouling_progression_generator):
        """Test detection of seasonal fouling patterns."""
        calculator = FoulingCalculator()

        fouling_data = fouling_progression_generator(
            num_days=365,
            initial_fouling=0.0001,
            fouling_rate_per_day=0.000003
        )

        trend = calculator.analyze_trend(fouling_data)

        # Should detect gradual fouling increase
        assert trend.trend_slope > 0
        assert trend.days_to_threshold >= 0

    def test_post_cleaning_reset(self):
        """Test fouling calculation after cleaning."""
        calculator = FoulingCalculator()

        # Before cleaning - severely fouled
        result_before = calculator.calculate_all(
            clean_u_value=45.0,
            current_u_value=28.0,
            days_since_cleaning=300
        )

        assert result_before.cleaning_recommended

        # After cleaning - clean
        result_after = calculator.calculate_all(
            clean_u_value=45.0,
            current_u_value=44.5,
            days_since_cleaning=1
        )

        assert not result_after.cleaning_recommended
        assert result_after.fouling_level == FoulingLevel.CLEAN

    def test_different_economizer_designs(
        self,
        bare_tube_economizer,
        finned_tube_economizer,
        extended_surface_economizer
    ):
        """Test fouling calculations for different economizer designs."""
        calculator = FoulingCalculator()

        for economizer in [bare_tube_economizer, finned_tube_economizer, extended_surface_economizer]:
            # Simulate 20% U-value reduction
            current_u = economizer.design_u_value_w_m2k * 0.8

            result = calculator.calculate_all(
                clean_u_value=economizer.design_u_value_w_m2k,
                current_u_value=current_u,
                days_since_cleaning=180
            )

            # 20% U-value reduction should give similar fouling factors
            # regardless of absolute U-value
            assert result.fouling_factor_m2k_w > 0
            assert result.cleanliness_factor == pytest.approx(0.8, rel=0.01)
