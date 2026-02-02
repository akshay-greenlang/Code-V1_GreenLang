# -*- coding: utf-8 -*-
"""
Unit Tests: Fouling Calculator

Comprehensive tests for fouling prediction calculations including:
- CF degradation modeling
- Time-to-threshold calculation
- Fouling rate estimation
- Cleaning optimization

Target Coverage: 85%+
Author: GL-TestEngineer
Date: December 2025
"""

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import (
    FoulingHistoryEntry,
    FoulingPredictionResult,
    CleaningMethod,
    CondenserConfig,
    TubeMaterial,
    WaterSource,
    AssertionHelpers,
    ProvenanceCalculator,
    OPERATING_LIMITS,
    TEST_SEED,
)


# =============================================================================
# FOULING CALCULATOR IMPLEMENTATION FOR TESTING
# =============================================================================

class FoulingCalculator:
    """
    Condenser fouling prediction and cleaning optimization calculator.

    Predicts cleanliness factor degradation over time and recommends
    optimal cleaning schedules based on economic analysis.
    """

    VERSION = "1.0.0"

    # Fouling model parameters
    DEFAULT_DECAY_RATE = 0.002  # CF loss per day
    ASYMPTOTIC_CF = 0.60  # Minimum CF (fully fouled)

    # Threshold values
    WARNING_CF = 0.80
    CRITICAL_CF = 0.75
    MINIMUM_CF = 0.70

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize fouling calculator."""
        self.config = config or {}
        self.warning_threshold = self.config.get("cf_threshold_warning", self.WARNING_CF)
        self.critical_threshold = self.config.get("cf_threshold_critical", self.CRITICAL_CF)
        self.prediction_horizon = self.config.get("prediction_horizon_days", 90)

    def estimate_fouling_rate(
        self,
        history: List[FoulingHistoryEntry],
        method: str = "linear"
    ) -> Tuple[float, float]:
        """
        Estimate fouling rate from historical data.

        Args:
            history: List of historical CF readings
            method: Estimation method ("linear", "exponential")

        Returns:
            Tuple of (rate, r_squared)
        """
        if len(history) < 2:
            return self.DEFAULT_DECAY_RATE, 0.0

        # Sort by timestamp (most recent first)
        sorted_history = sorted(history, key=lambda x: x.timestamp, reverse=True)

        # Extract days from most recent and CF values
        base_time = sorted_history[0].timestamp
        days = []
        cf_values = []

        for entry in sorted_history:
            delta = (base_time - entry.timestamp).total_seconds() / 86400
            days.append(delta)
            cf_values.append(entry.cleanliness_factor)

        days = np.array(days)
        cf_values = np.array(cf_values)

        if method == "linear":
            # Linear regression: CF = a - b*days
            if len(days) >= 2:
                coeffs = np.polyfit(days, cf_values, 1)
                rate = -coeffs[0]  # Negative slope = positive decay rate

                # Calculate R-squared
                predicted = np.polyval(coeffs, days)
                ss_res = np.sum((cf_values - predicted) ** 2)
                ss_tot = np.sum((cf_values - np.mean(cf_values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                return max(0.0, rate), max(0.0, r_squared)

        elif method == "exponential":
            # Exponential decay: CF = CF_asymptotic + (CF_0 - CF_asymptotic) * exp(-k*t)
            # Simplified: estimate decay constant k
            if cf_values[0] > self.ASYMPTOTIC_CF:
                # Use first and last points
                cf_range = cf_values[0] - self.ASYMPTOTIC_CF
                if cf_range > 0 and len(days) > 1:
                    cf_last = max(cf_values[-1], self.ASYMPTOTIC_CF + 0.01)
                    cf_ratio = (cf_last - self.ASYMPTOTIC_CF) / cf_range
                    if 0 < cf_ratio < 1:
                        k = -math.log(cf_ratio) / days[-1] if days[-1] > 0 else self.DEFAULT_DECAY_RATE
                        return k, 0.8  # Approximate R-squared

        return self.DEFAULT_DECAY_RATE, 0.0

    def predict_cf(
        self,
        current_cf: float,
        days_ahead: int,
        decay_rate: float = None,
        model: str = "linear"
    ) -> float:
        """
        Predict future CF value.

        Args:
            current_cf: Current cleanliness factor
            days_ahead: Days to predict ahead
            decay_rate: CF decay rate per day
            model: Prediction model ("linear", "exponential")

        Returns:
            Predicted CF value
        """
        rate = decay_rate if decay_rate is not None else self.DEFAULT_DECAY_RATE

        if model == "linear":
            predicted = current_cf - rate * days_ahead
        elif model == "exponential":
            # Exponential decay toward asymptote
            predicted = self.ASYMPTOTIC_CF + (current_cf - self.ASYMPTOTIC_CF) * math.exp(-rate * days_ahead)
        else:
            predicted = current_cf - rate * days_ahead

        # Clamp to valid range
        return max(self.ASYMPTOTIC_CF, min(1.0, predicted))

    def calculate_days_to_threshold(
        self,
        current_cf: float,
        threshold_cf: float,
        decay_rate: float = None,
        model: str = "linear"
    ) -> int:
        """
        Calculate days until CF reaches threshold.

        Args:
            current_cf: Current cleanliness factor
            threshold_cf: Target threshold CF
            decay_rate: CF decay rate per day
            model: Prediction model

        Returns:
            Days until threshold (or -1 if never reached)
        """
        if current_cf <= threshold_cf:
            return 0

        rate = decay_rate if decay_rate is not None else self.DEFAULT_DECAY_RATE

        if rate <= 0:
            return -1  # Never reaches threshold

        if model == "linear":
            days = (current_cf - threshold_cf) / rate
        elif model == "exponential":
            # Solve: threshold = asymptotic + (current - asymptotic) * exp(-k*t)
            if threshold_cf <= self.ASYMPTOTIC_CF:
                return -1  # Never reaches below asymptote

            cf_range = current_cf - self.ASYMPTOTIC_CF
            target_range = threshold_cf - self.ASYMPTOTIC_CF

            if cf_range > 0 and target_range > 0:
                ratio = target_range / cf_range
                if 0 < ratio < 1:
                    days = -math.log(ratio) / rate
                else:
                    return -1
            else:
                return -1
        else:
            days = (current_cf - threshold_cf) / rate

        return max(0, int(math.ceil(days)))

    def estimate_cleaning_cost(
        self,
        method: CleaningMethod,
        condenser_config: CondenserConfig = None
    ) -> float:
        """
        Estimate cleaning cost.

        Args:
            method: Cleaning method
            condenser_config: Condenser configuration (for scaling)

        Returns:
            Estimated cost (USD)
        """
        base_costs = {
            CleaningMethod.ONLINE_BALL: 100.0,
            CleaningMethod.ONLINE_BRUSH: 150.0,
            CleaningMethod.OFFLINE_HYDROLANCE: 25000.0,
            CleaningMethod.OFFLINE_CHEMICAL: 75000.0,
            CleaningMethod.NONE: 0.0,
        }

        base_cost = base_costs.get(method, 10000.0)

        # Scale by condenser size if config provided
        if condenser_config is not None:
            size_factor = condenser_config.surface_area_m2 / 25000.0  # Normalize to typical
            base_cost *= max(0.5, min(2.0, size_factor))

        return base_cost

    def estimate_cf_recovery(self, method: CleaningMethod) -> float:
        """
        Estimate CF recovery from cleaning.

        Args:
            method: Cleaning method

        Returns:
            Expected CF after cleaning (as fraction of clean CF)
        """
        recovery_rates = {
            CleaningMethod.ONLINE_BALL: 0.90,
            CleaningMethod.ONLINE_BRUSH: 0.92,
            CleaningMethod.OFFLINE_HYDROLANCE: 0.95,
            CleaningMethod.OFFLINE_CHEMICAL: 0.98,
            CleaningMethod.NONE: 0.0,
        }
        return recovery_rates.get(method, 0.85)

    def calculate_lost_generation_cost(
        self,
        cf_current: float,
        cf_clean: float,
        unit_load_mw: float,
        electricity_price_usd_mwh: float,
        hours: float
    ) -> float:
        """
        Calculate cost of lost generation due to fouling.

        Args:
            cf_current: Current CF
            cf_clean: Clean CF
            unit_load_mw: Unit load
            electricity_price_usd_mwh: Electricity price
            hours: Hours of operation

        Returns:
            Lost generation cost (USD)
        """
        if cf_clean <= 0:
            return 0.0

        # Approximate MW loss from CF degradation
        # Typical: 0.5-1% MW loss per 1% CF reduction
        mw_loss_factor = 0.01  # 1% MW per 1% CF
        cf_loss = cf_clean - cf_current
        mw_loss = unit_load_mw * cf_loss * mw_loss_factor

        return mw_loss * electricity_price_usd_mwh * hours

    def recommend_cleaning(
        self,
        current_cf: float,
        history: List[FoulingHistoryEntry],
        condenser_config: CondenserConfig = None,
        unit_load_mw: float = 500.0,
        electricity_price_usd_mwh: float = 50.0
    ) -> FoulingPredictionResult:
        """
        Recommend cleaning schedule based on economic analysis.

        Args:
            current_cf: Current cleanliness factor
            history: Historical CF data
            condenser_config: Condenser configuration
            unit_load_mw: Unit load (MW)
            electricity_price_usd_mwh: Electricity price

        Returns:
            FoulingPredictionResult with recommendations
        """
        # Estimate fouling rate
        decay_rate, r_squared = self.estimate_fouling_rate(history)

        # Calculate days to critical threshold
        days_to_critical = self.calculate_days_to_threshold(
            current_cf,
            self.critical_threshold,
            decay_rate
        )

        # Predict CF at horizon
        predicted_cf = self.predict_cf(current_cf, self.prediction_horizon, decay_rate)

        # Determine recommended cleaning method
        if current_cf < self.MINIMUM_CF:
            method = CleaningMethod.OFFLINE_CHEMICAL
        elif current_cf < self.critical_threshold:
            method = CleaningMethod.OFFLINE_HYDROLANCE
        elif current_cf < self.warning_threshold:
            method = CleaningMethod.ONLINE_BRUSH
        else:
            method = CleaningMethod.ONLINE_BALL

        # Calculate cleaning cost
        cleaning_cost = self.estimate_cleaning_cost(method, condenser_config)

        # Calculate lost generation cost over prediction horizon
        cf_clean = 0.92  # Typical clean CF
        lost_gen_cost = self.calculate_lost_generation_cost(
            current_cf, cf_clean, unit_load_mw, electricity_price_usd_mwh,
            self.prediction_horizon * 24
        )

        # Calculate ROI
        roi = lost_gen_cost / cleaning_cost if cleaning_cost > 0 else 0.0

        # Recommended cleaning date
        if days_to_critical > 0:
            # Clean before critical threshold
            clean_date = datetime.now(timezone.utc) + timedelta(days=max(1, days_to_critical - 7))
        else:
            # Immediate cleaning recommended
            clean_date = datetime.now(timezone.utc) + timedelta(days=1)

        # Generate provenance hash
        input_data = {
            "current_cf": current_cf,
            "decay_rate": decay_rate,
            "unit_load_mw": unit_load_mw,
            "electricity_price": electricity_price_usd_mwh,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()

        return FoulingPredictionResult(
            current_cf=current_cf,
            predicted_cf=predicted_cf,
            days_to_threshold=days_to_critical,
            cf_decay_rate_per_day=decay_rate,
            recommended_cleaning_date=clean_date,
            cleaning_method=method,
            cleaning_cost_usd=cleaning_cost,
            lost_generation_cost_usd=lost_gen_cost,
            roi_cleaning=roi,
            provenance_hash=provenance_hash,
        )


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def calculator() -> FoulingCalculator:
    """Create fouling calculator instance."""
    return FoulingCalculator()


@pytest.fixture
def calculator_with_config() -> FoulingCalculator:
    """Create fouling calculator with configuration."""
    config = {
        "cf_threshold_warning": 0.82,
        "cf_threshold_critical": 0.76,
        "prediction_horizon_days": 60,
    }
    return FoulingCalculator(config)


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestFoulingRateEstimation:
    """Tests for fouling rate estimation."""

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_rate_estimation_linear_decline(
        self,
        calculator: FoulingCalculator,
        fouling_history_clean: List[FoulingHistoryEntry]
    ):
        """Test rate estimation with linear decline."""
        rate, r_squared = calculator.estimate_fouling_rate(
            fouling_history_clean, method="linear"
        )

        assert rate > 0  # Positive decay rate
        assert 0 <= r_squared <= 1  # Valid R-squared

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_rate_estimation_rapid_decline(
        self,
        calculator: FoulingCalculator,
        fouling_history_rapid_decline: List[FoulingHistoryEntry]
    ):
        """Test rate estimation with rapid decline."""
        rate, r_squared = calculator.estimate_fouling_rate(
            fouling_history_rapid_decline, method="linear"
        )

        # Rapid decline should show higher rate
        assert rate > 0.005

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_rate_estimation_empty_history(self, calculator: FoulingCalculator):
        """Test rate estimation with empty history."""
        rate, r_squared = calculator.estimate_fouling_rate([])

        # Should return default rate
        assert rate == calculator.DEFAULT_DECAY_RATE
        assert r_squared == 0.0

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_rate_estimation_single_point(self, calculator: FoulingCalculator):
        """Test rate estimation with single data point."""
        history = [
            FoulingHistoryEntry(
                timestamp=datetime.now(timezone.utc),
                cleanliness_factor=0.85,
                heat_duty_mw=350.0,
                cw_inlet_temp_c=25.0,
            )
        ]

        rate, r_squared = calculator.estimate_fouling_rate(history)

        # Should return default rate
        assert rate == calculator.DEFAULT_DECAY_RATE

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_rate_estimation_constant_cf(self, calculator: FoulingCalculator):
        """Test rate estimation with constant CF (no fouling)."""
        base_time = datetime.now(timezone.utc)
        history = [
            FoulingHistoryEntry(
                timestamp=base_time - timedelta(days=i),
                cleanliness_factor=0.85,  # Constant
                heat_duty_mw=350.0,
                cw_inlet_temp_c=25.0,
            )
            for i in range(30)
        ]

        rate, r_squared = calculator.estimate_fouling_rate(history, method="linear")

        # Rate should be near zero
        assert abs(rate) < 0.001


class TestCFPrediction:
    """Tests for CF prediction."""

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_predict_cf_linear(self, calculator: FoulingCalculator):
        """Test linear CF prediction."""
        current_cf = 0.90
        days = 30
        rate = 0.002

        predicted = calculator.predict_cf(current_cf, days, rate, model="linear")

        expected = current_cf - rate * days  # 0.90 - 0.06 = 0.84
        assert abs(predicted - expected) < 0.001

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_predict_cf_exponential(self, calculator: FoulingCalculator):
        """Test exponential CF prediction."""
        current_cf = 0.90
        days = 30
        rate = 0.02

        predicted = calculator.predict_cf(current_cf, days, rate, model="exponential")

        # Should be above asymptote
        assert predicted > calculator.ASYMPTOTIC_CF
        assert predicted < current_cf

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_predict_cf_clamped_minimum(self, calculator: FoulingCalculator):
        """Test CF prediction is clamped to minimum."""
        current_cf = 0.70
        days = 100
        rate = 0.01  # Would go below asymptote

        predicted = calculator.predict_cf(current_cf, days, rate, model="linear")

        # Should be clamped to asymptote
        assert predicted >= calculator.ASYMPTOTIC_CF

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_predict_cf_zero_days(self, calculator: FoulingCalculator):
        """Test CF prediction for zero days."""
        current_cf = 0.85

        predicted = calculator.predict_cf(current_cf, 0, 0.002)

        assert predicted == current_cf

    @pytest.mark.unit
    @pytest.mark.fouling
    @pytest.mark.parametrize("days,expected_cf", [
        (0, 0.90),
        (10, 0.88),
        (30, 0.84),
        (50, 0.80),
        (100, 0.70),  # Clamped to asymptote
    ])
    def test_predict_cf_timeline(self, calculator: FoulingCalculator, days, expected_cf):
        """Test CF prediction over time."""
        predicted = calculator.predict_cf(0.90, days, 0.002, model="linear")

        # Allow for clamping
        expected = max(calculator.ASYMPTOTIC_CF, expected_cf)
        assert abs(predicted - expected) < 0.01


class TestDaysToThreshold:
    """Tests for days-to-threshold calculation."""

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_days_to_threshold_linear(self, calculator: FoulingCalculator):
        """Test days to threshold with linear model."""
        current_cf = 0.90
        threshold = 0.80
        rate = 0.002

        days = calculator.calculate_days_to_threshold(
            current_cf, threshold, rate, model="linear"
        )

        # 0.90 - 0.80 = 0.10, at 0.002/day = 50 days
        assert days == 50

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_days_to_threshold_already_below(self, calculator: FoulingCalculator):
        """Test days to threshold when already below."""
        current_cf = 0.75
        threshold = 0.80

        days = calculator.calculate_days_to_threshold(current_cf, threshold, 0.002)

        assert days == 0

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_days_to_threshold_zero_rate(self, calculator: FoulingCalculator):
        """Test days to threshold with zero rate."""
        current_cf = 0.90
        threshold = 0.80

        days = calculator.calculate_days_to_threshold(current_cf, threshold, 0.0)

        assert days == -1  # Never reaches

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_days_to_threshold_exponential(self, calculator: FoulingCalculator):
        """Test days to threshold with exponential model."""
        current_cf = 0.90
        threshold = 0.70
        rate = 0.02

        days = calculator.calculate_days_to_threshold(
            current_cf, threshold, rate, model="exponential"
        )

        assert days > 0
        # Verify by back-calculation
        predicted = calculator.predict_cf(current_cf, days, rate, model="exponential")
        assert predicted <= threshold + 0.01

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_days_to_threshold_below_asymptote(self, calculator: FoulingCalculator):
        """Test days to threshold below asymptote."""
        current_cf = 0.70
        threshold = 0.50  # Below asymptote

        days = calculator.calculate_days_to_threshold(
            current_cf, threshold, 0.02, model="exponential"
        )

        assert days == -1  # Never reaches below asymptote


class TestCleaningCostEstimation:
    """Tests for cleaning cost estimation."""

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_cost_online_ball(self, calculator: FoulingCalculator):
        """Test cost for online ball cleaning."""
        cost = calculator.estimate_cleaning_cost(CleaningMethod.ONLINE_BALL)
        assert cost > 0
        assert cost < 1000  # Should be low cost

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_cost_offline_chemical(self, calculator: FoulingCalculator):
        """Test cost for offline chemical cleaning."""
        cost = calculator.estimate_cleaning_cost(CleaningMethod.OFFLINE_CHEMICAL)
        assert cost > 50000  # Should be expensive

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_cost_none(self, calculator: FoulingCalculator):
        """Test cost for no cleaning."""
        cost = calculator.estimate_cleaning_cost(CleaningMethod.NONE)
        assert cost == 0

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_cost_scales_with_condenser_size(
        self,
        calculator: FoulingCalculator,
        sample_condenser_config: CondenserConfig,
        large_condenser_config: CondenserConfig
    ):
        """Test cost scales with condenser size."""
        cost_normal = calculator.estimate_cleaning_cost(
            CleaningMethod.OFFLINE_HYDROLANCE, sample_condenser_config
        )
        cost_large = calculator.estimate_cleaning_cost(
            CleaningMethod.OFFLINE_HYDROLANCE, large_condenser_config
        )

        # Larger condenser should cost more
        assert cost_large > cost_normal


class TestCFRecovery:
    """Tests for CF recovery estimation."""

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_recovery_online_ball(self, calculator: FoulingCalculator):
        """Test recovery for online ball cleaning."""
        recovery = calculator.estimate_cf_recovery(CleaningMethod.ONLINE_BALL)
        assert 0.85 <= recovery <= 0.95

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_recovery_offline_chemical(self, calculator: FoulingCalculator):
        """Test recovery for offline chemical cleaning."""
        recovery = calculator.estimate_cf_recovery(CleaningMethod.OFFLINE_CHEMICAL)
        assert recovery >= 0.95  # Should be nearly complete recovery

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_recovery_ordering(self, calculator: FoulingCalculator):
        """Test that more aggressive methods have higher recovery."""
        recovery_ball = calculator.estimate_cf_recovery(CleaningMethod.ONLINE_BALL)
        recovery_brush = calculator.estimate_cf_recovery(CleaningMethod.ONLINE_BRUSH)
        recovery_hydro = calculator.estimate_cf_recovery(CleaningMethod.OFFLINE_HYDROLANCE)
        recovery_chem = calculator.estimate_cf_recovery(CleaningMethod.OFFLINE_CHEMICAL)

        assert recovery_ball <= recovery_brush <= recovery_hydro <= recovery_chem


class TestLostGenerationCost:
    """Tests for lost generation cost calculation."""

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_lost_generation_cost_basic(self, calculator: FoulingCalculator):
        """Test basic lost generation cost calculation."""
        cost = calculator.calculate_lost_generation_cost(
            cf_current=0.80,
            cf_clean=0.90,
            unit_load_mw=500.0,
            electricity_price_usd_mwh=50.0,
            hours=720  # 30 days
        )

        assert cost > 0

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_lost_generation_cost_no_degradation(self, calculator: FoulingCalculator):
        """Test lost generation cost with no degradation."""
        cost = calculator.calculate_lost_generation_cost(
            cf_current=0.90,
            cf_clean=0.90,
            unit_load_mw=500.0,
            electricity_price_usd_mwh=50.0,
            hours=720
        )

        assert cost == 0.0

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_lost_generation_scales_with_price(self, calculator: FoulingCalculator):
        """Test lost generation cost scales with electricity price."""
        cost_low = calculator.calculate_lost_generation_cost(
            0.80, 0.90, 500.0, 30.0, 720
        )
        cost_high = calculator.calculate_lost_generation_cost(
            0.80, 0.90, 500.0, 60.0, 720
        )

        assert cost_high == 2 * cost_low


class TestCleaningRecommendation:
    """Tests for cleaning recommendation."""

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_recommend_cleaning_clean_condenser(
        self,
        calculator: FoulingCalculator,
        fouling_history_clean: List[FoulingHistoryEntry]
    ):
        """Test recommendation for clean condenser."""
        result = calculator.recommend_cleaning(
            current_cf=0.88,
            history=fouling_history_clean
        )

        assert isinstance(result, FoulingPredictionResult)
        assert result.cleaning_method == CleaningMethod.ONLINE_BALL
        assert result.days_to_threshold > 0

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_recommend_cleaning_fouled_condenser(
        self,
        calculator: FoulingCalculator,
        fouling_history_rapid_decline: List[FoulingHistoryEntry]
    ):
        """Test recommendation for fouled condenser."""
        result = calculator.recommend_cleaning(
            current_cf=0.72,
            history=fouling_history_rapid_decline
        )

        # Should recommend aggressive cleaning
        assert result.cleaning_method in [
            CleaningMethod.OFFLINE_HYDROLANCE,
            CleaningMethod.OFFLINE_CHEMICAL
        ]

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_recommend_cleaning_has_provenance(
        self,
        calculator: FoulingCalculator,
        fouling_history_clean: List[FoulingHistoryEntry]
    ):
        """Test recommendation includes provenance hash."""
        result = calculator.recommend_cleaning(
            current_cf=0.85,
            history=fouling_history_clean
        )

        assert len(result.provenance_hash) == 64

    @pytest.mark.unit
    @pytest.mark.fouling
    def test_recommend_cleaning_roi_positive_for_fouled(
        self,
        calculator: FoulingCalculator,
        fouling_history_rapid_decline: List[FoulingHistoryEntry]
    ):
        """Test ROI is positive for fouled condenser."""
        result = calculator.recommend_cleaning(
            current_cf=0.70,
            history=fouling_history_rapid_decline,
            electricity_price_usd_mwh=100.0  # High price
        )

        # Cleaning should pay off
        assert result.roi_cleaning > 0


class TestDeterminism:
    """Tests for deterministic behavior."""

    @pytest.mark.unit
    @pytest.mark.fouling
    @pytest.mark.golden
    def test_prediction_is_deterministic(self, calculator: FoulingCalculator):
        """Test CF prediction is deterministic."""
        results = [
            calculator.predict_cf(0.85, 30, 0.002)
            for _ in range(100)
        ]

        assert len(set(results)) == 1

    @pytest.mark.unit
    @pytest.mark.fouling
    @pytest.mark.golden
    def test_days_to_threshold_is_deterministic(self, calculator: FoulingCalculator):
        """Test days to threshold is deterministic."""
        results = [
            calculator.calculate_days_to_threshold(0.90, 0.80, 0.002)
            for _ in range(100)
        ]

        assert len(set(results)) == 1

    @pytest.mark.unit
    @pytest.mark.fouling
    @pytest.mark.golden
    def test_recommendation_hash_is_deterministic(
        self,
        calculator: FoulingCalculator,
        fouling_history_clean: List[FoulingHistoryEntry]
    ):
        """Test recommendation hash is deterministic."""
        results = [
            calculator.recommend_cleaning(0.85, fouling_history_clean)
            for _ in range(10)
        ]

        hashes = [r.provenance_hash for r in results]
        assert len(set(hashes)) == 1


class TestPerformance:
    """Performance tests."""

    @pytest.mark.unit
    @pytest.mark.fouling
    @pytest.mark.performance
    def test_prediction_speed(
        self,
        calculator: FoulingCalculator,
        performance_timer
    ):
        """Test prediction calculation speed."""
        timer = performance_timer()

        with timer:
            for _ in range(10000):
                calculator.predict_cf(0.85, 30, 0.002)

        assert timer.elapsed < 1.0

    @pytest.mark.unit
    @pytest.mark.fouling
    @pytest.mark.performance
    def test_rate_estimation_speed(
        self,
        calculator: FoulingCalculator,
        fouling_history_clean: List[FoulingHistoryEntry],
        performance_timer
    ):
        """Test rate estimation speed."""
        timer = performance_timer()

        with timer:
            for _ in range(1000):
                calculator.estimate_fouling_rate(fouling_history_clean)

        assert timer.elapsed < 2.0
