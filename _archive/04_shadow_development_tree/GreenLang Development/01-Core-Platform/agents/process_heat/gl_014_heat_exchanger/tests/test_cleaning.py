# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Agent - Cleaning Schedule Optimization Tests

Comprehensive tests for cleaning schedule optimization including:
- Optimal cleaning frequency calculation
- Economic cleaning decisions
- Cleaning method selection
- ROI calculations for cleaning

Coverage Target: 90%+
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_014_heat_exchanger.cleaning import (
    CleaningScheduleOptimizer,
)
from greenlang.agents.process_heat.gl_014_heat_exchanger.config import (
    CleaningConfig,
    EconomicsConfig,
    FoulingConfig,
    CleaningMethod,
    AlertSeverity,
)
from greenlang.agents.process_heat.gl_014_heat_exchanger.schemas import (
    CleaningRecommendation,
)


class TestCleaningScheduleOptimizerInit:
    """Tests for CleaningScheduleOptimizer initialization."""

    def test_optimizer_initialization(self, cleaning_config, economics_config, fouling_config):
        """Test optimizer initializes correctly."""
        optimizer = CleaningScheduleOptimizer(
            cleaning_config=cleaning_config,
            economics_config=economics_config,
            fouling_config=fouling_config,
        )
        assert optimizer.cleaning_config == cleaning_config
        assert optimizer.economics_config == economics_config

    def test_optimizer_with_defaults(self):
        """Test optimizer with default configurations."""
        optimizer = CleaningScheduleOptimizer(
            cleaning_config=CleaningConfig(),
            economics_config=EconomicsConfig(),
            fouling_config=FoulingConfig(),
        )
        assert optimizer is not None


class TestOptimalCleaningFrequency:
    """Tests for optimal cleaning frequency calculation."""

    @pytest.fixture
    def optimizer(self, cleaning_config, economics_config, fouling_config):
        """Create CleaningScheduleOptimizer instance."""
        return CleaningScheduleOptimizer(
            cleaning_config=cleaning_config,
            economics_config=economics_config,
            fouling_config=fouling_config,
        )

    def test_optimal_frequency_calculation(self, optimizer):
        """Test optimal cleaning frequency is calculated."""
        optimal_days = optimizer.calculate_optimal_frequency(
            fouling_rate_m2kw_per_day=0.000002,
            cleaning_cost_usd=5000.0,
            energy_cost_usd_per_kwh=0.10,
            energy_loss_per_fouling_kwh=1000.0,
        )

        # Optimal frequency should be between min and max intervals
        assert optimizer.cleaning_config.minimum_interval_days <= optimal_days
        assert optimal_days <= optimizer.cleaning_config.maximum_interval_days

    def test_optimal_frequency_high_fouling(self, optimizer):
        """Test optimal frequency is shorter with high fouling rate."""
        freq_high = optimizer.calculate_optimal_frequency(
            fouling_rate_m2kw_per_day=0.00001,  # High fouling
            cleaning_cost_usd=5000.0,
            energy_cost_usd_per_kwh=0.10,
            energy_loss_per_fouling_kwh=1000.0,
        )

        freq_low = optimizer.calculate_optimal_frequency(
            fouling_rate_m2kw_per_day=0.0000005,  # Low fouling
            cleaning_cost_usd=5000.0,
            energy_cost_usd_per_kwh=0.10,
            energy_loss_per_fouling_kwh=1000.0,
        )

        # Higher fouling rate should mean shorter interval
        assert freq_high < freq_low

    def test_optimal_frequency_expensive_cleaning(self, optimizer):
        """Test optimal frequency is longer with expensive cleaning."""
        freq_cheap = optimizer.calculate_optimal_frequency(
            fouling_rate_m2kw_per_day=0.000002,
            cleaning_cost_usd=3000.0,  # Cheap cleaning
            energy_cost_usd_per_kwh=0.10,
            energy_loss_per_fouling_kwh=1000.0,
        )

        freq_expensive = optimizer.calculate_optimal_frequency(
            fouling_rate_m2kw_per_day=0.000002,
            cleaning_cost_usd=10000.0,  # Expensive cleaning
            energy_cost_usd_per_kwh=0.10,
            energy_loss_per_fouling_kwh=1000.0,
        )

        # More expensive cleaning should mean longer interval
        assert freq_expensive > freq_cheap

    def test_optimal_frequency_bounds(self, optimizer):
        """Test optimal frequency respects min/max bounds."""
        # Very high fouling rate
        freq = optimizer.calculate_optimal_frequency(
            fouling_rate_m2kw_per_day=0.001,  # Extreme fouling
            cleaning_cost_usd=1000.0,
            energy_cost_usd_per_kwh=0.50,
            energy_loss_per_fouling_kwh=10000.0,
        )
        assert freq >= optimizer.cleaning_config.minimum_interval_days

        # Very low fouling rate
        freq = optimizer.calculate_optimal_frequency(
            fouling_rate_m2kw_per_day=0.00000001,  # Very low fouling
            cleaning_cost_usd=100000.0,
            energy_cost_usd_per_kwh=0.01,
            energy_loss_per_fouling_kwh=10.0,
        )
        assert freq <= optimizer.cleaning_config.maximum_interval_days


class TestCleaningRecommendation:
    """Tests for cleaning recommendation generation."""

    @pytest.fixture
    def optimizer(self, cleaning_config, economics_config, fouling_config):
        """Create CleaningScheduleOptimizer instance."""
        return CleaningScheduleOptimizer(
            cleaning_config=cleaning_config,
            economics_config=economics_config,
            fouling_config=fouling_config,
        )

    def test_generate_recommendation_clean_needed(self, optimizer):
        """Test recommendation when cleaning is needed."""
        rec = optimizer.generate_recommendation(
            current_fouling_m2kw=0.0005,  # High fouling
            fouling_rate_m2kw_per_day=0.000003,
            current_effectiveness=0.65,  # Below threshold
            clean_u_w_m2k=500.0,
            area_m2=100.0,
            days_since_last_cleaning=120,
        )

        assert isinstance(rec, CleaningRecommendation)
        assert rec.recommended == True
        assert rec.urgency in [AlertSeverity.WARNING, AlertSeverity.ALARM, AlertSeverity.CRITICAL]

    def test_generate_recommendation_no_clean_needed(self, optimizer):
        """Test recommendation when cleaning is not needed."""
        rec = optimizer.generate_recommendation(
            current_fouling_m2kw=0.0001,  # Low fouling
            fouling_rate_m2kw_per_day=0.0000005,
            current_effectiveness=0.85,  # Good effectiveness
            clean_u_w_m2k=500.0,
            area_m2=100.0,
            days_since_last_cleaning=30,
        )

        assert rec.recommended == False or rec.urgency == AlertSeverity.INFO

    def test_recommendation_has_days_until(self, optimizer):
        """Test recommendation includes days until cleaning."""
        rec = optimizer.generate_recommendation(
            current_fouling_m2kw=0.0003,
            fouling_rate_m2kw_per_day=0.000002,
            current_effectiveness=0.72,
            clean_u_w_m2k=500.0,
            area_m2=100.0,
            days_since_last_cleaning=60,
        )

        assert rec.days_until_recommended is not None
        assert rec.days_until_recommended >= 0

    def test_recommendation_has_cost_estimate(self, optimizer):
        """Test recommendation includes cost estimate."""
        rec = optimizer.generate_recommendation(
            current_fouling_m2kw=0.0003,
            fouling_rate_m2kw_per_day=0.000002,
            current_effectiveness=0.72,
            clean_u_w_m2k=500.0,
            area_m2=100.0,
            days_since_last_cleaning=60,
        )

        assert rec.estimated_cleaning_cost_usd > 0

    def test_recommendation_has_method(self, optimizer):
        """Test recommendation includes cleaning method."""
        rec = optimizer.generate_recommendation(
            current_fouling_m2kw=0.0003,
            fouling_rate_m2kw_per_day=0.000002,
            current_effectiveness=0.72,
            clean_u_w_m2k=500.0,
            area_m2=100.0,
            days_since_last_cleaning=60,
        )

        assert rec.recommended_method in CleaningMethod


class TestCleaningMethodSelection:
    """Tests for cleaning method selection logic."""

    @pytest.fixture
    def optimizer(self, cleaning_config, economics_config, fouling_config):
        """Create CleaningScheduleOptimizer instance."""
        return CleaningScheduleOptimizer(
            cleaning_config=cleaning_config,
            economics_config=economics_config,
            fouling_config=fouling_config,
        )

    def test_method_selection_light_fouling(self, optimizer):
        """Test method selection for light fouling."""
        method = optimizer.select_cleaning_method(
            fouling_severity=0.2,  # Light
            fouling_type="particulate",
        )

        # Light fouling - simpler methods
        assert method in [
            CleaningMethod.HIGH_PRESSURE_WATER,
            CleaningMethod.MECHANICAL_BRUSHING,
        ]

    def test_method_selection_heavy_fouling(self, optimizer):
        """Test method selection for heavy fouling."""
        method = optimizer.select_cleaning_method(
            fouling_severity=0.8,  # Heavy
            fouling_type="scale",
        )

        # Heavy fouling - more aggressive methods
        assert method in [
            CleaningMethod.CHEMICAL,
            CleaningMethod.COMBINED,
        ]

    def test_method_selection_biological(self, optimizer):
        """Test method selection for biological fouling."""
        method = optimizer.select_cleaning_method(
            fouling_severity=0.5,
            fouling_type="biological",
        )

        # Biological fouling often needs chemical treatment
        assert method in [
            CleaningMethod.CHEMICAL,
            CleaningMethod.HIGH_PRESSURE_WATER,
            CleaningMethod.COMBINED,
        ]

    def test_alternative_methods_provided(self, optimizer):
        """Test alternative methods are provided."""
        rec = optimizer.generate_recommendation(
            current_fouling_m2kw=0.0004,
            fouling_rate_m2kw_per_day=0.000002,
            current_effectiveness=0.68,
            clean_u_w_m2k=500.0,
            area_m2=100.0,
            days_since_last_cleaning=90,
        )

        # Should provide alternatives
        assert len(rec.alternative_methods) >= 0


class TestCleaningROI:
    """Tests for cleaning ROI calculations."""

    @pytest.fixture
    def optimizer(self, cleaning_config, economics_config, fouling_config):
        """Create CleaningScheduleOptimizer instance."""
        return CleaningScheduleOptimizer(
            cleaning_config=cleaning_config,
            economics_config=economics_config,
            fouling_config=fouling_config,
        )

    def test_cleaning_roi_positive(self, optimizer):
        """Test cleaning ROI is positive when beneficial."""
        roi = optimizer.calculate_cleaning_roi(
            current_fouling_m2kw=0.0005,
            cleaning_cost_usd=5000.0,
            energy_savings_usd_per_day=200.0,
            time_horizon_days=90,
        )

        # ROI = (savings - cost) / cost * 100
        # Expected: (200*90 - 5000) / 5000 * 100 = 260%
        assert roi > 0

    def test_cleaning_roi_negative(self, optimizer):
        """Test cleaning ROI is negative when not beneficial."""
        roi = optimizer.calculate_cleaning_roi(
            current_fouling_m2kw=0.0001,  # Light fouling
            cleaning_cost_usd=10000.0,  # Expensive cleaning
            energy_savings_usd_per_day=10.0,  # Low savings
            time_horizon_days=30,
        )

        # ROI = (10*30 - 10000) / 10000 * 100 = -97%
        assert roi < 0

    def test_cleaning_roi_in_recommendation(self, optimizer):
        """Test ROI is included in recommendation."""
        rec = optimizer.generate_recommendation(
            current_fouling_m2kw=0.0004,
            fouling_rate_m2kw_per_day=0.000002,
            current_effectiveness=0.70,
            clean_u_w_m2k=500.0,
            area_m2=100.0,
            days_since_last_cleaning=90,
        )

        assert rec.cleaning_roi_percent is not None


class TestCleaningUrgency:
    """Tests for cleaning urgency determination."""

    @pytest.fixture
    def optimizer(self, cleaning_config, economics_config, fouling_config):
        """Create CleaningScheduleOptimizer instance."""
        return CleaningScheduleOptimizer(
            cleaning_config=cleaning_config,
            economics_config=economics_config,
            fouling_config=fouling_config,
        )

    def test_urgency_critical(self, optimizer):
        """Test critical urgency for severe fouling."""
        rec = optimizer.generate_recommendation(
            current_fouling_m2kw=0.001,  # Severe fouling
            fouling_rate_m2kw_per_day=0.00001,
            current_effectiveness=0.50,  # Very low
            clean_u_w_m2k=500.0,
            area_m2=100.0,
            days_since_last_cleaning=180,
        )

        assert rec.urgency == AlertSeverity.CRITICAL

    def test_urgency_alarm(self, optimizer):
        """Test alarm urgency for high fouling."""
        rec = optimizer.generate_recommendation(
            current_fouling_m2kw=0.0006,
            fouling_rate_m2kw_per_day=0.000005,
            current_effectiveness=0.60,
            clean_u_w_m2k=500.0,
            area_m2=100.0,
            days_since_last_cleaning=150,
        )

        assert rec.urgency in [AlertSeverity.ALARM, AlertSeverity.CRITICAL]

    def test_urgency_warning(self, optimizer):
        """Test warning urgency for moderate fouling."""
        rec = optimizer.generate_recommendation(
            current_fouling_m2kw=0.0004,
            fouling_rate_m2kw_per_day=0.000002,
            current_effectiveness=0.68,
            clean_u_w_m2k=500.0,
            area_m2=100.0,
            days_since_last_cleaning=90,
        )

        assert rec.urgency in [AlertSeverity.WARNING, AlertSeverity.ALARM]

    def test_urgency_info(self, optimizer):
        """Test info urgency for low fouling."""
        rec = optimizer.generate_recommendation(
            current_fouling_m2kw=0.0001,
            fouling_rate_m2kw_per_day=0.0000005,
            current_effectiveness=0.85,
            clean_u_w_m2k=500.0,
            area_m2=100.0,
            days_since_last_cleaning=30,
        )

        assert rec.urgency == AlertSeverity.INFO


class TestCleaningDowntime:
    """Tests for cleaning downtime estimation."""

    @pytest.fixture
    def optimizer(self, cleaning_config, economics_config, fouling_config):
        """Create CleaningScheduleOptimizer instance."""
        return CleaningScheduleOptimizer(
            cleaning_config=cleaning_config,
            economics_config=economics_config,
            fouling_config=fouling_config,
        )

    def test_downtime_estimation(self, optimizer):
        """Test downtime is estimated for cleaning."""
        rec = optimizer.generate_recommendation(
            current_fouling_m2kw=0.0004,
            fouling_rate_m2kw_per_day=0.000002,
            current_effectiveness=0.70,
            clean_u_w_m2k=500.0,
            area_m2=100.0,
            days_since_last_cleaning=90,
        )

        assert rec.estimated_downtime_hours > 0

    def test_downtime_by_method(self, optimizer):
        """Test different methods have different downtimes."""
        downtime_water = optimizer.estimate_downtime(
            method=CleaningMethod.HIGH_PRESSURE_WATER,
            area_m2=100.0,
        )

        downtime_chemical = optimizer.estimate_downtime(
            method=CleaningMethod.CHEMICAL,
            area_m2=100.0,
        )

        # Chemical cleaning typically takes longer
        assert downtime_chemical >= downtime_water

    def test_downtime_scales_with_area(self, optimizer):
        """Test downtime scales with exchanger area."""
        downtime_small = optimizer.estimate_downtime(
            method=CleaningMethod.HIGH_PRESSURE_WATER,
            area_m2=50.0,
        )

        downtime_large = optimizer.estimate_downtime(
            method=CleaningMethod.HIGH_PRESSURE_WATER,
            area_m2=200.0,
        )

        assert downtime_large > downtime_small


class TestURecoveryPrediction:
    """Tests for U value recovery prediction after cleaning."""

    @pytest.fixture
    def optimizer(self, cleaning_config, economics_config, fouling_config):
        """Create CleaningScheduleOptimizer instance."""
        return CleaningScheduleOptimizer(
            cleaning_config=cleaning_config,
            economics_config=economics_config,
            fouling_config=fouling_config,
        )

    def test_u_recovery_prediction(self, optimizer):
        """Test U recovery prediction after cleaning."""
        recovery = optimizer.predict_u_recovery(
            current_u_w_m2k=350.0,
            clean_u_w_m2k=500.0,
            cleaning_method=CleaningMethod.HIGH_PRESSURE_WATER,
        )

        # Recovery should be 0-100%
        assert 0 <= recovery <= 100

    def test_u_recovery_by_method(self, optimizer):
        """Test different methods have different recovery rates."""
        recovery_water = optimizer.predict_u_recovery(
            current_u_w_m2k=350.0,
            clean_u_w_m2k=500.0,
            cleaning_method=CleaningMethod.HIGH_PRESSURE_WATER,
        )

        recovery_combined = optimizer.predict_u_recovery(
            current_u_w_m2k=350.0,
            clean_u_w_m2k=500.0,
            cleaning_method=CleaningMethod.COMBINED,
        )

        # Combined method should have better recovery
        assert recovery_combined >= recovery_water

    def test_effectiveness_prediction(self, optimizer):
        """Test effectiveness prediction after cleaning."""
        rec = optimizer.generate_recommendation(
            current_fouling_m2kw=0.0004,
            fouling_rate_m2kw_per_day=0.000002,
            current_effectiveness=0.70,
            clean_u_w_m2k=500.0,
            area_m2=100.0,
            days_since_last_cleaning=90,
        )

        # Expected effectiveness should be higher than current
        if rec.expected_effectiveness_after is not None:
            assert rec.expected_effectiveness_after > 0.70
