"""
GL-017 CONDENSYNC Agent - Tube Fouling Detector Tests

Unit tests for TubeFoulingDetector including backpressure analysis,
fouling severity determination, and economic impact calculations.

Coverage targets:
    - Baseline curve generation
    - Expected backpressure calculation
    - Fouling severity classification
    - Heat rate penalty calculation
    - Cleaning recommendation logic
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_017_condenser_optimization.tube_fouling import (
    TubeFoulingDetector,
    BackpressureConstants,
    BackpressureDataPoint,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.config import (
    TubeFoulingConfig,
    PerformanceConfig,
)
from greenlang.agents.process_heat.gl_017_condenser_optimization.schemas import (
    TubeFoulingResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def fouling_config():
    """Create default tube fouling configuration."""
    return TubeFoulingConfig()


@pytest.fixture
def performance_config():
    """Create default performance configuration."""
    return PerformanceConfig()


@pytest.fixture
def detector(fouling_config, performance_config):
    """Create TubeFoulingDetector instance."""
    return TubeFoulingDetector(fouling_config, performance_config)


# =============================================================================
# CONSTANTS TESTS
# =============================================================================

class TestBackpressureConstants:
    """Test BackpressureConstants values."""

    def test_heat_rate_penalty(self):
        """Test heat rate penalty constant."""
        assert BackpressureConstants.HEAT_RATE_PENALTY_BTU_KWH_PER_INHG == 80.0

    def test_capacity_loss_constant(self):
        """Test capacity loss constant."""
        assert BackpressureConstants.CAPACITY_LOSS_PCT_PER_INHG == 0.75

    def test_saturation_temp_to_vacuum(self):
        """Test saturation temperature to vacuum conversion."""
        vacuum = BackpressureConstants.saturation_temp_to_vacuum(101.0)
        assert vacuum > 0

    def test_vacuum_to_saturation_temp(self):
        """Test vacuum to saturation temperature conversion."""
        temp = BackpressureConstants.vacuum_to_saturation_temp(1.5)
        # Should be around 100F for 1.5 inHgA
        assert 90.0 < temp < 120.0

    def test_round_trip_conversion(self):
        """Test round-trip temperature-vacuum conversion."""
        original_temp = 101.0
        vacuum = BackpressureConstants.saturation_temp_to_vacuum(original_temp)
        recovered_temp = BackpressureConstants.vacuum_to_saturation_temp(vacuum)

        # Should be close to original
        assert abs(recovered_temp - original_temp) < 5.0


# =============================================================================
# DETECTOR INITIALIZATION TESTS
# =============================================================================

class TestDetectorInitialization:
    """Test detector initialization."""

    def test_basic_initialization(self, fouling_config, performance_config):
        """Test detector initializes correctly."""
        detector = TubeFoulingDetector(fouling_config, performance_config)
        assert detector is not None
        assert detector._baseline_curve is not None

    def test_baseline_curve_built(self, detector):
        """Test baseline curve is built on initialization."""
        assert detector._baseline_curve is not None
        assert len(detector._baseline_curve) > 0

    def test_baseline_curve_structure(self, detector):
        """Test baseline curve has expected structure."""
        # Should have load keys
        assert 100 in detector._baseline_curve

        # Each load should have temperature keys
        assert 70 in detector._baseline_curve[100]

        # Values should be positive backpressures
        assert detector._baseline_curve[100][70] > 0


# =============================================================================
# EXPECTED BACKPRESSURE TESTS
# =============================================================================

class TestExpectedBackpressure:
    """Test expected backpressure calculation."""

    def test_design_conditions(self, detector):
        """Test expected BP at design conditions."""
        expected_bp = detector._calculate_expected_backpressure(
            load_pct=100.0,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=100000.0,
        )

        # Should be close to design backpressure (1.5 inHgA)
        assert 1.0 < expected_bp < 2.0

    def test_higher_load_higher_bp(self, detector):
        """Test higher load gives higher backpressure."""
        bp_low = detector._calculate_expected_backpressure(
            load_pct=50.0,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=100000.0,
        )

        bp_high = detector._calculate_expected_backpressure(
            load_pct=100.0,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=100000.0,
        )

        assert bp_high > bp_low

    def test_higher_inlet_temp_higher_bp(self, detector):
        """Test higher inlet temperature gives higher backpressure."""
        bp_cold = detector._calculate_expected_backpressure(
            load_pct=85.0,
            cw_inlet_temp_f=60.0,
            cw_flow_gpm=100000.0,
        )

        bp_hot = detector._calculate_expected_backpressure(
            load_pct=85.0,
            cw_inlet_temp_f=90.0,
            cw_flow_gpm=100000.0,
        )

        assert bp_hot > bp_cold

    def test_lower_flow_higher_bp(self, detector):
        """Test lower flow gives higher backpressure."""
        bp_high_flow = detector._calculate_expected_backpressure(
            load_pct=85.0,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=100000.0,
        )

        bp_low_flow = detector._calculate_expected_backpressure(
            load_pct=85.0,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=80000.0,
        )

        assert bp_low_flow >= bp_high_flow

    def test_interpolation(self, detector):
        """Test interpolation between curve points."""
        # Test a point between curve points
        bp = detector._calculate_expected_backpressure(
            load_pct=75.0,  # Between 70 and 80
            cw_inlet_temp_f=75.0,  # Between 70 and 80
            cw_flow_gpm=100000.0,
        )

        assert 0.5 < bp < 5.0


# =============================================================================
# FOULING ANALYSIS TESTS
# =============================================================================

class TestFoulingAnalysis:
    """Test fouling analysis method."""

    def test_analyze_no_fouling(self, detector):
        """Test analysis with no fouling."""
        result = detector.analyze_fouling(
            current_backpressure_inhga=1.5,  # At expected
            load_pct=100.0,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=100000.0,
        )

        assert isinstance(result, TubeFoulingResult)
        assert result.fouling_severity in ["none", "light"]

    def test_analyze_moderate_fouling(self, detector):
        """Test analysis with moderate fouling."""
        result = detector.analyze_fouling(
            current_backpressure_inhga=2.0,  # 0.5 inHg above expected
            load_pct=100.0,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=100000.0,
        )

        assert result.fouling_detected is True
        assert result.backpressure_penalty_inhg > 0

    def test_analyze_severe_fouling(self, detector):
        """Test analysis with severe fouling."""
        result = detector.analyze_fouling(
            current_backpressure_inhga=3.0,  # Far above expected
            load_pct=100.0,
            cw_inlet_temp_f=70.0,
            cw_flow_gpm=100000.0,
        )

        assert result.fouling_detected is True
        assert result.fouling_severity in ["moderate", "severe"]

    def test_result_components(self, detector):
        """Test all result components are populated."""
        result = detector.analyze_fouling(
            current_backpressure_inhga=1.8,
            load_pct=85.0,
            cw_inlet_temp_f=75.0,
            cw_flow_gpm=90000.0,
        )

        assert result.current_backpressure_inhga is not None
        assert result.expected_backpressure_inhga is not None
        assert result.backpressure_penalty_inhg is not None
        assert result.backpressure_deviation_pct is not None
        assert result.heat_rate_penalty_btu_kwh is not None
        assert result.efficiency_loss_pct is not None


# =============================================================================
# FOULING SEVERITY TESTS
# =============================================================================

class TestFoulingSeverity:
    """Test fouling severity determination."""

    def test_severity_none(self, detector):
        """Test none severity for low penalty."""
        severity = detector._determine_fouling_severity(0.1)
        assert severity == "none"

    def test_severity_light(self, detector):
        """Test light severity."""
        severity = detector._determine_fouling_severity(0.35)
        assert severity == "light"

    def test_severity_moderate(self, detector):
        """Test moderate severity."""
        severity = detector._determine_fouling_severity(0.55)
        assert severity == "moderate"

    def test_severity_severe(self, detector):
        """Test severe severity."""
        severity = detector._determine_fouling_severity(1.0)
        assert severity == "severe"

    @pytest.mark.parametrize("penalty,expected", [
        (0.0, "none"),
        (0.2, "none"),
        (0.35, "light"),
        (0.45, "light"),
        (0.6, "moderate"),
        (0.8, "severe"),
    ])
    def test_severity_thresholds(self, detector, penalty, expected):
        """Test severity at various penalty levels."""
        severity = detector._determine_fouling_severity(penalty)
        assert severity == expected


# =============================================================================
# HEAT RATE PENALTY TESTS
# =============================================================================

class TestHeatRatePenalty:
    """Test heat rate penalty calculations."""

    def test_no_penalty_for_no_deviation(self, detector):
        """Test no penalty when no backpressure deviation."""
        penalty = detector._calculate_heat_rate_penalty(0.0)
        assert penalty == 0.0

    def test_no_penalty_for_negative_deviation(self, detector):
        """Test no penalty for negative deviation (better than expected)."""
        penalty = detector._calculate_heat_rate_penalty(-0.1)
        assert penalty == 0.0

    def test_penalty_calculation(self, detector):
        """Test heat rate penalty calculation."""
        penalty = detector._calculate_heat_rate_penalty(0.5)

        # 0.5 inHg * 80 BTU/kWh/inHg = 40 BTU/kWh
        expected = 0.5 * BackpressureConstants.HEAT_RATE_PENALTY_BTU_KWH_PER_INHG
        assert penalty == pytest.approx(expected, rel=0.01)

    @pytest.mark.parametrize("bp_penalty,expected_hr_penalty", [
        (0.0, 0.0),
        (0.5, 40.0),
        (1.0, 80.0),
        (2.0, 160.0),
    ])
    def test_penalty_linearity(self, detector, bp_penalty, expected_hr_penalty):
        """Test heat rate penalty is linear with backpressure."""
        penalty = detector._calculate_heat_rate_penalty(bp_penalty)
        assert penalty == pytest.approx(expected_hr_penalty, rel=0.01)


# =============================================================================
# CAPACITY LOSS TESTS
# =============================================================================

class TestCapacityLoss:
    """Test capacity loss calculations."""

    def test_no_loss_for_no_deviation(self, detector):
        """Test no capacity loss when no deviation."""
        loss = detector._calculate_lost_capacity(0.0, 500.0)
        assert loss == 0.0

    def test_capacity_loss_calculation(self, detector):
        """Test capacity loss calculation."""
        loss = detector._calculate_lost_capacity(0.5, 500.0)

        # 0.5 inHg * 0.75%/inHg = 0.375%
        # 500 MW * 0.375% = 1.875 MW
        expected = 500.0 * 0.5 * BackpressureConstants.CAPACITY_LOSS_PCT_PER_INHG / 100
        assert loss == pytest.approx(expected, rel=0.01)


# =============================================================================
# DAILY COST TESTS
# =============================================================================

class TestDailyCost:
    """Test daily cost calculations."""

    def test_no_cost_for_no_loss(self, detector):
        """Test no cost when no capacity loss."""
        cost = detector._calculate_daily_cost(0.0, 85.0, 50.0)
        assert cost == 0.0

    def test_daily_cost_calculation(self, detector):
        """Test daily cost calculation."""
        cost = detector._calculate_daily_cost(
            lost_capacity_mw=2.0,
            load_pct=100.0,
            price_usd_mwh=50.0,
        )

        # 2 MW * 24 hours * $50/MWh = $2400
        expected = 2.0 * 24.0 * 50.0
        assert cost == pytest.approx(expected, rel=0.01)

    def test_daily_cost_adjusted_for_load(self, detector):
        """Test daily cost is adjusted for load."""
        cost_full = detector._calculate_daily_cost(2.0, 100.0, 50.0)
        cost_half = detector._calculate_daily_cost(2.0, 50.0, 50.0)

        # Half load should have half the cost
        assert cost_half == pytest.approx(cost_full / 2, rel=0.01)


# =============================================================================
# CLEANING RECOMMENDATION TESTS
# =============================================================================

class TestCleaningRecommendation:
    """Test cleaning recommendation logic."""

    def test_recommend_cleaning_for_high_cost(self, detector):
        """Test cleaning recommended for high daily cost."""
        should_clean = detector._should_recommend_cleaning(
            bp_penalty_inhg=0.3,
            daily_cost_usd=3000.0,  # High cost
            cleaning_cost_usd=50000.0,
        )

        # Payback = 50000 / 3000 = 16.7 days < 30 days
        assert should_clean is True

    def test_no_recommendation_for_low_cost(self, detector):
        """Test no cleaning recommended for low cost."""
        should_clean = detector._should_recommend_cleaning(
            bp_penalty_inhg=0.1,
            daily_cost_usd=500.0,  # Low cost
            cleaning_cost_usd=50000.0,
        )

        # Payback = 50000 / 500 = 100 days > 30 days
        assert should_clean is False

    def test_recommend_for_high_penalty(self, detector):
        """Test cleaning recommended for high penalty regardless of cost."""
        should_clean = detector._should_recommend_cleaning(
            bp_penalty_inhg=1.0,  # Above alarm threshold
            daily_cost_usd=100.0,  # Low cost
            cleaning_cost_usd=50000.0,
        )

        assert should_clean is True


# =============================================================================
# CLEANING METHOD TESTS
# =============================================================================

class TestCleaningMethod:
    """Test cleaning method recommendations."""

    def test_no_cleaning_for_none(self, detector):
        """Test no cleaning method for none severity."""
        method = detector._recommend_cleaning_method("none")
        assert method is None

    def test_online_cleaning_for_light(self, detector):
        """Test online cleaning for light severity."""
        method = detector._recommend_cleaning_method("light")
        assert "online" in method.lower()

    def test_intensive_cleaning_for_moderate(self, detector):
        """Test intensive cleaning for moderate severity."""
        method = detector._recommend_cleaning_method("moderate")
        assert "online" in method.lower()

    def test_offline_cleaning_for_severe(self, detector):
        """Test offline cleaning for severe severity."""
        method = detector._recommend_cleaning_method("severe")
        assert "offline" in method.lower()


# =============================================================================
# TREND ANALYSIS TESTS
# =============================================================================

class TestTrendAnalysis:
    """Test fouling trend analysis."""

    def test_trend_insufficient_data(self, detector):
        """Test trend with insufficient data."""
        trend = detector._analyze_trend()
        assert trend == "insufficient_data"

    def test_trend_stable(self, detector):
        """Test stable trend detection."""
        # Add stable readings
        for _ in range(20):
            detector._record_data_point(1.6, 85.0, 75.0, 90000.0)

        trend = detector._analyze_trend()
        assert trend == "stable"

    def test_trend_degrading(self, detector):
        """Test degrading trend detection."""
        # Add degrading readings
        base_time = datetime.now(timezone.utc)
        for i in range(20):
            # Increasing backpressure
            bp = 1.5 + (i * 0.02)
            detector._history.append(BackpressureDataPoint(
                timestamp=base_time + timedelta(hours=i),
                backpressure_inhga=bp,
                load_pct=85.0,
                cw_inlet_temp_f=75.0,
                cw_flow_gpm=90000.0,
            ))

        trend = detector._analyze_trend()
        assert trend == "degrading"

    def test_trend_improving(self, detector):
        """Test improving trend detection (after cleaning)."""
        base_time = datetime.now(timezone.utc)
        for i in range(20):
            # Decreasing backpressure
            bp = 2.0 - (i * 0.02)
            detector._history.append(BackpressureDataPoint(
                timestamp=base_time + timedelta(hours=i),
                backpressure_inhga=bp,
                load_pct=85.0,
                cw_inlet_temp_f=75.0,
                cw_flow_gpm=90000.0,
            ))

        trend = detector._analyze_trend()
        assert trend == "improving"


# =============================================================================
# HISTORY TESTS
# =============================================================================

class TestHistory:
    """Test historical data management."""

    def test_record_data_point(self, detector):
        """Test recording data points."""
        detector._record_data_point(1.6, 85.0, 75.0, 90000.0)

        assert len(detector._history) == 1

    def test_history_trimming(self, detector):
        """Test old history is trimmed."""
        # Add old data point
        old_dp = BackpressureDataPoint(
            timestamp=datetime.now(timezone.utc) - timedelta(days=40),
            backpressure_inhga=1.6,
            load_pct=85.0,
            cw_inlet_temp_f=75.0,
            cw_flow_gpm=90000.0,
        )
        detector._history.append(old_dp)

        # Record new data - triggers trimming
        detector._record_data_point(1.65, 85.0, 75.0, 90000.0)

        # Old entry should be removed (older than 30 days)
        assert len(detector._history) == 1

    def test_get_historical_penalties(self, detector):
        """Test retrieving historical penalties."""
        # Add some history
        for i in range(5):
            detector._record_data_point(1.6 + (i * 0.05), 85.0, 75.0, 90000.0)

        penalties = detector.get_historical_penalties(days=30)

        assert isinstance(penalties, list)
        assert len(penalties) == 5


# =============================================================================
# INPUT VALIDATION TESTS
# =============================================================================

class TestInputValidation:
    """Test input validation."""

    def test_invalid_backpressure(self, detector):
        """Test invalid backpressure raises error."""
        with pytest.raises(ValueError):
            detector.analyze_fouling(
                current_backpressure_inhga=0.0,
                load_pct=85.0,
                cw_inlet_temp_f=75.0,
                cw_flow_gpm=90000.0,
            )

    def test_invalid_load(self, detector):
        """Test invalid load raises error."""
        with pytest.raises(ValueError):
            detector.analyze_fouling(
                current_backpressure_inhga=1.5,
                load_pct=150.0,  # Too high
                cw_inlet_temp_f=75.0,
                cw_flow_gpm=90000.0,
            )

    def test_invalid_flow(self, detector):
        """Test invalid flow raises error."""
        with pytest.raises(ValueError):
            detector.analyze_fouling(
                current_backpressure_inhga=1.5,
                load_pct=85.0,
                cw_inlet_temp_f=75.0,
                cw_flow_gpm=0.0,  # Invalid
            )


# =============================================================================
# CALCULATION COUNT TESTS
# =============================================================================

class TestCalculationCount:
    """Test calculation counting."""

    def test_count_increments(self, detector):
        """Test calculation count increments."""
        initial = detector.calculation_count

        detector.analyze_fouling(
            current_backpressure_inhga=1.5,
            load_pct=85.0,
            cw_inlet_temp_f=75.0,
            cw_flow_gpm=90000.0,
        )

        assert detector.calculation_count == initial + 1
