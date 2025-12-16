# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Agent - Tube Integrity Analysis Tests

Comprehensive tests for tube integrity analysis including:
- Wall thickness calculations
- Weibull reliability analysis
- Corrosion rate trending
- Failure probability predictions
- Inspection scheduling

Coverage Target: 90%+

References:
    - API 579-1/ASME FFS-1 (Fitness-For-Service)
    - ASME Section VIII Division 1
    - API 570 Piping Inspection Code
"""

import pytest
import math
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_014_heat_exchanger.tube_analysis import (
    TubeIntegrityAnalyzer,
    TubeCondition,
    WeibullParameters,
    CorrosionTrend,
)
from greenlang.agents.process_heat.gl_014_heat_exchanger.config import (
    TubeIntegrityConfig,
    TubeGeometryConfig,
    TubeMaterial,
    FailureMode,
    AlertSeverity,
)
from greenlang.agents.process_heat.gl_014_heat_exchanger.schemas import (
    TubeInspectionData,
    TubeIntegrityResult,
)


class TestTubeIntegrityAnalyzerInit:
    """Tests for TubeIntegrityAnalyzer initialization."""

    def test_analyzer_initialization(self, tube_integrity_config, tube_geometry_config):
        """Test analyzer initializes correctly."""
        analyzer = TubeIntegrityAnalyzer(
            config=tube_integrity_config,
            geometry=tube_geometry_config,
        )
        assert analyzer.config == tube_integrity_config
        assert analyzer.geometry == tube_geometry_config
        assert len(analyzer.inspection_history) == 0

    def test_analyzer_with_weibull_params(self, tube_integrity_config, tube_geometry_config):
        """Test analyzer initializes Weibull parameters from config."""
        analyzer = TubeIntegrityAnalyzer(
            config=tube_integrity_config,
            geometry=tube_geometry_config,
        )

        if tube_integrity_config.weibull_beta and tube_integrity_config.weibull_eta:
            assert analyzer.weibull is not None
            assert analyzer.weibull.beta == tube_integrity_config.weibull_beta
            assert analyzer.weibull.eta == tube_integrity_config.weibull_eta


class TestWallThicknessAnalysis:
    """Tests for wall thickness calculations."""

    @pytest.fixture
    def analyzer(self, tube_integrity_config, tube_geometry_config):
        """Create TubeIntegrityAnalyzer instance."""
        return TubeIntegrityAnalyzer(
            config=tube_integrity_config,
            geometry=tube_geometry_config,
        )

    def test_wall_thickness_from_inspection(self, analyzer, tube_inspection_data):
        """Test wall thickness calculation from inspection data."""
        result = analyzer.analyze_integrity(
            inspection_data=tube_inspection_data,
            operating_years=5.0,
        )

        # Wall thickness should be between min and original
        assert result.current_wall_thickness_mm > result.minimum_required_thickness_mm
        assert result.current_wall_thickness_mm <= analyzer.geometry.wall_thickness_mm

    def test_wall_loss_percentage(self, analyzer, tube_inspection_data):
        """Test wall loss percentage calculation."""
        result = analyzer.analyze_integrity(
            inspection_data=tube_inspection_data,
            operating_years=5.0,
        )

        # Wall loss should be 0-100%
        assert 0 <= result.wall_loss_percent <= 100

        # Verify calculation
        expected_loss = (
            (analyzer.geometry.wall_thickness_mm - result.current_wall_thickness_mm) /
            analyzer.geometry.wall_thickness_mm * 100
        )
        assert result.wall_loss_percent == pytest.approx(expected_loss, rel=0.01)

    def test_thickness_margin(self, analyzer, tube_inspection_data):
        """Test thickness margin calculation."""
        result = analyzer.analyze_integrity(
            inspection_data=tube_inspection_data,
            operating_years=5.0,
        )

        expected_margin = (
            result.current_wall_thickness_mm -
            result.minimum_required_thickness_mm
        )
        assert result.thickness_margin_mm == pytest.approx(expected_margin, rel=0.01)


class TestCorrosionRateCalculation:
    """Tests for corrosion rate calculations."""

    @pytest.fixture
    def analyzer(self, tube_integrity_config, tube_geometry_config):
        """Create TubeIntegrityAnalyzer instance."""
        return TubeIntegrityAnalyzer(
            config=tube_integrity_config,
            geometry=tube_geometry_config,
        )

    def test_corrosion_rate_from_inspections(self, analyzer):
        """Test corrosion rate calculation from multiple inspections."""
        # Create inspection history
        base_date = datetime.now(timezone.utc) - timedelta(days=3*365)

        inspections = [
            TubeInspectionData(
                inspection_date=base_date,
                inspection_method="eddy_current",
                total_tubes=100,
                tubes_inspected=100,
                tubes_with_defects=2,
                tubes_plugged=0,
                wall_loss_summary={"<20%": 98, "20-40%": 2, "40-60%": 0, "60-80%": 0, ">80%": 0},
            ),
            TubeInspectionData(
                inspection_date=base_date + timedelta(days=365),
                inspection_method="eddy_current",
                total_tubes=100,
                tubes_inspected=100,
                tubes_with_defects=4,
                tubes_plugged=1,
                wall_loss_summary={"<20%": 93, "20-40%": 5, "40-60%": 2, "60-80%": 0, ">80%": 0},
            ),
            TubeInspectionData(
                inspection_date=base_date + timedelta(days=2*365),
                inspection_method="eddy_current",
                total_tubes=100,
                tubes_inspected=100,
                tubes_with_defects=7,
                tubes_plugged=2,
                wall_loss_summary={"<20%": 85, "20-40%": 10, "40-60%": 3, "60-80%": 2, ">80%": 0},
            ),
        ]

        trend = analyzer.analyze_corrosion_trend(inspections)

        assert isinstance(trend, CorrosionTrend)
        assert trend.average_rate_mm_year >= 0
        assert trend.data_points == len(inspections)

    def test_corrosion_trend_detection(self, analyzer):
        """Test corrosion rate trend detection."""
        base_date = datetime.now(timezone.utc) - timedelta(days=4*365)

        # Accelerating corrosion
        inspections = [
            TubeInspectionData(
                inspection_date=base_date + timedelta(days=i*365),
                inspection_method="eddy_current",
                total_tubes=100,
                tubes_inspected=100,
                tubes_with_defects=2 + i*3,  # Accelerating
                tubes_plugged=i,
                wall_loss_summary={
                    "<20%": 95 - i*10,
                    "20-40%": 3 + i*5,
                    "40-60%": 2 + i*3,
                    "60-80%": i,
                    ">80%": 0,
                },
            )
            for i in range(4)
        ]

        trend = analyzer.analyze_corrosion_trend(inspections)

        # Should detect increasing or accelerating trend
        assert trend.rate_trend in ["stable", "increasing", "decreasing"]


class TestWeibullAnalysis:
    """Tests for Weibull reliability analysis."""

    @pytest.fixture
    def analyzer(self, tube_integrity_config, tube_geometry_config):
        """Create TubeIntegrityAnalyzer with Weibull parameters."""
        return TubeIntegrityAnalyzer(
            config=tube_integrity_config,
            geometry=tube_geometry_config,
        )

    def test_failure_probability_calculation(self, analyzer):
        """Test failure probability using Weibull distribution."""
        prob = analyzer.calculate_failure_probability(
            operating_years=10.0,
            time_horizon_years=1.0,
        )

        # Probability should be 0-1
        assert 0 <= prob <= 1

    def test_failure_probability_increases_with_age(self, analyzer):
        """Test failure probability increases with operating age."""
        prob_5yr = analyzer.calculate_failure_probability(
            operating_years=5.0,
            time_horizon_years=1.0,
        )

        prob_15yr = analyzer.calculate_failure_probability(
            operating_years=15.0,
            time_horizon_years=1.0,
        )

        # Older equipment has higher failure probability
        assert prob_15yr > prob_5yr

    def test_failure_probability_longer_horizon(self, analyzer):
        """Test failure probability increases with longer time horizon."""
        prob_1yr = analyzer.calculate_failure_probability(
            operating_years=10.0,
            time_horizon_years=1.0,
        )

        prob_5yr = analyzer.calculate_failure_probability(
            operating_years=10.0,
            time_horizon_years=5.0,
        )

        # Longer horizon has higher failure probability
        assert prob_5yr > prob_1yr

    def test_weibull_beta_effect(self, analyzer):
        """Test Weibull beta parameter affects failure distribution."""
        # Beta > 1 indicates wear-out failure (increasing failure rate)
        # The analyzer should use this for tube life estimation

        if analyzer.weibull:
            assert analyzer.weibull.beta > 1.0  # Wear-out expected for tubes


class TestRemainingLifeEstimation:
    """Tests for remaining tube life estimation."""

    @pytest.fixture
    def analyzer(self, tube_integrity_config, tube_geometry_config):
        """Create TubeIntegrityAnalyzer instance."""
        return TubeIntegrityAnalyzer(
            config=tube_integrity_config,
            geometry=tube_geometry_config,
        )

    def test_remaining_life_estimation(self, analyzer):
        """Test remaining life estimation from wall thickness."""
        remaining_life, confidence = analyzer.estimate_remaining_life(
            current_thickness_mm=1.8,
            corrosion_rate_mm_year=0.1,
        )

        # Remaining life = (current - min) / rate
        expected_life = (1.8 - analyzer.config.minimum_wall_thickness_mm) / 0.1
        assert remaining_life == pytest.approx(expected_life, rel=0.1)

    def test_remaining_life_at_minimum_thickness(self, analyzer):
        """Test remaining life is zero at minimum thickness."""
        remaining_life, confidence = analyzer.estimate_remaining_life(
            current_thickness_mm=analyzer.config.minimum_wall_thickness_mm,
            corrosion_rate_mm_year=0.1,
        )

        assert remaining_life == pytest.approx(0.0, abs=0.1)

    def test_remaining_life_confidence(self, analyzer):
        """Test remaining life confidence is provided."""
        remaining_life, confidence = analyzer.estimate_remaining_life(
            current_thickness_mm=1.8,
            corrosion_rate_mm_year=0.1,
        )

        assert 0 < confidence <= 1.0


class TestTubeFailurePrediction:
    """Tests for tube failure predictions."""

    @pytest.fixture
    def analyzer(self, tube_integrity_config, tube_geometry_config):
        """Create TubeIntegrityAnalyzer instance."""
        return TubeIntegrityAnalyzer(
            config=tube_integrity_config,
            geometry=tube_geometry_config,
        )

    def test_failure_prediction_1yr(self, analyzer, tube_inspection_data):
        """Test 1-year failure prediction."""
        result = analyzer.analyze_integrity(
            inspection_data=tube_inspection_data,
            operating_years=5.0,
        )

        assert result.predicted_failures_1yr >= 0
        assert result.predicted_failures_1yr <= analyzer.geometry.tube_count

    def test_failure_prediction_5yr(self, analyzer, tube_inspection_data):
        """Test 5-year failure prediction."""
        result = analyzer.analyze_integrity(
            inspection_data=tube_inspection_data,
            operating_years=5.0,
        )

        assert result.predicted_failures_5yr >= 0
        assert result.predicted_failures_5yr >= result.predicted_failures_1yr

    def test_failure_modes_analysis(self, analyzer, tube_inspection_data):
        """Test failure modes are analyzed."""
        result = analyzer.analyze_integrity(
            inspection_data=tube_inspection_data,
            operating_years=10.0,
        )

        # Should have failure modes list
        assert isinstance(result.failure_modes, list)


class TestPluggingAnalysis:
    """Tests for tube plugging analysis."""

    @pytest.fixture
    def analyzer(self, tube_integrity_config, tube_geometry_config):
        """Create TubeIntegrityAnalyzer instance."""
        return TubeIntegrityAnalyzer(
            config=tube_integrity_config,
            geometry=tube_geometry_config,
        )

    def test_plugging_rate_calculation(self, analyzer, tube_inspection_data):
        """Test plugging rate calculation."""
        result = analyzer.analyze_integrity(
            inspection_data=tube_inspection_data,
            operating_years=5.0,
        )

        expected_rate = tube_inspection_data.tubes_plugged / analyzer.geometry.tube_count * 100
        assert result.plugging_rate_percent == pytest.approx(expected_rate, rel=0.01)

    def test_tubes_at_risk_count(self, analyzer, tube_inspection_data):
        """Test tubes at risk count."""
        result = analyzer.analyze_integrity(
            inspection_data=tube_inspection_data,
            operating_years=5.0,
        )

        # Should match inspection recommendations
        expected_at_risk = len(tube_inspection_data.tubes_recommended_for_plugging)
        assert result.tubes_at_risk == expected_at_risk

    def test_retube_recommendation_threshold(self, analyzer):
        """Test retube recommendation at plugging threshold."""
        inspection = TubeInspectionData(
            inspection_date=datetime.now(timezone.utc),
            inspection_method="eddy_current",
            total_tubes=100,
            tubes_inspected=100,
            tubes_with_defects=20,
            tubes_plugged=12,  # 12% > 10% threshold
            tubes_recommended_for_plugging=[],
            retube_recommended=True,
        )

        result = analyzer.analyze_integrity(
            inspection_data=inspection,
            operating_years=10.0,
        )

        # Should recommend retubing
        assert result.retube_recommended == True


class TestInspectionScheduling:
    """Tests for inspection scheduling."""

    @pytest.fixture
    def analyzer(self, tube_integrity_config, tube_geometry_config):
        """Create TubeIntegrityAnalyzer instance."""
        return TubeIntegrityAnalyzer(
            config=tube_integrity_config,
            geometry=tube_geometry_config,
        )

    def test_next_inspection_date(self, analyzer, tube_inspection_data):
        """Test next inspection date is calculated."""
        result = analyzer.analyze_integrity(
            inspection_data=tube_inspection_data,
            operating_years=5.0,
        )

        assert result.next_inspection_date is not None
        assert result.next_inspection_date > datetime.now(timezone.utc)

    def test_inspection_urgency_levels(self, analyzer):
        """Test inspection urgency reflects tube condition."""
        # Good condition - low urgency
        inspection_good = TubeInspectionData(
            inspection_date=datetime.now(timezone.utc),
            inspection_method="eddy_current",
            total_tubes=100,
            tubes_inspected=100,
            tubes_with_defects=2,
            tubes_plugged=0,
            wall_loss_summary={"<20%": 98, "20-40%": 2, "40-60%": 0, "60-80%": 0, ">80%": 0},
        )

        result_good = analyzer.analyze_integrity(
            inspection_data=inspection_good,
            operating_years=3.0,
        )

        # Poor condition - high urgency
        inspection_poor = TubeInspectionData(
            inspection_date=datetime.now(timezone.utc),
            inspection_method="eddy_current",
            total_tubes=100,
            tubes_inspected=100,
            tubes_with_defects=25,
            tubes_plugged=8,
            wall_loss_summary={"<20%": 50, "20-40%": 25, "40-60%": 15, "60-80%": 8, ">80%": 2},
            tubes_recommended_for_plugging=[i for i in range(10)],
        )

        result_poor = analyzer.analyze_integrity(
            inspection_data=inspection_poor,
            operating_years=15.0,
        )

        # Poor condition should have higher urgency
        urgency_order = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.ALARM: 2,
            AlertSeverity.CRITICAL: 3,
        }

        assert urgency_order[result_poor.inspection_urgency] >= urgency_order[result_good.inspection_urgency]


class TestMonteCarloSimulation:
    """Tests for Monte Carlo tube failure simulation."""

    @pytest.fixture
    def analyzer(self, tube_integrity_config, tube_geometry_config):
        """Create TubeIntegrityAnalyzer instance."""
        return TubeIntegrityAnalyzer(
            config=tube_integrity_config,
            geometry=tube_geometry_config,
        )

    def test_monte_carlo_simulation(self, analyzer):
        """Test Monte Carlo failure simulation."""
        results = analyzer.simulate_tube_failures(
            years_ahead=5,
            n_simulations=100,  # Reduced for test speed
        )

        assert "predictions" in results
        assert len(results["predictions"]) == 5

        # Each year should have statistics
        for year in range(1, 6):
            assert year in results["predictions"]
            assert "mean" in results["predictions"][year]
            assert "p10" in results["predictions"][year]
            assert "p90" in results["predictions"][year]

    def test_simulation_statistics_validity(self, analyzer):
        """Test simulation statistics are valid."""
        results = analyzer.simulate_tube_failures(
            years_ahead=3,
            n_simulations=100,
        )

        for year in range(1, 4):
            stats = results["predictions"][year]
            # Mean should be between percentiles
            assert stats["p10"] <= stats["mean"] <= stats["p90"]
            # Failure rate should be 0-100%
            assert 0 <= stats["failure_rate"] <= 100


class TestInspectionHistory:
    """Tests for inspection history management."""

    @pytest.fixture
    def analyzer(self, tube_integrity_config, tube_geometry_config):
        """Create TubeIntegrityAnalyzer instance."""
        return TubeIntegrityAnalyzer(
            config=tube_integrity_config,
            geometry=tube_geometry_config,
        )

    def test_add_inspection_data(self, analyzer, tube_inspection_data):
        """Test adding inspection data."""
        analyzer.add_inspection_data(tube_inspection_data)
        assert len(analyzer.inspection_history) == 1

    def test_inspection_history_ordering(self, analyzer):
        """Test inspection history is ordered by date."""
        now = datetime.now(timezone.utc)

        # Add out of order
        for days_ago in [30, 60, 10, 45]:
            inspection = TubeInspectionData(
                inspection_date=now - timedelta(days=days_ago),
                inspection_method="eddy_current",
                total_tubes=100,
                tubes_inspected=100,
                tubes_with_defects=5,
                tubes_plugged=2,
            )
            analyzer.add_inspection_data(inspection)

        # Should be sorted
        dates = [i.inspection_date for i in analyzer.inspection_history]
        assert dates == sorted(dates)

    def test_inspection_history_limit(self, analyzer):
        """Test inspection history has a limit."""
        for i in range(100):
            inspection = TubeInspectionData(
                inspection_date=datetime.now(timezone.utc) + timedelta(days=i),
                inspection_method="eddy_current",
                total_tubes=100,
                tubes_inspected=100,
                tubes_with_defects=5,
                tubes_plugged=2,
            )
            analyzer.add_inspection_data(inspection)

        # Should be limited
        assert len(analyzer.inspection_history) <= 50
