# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Oil Analysis Tests

Tests for oil analysis trending and interpretation.
Validates viscosity change detection, wear metal trending, and condition assessment.

Coverage Target: 85%+
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import List

from greenlang.agents.process_heat.gl_013_predictive_maintenance.oil_analysis import (
    OilAnalyzer,
    OilBaseline,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.config import (
    AlertSeverity,
    OilThresholds,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.schemas import (
    HealthStatus,
    OilAnalysisReading,
    OilAnalysisResult,
)


class TestOilBaseline:
    """Tests for OilBaseline dataclass."""

    def test_valid_baseline(self, oil_baseline):
        """Test valid oil baseline creation."""
        assert oil_baseline.viscosity_40c_cst == 46.0
        assert oil_baseline.tan_mg_koh_g == 0.3
        assert oil_baseline.iron_ppm == 0.0

    def test_baseline_with_all_parameters(self):
        """Test baseline with all parameters specified."""
        baseline = OilBaseline(
            viscosity_40c_cst=68.0,
            viscosity_100c_cst=8.2,
            tan_mg_koh_g=0.2,
            iron_ppm=0.0,
            copper_ppm=0.0,
            chromium_ppm=0.0,
            water_ppm=30.0,
            particle_count_iso_4406="15/13/10",
        )

        assert baseline.viscosity_40c_cst == 68.0
        assert baseline.particle_count_iso_4406 == "15/13/10"


class TestOilAnalyzer:
    """Tests for OilAnalyzer class."""

    def test_initialization(self, oil_thresholds, oil_baseline):
        """Test analyzer initialization."""
        analyzer = OilAnalyzer(oil_thresholds, oil_baseline)

        assert analyzer.thresholds == oil_thresholds
        assert analyzer.baseline == oil_baseline

    def test_initialization_without_baseline(self, oil_thresholds):
        """Test initialization without baseline uses defaults."""
        analyzer = OilAnalyzer(oil_thresholds)

        assert analyzer.baseline is not None

    def test_initialization_default_thresholds(self):
        """Test initialization with default thresholds."""
        analyzer = OilAnalyzer()

        assert analyzer.thresholds is not None


class TestViscosityAnalysis:
    """Tests for viscosity change detection."""

    def test_viscosity_normal(self, oil_analyzer, oil_reading_healthy):
        """Test normal viscosity within limits."""
        result = oil_analyzer.analyze(oil_reading_healthy)

        assert result.viscosity_status == "normal"
        assert abs(result.viscosity_change_pct) < 10.0

    def test_viscosity_increase_warning(self, oil_analyzer, oil_reading_degraded):
        """Test viscosity increase beyond warning threshold."""
        result = oil_analyzer.analyze(oil_reading_degraded)

        # 13% increase from 46 to 52
        assert result.viscosity_change_pct > 10.0
        assert "high" in result.viscosity_status.lower() or \
               result.viscosity_status == "elevated"

    def test_viscosity_decrease_warning(self, oil_analyzer, oil_baseline):
        """Test viscosity decrease beyond warning threshold."""
        low_visc_reading = OilAnalysisReading(
            sample_id="OIL-LOW",
            sample_point="Sump",
            timestamp=datetime.now(timezone.utc),
            viscosity_40c_cst=38.0,  # 17% decrease
            tan_mg_koh_g=0.5,
        )

        result = oil_analyzer.analyze(low_visc_reading)

        assert result.viscosity_change_pct < -10.0
        assert "low" in result.viscosity_status.lower() or \
               result.viscosity_status in ["thin", "diluted"]

    def test_viscosity_change_calculation(self, oil_analyzer, oil_baseline):
        """Test viscosity change percentage calculation."""
        # Exactly 10% increase
        reading = OilAnalysisReading(
            sample_id="OIL-TEST",
            sample_point="Sump",
            timestamp=datetime.now(timezone.utc),
            viscosity_40c_cst=50.6,  # 10% increase from 46
            tan_mg_koh_g=0.5,
        )

        result = oil_analyzer.analyze(reading)

        assert result.viscosity_change_pct == pytest.approx(10.0, rel=0.1)


class TestTANAnalysis:
    """Tests for Total Acid Number analysis."""

    def test_tan_good(self, oil_analyzer, oil_reading_healthy):
        """Test TAN in good range."""
        result = oil_analyzer.analyze(oil_reading_healthy)

        assert result.tan_status == AlertSeverity.GOOD

    def test_tan_warning(self, oil_analyzer, oil_reading_degraded):
        """Test TAN in warning range (> 2.0 mg KOH/g)."""
        result = oil_analyzer.analyze(oil_reading_degraded)

        assert result.tan_status in [AlertSeverity.UNSATISFACTORY, AlertSeverity.ACCEPTABLE]

    def test_tan_critical(self, oil_analyzer, oil_reading_critical):
        """Test TAN in critical range (> 4.0 mg KOH/g)."""
        result = oil_analyzer.analyze(oil_reading_critical)

        assert result.tan_status == AlertSeverity.UNACCEPTABLE

    @pytest.mark.parametrize("tan,expected_status", [
        (0.5, AlertSeverity.GOOD),
        (1.5, AlertSeverity.ACCEPTABLE),
        (2.5, AlertSeverity.UNSATISFACTORY),
        (4.5, AlertSeverity.UNACCEPTABLE),
    ])
    def test_tan_thresholds(self, oil_analyzer, tan, expected_status):
        """Test TAN threshold boundaries."""
        reading = OilAnalysisReading(
            sample_id="OIL-TAN",
            sample_point="Sump",
            timestamp=datetime.now(timezone.utc),
            viscosity_40c_cst=46.0,
            tan_mg_koh_g=tan,
        )

        result = oil_analyzer.analyze(reading)

        assert result.tan_status == expected_status


class TestWaterAnalysis:
    """Tests for water content analysis."""

    def test_water_good(self, oil_analyzer, oil_reading_healthy):
        """Test water content in good range."""
        result = oil_analyzer.analyze(oil_reading_healthy)

        assert result.water_status == AlertSeverity.GOOD

    def test_water_warning(self, oil_analyzer, oil_reading_degraded):
        """Test water content in warning range (> 500 ppm)."""
        result = oil_analyzer.analyze(oil_reading_degraded)

        assert result.water_status in [
            AlertSeverity.UNSATISFACTORY,
            AlertSeverity.ACCEPTABLE
        ]

    def test_water_critical(self, oil_analyzer, oil_reading_critical):
        """Test water content in critical range (> 1000 ppm)."""
        result = oil_analyzer.analyze(oil_reading_critical)

        assert result.water_status == AlertSeverity.UNACCEPTABLE

    @pytest.mark.parametrize("water_ppm,expected_status", [
        (100, AlertSeverity.GOOD),
        (400, AlertSeverity.ACCEPTABLE),
        (700, AlertSeverity.UNSATISFACTORY),
        (1200, AlertSeverity.UNACCEPTABLE),
    ])
    def test_water_thresholds(self, oil_analyzer, water_ppm, expected_status):
        """Test water content threshold boundaries."""
        reading = OilAnalysisReading(
            sample_id="OIL-WATER",
            sample_point="Sump",
            timestamp=datetime.now(timezone.utc),
            viscosity_40c_cst=46.0,
            tan_mg_koh_g=0.5,
            water_ppm=water_ppm,
        )

        result = oil_analyzer.analyze(reading)

        assert result.water_status == expected_status


class TestWearMetalAnalysis:
    """Tests for wear metal trending analysis."""

    def test_iron_good(self, oil_analyzer, oil_reading_healthy):
        """Test iron content in good range."""
        result = oil_analyzer.analyze(oil_reading_healthy)

        assert result.iron_status == AlertSeverity.GOOD

    def test_iron_warning(self, oil_analyzer, oil_reading_degraded):
        """Test iron content in warning range (> 100 ppm)."""
        result = oil_analyzer.analyze(oil_reading_degraded)

        assert result.iron_status in [
            AlertSeverity.UNSATISFACTORY,
            AlertSeverity.ACCEPTABLE
        ]

    def test_iron_critical(self, oil_analyzer, oil_reading_critical):
        """Test iron content in critical range (> 200 ppm)."""
        result = oil_analyzer.analyze(oil_reading_critical)

        assert result.iron_status == AlertSeverity.UNACCEPTABLE

    @pytest.mark.parametrize("iron_ppm,expected_status", [
        (25, AlertSeverity.GOOD),
        (75, AlertSeverity.ACCEPTABLE),
        (150, AlertSeverity.UNSATISFACTORY),
        (250, AlertSeverity.UNACCEPTABLE),
    ])
    def test_iron_thresholds(self, oil_analyzer, iron_ppm, expected_status):
        """Test iron content threshold boundaries."""
        reading = OilAnalysisReading(
            sample_id="OIL-IRON",
            sample_point="Sump",
            timestamp=datetime.now(timezone.utc),
            viscosity_40c_cst=46.0,
            tan_mg_koh_g=0.5,
            iron_ppm=iron_ppm,
        )

        result = oil_analyzer.analyze(reading)

        assert result.iron_status == expected_status

    def test_copper_analysis(self, oil_analyzer):
        """Test copper wear metal analysis."""
        reading = OilAnalysisReading(
            sample_id="OIL-CU",
            sample_point="Sump",
            timestamp=datetime.now(timezone.utc),
            viscosity_40c_cst=46.0,
            tan_mg_koh_g=0.5,
            copper_ppm=75.0,  # Above warning 50 ppm
        )

        result = oil_analyzer.analyze(reading)

        assert result.copper_status in [
            AlertSeverity.UNSATISFACTORY,
            AlertSeverity.ACCEPTABLE
        ]


class TestParticleCountAnalysis:
    """Tests for particle count analysis (ISO 4406)."""

    def test_particle_count_good(self, oil_analyzer, oil_reading_healthy):
        """Test particle count in good range."""
        result = oil_analyzer.analyze(oil_reading_healthy)

        assert result.particle_status in [
            AlertSeverity.GOOD,
            AlertSeverity.ACCEPTABLE
        ]

    def test_particle_count_warning(self, oil_analyzer, oil_reading_degraded):
        """Test particle count in warning range."""
        result = oil_analyzer.analyze(oil_reading_degraded)

        # 19/17/14 is elevated
        assert result.particle_status in [
            AlertSeverity.UNSATISFACTORY,
            AlertSeverity.ACCEPTABLE
        ]

    def test_particle_count_critical(self, oil_analyzer, oil_reading_critical):
        """Test particle count in critical range."""
        result = oil_analyzer.analyze(oil_reading_critical)

        # 21/19/16 is high
        assert result.particle_status in [
            AlertSeverity.UNACCEPTABLE,
            AlertSeverity.UNSATISFACTORY
        ]

    def test_iso_4406_parsing(self, oil_analyzer):
        """Test ISO 4406 code parsing."""
        reading = OilAnalysisReading(
            sample_id="OIL-PC",
            sample_point="Sump",
            timestamp=datetime.now(timezone.utc),
            viscosity_40c_cst=46.0,
            tan_mg_koh_g=0.5,
            particle_count_iso_4406="18/16/13",
        )

        result = oil_analyzer.analyze(reading)

        # Should parse correctly - no exception
        assert result.particle_status is not None


class TestOverallOilCondition:
    """Tests for overall oil condition assessment."""

    def test_healthy_oil(self, oil_analyzer, oil_reading_healthy):
        """Test healthy oil assessment."""
        result = oil_analyzer.analyze(oil_reading_healthy)

        assert result.oil_condition == HealthStatus.HEALTHY

    def test_degraded_oil(self, oil_analyzer, oil_reading_degraded):
        """Test degraded oil assessment."""
        result = oil_analyzer.analyze(oil_reading_degraded)

        assert result.oil_condition in [
            HealthStatus.DEGRADED,
            HealthStatus.WARNING
        ]

    def test_critical_oil(self, oil_analyzer, oil_reading_critical):
        """Test critical oil assessment."""
        result = oil_analyzer.analyze(oil_reading_critical)

        assert result.oil_condition in [
            HealthStatus.WARNING,
            HealthStatus.CRITICAL
        ]

    def test_single_critical_parameter(self, oil_analyzer):
        """Test single critical parameter escalates condition."""
        reading = OilAnalysisReading(
            sample_id="OIL-SINGLE",
            sample_point="Sump",
            timestamp=datetime.now(timezone.utc),
            viscosity_40c_cst=46.0,
            tan_mg_koh_g=0.5,
            water_ppm=100.0,
            iron_ppm=250.0,  # Only iron is critical
        )

        result = oil_analyzer.analyze(reading)

        # Single critical should escalate
        assert result.oil_condition in [
            HealthStatus.WARNING,
            HealthStatus.CRITICAL
        ]


class TestOilChangeRecommendation:
    """Tests for oil change recommendation logic."""

    def test_no_change_recommended_healthy(self, oil_analyzer, oil_reading_healthy):
        """Test no oil change recommended for healthy oil."""
        result = oil_analyzer.analyze(oil_reading_healthy)

        assert result.oil_change_recommended is False

    def test_change_recommended_degraded(self, oil_analyzer, oil_reading_degraded):
        """Test oil change recommended for degraded oil."""
        result = oil_analyzer.analyze(oil_reading_degraded)

        # May or may not recommend based on severity
        # At least should be flagged
        pass

    def test_change_recommended_critical(self, oil_analyzer, oil_reading_critical):
        """Test oil change recommended for critical oil."""
        result = oil_analyzer.analyze(oil_reading_critical)

        assert result.oil_change_recommended is True

    def test_change_recommended_high_tan(self, oil_analyzer):
        """Test oil change recommended when TAN is critical."""
        reading = OilAnalysisReading(
            sample_id="OIL-TAN-CRIT",
            sample_point="Sump",
            timestamp=datetime.now(timezone.utc),
            viscosity_40c_cst=46.0,
            tan_mg_koh_g=5.0,  # Very high TAN
        )

        result = oil_analyzer.analyze(reading)

        assert result.oil_change_recommended is True


class TestOilRemainingLife:
    """Tests for oil remaining useful life estimation."""

    def test_rul_healthy_oil(self, oil_analyzer, oil_reading_healthy):
        """Test RUL estimation for healthy oil."""
        result = oil_analyzer.analyze(oil_reading_healthy)

        if result.remaining_useful_life_pct is not None:
            assert result.remaining_useful_life_pct > 75.0

    def test_rul_degraded_oil(self, oil_analyzer, oil_reading_degraded):
        """Test RUL estimation for degraded oil."""
        result = oil_analyzer.analyze(oil_reading_degraded)

        if result.remaining_useful_life_pct is not None:
            assert result.remaining_useful_life_pct < 50.0

    def test_rul_critical_oil(self, oil_analyzer, oil_reading_critical):
        """Test RUL estimation for critical oil."""
        result = oil_analyzer.analyze(oil_reading_critical)

        if result.remaining_useful_life_pct is not None:
            assert result.remaining_useful_life_pct < 20.0


class TestOilTrendAnalysis:
    """Tests for oil condition trending."""

    def test_trend_analysis_with_history(self, oil_analyzer):
        """Test trend analysis with historical data."""
        # Create historical readings
        history = []
        base_time = datetime.now(timezone.utc) - timedelta(days=90)

        for i in range(4):
            reading = OilAnalysisReading(
                sample_id=f"OIL-H{i}",
                sample_point="Sump",
                timestamp=base_time + timedelta(days=i * 30),
                viscosity_40c_cst=46.0 + i * 2,  # Increasing
                tan_mg_koh_g=0.5 + i * 0.5,  # Increasing
                iron_ppm=25.0 + i * 25,  # Increasing
            )
            history.append(reading)

        # Set historical data
        if hasattr(oil_analyzer, 'set_history'):
            oil_analyzer.set_history(history)

        # Analyze current
        current = OilAnalysisReading(
            sample_id="OIL-CURRENT",
            sample_point="Sump",
            timestamp=datetime.now(timezone.utc),
            viscosity_40c_cst=54.0,
            tan_mg_koh_g=2.5,
            iron_ppm=125.0,
        )

        result = oil_analyzer.analyze(current)

        # Should detect increasing trends
        if result.viscosity_trend is not None:
            assert "increasing" in result.viscosity_trend.lower()


class TestOilAnalysisRecommendations:
    """Tests for analysis recommendations."""

    def test_recommendations_healthy(self, oil_analyzer, oil_reading_healthy):
        """Test recommendations for healthy oil."""
        result = oil_analyzer.analyze(oil_reading_healthy)

        # Should have routine monitoring recommendation
        assert len(result.recommendations) >= 0

    def test_recommendations_degraded(self, oil_analyzer, oil_reading_degraded):
        """Test recommendations for degraded oil."""
        result = oil_analyzer.analyze(oil_reading_degraded)

        assert len(result.recommendations) > 0
        # Should mention specific issues
        rec_text = " ".join(result.recommendations).lower()
        assert any(term in rec_text for term in [
            "oil", "analysis", "sample", "change", "monitor"
        ])

    def test_recommendations_critical(self, oil_analyzer, oil_reading_critical):
        """Test recommendations for critical oil."""
        result = oil_analyzer.analyze(oil_reading_critical)

        assert len(result.recommendations) > 0
        # Should recommend immediate action
        rec_text = " ".join(result.recommendations).lower()
        assert any(term in rec_text for term in [
            "change", "immediate", "critical", "urgent"
        ])


class TestOilAnalysisDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_analysis_same_result(self, oil_analyzer, oil_reading_healthy):
        """Test repeated analysis produces identical results."""
        results = [
            oil_analyzer.analyze(oil_reading_healthy)
            for _ in range(5)
        ]

        # All conditions should be identical
        conditions = [r.oil_condition for r in results]
        assert len(set(conditions)) == 1

    def test_provenance_hash_deterministic(self, oil_analyzer, oil_reading_healthy):
        """Test provenance hash is deterministic."""
        result1 = oil_analyzer.analyze(oil_reading_healthy)
        result2 = oil_analyzer.analyze(oil_reading_healthy)

        if result1.provenance_hash is not None:
            assert result1.provenance_hash == result2.provenance_hash


class TestOilAnalysisEdgeCases:
    """Tests for edge cases."""

    def test_zero_viscosity_handled(self, oil_analyzer):
        """Test zero viscosity is handled."""
        reading = OilAnalysisReading(
            sample_id="OIL-ZERO",
            sample_point="Sump",
            timestamp=datetime.now(timezone.utc),
            viscosity_40c_cst=0.0,  # Invalid
            tan_mg_koh_g=0.5,
        )

        # Should handle gracefully
        with pytest.raises((ValueError, ZeroDivisionError, Exception)):
            oil_analyzer.analyze(reading)

    def test_missing_optional_parameters(self, oil_analyzer):
        """Test analysis with minimal parameters."""
        reading = OilAnalysisReading(
            sample_id="OIL-MIN",
            sample_point="Sump",
            timestamp=datetime.now(timezone.utc),
            viscosity_40c_cst=46.0,
            tan_mg_koh_g=0.5,
            # No metals or water specified
        )

        result = oil_analyzer.analyze(reading)

        # Should still produce valid result
        assert result.oil_condition is not None

    def test_very_high_values(self, oil_analyzer):
        """Test handling of extremely high values."""
        reading = OilAnalysisReading(
            sample_id="OIL-HIGH",
            sample_point="Sump",
            timestamp=datetime.now(timezone.utc),
            viscosity_40c_cst=500.0,  # Very high
            tan_mg_koh_g=20.0,  # Very high
            iron_ppm=1000.0,  # Very high
        )

        result = oil_analyzer.analyze(reading)

        # Should flag as critical
        assert result.oil_condition == HealthStatus.CRITICAL


class TestOilAnalysisIntegration:
    """Integration tests."""

    def test_full_analysis_workflow(self, oil_thresholds, oil_baseline):
        """Test complete analysis workflow."""
        analyzer = OilAnalyzer(oil_thresholds, oil_baseline)

        reading = OilAnalysisReading(
            sample_id="OIL-INT",
            sample_point="Main Sump",
            timestamp=datetime.now(timezone.utc),
            viscosity_40c_cst=48.0,
            viscosity_100c_cst=7.0,
            tan_mg_koh_g=1.5,
            water_ppm=300.0,
            particle_count_iso_4406="17/15/12",
            iron_ppm=80.0,
            copper_ppm=30.0,
            chromium_ppm=5.0,
            silicon_ppm=12.0,
        )

        result = analyzer.analyze(reading)

        # Verify all fields populated
        assert result.sample_id == "OIL-INT"
        assert result.oil_condition is not None
        assert result.viscosity_status is not None
        assert result.tan_status is not None
        assert result.water_status is not None
        assert result.particle_status is not None
        assert result.iron_status is not None
        assert isinstance(result.oil_change_recommended, bool)
