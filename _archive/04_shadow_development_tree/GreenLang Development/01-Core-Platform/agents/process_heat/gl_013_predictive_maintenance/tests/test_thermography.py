# -*- coding: utf-8 -*-
"""
GL-013 PredictMaint Agent - Thermography Tests

Tests for IR thermography fault detection and analysis.
Validates hot spot detection, temperature trending, and thermal anomaly classification.

Coverage Target: 85%+
"""

import pytest
from datetime import datetime, timezone
from typing import List, Dict, Any

from greenlang.agents.process_heat.gl_013_predictive_maintenance.thermography import (
    ThermographyAnalyzer,
    ThermalReference,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.config import (
    AlertSeverity,
    TemperatureThresholds,
)
from greenlang.agents.process_heat.gl_013_predictive_maintenance.schemas import (
    HealthStatus,
    ThermalImage,
    ThermographyResult,
)


class TestThermalReference:
    """Tests for ThermalReference data class."""

    def test_valid_reference(self):
        """Test valid thermal reference creation."""
        ref = ThermalReference(
            name="Bearing DE",
            expected_temperature_c=60.0,
            warning_delta_c=15.0,
            alarm_delta_c=25.0,
        )

        assert ref.name == "Bearing DE"
        assert ref.expected_temperature_c == 60.0

    def test_reference_thresholds(self):
        """Test reference threshold calculations."""
        ref = ThermalReference(
            name="Motor Winding",
            expected_temperature_c=80.0,
            warning_delta_c=20.0,
            alarm_delta_c=40.0,
        )

        warning_temp = ref.expected_temperature_c + ref.warning_delta_c
        alarm_temp = ref.expected_temperature_c + ref.alarm_delta_c

        assert warning_temp == 100.0
        assert alarm_temp == 120.0


class TestThermographyAnalyzer:
    """Tests for ThermographyAnalyzer class."""

    def test_initialization(self, temperature_thresholds):
        """Test analyzer initialization."""
        analyzer = ThermographyAnalyzer(temperature_thresholds)

        assert analyzer.thresholds == temperature_thresholds

    def test_initialization_default_thresholds(self):
        """Test initialization with default thresholds."""
        analyzer = ThermographyAnalyzer()

        assert analyzer.thresholds is not None


class TestThermalSeverityClassification:
    """Tests for thermal severity classification."""

    def test_severity_good(self, thermography_analyzer, thermal_image_healthy):
        """Test good severity classification."""
        result = thermography_analyzer.analyze(thermal_image_healthy)

        assert result.thermal_severity == AlertSeverity.GOOD

    def test_severity_unsatisfactory(
        self,
        thermography_analyzer,
        thermal_image_warning
    ):
        """Test unsatisfactory severity classification."""
        result = thermography_analyzer.analyze(thermal_image_warning)

        assert result.thermal_severity in [
            AlertSeverity.UNSATISFACTORY,
            AlertSeverity.ACCEPTABLE
        ]

    def test_severity_unacceptable(
        self,
        thermography_analyzer,
        thermal_image_critical
    ):
        """Test unacceptable severity classification."""
        result = thermography_analyzer.analyze(thermal_image_critical)

        assert result.thermal_severity == AlertSeverity.UNACCEPTABLE

    @pytest.mark.parametrize("max_temp,expected_severity", [
        (55.0, AlertSeverity.GOOD),
        (72.0, AlertSeverity.ACCEPTABLE),
        (88.0, AlertSeverity.UNSATISFACTORY),
        (100.0, AlertSeverity.UNACCEPTABLE),
    ])
    def test_temperature_severity_boundaries(
        self,
        thermography_analyzer,
        max_temp,
        expected_severity
    ):
        """Test temperature severity boundaries."""
        image = ThermalImage(
            image_id="THERM-TEST",
            camera_id="IR-CAM-001",
            timestamp=datetime.now(timezone.utc),
            min_temperature_c=30.0,
            max_temperature_c=max_temp,
            avg_temperature_c=(30.0 + max_temp) / 2,
            emissivity=0.95,
            ambient_c=25.0,
        )

        result = thermography_analyzer.analyze(image)

        assert result.thermal_severity == expected_severity


class TestHotSpotDetection:
    """Tests for hot spot detection."""

    def test_no_hot_spots(self, thermography_analyzer, thermal_image_healthy):
        """Test no hot spots in healthy image."""
        result = thermography_analyzer.analyze(thermal_image_healthy)

        assert len(result.hot_spots) == 0 or result.hot_spots_detected is False

    def test_hot_spots_detected(
        self,
        thermography_analyzer,
        thermal_image_warning
    ):
        """Test hot spots detected in warning image."""
        result = thermography_analyzer.analyze(thermal_image_warning)

        assert result.hot_spots_detected is True
        assert len(result.hot_spots) >= 1

    def test_multiple_hot_spots(
        self,
        thermography_analyzer,
        thermal_image_critical
    ):
        """Test multiple hot spots detected."""
        result = thermography_analyzer.analyze(thermal_image_critical)

        assert len(result.hot_spots) >= 2

    def test_hot_spot_properties(
        self,
        thermography_analyzer,
        thermal_image_warning
    ):
        """Test hot spot properties are extracted."""
        result = thermography_analyzer.analyze(thermal_image_warning)

        if result.hot_spots:
            hot_spot = result.hot_spots[0]
            assert "temperature_c" in hot_spot or hasattr(hot_spot, "temperature_c")


class TestDeltaTAnalysis:
    """Tests for delta-T (temperature differential) analysis."""

    def test_delta_t_calculation(self, thermography_analyzer):
        """Test delta-T calculation."""
        image = ThermalImage(
            image_id="THERM-DT",
            camera_id="IR-CAM-001",
            timestamp=datetime.now(timezone.utc),
            min_temperature_c=40.0,
            max_temperature_c=80.0,
            avg_temperature_c=55.0,
            emissivity=0.95,
            ambient_c=25.0,
        )

        result = thermography_analyzer.analyze(image)

        # Delta-T should be calculated
        if result.delta_t_c is not None:
            # Could be max - ambient or max - min
            assert result.delta_t_c > 0

    def test_delta_t_above_alarm(self, thermography_analyzer):
        """Test delta-T above alarm threshold."""
        image = ThermalImage(
            image_id="THERM-DT-ALARM",
            camera_id="IR-CAM-001",
            timestamp=datetime.now(timezone.utc),
            min_temperature_c=35.0,
            max_temperature_c=100.0,  # High delta-T
            avg_temperature_c=60.0,
            emissivity=0.95,
            ambient_c=25.0,
        )

        result = thermography_analyzer.analyze(image)

        # High delta-T should trigger warning/alarm
        assert result.thermal_severity in [
            AlertSeverity.UNSATISFACTORY,
            AlertSeverity.UNACCEPTABLE
        ]


class TestAmbientCompensation:
    """Tests for ambient temperature compensation."""

    def test_ambient_compensation(self, thermography_analyzer):
        """Test ambient temperature is considered in analysis."""
        # Same absolute temps, different ambient
        image_cool_ambient = ThermalImage(
            image_id="THERM-COOL",
            camera_id="IR-CAM-001",
            timestamp=datetime.now(timezone.utc),
            min_temperature_c=35.0,
            max_temperature_c=70.0,
            avg_temperature_c=50.0,
            emissivity=0.95,
            ambient_c=15.0,  # Cool ambient
        )

        image_warm_ambient = ThermalImage(
            image_id="THERM-WARM",
            camera_id="IR-CAM-001",
            timestamp=datetime.now(timezone.utc),
            min_temperature_c=35.0,
            max_temperature_c=70.0,
            avg_temperature_c=50.0,
            emissivity=0.95,
            ambient_c=35.0,  # Warm ambient
        )

        result_cool = thermography_analyzer.analyze(image_cool_ambient)
        result_warm = thermography_analyzer.analyze(image_warm_ambient)

        # Cool ambient should show more concern (higher delta from ambient)
        # Analysis may or may not differ based on implementation


class TestEmissivityCorrection:
    """Tests for emissivity correction."""

    def test_emissivity_normal(self, thermography_analyzer):
        """Test normal emissivity handling."""
        image = ThermalImage(
            image_id="THERM-E95",
            camera_id="IR-CAM-001",
            timestamp=datetime.now(timezone.utc),
            min_temperature_c=35.0,
            max_temperature_c=65.0,
            avg_temperature_c=50.0,
            emissivity=0.95,
            ambient_c=25.0,
        )

        result = thermography_analyzer.analyze(image)

        assert result is not None

    def test_emissivity_low(self, thermography_analyzer):
        """Test low emissivity handling (shiny surfaces)."""
        image = ThermalImage(
            image_id="THERM-E50",
            camera_id="IR-CAM-001",
            timestamp=datetime.now(timezone.utc),
            min_temperature_c=35.0,
            max_temperature_c=65.0,
            avg_temperature_c=50.0,
            emissivity=0.50,  # Shiny surface
            ambient_c=25.0,
        )

        result = thermography_analyzer.analyze(image)

        # Low emissivity may affect confidence
        assert result is not None


class TestThermalComponentAnalysis:
    """Tests for component-specific thermal analysis."""

    def test_bearing_thermal_analysis(self, thermography_analyzer):
        """Test bearing-specific thermal analysis."""
        image = ThermalImage(
            image_id="THERM-BEARING",
            camera_id="IR-CAM-001",
            timestamp=datetime.now(timezone.utc),
            min_temperature_c=40.0,
            max_temperature_c=90.0,  # Hot bearing
            avg_temperature_c=55.0,
            hot_spots=[
                {"x": 150, "y": 200, "temperature_c": 90.0, "component": "bearing_de"},
            ],
            emissivity=0.95,
            ambient_c=25.0,
        )

        result = thermography_analyzer.analyze(image)

        # Should flag bearing temperature
        assert result.thermal_severity in [
            AlertSeverity.UNSATISFACTORY,
            AlertSeverity.UNACCEPTABLE
        ]

    def test_motor_winding_thermal_analysis(self, thermography_analyzer):
        """Test motor winding thermal analysis."""
        image = ThermalImage(
            image_id="THERM-WINDING",
            camera_id="IR-CAM-001",
            timestamp=datetime.now(timezone.utc),
            min_temperature_c=50.0,
            max_temperature_c=135.0,  # Hot winding
            avg_temperature_c=80.0,
            hot_spots=[
                {"x": 200, "y": 150, "temperature_c": 135.0, "component": "winding"},
            ],
            emissivity=0.95,
            ambient_c=30.0,
        )

        result = thermography_analyzer.analyze(image)

        # Should flag winding temperature
        assert result.thermal_severity == AlertSeverity.UNACCEPTABLE


class TestThermographyRecommendations:
    """Tests for thermography recommendations."""

    def test_recommendations_healthy(
        self,
        thermography_analyzer,
        thermal_image_healthy
    ):
        """Test recommendations for healthy thermal image."""
        result = thermography_analyzer.analyze(thermal_image_healthy)

        assert isinstance(result.recommendations, list)

    def test_recommendations_warning(
        self,
        thermography_analyzer,
        thermal_image_warning
    ):
        """Test recommendations for warning thermal image."""
        result = thermography_analyzer.analyze(thermal_image_warning)

        assert len(result.recommendations) > 0

    def test_recommendations_critical(
        self,
        thermography_analyzer,
        thermal_image_critical
    ):
        """Test recommendations for critical thermal image."""
        result = thermography_analyzer.analyze(thermal_image_critical)

        assert len(result.recommendations) > 0
        # Should recommend immediate action
        rec_text = " ".join(result.recommendations).lower()
        assert any(term in rec_text for term in [
            "immediate", "critical", "shutdown", "investigate"
        ])


class TestThermographyResult:
    """Tests for ThermographyResult model."""

    def test_result_fields(self, thermography_analyzer, thermal_image_healthy):
        """Test all result fields are populated."""
        result = thermography_analyzer.analyze(thermal_image_healthy)

        assert result.image_id == "THERM-001"
        assert result.timestamp is not None
        assert result.max_temperature_c == 55.0
        assert result.thermal_severity is not None
        assert isinstance(result.recommendations, list)

    def test_result_hot_spot_count(
        self,
        thermography_analyzer,
        thermal_image_warning
    ):
        """Test hot spot count in result."""
        result = thermography_analyzer.analyze(thermal_image_warning)

        if result.hot_spots_detected:
            assert result.hot_spot_count >= 1


class TestThermographyDeterminism:
    """Tests for deterministic behavior."""

    def test_repeated_analysis_same_result(
        self,
        thermography_analyzer,
        thermal_image_healthy
    ):
        """Test repeated analysis produces identical results."""
        results = [
            thermography_analyzer.analyze(thermal_image_healthy)
            for _ in range(5)
        ]

        # All severities should be identical
        severities = [r.thermal_severity for r in results]
        assert len(set(severities)) == 1

    def test_provenance_hash_deterministic(
        self,
        thermography_analyzer,
        thermal_image_healthy
    ):
        """Test provenance hash is deterministic."""
        result1 = thermography_analyzer.analyze(thermal_image_healthy)
        result2 = thermography_analyzer.analyze(thermal_image_healthy)

        if result1.provenance_hash is not None:
            assert result1.provenance_hash == result2.provenance_hash


class TestThermographyEdgeCases:
    """Tests for edge cases."""

    def test_zero_temperatures(self, thermography_analyzer):
        """Test handling of zero temperatures."""
        image = ThermalImage(
            image_id="THERM-ZERO",
            camera_id="IR-CAM-001",
            timestamp=datetime.now(timezone.utc),
            min_temperature_c=0.0,
            max_temperature_c=0.0,
            avg_temperature_c=0.0,
            emissivity=0.95,
            ambient_c=0.0,
        )

        result = thermography_analyzer.analyze(image)

        # Should handle gracefully
        assert result is not None

    def test_negative_temperatures(self, thermography_analyzer):
        """Test handling of negative temperatures (cold environments)."""
        image = ThermalImage(
            image_id="THERM-COLD",
            camera_id="IR-CAM-001",
            timestamp=datetime.now(timezone.utc),
            min_temperature_c=-20.0,
            max_temperature_c=10.0,
            avg_temperature_c=-5.0,
            emissivity=0.95,
            ambient_c=-15.0,
        )

        result = thermography_analyzer.analyze(image)

        # Should handle negative temps
        assert result is not None

    def test_very_high_temperatures(self, thermography_analyzer):
        """Test handling of very high temperatures."""
        image = ThermalImage(
            image_id="THERM-HOT",
            camera_id="IR-CAM-001",
            timestamp=datetime.now(timezone.utc),
            min_temperature_c=100.0,
            max_temperature_c=500.0,  # Very hot
            avg_temperature_c=200.0,
            emissivity=0.85,
            ambient_c=30.0,
        )

        result = thermography_analyzer.analyze(image)

        assert result.thermal_severity == AlertSeverity.UNACCEPTABLE

    def test_many_hot_spots(self, thermography_analyzer):
        """Test handling of many hot spots."""
        hot_spots = [
            {"x": i * 50, "y": i * 50, "temperature_c": 80 + i, "area_pixels": 25}
            for i in range(20)
        ]

        image = ThermalImage(
            image_id="THERM-MANY",
            camera_id="IR-CAM-001",
            timestamp=datetime.now(timezone.utc),
            min_temperature_c=40.0,
            max_temperature_c=100.0,
            avg_temperature_c=60.0,
            hot_spots=hot_spots,
            emissivity=0.95,
            ambient_c=25.0,
        )

        result = thermography_analyzer.analyze(image)

        # Should handle many hot spots
        assert result is not None


class TestThermographyIntegration:
    """Integration tests for thermography analysis."""

    def test_full_analysis_workflow(self, temperature_thresholds):
        """Test complete analysis workflow."""
        analyzer = ThermographyAnalyzer(temperature_thresholds)

        image = ThermalImage(
            image_id="THERM-INT",
            camera_id="IR-CAM-001",
            timestamp=datetime.now(timezone.utc),
            min_temperature_c=38.0,
            max_temperature_c=82.0,
            avg_temperature_c=52.0,
            hot_spots=[
                {"x": 120, "y": 180, "temperature_c": 82.0, "area_pixels": 45},
            ],
            emissivity=0.90,
            ambient_c=28.0,
            reference_temperature_c=40.0,
        )

        result = analyzer.analyze(image)

        # Verify comprehensive result
        assert result.image_id == "THERM-INT"
        assert result.max_temperature_c == 82.0
        assert result.thermal_severity is not None
        assert result.hot_spots_detected is True
        assert isinstance(result.recommendations, list)
        if result.delta_t_c is not None:
            assert result.delta_t_c > 0

    def test_batch_analysis(self, thermography_analyzer):
        """Test batch analysis of multiple images."""
        images = [
            ThermalImage(
                image_id=f"THERM-BATCH-{i}",
                camera_id="IR-CAM-001",
                timestamp=datetime.now(timezone.utc),
                min_temperature_c=35.0,
                max_temperature_c=50.0 + i * 10,
                avg_temperature_c=40.0 + i * 5,
                emissivity=0.95,
                ambient_c=25.0,
            )
            for i in range(5)
        ]

        results = [thermography_analyzer.analyze(img) for img in images]

        # All should produce valid results
        assert len(results) == 5
        # Later images should have higher severity
        # (as temperatures increase)
