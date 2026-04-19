# -*- coding: utf-8 -*-
"""
Unit Tests: Condenser State Classifier

Comprehensive tests for condenser state classification including:
- Failure mode detection (fouling, air leak, tube issues)
- Severity classification
- Confidence scoring
- Recommendation generation

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
    FailureMode,
    FailureSeverity,
    CondenserReading,
    CondenserConfig,
    CondenserStateResult,
    OperatingMode,
    TubeMaterial,
    WaterSource,
    AssertionHelpers,
    ProvenanceCalculator,
    OPERATING_LIMITS,
    TEST_SEED,
)


# =============================================================================
# CONDENSER STATE CLASSIFIER IMPLEMENTATION FOR TESTING
# =============================================================================

class CondenserStateClassifier:
    """
    Condenser state classifier using sensor data analysis.

    Classifies condenser operating state and identifies failure modes
    based on sensor readings and derived metrics.
    """

    VERSION = "1.0.0"

    # Threshold values for classification
    THRESHOLDS = {
        "cf_warning": 0.80,
        "cf_critical": 0.75,
        "cf_minimum": 0.70,
        "ttd_warning": 8.0,  # C
        "ttd_critical": 12.0,  # C
        "air_ingress_warning": 5.0,  # SCFM
        "air_ingress_critical": 10.0,  # SCFM
        "do_warning": 20.0,  # ppb
        "do_critical": 50.0,  # ppb
        "subcooling_warning": 3.0,  # C
        "subcooling_critical": 5.0,  # C
        "vacuum_deviation_warning": 1.0,  # kPa
        "vacuum_deviation_critical": 2.0,  # kPa
    }

    # Failure mode signatures
    SIGNATURES = {
        FailureMode.FOULING_BIOLOGICAL: {
            "ttd_elevated": True,
            "cf_degraded": True,
            "air_ingress_normal": True,
            "seasonal_correlation": True,
        },
        FailureMode.FOULING_SCALE: {
            "ttd_elevated": True,
            "cf_degraded": True,
            "gradual_progression": True,
        },
        FailureMode.AIR_LEAK_MINOR: {
            "air_ingress_elevated": True,
            "do_elevated": True,
            "subcooling_elevated": True,
            "cf_slight_degradation": True,
        },
        FailureMode.AIR_LEAK_MAJOR: {
            "air_ingress_high": True,
            "do_high": True,
            "subcooling_high": True,
            "vacuum_degraded": True,
        },
        FailureMode.TUBE_LEAK: {
            "conductivity_elevated": True,
            "hotwell_level_change": True,
        },
        FailureMode.TUBE_PLUGGED: {
            "cw_flow_reduced": True,
            "pressure_drop_elevated": True,
            "ttd_elevated": True,
        },
        FailureMode.CW_PUMP_DEGRADED: {
            "cw_flow_reduced": True,
            "pump_current_elevated": True,
        },
    }

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize classifier."""
        self.config = config or {}
        for key, value in self.THRESHOLDS.items():
            if key in self.config:
                self.THRESHOLDS[key] = self.config[key]

    def calculate_expected_vacuum(
        self,
        cw_inlet_temp_c: float,
        cw_flow_m3_s: float,
        heat_duty_mw: float,
        cf: float = 0.85
    ) -> float:
        """
        Calculate expected vacuum based on operating conditions.

        Args:
            cw_inlet_temp_c: CW inlet temperature
            cw_flow_m3_s: CW flow rate
            heat_duty_mw: Heat duty
            cf: Cleanliness factor

        Returns:
            Expected vacuum pressure (kPa absolute)
        """
        # Simplified model: vacuum depends on CW inlet + heat load + cleanliness
        # Higher CW inlet -> higher vacuum (worse)
        # Lower CF -> higher vacuum (worse)

        base_approach = 10.0  # C, base approach temp
        approach = base_approach / cf  # Approach increases with fouling

        expected_sat_temp = cw_inlet_temp_c + approach

        # Convert to pressure (simplified Antoine)
        from conftest import pressure_from_saturation_temp
        return pressure_from_saturation_temp(expected_sat_temp)

    def extract_features(self, reading: CondenserReading) -> Dict[str, float]:
        """
        Extract classification features from reading.

        Args:
            reading: Condenser sensor reading

        Returns:
            Dictionary of feature values
        """
        return {
            "cw_temp_rise": reading.cw_temp_rise_c,
            "ttd": reading.terminal_temp_diff_c,
            "approach": reading.approach_temp_c,
            "subcooling": reading.subcooling_c,
            "air_ingress": reading.air_ingress_scfm,
            "dissolved_oxygen": reading.dissolved_oxygen_ppb,
            "vacuum_pressure": reading.vacuum_pressure_kpa_abs,
            "cw_flow": reading.cw_flow_m3_s,
            "unit_load": reading.unit_load_mw,
        }

    def check_fouling_indicators(
        self,
        features: Dict[str, float],
        expected_vacuum: float
    ) -> Tuple[bool, float]:
        """
        Check for fouling indicators.

        Args:
            features: Extracted features
            expected_vacuum: Expected vacuum pressure

        Returns:
            Tuple of (is_fouling_detected, confidence)
        """
        indicators = []
        confidence_scores = []

        # High TTD indicates fouling
        if features["ttd"] > self.THRESHOLDS["ttd_warning"]:
            indicators.append("ttd_elevated")
            conf = min(1.0, features["ttd"] / self.THRESHOLDS["ttd_critical"])
            confidence_scores.append(conf)

        # Elevated vacuum (worse than expected)
        vacuum_deviation = features["vacuum_pressure"] - expected_vacuum
        if vacuum_deviation > self.THRESHOLDS["vacuum_deviation_warning"]:
            indicators.append("vacuum_degraded")
            conf = min(1.0, vacuum_deviation / self.THRESHOLDS["vacuum_deviation_critical"])
            confidence_scores.append(conf)

        # High approach temperature
        if features["approach"] > features["cw_temp_rise"] * 1.5:
            indicators.append("approach_elevated")
            confidence_scores.append(0.7)

        is_fouling = len(indicators) >= 2
        confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        return is_fouling, confidence

    def check_air_leak_indicators(self, features: Dict[str, float]) -> Tuple[bool, float, str]:
        """
        Check for air in-leakage indicators.

        Args:
            features: Extracted features

        Returns:
            Tuple of (is_air_leak, confidence, severity)
        """
        indicators = []
        confidence_scores = []

        # High air ingress
        if features["air_ingress"] > self.THRESHOLDS["air_ingress_critical"]:
            indicators.append("air_ingress_high")
            confidence_scores.append(0.95)
            severity = "major"
        elif features["air_ingress"] > self.THRESHOLDS["air_ingress_warning"]:
            indicators.append("air_ingress_elevated")
            confidence_scores.append(0.8)
            severity = "minor"
        else:
            severity = "none"

        # High dissolved oxygen
        if features["dissolved_oxygen"] > self.THRESHOLDS["do_critical"]:
            indicators.append("do_high")
            confidence_scores.append(0.9)
        elif features["dissolved_oxygen"] > self.THRESHOLDS["do_warning"]:
            indicators.append("do_elevated")
            confidence_scores.append(0.7)

        # Elevated subcooling
        if features["subcooling"] > self.THRESHOLDS["subcooling_critical"]:
            indicators.append("subcooling_high")
            confidence_scores.append(0.85)
        elif features["subcooling"] > self.THRESHOLDS["subcooling_warning"]:
            indicators.append("subcooling_elevated")
            confidence_scores.append(0.6)

        is_air_leak = len(indicators) >= 2
        confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        return is_air_leak, confidence, severity

    def determine_severity(
        self,
        failure_mode: FailureMode,
        features: Dict[str, float]
    ) -> FailureSeverity:
        """
        Determine severity of detected failure mode.

        Args:
            failure_mode: Detected failure mode
            features: Extracted features

        Returns:
            Severity classification
        """
        if failure_mode == FailureMode.NORMAL:
            return FailureSeverity.NONE

        # Fouling severity based on TTD
        if failure_mode in [FailureMode.FOULING_BIOLOGICAL, FailureMode.FOULING_SCALE]:
            ttd = features["ttd"]
            if ttd > self.THRESHOLDS["ttd_critical"]:
                return FailureSeverity.HIGH
            elif ttd > self.THRESHOLDS["ttd_warning"]:
                return FailureSeverity.MEDIUM
            else:
                return FailureSeverity.LOW

        # Air leak severity
        if failure_mode in [FailureMode.AIR_LEAK_MINOR, FailureMode.AIR_LEAK_MAJOR]:
            air = features["air_ingress"]
            if air > self.THRESHOLDS["air_ingress_critical"]:
                return FailureSeverity.HIGH
            elif air > self.THRESHOLDS["air_ingress_warning"]:
                return FailureSeverity.MEDIUM
            else:
                return FailureSeverity.LOW

        return FailureSeverity.MEDIUM

    def generate_recommendations(
        self,
        failure_mode: FailureMode,
        severity: FailureSeverity
    ) -> List[str]:
        """
        Generate action recommendations.

        Args:
            failure_mode: Detected failure mode
            severity: Severity level

        Returns:
            List of recommended actions
        """
        recommendations = []

        if failure_mode == FailureMode.NORMAL:
            recommendations.append("Continue normal monitoring")
            return recommendations

        # Fouling recommendations
        if failure_mode in [FailureMode.FOULING_BIOLOGICAL, FailureMode.FOULING_SCALE]:
            if severity == FailureSeverity.HIGH:
                recommendations.append("Schedule offline cleaning within 7 days")
                recommendations.append("Increase online cleaning frequency")
                recommendations.append("Review chemical treatment program")
            elif severity == FailureSeverity.MEDIUM:
                recommendations.append("Increase online ball/brush cleaning frequency")
                recommendations.append("Plan offline cleaning at next outage")
            else:
                recommendations.append("Monitor fouling trend")
                recommendations.append("Ensure online cleaning system is operational")

        # Air leak recommendations
        elif failure_mode in [FailureMode.AIR_LEAK_MINOR, FailureMode.AIR_LEAK_MAJOR]:
            if severity == FailureSeverity.HIGH:
                recommendations.append("Conduct emergency air leak search")
                recommendations.append("Check turbine gland seals")
                recommendations.append("Verify air ejector capacity")
            elif severity == FailureSeverity.MEDIUM:
                recommendations.append("Schedule air leak detection survey")
                recommendations.append("Check condenser expansion joint seals")
            else:
                recommendations.append("Monitor air ingress trend")

        # Tube issues
        elif failure_mode == FailureMode.TUBE_LEAK:
            recommendations.append("Perform chemistry analysis")
            recommendations.append("Schedule eddy current testing")
            recommendations.append("Identify and plug leaking tubes")

        elif failure_mode == FailureMode.TUBE_PLUGGED:
            recommendations.append("Review waterbox debris screens")
            recommendations.append("Schedule offline cleaning")

        return recommendations

    def estimate_mw_impact(
        self,
        failure_mode: FailureMode,
        severity: FailureSeverity,
        unit_load_mw: float
    ) -> float:
        """
        Estimate MW impact of failure mode.

        Args:
            failure_mode: Detected failure mode
            severity: Severity level
            unit_load_mw: Current unit load

        Returns:
            Estimated MW loss
        """
        # Base impact factors (% of load)
        impact_factors = {
            (FailureMode.FOULING_BIOLOGICAL, FailureSeverity.HIGH): 0.02,
            (FailureMode.FOULING_BIOLOGICAL, FailureSeverity.MEDIUM): 0.01,
            (FailureMode.FOULING_BIOLOGICAL, FailureSeverity.LOW): 0.005,
            (FailureMode.FOULING_SCALE, FailureSeverity.HIGH): 0.015,
            (FailureMode.FOULING_SCALE, FailureSeverity.MEDIUM): 0.008,
            (FailureMode.AIR_LEAK_MAJOR, FailureSeverity.HIGH): 0.025,
            (FailureMode.AIR_LEAK_MINOR, FailureSeverity.MEDIUM): 0.01,
            (FailureMode.TUBE_PLUGGED, FailureSeverity.HIGH): 0.012,
        }

        factor = impact_factors.get((failure_mode, severity), 0.005)
        return unit_load_mw * factor

    def classify(
        self,
        reading: CondenserReading,
        config: CondenserConfig = None
    ) -> CondenserStateResult:
        """
        Classify condenser state from reading.

        Args:
            reading: Condenser sensor reading
            config: Condenser configuration

        Returns:
            CondenserStateResult with classification
        """
        # Extract features
        features = self.extract_features(reading)

        # Calculate expected vacuum
        heat_duty_mw = (
            features["cw_flow"] * 1000 * 4.186 * features["cw_temp_rise"] / 1000
        )
        expected_vacuum = self.calculate_expected_vacuum(
            reading.cw_inlet_temp_c,
            reading.cw_flow_m3_s,
            heat_duty_mw
        )

        # Check for different failure modes
        failure_mode = FailureMode.NORMAL
        confidence = 1.0
        contributing_factors = []

        # Check fouling
        is_fouling, fouling_conf = self.check_fouling_indicators(features, expected_vacuum)
        if is_fouling:
            failure_mode = FailureMode.FOULING_BIOLOGICAL  # Simplification
            confidence = fouling_conf
            contributing_factors.append("Elevated TTD")
            if features["vacuum_pressure"] > expected_vacuum:
                contributing_factors.append("Degraded vacuum")

        # Check air leakage (may override or combine with fouling)
        is_air_leak, air_conf, air_severity = self.check_air_leak_indicators(features)
        if is_air_leak and air_conf > confidence:
            failure_mode = (
                FailureMode.AIR_LEAK_MAJOR if air_severity == "major"
                else FailureMode.AIR_LEAK_MINOR
            )
            confidence = air_conf
            contributing_factors = ["Elevated air ingress"]
            if features["dissolved_oxygen"] > self.THRESHOLDS["do_warning"]:
                contributing_factors.append("High dissolved oxygen")
            if features["subcooling"] > self.THRESHOLDS["subcooling_warning"]:
                contributing_factors.append("Excessive subcooling")

        # Determine severity
        severity = self.determine_severity(failure_mode, features)

        # Generate recommendations
        recommendations = self.generate_recommendations(failure_mode, severity)

        # Estimate MW impact
        mw_impact = self.estimate_mw_impact(failure_mode, severity, reading.unit_load_mw)

        # Generate provenance hash
        input_data = reading.to_dict()
        provenance_hash = hashlib.sha256(
            json.dumps(input_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        return CondenserStateResult(
            condenser_id=reading.condenser_id,
            timestamp=reading.timestamp,
            failure_mode=failure_mode,
            severity=severity,
            confidence=round(confidence, 3),
            contributing_factors=contributing_factors,
            recommended_actions=recommendations,
            estimated_impact_mw=round(mw_impact, 3),
            provenance_hash=provenance_hash,
        )


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def classifier() -> CondenserStateClassifier:
    """Create classifier instance."""
    return CondenserStateClassifier()


@pytest.fixture
def classifier_with_config() -> CondenserStateClassifier:
    """Create classifier with custom config."""
    config = {
        "cf_warning": 0.82,
        "cf_critical": 0.76,
        "ttd_warning": 7.0,
        "ttd_critical": 10.0,
    }
    return CondenserStateClassifier(config)


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestFeatureExtraction:
    """Tests for feature extraction."""

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_extract_features_basic(
        self,
        classifier: CondenserStateClassifier,
        healthy_condenser_reading: CondenserReading
    ):
        """Test basic feature extraction."""
        features = classifier.extract_features(healthy_condenser_reading)

        assert "ttd" in features
        assert "cw_temp_rise" in features
        assert "air_ingress" in features
        assert "vacuum_pressure" in features

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_extract_features_values_correct(
        self,
        classifier: CondenserStateClassifier,
        healthy_condenser_reading: CondenserReading
    ):
        """Test extracted feature values are correct."""
        features = classifier.extract_features(healthy_condenser_reading)

        expected_ttd = (
            healthy_condenser_reading.saturation_temp_c -
            healthy_condenser_reading.cw_outlet_temp_c
        )
        assert abs(features["ttd"] - expected_ttd) < 0.01

        expected_rise = (
            healthy_condenser_reading.cw_outlet_temp_c -
            healthy_condenser_reading.cw_inlet_temp_c
        )
        assert abs(features["cw_temp_rise"] - expected_rise) < 0.01


class TestFoulingDetection:
    """Tests for fouling detection."""

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_detect_fouling_healthy(
        self,
        classifier: CondenserStateClassifier,
        healthy_condenser_reading: CondenserReading
    ):
        """Test fouling detection for healthy condenser."""
        features = classifier.extract_features(healthy_condenser_reading)
        expected_vacuum = 5.0  # kPa

        is_fouling, confidence = classifier.check_fouling_indicators(
            features, expected_vacuum
        )

        # Healthy condenser should not show fouling
        assert not is_fouling or confidence < 0.5

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_detect_fouling_fouled(
        self,
        classifier: CondenserStateClassifier,
        fouled_condenser_reading: CondenserReading
    ):
        """Test fouling detection for fouled condenser."""
        features = classifier.extract_features(fouled_condenser_reading)
        expected_vacuum = 5.0  # kPa

        is_fouling, confidence = classifier.check_fouling_indicators(
            features, expected_vacuum
        )

        # Fouled condenser should be detected
        assert is_fouling
        assert confidence > 0.5

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_fouling_confidence_increases_with_severity(
        self,
        classifier: CondenserStateClassifier
    ):
        """Test fouling confidence increases with TTD."""
        # Moderate fouling
        features_moderate = {"ttd": 9.0, "approach": 15.0, "cw_temp_rise": 10.0}
        _, conf_moderate = classifier.check_fouling_indicators(features_moderate, 5.0)

        # Severe fouling
        features_severe = {"ttd": 15.0, "approach": 20.0, "cw_temp_rise": 10.0}
        _, conf_severe = classifier.check_fouling_indicators(features_severe, 5.0)

        assert conf_severe > conf_moderate


class TestAirLeakDetection:
    """Tests for air leak detection."""

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_detect_air_leak_healthy(
        self,
        classifier: CondenserStateClassifier,
        healthy_condenser_reading: CondenserReading
    ):
        """Test air leak detection for healthy condenser."""
        features = classifier.extract_features(healthy_condenser_reading)

        is_leak, confidence, severity = classifier.check_air_leak_indicators(features)

        assert not is_leak or severity == "none"

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_detect_air_leak_minor(
        self,
        classifier: CondenserStateClassifier
    ):
        """Test detection of minor air leak."""
        features = {
            "air_ingress": 7.0,  # Above warning
            "dissolved_oxygen": 30.0,  # Elevated
            "subcooling": 4.0,  # Elevated
        }

        is_leak, confidence, severity = classifier.check_air_leak_indicators(features)

        assert is_leak
        assert severity == "minor"

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_detect_air_leak_major(
        self,
        classifier: CondenserStateClassifier,
        air_leak_condenser_reading: CondenserReading
    ):
        """Test detection of major air leak."""
        features = classifier.extract_features(air_leak_condenser_reading)

        is_leak, confidence, severity = classifier.check_air_leak_indicators(features)

        assert is_leak
        assert confidence > 0.7


class TestSeverityClassification:
    """Tests for severity classification."""

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_severity_normal(self, classifier: CondenserStateClassifier):
        """Test severity for normal operation."""
        features = {"ttd": 3.0, "air_ingress": 2.0}

        severity = classifier.determine_severity(FailureMode.NORMAL, features)

        assert severity == FailureSeverity.NONE

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_severity_fouling_high(self, classifier: CondenserStateClassifier):
        """Test high severity for severe fouling."""
        features = {"ttd": 15.0}

        severity = classifier.determine_severity(
            FailureMode.FOULING_BIOLOGICAL, features
        )

        assert severity == FailureSeverity.HIGH

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_severity_fouling_medium(self, classifier: CondenserStateClassifier):
        """Test medium severity for moderate fouling."""
        features = {"ttd": 10.0}

        severity = classifier.determine_severity(
            FailureMode.FOULING_BIOLOGICAL, features
        )

        assert severity == FailureSeverity.MEDIUM

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_severity_air_leak_high(self, classifier: CondenserStateClassifier):
        """Test high severity for major air leak."""
        features = {"air_ingress": 15.0}

        severity = classifier.determine_severity(
            FailureMode.AIR_LEAK_MAJOR, features
        )

        assert severity == FailureSeverity.HIGH


class TestRecommendationGeneration:
    """Tests for recommendation generation."""

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_recommendations_normal(self, classifier: CondenserStateClassifier):
        """Test recommendations for normal operation."""
        recommendations = classifier.generate_recommendations(
            FailureMode.NORMAL, FailureSeverity.NONE
        )

        assert len(recommendations) > 0
        assert any("monitoring" in r.lower() for r in recommendations)

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_recommendations_fouling_high(self, classifier: CondenserStateClassifier):
        """Test recommendations for severe fouling."""
        recommendations = classifier.generate_recommendations(
            FailureMode.FOULING_BIOLOGICAL, FailureSeverity.HIGH
        )

        assert len(recommendations) > 1
        assert any("cleaning" in r.lower() for r in recommendations)

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_recommendations_air_leak_high(self, classifier: CondenserStateClassifier):
        """Test recommendations for major air leak."""
        recommendations = classifier.generate_recommendations(
            FailureMode.AIR_LEAK_MAJOR, FailureSeverity.HIGH
        )

        assert len(recommendations) > 1
        assert any("leak" in r.lower() for r in recommendations)


class TestMWImpactEstimation:
    """Tests for MW impact estimation."""

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_mw_impact_normal(self, classifier: CondenserStateClassifier):
        """Test MW impact for normal operation."""
        impact = classifier.estimate_mw_impact(
            FailureMode.NORMAL, FailureSeverity.NONE, 500.0
        )

        assert impact == 0.0 or impact < 1.0  # Minimal impact

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_mw_impact_fouling_high(self, classifier: CondenserStateClassifier):
        """Test MW impact for severe fouling."""
        impact = classifier.estimate_mw_impact(
            FailureMode.FOULING_BIOLOGICAL, FailureSeverity.HIGH, 500.0
        )

        # Should show significant impact (> 5 MW for 500 MW unit)
        assert impact > 5.0

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_mw_impact_scales_with_load(self, classifier: CondenserStateClassifier):
        """Test MW impact scales with unit load."""
        impact_low = classifier.estimate_mw_impact(
            FailureMode.FOULING_BIOLOGICAL, FailureSeverity.HIGH, 250.0
        )
        impact_high = classifier.estimate_mw_impact(
            FailureMode.FOULING_BIOLOGICAL, FailureSeverity.HIGH, 500.0
        )

        assert impact_high == 2 * impact_low


class TestFullClassification:
    """Tests for full classification."""

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_classify_healthy(
        self,
        classifier: CondenserStateClassifier,
        healthy_condenser_reading: CondenserReading
    ):
        """Test classification of healthy condenser."""
        result = classifier.classify(healthy_condenser_reading)

        assert isinstance(result, CondenserStateResult)
        assert result.failure_mode == FailureMode.NORMAL
        assert result.severity == FailureSeverity.NONE

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_classify_fouled(
        self,
        classifier: CondenserStateClassifier,
        fouled_condenser_reading: CondenserReading
    ):
        """Test classification of fouled condenser."""
        result = classifier.classify(fouled_condenser_reading)

        assert result.failure_mode in [
            FailureMode.FOULING_BIOLOGICAL,
            FailureMode.FOULING_SCALE,
            FailureMode.FOULING_DEBRIS,
        ]
        assert result.severity in [FailureSeverity.MEDIUM, FailureSeverity.HIGH]

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_classify_air_leak(
        self,
        classifier: CondenserStateClassifier,
        air_leak_condenser_reading: CondenserReading
    ):
        """Test classification of air leak condition."""
        result = classifier.classify(air_leak_condenser_reading)

        assert result.failure_mode in [
            FailureMode.AIR_LEAK_MINOR,
            FailureMode.AIR_LEAK_MAJOR,
        ]

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_classify_has_provenance(
        self,
        classifier: CondenserStateClassifier,
        healthy_condenser_reading: CondenserReading
    ):
        """Test classification includes provenance hash."""
        result = classifier.classify(healthy_condenser_reading)

        assert len(result.provenance_hash) == 64

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_classify_has_recommendations(
        self,
        classifier: CondenserStateClassifier,
        fouled_condenser_reading: CondenserReading
    ):
        """Test classification includes recommendations."""
        result = classifier.classify(fouled_condenser_reading)

        assert len(result.recommended_actions) > 0

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_classify_has_contributing_factors(
        self,
        classifier: CondenserStateClassifier,
        fouled_condenser_reading: CondenserReading
    ):
        """Test classification includes contributing factors."""
        result = classifier.classify(fouled_condenser_reading)

        assert len(result.contributing_factors) > 0

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_classify_fleet(
        self,
        classifier: CondenserStateClassifier,
        condenser_fleet: List[CondenserReading]
    ):
        """Test classification of entire fleet."""
        results = [classifier.classify(reading) for reading in condenser_fleet]

        assert len(results) == len(condenser_fleet)
        # Should detect different conditions in fleet
        failure_modes = set(r.failure_mode for r in results)
        assert len(failure_modes) >= 2  # At least healthy and one failure


class TestConfidenceScoring:
    """Tests for confidence scoring."""

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_confidence_in_valid_range(
        self,
        classifier: CondenserStateClassifier,
        healthy_condenser_reading: CondenserReading
    ):
        """Test confidence is in valid range."""
        result = classifier.classify(healthy_condenser_reading)

        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.unit
    @pytest.mark.classifier
    def test_confidence_high_for_clear_case(
        self,
        classifier: CondenserStateClassifier,
        air_leak_condenser_reading: CondenserReading
    ):
        """Test confidence is high for clear failure case."""
        result = classifier.classify(air_leak_condenser_reading)

        if result.failure_mode != FailureMode.NORMAL:
            assert result.confidence > 0.5


class TestDeterminism:
    """Tests for deterministic behavior."""

    @pytest.mark.unit
    @pytest.mark.classifier
    @pytest.mark.golden
    def test_classification_is_deterministic(
        self,
        classifier: CondenserStateClassifier,
        healthy_condenser_reading: CondenserReading
    ):
        """Test classification is deterministic."""
        results = [
            classifier.classify(healthy_condenser_reading)
            for _ in range(10)
        ]

        # All classifications should be identical
        modes = [r.failure_mode for r in results]
        assert len(set(modes)) == 1

        # All hashes should be identical
        hashes = [r.provenance_hash for r in results]
        assert len(set(hashes)) == 1

    @pytest.mark.unit
    @pytest.mark.classifier
    @pytest.mark.golden
    def test_different_inputs_different_hash(
        self,
        classifier: CondenserStateClassifier,
        healthy_condenser_reading: CondenserReading,
        fouled_condenser_reading: CondenserReading
    ):
        """Test different inputs produce different hashes."""
        result1 = classifier.classify(healthy_condenser_reading)
        result2 = classifier.classify(fouled_condenser_reading)

        assert result1.provenance_hash != result2.provenance_hash


class TestPerformance:
    """Performance tests."""

    @pytest.mark.unit
    @pytest.mark.classifier
    @pytest.mark.performance
    def test_classification_speed(
        self,
        classifier: CondenserStateClassifier,
        healthy_condenser_reading: CondenserReading,
        performance_timer
    ):
        """Test classification speed."""
        timer = performance_timer()

        with timer:
            for _ in range(1000):
                classifier.classify(healthy_condenser_reading)

        # 1000 classifications in < 1 second
        assert timer.elapsed < 1.0

    @pytest.mark.unit
    @pytest.mark.classifier
    @pytest.mark.performance
    def test_fleet_classification_throughput(
        self,
        classifier: CondenserStateClassifier,
        condenser_fleet: List[CondenserReading],
        throughput_measurer
    ):
        """Test fleet classification throughput."""
        measurer = throughput_measurer()

        with measurer:
            for _ in range(100):
                for reading in condenser_fleet:
                    classifier.classify(reading)
            measurer.add_items(100 * len(condenser_fleet))

        # Should achieve high throughput
        assert measurer.items_per_second >= 500
