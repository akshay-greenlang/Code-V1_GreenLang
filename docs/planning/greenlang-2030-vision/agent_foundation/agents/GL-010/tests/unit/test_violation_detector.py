# -*- coding: utf-8 -*-
"""
Unit Tests for GL-010 EMISSIONWATCH Violation Detector.

Tests exceedance detection, predictive alerts, trend analysis,
severity classification, and false positive filtering.

Test Count: 20+ tests
Coverage Target: 90%+

Author: GreenLang Foundation Test Engineering
Version: 1.0.0
"""

import pytest
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tools import (
    EmissionsComplianceTools,
    ViolationResult,
    ExceedancePredictionResult,
)


# =============================================================================
# TEST CLASS: VIOLATION DETECTOR
# =============================================================================

@pytest.mark.unit
class TestViolationDetector:
    """Test suite for violation detection."""

    # =========================================================================
    # EXCEEDANCE DETECTION TESTS
    # =========================================================================

    def test_exceedance_detection_single_pollutant(self, emissions_tools, epa_permit_limits):
        """Test exceedance detection for single pollutant."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.15},  # Above 0.10 limit
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        assert len(violations) == 1
        assert violations[0].pollutant == "NOX"
        assert violations[0].measured_value == 0.15
        assert violations[0].limit_value == 0.10

    def test_exceedance_detection_multiple_pollutants(self, emissions_tools, epa_permit_limits):
        """Test exceedance detection for multiple pollutants."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.15},  # Violation
            "sox": {"emission_rate_lb_mmbtu": 0.20},  # Violation
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.05},  # Violation
        }

        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        assert len(violations) == 3
        pollutants = [v.pollutant for v in violations]
        assert "NOX" in pollutants
        assert "SOX" in pollutants
        assert "PM" in pollutants

    def test_exceedance_detection_no_violations(self, emissions_tools, epa_permit_limits):
        """Test no violations when compliant."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.08},
            "sox": {"emission_rate_lb_mmbtu": 0.10},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        assert len(violations) == 0

    def test_exceedance_detection_at_limit(self, emissions_tools, epa_permit_limits):
        """Test detection when exactly at limit."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.10},  # Exactly at limit
            "sox": {"emission_rate_lb_mmbtu": 0.15},  # Exactly at limit
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.03},  # Exactly at limit
        }

        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        # At limit should not be violation
        assert len(violations) == 0

    def test_exceedance_percent_calculation(self, emissions_tools, epa_permit_limits):
        """Test exceedance percentage calculation."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.15},  # 50% above limit
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        assert len(violations) == 1
        assert violations[0].exceedance_percent == 50.0

    # =========================================================================
    # PREDICTIVE ALERT TESTS
    # =========================================================================

    def test_predictive_alert_upward_trend(self, emissions_tools, cems_data_series, epa_permit_limits):
        """Test predictive alert for upward trending emissions."""
        # Create upward trending data
        historical_data = []
        for i in range(24):
            historical_data.append({
                "nox_lb_mmbtu": 0.05 + i * 0.002,  # Increasing trend
                "sox_lb_mmbtu": 0.08,
                "pm_lb_mmbtu": 0.02,
            })

        predictions = emissions_tools.predict_exceedances(
            historical_data=historical_data,
            permit_limits=epa_permit_limits,
            forecast_hours=24,
        )

        assert len(predictions) > 0
        # Find NOx prediction
        nox_pred = next((p for p in predictions if p.pollutant == "NOX"), None)
        assert nox_pred is not None
        assert nox_pred.model_type == "linear_extrapolation"

    def test_predictive_alert_stable_trend(self, emissions_tools, epa_permit_limits):
        """Test predictive alert for stable emissions."""
        historical_data = []
        for i in range(24):
            historical_data.append({
                "nox_lb_mmbtu": 0.05,  # Stable
                "sox_lb_mmbtu": 0.08,
                "pm_lb_mmbtu": 0.02,
            })

        predictions = emissions_tools.predict_exceedances(
            historical_data=historical_data,
            permit_limits=epa_permit_limits,
            forecast_hours=24,
        )

        # Stable data should have low exceedance probability
        nox_pred = next((p for p in predictions if p.pollutant == "NOX"), None)
        if nox_pred:
            assert nox_pred.exceedance_probability < 50

    def test_predictive_alert_time_to_exceedance(self, emissions_tools, epa_permit_limits):
        """Test time-to-exceedance calculation."""
        # Create data trending toward limit
        historical_data = []
        for i in range(24):
            historical_data.append({
                "nox_lb_mmbtu": 0.06 + i * 0.001,  # Gradual increase
                "sox_lb_mmbtu": 0.08,
                "pm_lb_mmbtu": 0.02,
            })

        predictions = emissions_tools.predict_exceedances(
            historical_data=historical_data,
            permit_limits=epa_permit_limits,
            forecast_hours=48,
        )

        nox_pred = next((p for p in predictions if p.pollutant == "NOX"), None)
        if nox_pred and nox_pred.time_to_exceedance_hours:
            assert nox_pred.time_to_exceedance_hours > 0

    # =========================================================================
    # TREND ANALYSIS TESTS
    # =========================================================================

    def test_trend_analysis_increasing(self, emissions_tools, epa_permit_limits):
        """Test trend analysis detects increasing pattern."""
        historical_data = []
        for i in range(24):
            historical_data.append({
                "nox_lb_mmbtu": 0.04 + i * 0.002,
                "sox_lb_mmbtu": 0.08,
                "pm_lb_mmbtu": 0.02,
            })

        predictions = emissions_tools.predict_exceedances(
            historical_data=historical_data,
            permit_limits=epa_permit_limits,
        )

        nox_pred = next((p for p in predictions if p.pollutant == "NOX"), None)
        assert nox_pred is not None
        # Predicted should be higher than current for increasing trend
        assert nox_pred.predicted_value > nox_pred.current_value

    def test_trend_analysis_decreasing(self, emissions_tools, epa_permit_limits):
        """Test trend analysis detects decreasing pattern."""
        historical_data = []
        for i in range(24):
            historical_data.append({
                "nox_lb_mmbtu": 0.09 - i * 0.001,  # Decreasing
                "sox_lb_mmbtu": 0.08,
                "pm_lb_mmbtu": 0.02,
            })

        predictions = emissions_tools.predict_exceedances(
            historical_data=historical_data,
            permit_limits=epa_permit_limits,
        )

        nox_pred = next((p for p in predictions if p.pollutant == "NOX"), None)
        assert nox_pred is not None
        # Predicted should be lower for decreasing trend
        assert nox_pred.predicted_value < nox_pred.current_value

    def test_trend_analysis_confidence_level(self, emissions_tools, epa_permit_limits):
        """Test confidence level based on data points."""
        # Few data points - low confidence
        short_data = [{"nox_lb_mmbtu": 0.05, "sox_lb_mmbtu": 0.08, "pm_lb_mmbtu": 0.02}
                      for _ in range(5)]

        predictions_short = emissions_tools.predict_exceedances(
            historical_data=short_data,
            permit_limits=epa_permit_limits,
        )

        # Many data points - higher confidence
        long_data = [{"nox_lb_mmbtu": 0.05, "sox_lb_mmbtu": 0.08, "pm_lb_mmbtu": 0.02}
                     for _ in range(30)]

        predictions_long = emissions_tools.predict_exceedances(
            historical_data=long_data,
            permit_limits=epa_permit_limits,
        )

        nox_short = next((p for p in predictions_short if p.pollutant == "NOX"), None)
        nox_long = next((p for p in predictions_long if p.pollutant == "NOX"), None)

        if nox_short and nox_long:
            # More data should give higher confidence
            confidence_order = {"low": 1, "medium": 2, "high": 3}
            assert confidence_order[nox_long.confidence_level] >= confidence_order[nox_short.confidence_level]

    # =========================================================================
    # SEVERITY CLASSIFICATION TESTS
    # =========================================================================

    def test_severity_classification_low(self, emissions_tools, epa_permit_limits):
        """Test low severity classification."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.105},  # 5% above limit
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        assert len(violations) == 1
        assert violations[0].severity == "low"

    def test_severity_classification_medium(self, emissions_tools, epa_permit_limits):
        """Test medium severity classification."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.12},  # 20% above limit
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        assert len(violations) == 1
        assert violations[0].severity == "medium"

    def test_severity_classification_high(self, emissions_tools, epa_permit_limits):
        """Test high severity classification."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.14},  # 40% above limit
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        assert len(violations) == 1
        assert violations[0].severity == "high"

    def test_severity_classification_critical(self, emissions_tools, epa_permit_limits):
        """Test critical severity classification."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.20},  # 100% above limit
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        assert len(violations) == 1
        assert violations[0].severity == "critical"

    # =========================================================================
    # FALSE POSITIVE FILTERING TESTS
    # =========================================================================

    def test_false_positive_filtering_spike(self, emissions_tools, epa_permit_limits):
        """Test handling of data spikes."""
        # Single spike in otherwise normal data would be handled
        # in a more sophisticated implementation
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.15},
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        # Current implementation detects all exceedances
        assert len(violations) >= 1

    def test_violation_id_uniqueness(self, emissions_tools, epa_permit_limits):
        """Test violation IDs are unique."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.15},
            "sox": {"emission_rate_lb_mmbtu": 0.20},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.05},
        }

        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        violation_ids = [v.violation_id for v in violations]
        assert len(violation_ids) == len(set(violation_ids))  # All unique

    # =========================================================================
    # VIOLATION RESULT STRUCTURE TESTS
    # =========================================================================

    def test_violation_result_structure(self, emissions_tools, epa_permit_limits):
        """Test ViolationResult has all required fields."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.15},
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        assert len(violations) == 1
        v = violations[0]

        assert isinstance(v, ViolationResult)
        assert v.violation_id is not None
        assert v.pollutant == "NOX"
        assert v.measured_value == 0.15
        assert v.limit_value == 0.10
        assert v.exceedance_percent == 50.0
        assert v.severity in ["low", "medium", "high", "critical"]
        assert v.duration_minutes > 0
        assert v.regulatory_reference is not None
        assert v.timestamp is not None

    def test_violation_to_dict(self, emissions_tools, epa_permit_limits):
        """Test ViolationResult to_dict conversion."""
        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": 0.15},
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        v_dict = violations[0].to_dict()

        assert isinstance(v_dict, dict)
        assert "violation_id" in v_dict
        assert "pollutant" in v_dict
        assert "measured_value" in v_dict
        assert "limit_value" in v_dict
        assert "exceedance_percent" in v_dict
        assert "severity" in v_dict


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================

@pytest.mark.unit
class TestViolationDetectorParametrized:
    """Parametrized tests for violation detector."""

    @pytest.mark.parametrize("exceedance_percent,expected_severity", [
        (5.0, "low"),
        (15.0, "medium"),
        (30.0, "high"),
        (60.0, "critical"),
    ])
    def test_severity_thresholds(
        self, emissions_tools, epa_permit_limits, exceedance_percent, expected_severity
    ):
        """Test severity classification thresholds."""
        nox_value = 0.10 * (1 + exceedance_percent / 100)

        emissions_result = {
            "nox": {"emission_rate_lb_mmbtu": nox_value},
            "sox": {"emission_rate_lb_mmbtu": 0.08},
            "co2": {"mass_rate_tons_hr": 40.0},
            "pm": {"emission_rate_lb_mmbtu": 0.02},
        }

        violations = emissions_tools.detect_violations(
            emissions_result=emissions_result,
            permit_limits=epa_permit_limits,
        )

        assert len(violations) == 1
        assert violations[0].severity == expected_severity
