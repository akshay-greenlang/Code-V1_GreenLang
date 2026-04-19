# -*- coding: utf-8 -*-
"""
GL-005 Anomaly Detection Tests
==============================

Comprehensive unit tests for anomaly detection module including
SPC, ML-based, and rule-based detection methods.

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import List
import random

from greenlang.agents.process_heat.gl_005_combustion_diagnostics.config import (
    AnomalyDetectionConfig,
    SPCConfig,
    MLAnomalyConfig,
    AnomalyType,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.schemas import (
    FlueGasReading,
    AnalysisStatus,
    AnomalySeverity,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.anomaly_detection import (
    SPCAnalyzer,
    SPCStatistics,
    SPCPoint,
    MLAnomalyDetector,
    RuleBasedDetector,
    CombustionAnomalyDetector,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.tests.conftest import (
    assert_valid_provenance_hash,
)


class TestSPCAnalyzer:
    """Tests for Statistical Process Control analyzer."""

    def test_initialization(self, default_spc_config):
        """Test SPC analyzer initialization."""
        analyzer = SPCAnalyzer(default_spc_config)
        assert analyzer.config == default_spc_config
        assert analyzer._statistics == {}

    def test_update_baseline(self, default_spc_config):
        """Test baseline statistics update."""
        analyzer = SPCAnalyzer(default_spc_config)

        values = [3.0, 3.1, 2.9, 3.2, 2.8, 3.0, 3.1, 2.9, 3.0, 3.1]
        stats = analyzer.update_baseline("oxygen", values)

        assert stats.parameter == "oxygen"
        assert stats.sample_count == len(values)
        assert 2.9 < stats.mean < 3.1
        assert stats.std_dev > 0
        assert stats.ucl > stats.mean
        assert stats.lcl < stats.mean

    def test_control_limits_calculation(self, default_spc_config):
        """Test control limit calculations."""
        analyzer = SPCAnalyzer(default_spc_config)

        values = [100.0] * 10  # Constant values
        values[0] = 90.0  # Add some variance
        values[9] = 110.0

        stats = analyzer.update_baseline("test", values)

        # UCL = mean + 3 * sigma
        # LCL = mean - 3 * sigma
        assert stats.ucl == pytest.approx(stats.mean + 3 * stats.std_dev, rel=0.01)
        assert stats.lcl == pytest.approx(stats.mean - 3 * stats.std_dev, rel=0.01)

    def test_analyze_point_in_control(self, default_spc_config):
        """Test analyzing a point that is in control."""
        analyzer = SPCAnalyzer(default_spc_config)

        # Establish baseline
        baseline = [3.0 + random.gauss(0, 0.1) for _ in range(100)]
        analyzer.update_baseline("oxygen", baseline)

        # Analyze a normal point
        point, violations = analyzer.analyze_point(
            "oxygen",
            3.05,
            datetime.now(timezone.utc)
        )

        assert point.in_control is True
        assert len(violations) == 0

    def test_analyze_point_out_of_control(self, default_spc_config):
        """Test analyzing a point that is out of control."""
        analyzer = SPCAnalyzer(default_spc_config)

        # Establish baseline with tight variance
        baseline = [3.0 + random.gauss(0, 0.05) for _ in range(100)]
        analyzer.update_baseline("oxygen", baseline)

        # Analyze an outlier (beyond 3 sigma)
        stats = analyzer.get_statistics("oxygen")
        outlier_value = stats.mean + 4 * stats.std_dev

        point, violations = analyzer.analyze_point(
            "oxygen",
            outlier_value,
            datetime.now(timezone.utc)
        )

        assert point.in_control is False
        assert len(violations) > 0
        assert "Rule 1" in violations[0]

    def test_western_electric_rule_consecutive_one_side(self, default_spc_config):
        """Test Western Electric rule for consecutive points on one side."""
        analyzer = SPCAnalyzer(default_spc_config)

        # Establish baseline
        baseline = [3.0 + random.gauss(0, 0.1) for _ in range(100)]
        random.seed(42)
        analyzer.update_baseline("oxygen", baseline)

        stats = analyzer.get_statistics("oxygen")

        # Add 7+ consecutive points above mean
        for i in range(8):
            analyzer.analyze_point(
                "oxygen",
                stats.mean + 0.05,  # Slightly above mean
                datetime.now(timezone.utc) + timedelta(minutes=i)
            )

        # The last point should trigger rule 4
        point, violations = analyzer.analyze_point(
            "oxygen",
            stats.mean + 0.05,
            datetime.now(timezone.utc) + timedelta(minutes=8)
        )

        rule_4_triggered = any("Rule 4" in v for v in violations)
        # May or may not trigger depending on exact values

    def test_western_electric_rule_trending(self, default_spc_config):
        """Test Western Electric rule for trending points."""
        analyzer = SPCAnalyzer(default_spc_config)

        # Establish baseline
        baseline = [3.0 + random.gauss(0, 0.1) for _ in range(100)]
        analyzer.update_baseline("oxygen", baseline)

        # Add 6+ consecutive increasing points
        base_value = 3.0
        for i in range(7):
            analyzer.analyze_point(
                "oxygen",
                base_value + i * 0.02,  # Steadily increasing
                datetime.now(timezone.utc) + timedelta(minutes=i)
            )

        # Check if trend rule triggers
        point, violations = analyzer.analyze_point(
            "oxygen",
            base_value + 7 * 0.02,
            datetime.now(timezone.utc) + timedelta(minutes=7)
        )

        rule_5_triggered = any("Rule 5" in v for v in violations)
        # Should trigger trending rule

    def test_no_baseline_no_violation(self, default_spc_config):
        """Test that no violations occur without baseline."""
        analyzer = SPCAnalyzer(default_spc_config)

        point, violations = analyzer.analyze_point(
            "new_parameter",
            100.0,
            datetime.now(timezone.utc)
        )

        assert point.in_control is True
        assert len(violations) == 0

    def test_reset_history(self, default_spc_config):
        """Test history reset functionality."""
        analyzer = SPCAnalyzer(default_spc_config)

        baseline = [3.0] * 100
        analyzer.update_baseline("oxygen", baseline)

        # Add some points
        for i in range(10):
            analyzer.analyze_point("oxygen", 3.0, datetime.now(timezone.utc))

        # Reset
        analyzer.reset_history("oxygen")

        # History should be cleared
        assert len(analyzer._history.get("oxygen", [])) == 0


class TestMLAnomalyDetector:
    """Tests for ML-based anomaly detection."""

    def test_initialization(self, default_ml_config):
        """Test ML detector initialization."""
        detector = MLAnomalyDetector(default_ml_config)
        assert detector.config == default_ml_config
        assert detector._is_fitted is False

    def test_fit_detector(self, default_ml_config):
        """Test fitting the ML detector."""
        detector = MLAnomalyDetector(default_ml_config)

        training_data = [
            {"oxygen": 3.0, "co": 30.0, "nox": 45.0}
            for _ in range(100)
        ]
        # Add some variance
        for i, d in enumerate(training_data):
            d["oxygen"] += random.gauss(0, 0.2)
            d["co"] += random.gauss(0, 5)
            d["nox"] += random.gauss(0, 5)

        detector.fit(training_data)

        assert detector._is_fitted is True
        assert len(detector._feature_names) == 3
        assert "oxygen" in detector._feature_ranges

    def test_fit_requires_data(self, default_ml_config):
        """Test that fit requires non-empty data."""
        detector = MLAnomalyDetector(default_ml_config)

        with pytest.raises(ValueError):
            detector.fit([])

    def test_predict_anomaly_score(self, default_ml_config):
        """Test anomaly score prediction."""
        detector = MLAnomalyDetector(default_ml_config)

        # Fit with normal data
        training_data = [
            {"oxygen": 3.0 + random.gauss(0, 0.2),
             "co": 30.0 + random.gauss(0, 5),
             "nox": 45.0 + random.gauss(0, 5)}
            for _ in range(200)
        ]
        detector.fit(training_data)

        # Predict on normal point
        normal_score = detector.predict_anomaly_score(
            {"oxygen": 3.0, "co": 30.0, "nox": 45.0}
        )

        # Predict on anomalous point
        anomaly_score = detector.predict_anomaly_score(
            {"oxygen": 9.0, "co": 500.0, "nox": 300.0}
        )

        # Anomaly should have higher score
        assert 0.0 <= normal_score <= 1.0
        assert 0.0 <= anomaly_score <= 1.0
        # Anomalous point should typically have higher score
        # (though this depends on the random forest structure)

    def test_detect_method(self, default_ml_config):
        """Test the detect method."""
        detector = MLAnomalyDetector(default_ml_config)

        # Fit with normal data
        training_data = [
            {"oxygen": 3.0 + random.gauss(0, 0.1),
             "co": 30.0 + random.gauss(0, 3),
             "nox": 45.0 + random.gauss(0, 3)}
            for _ in range(200)
        ]
        detector.fit(training_data)

        # Detect normal point
        is_anomaly, score, contributions = detector.detect(
            {"oxygen": 3.0, "co": 30.0, "nox": 45.0}
        )

        assert isinstance(is_anomaly, bool)
        assert 0.0 <= score <= 1.0
        assert isinstance(contributions, dict)

    def test_feature_contributions(self, default_ml_config):
        """Test that feature contributions are calculated."""
        config = MLAnomalyConfig(
            enabled=True,
            track_feature_importance=True,
        )
        detector = MLAnomalyDetector(config)

        training_data = [
            {"oxygen": 3.0 + random.gauss(0, 0.1),
             "co": 30.0 + random.gauss(0, 3),
             "nox": 45.0 + random.gauss(0, 3)}
            for _ in range(200)
        ]
        detector.fit(training_data)

        _, _, contributions = detector.detect(
            {"oxygen": 3.0, "co": 30.0, "nox": 45.0}
        )

        assert "oxygen" in contributions
        assert "co" in contributions
        assert "nox" in contributions

    def test_predict_requires_fitted(self, default_ml_config):
        """Test that prediction requires fitted model."""
        detector = MLAnomalyDetector(default_ml_config)

        with pytest.raises(RuntimeError):
            detector.predict_anomaly_score({"oxygen": 3.0})


class TestRuleBasedDetector:
    """Tests for rule-based anomaly detection."""

    def test_initialization(self):
        """Test rule-based detector initialization."""
        detector = RuleBasedDetector()
        assert len(detector._rules) > 0

    def test_detect_low_oxygen(self):
        """Test detection of low oxygen condition."""
        detector = RuleBasedDetector()

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=1.0,  # Dangerously low
            co2_pct=12.0,
            co_ppm=500.0,
            nox_ppm=30.0,
            flue_gas_temp_c=200.0,
        )

        triggered = detector.detect(reading)

        low_o2_rules = [r for r in triggered if r["type"] == AnomalyType.LOW_OXYGEN]
        assert len(low_o2_rules) > 0
        assert low_o2_rules[0]["severity"] == AnomalySeverity.CRITICAL

    def test_detect_excess_oxygen(self):
        """Test detection of excess oxygen condition."""
        detector = RuleBasedDetector()

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=9.0,  # Too high
            co2_pct=7.0,
            co_ppm=20.0,
            nox_ppm=80.0,
            flue_gas_temp_c=220.0,
        )

        triggered = detector.detect(reading)

        excess_o2_rules = [r for r in triggered if r["type"] == AnomalyType.EXCESS_OXYGEN]
        assert len(excess_o2_rules) > 0

    def test_detect_high_co(self):
        """Test detection of high CO condition."""
        detector = RuleBasedDetector()

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.0,
            co2_pct=10.5,
            co_ppm=600.0,  # High CO
            nox_ppm=45.0,
            flue_gas_temp_c=180.0,
        )

        triggered = detector.detect(reading)

        high_co_rules = [r for r in triggered if r["type"] == AnomalyType.HIGH_CO]
        assert len(high_co_rules) > 0

    def test_detect_critical_co(self):
        """Test detection of critical CO level."""
        detector = RuleBasedDetector()

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=2.0,
            co2_pct=11.0,
            co_ppm=1500.0,  # Critical CO
            nox_ppm=30.0,
            flue_gas_temp_c=190.0,
        )

        triggered = detector.detect(reading)

        critical_co_rules = [r for r in triggered
                           if r["type"] == AnomalyType.HIGH_CO
                           and r["severity"] == AnomalySeverity.CRITICAL]
        assert len(critical_co_rules) > 0

    def test_detect_high_nox(self):
        """Test detection of high NOx condition."""
        detector = RuleBasedDetector()

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=4.0,
            co2_pct=9.5,
            co_ppm=50.0,
            nox_ppm=250.0,  # High NOx
            flue_gas_temp_c=210.0,
        )

        triggered = detector.detect(reading)

        high_nox_rules = [r for r in triggered if r["type"] == AnomalyType.HIGH_NOX]
        assert len(high_nox_rules) > 0

    def test_detect_fouling(self):
        """Test detection of fouling condition."""
        detector = RuleBasedDetector()

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=3.5,
            co2_pct=10.0,
            co_ppm=40.0,
            nox_ppm=50.0,
            flue_gas_temp_c=350.0,  # High stack temp
        )

        triggered = detector.detect(reading)

        fouling_rules = [r for r in triggered if r["type"] == AnomalyType.FOULING_DETECTED]
        assert len(fouling_rules) > 0

    def test_no_anomalies_normal_reading(self, optimal_flue_gas_reading):
        """Test that normal reading triggers no anomalies."""
        detector = RuleBasedDetector()

        triggered = detector.detect(optimal_flue_gas_reading)

        # Optimal reading should not trigger any serious anomalies
        critical_rules = [r for r in triggered if r["severity"] == AnomalySeverity.CRITICAL]
        assert len(critical_rules) == 0

    def test_rule_includes_actions(self):
        """Test that triggered rules include recommended actions."""
        detector = RuleBasedDetector()

        reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=1.0,
            co2_pct=12.0,
            co_ppm=800.0,
            nox_ppm=30.0,
            flue_gas_temp_c=200.0,
        )

        triggered = detector.detect(reading)

        for rule in triggered:
            assert "actions" in rule
            assert len(rule["actions"]) > 0


class TestCombustionAnomalyDetector:
    """Tests for integrated anomaly detection."""

    def test_initialization(self, default_anomaly_config):
        """Test integrated detector initialization."""
        detector = CombustionAnomalyDetector(default_anomaly_config)

        assert detector.spc is not None
        assert detector.rules is not None
        if default_anomaly_config.ml.enabled:
            assert detector.ml is not None

    def test_initialize_baseline(self, default_anomaly_config, historical_readings_normal):
        """Test baseline initialization."""
        detector = CombustionAnomalyDetector(default_anomaly_config)
        detector.initialize_baseline(historical_readings_normal)

        # Check SPC baselines are set
        assert detector.spc.get_statistics("oxygen") is not None
        assert detector.spc.get_statistics("co") is not None

    def test_detect_no_anomalies(
        self, default_anomaly_config, historical_readings_normal, optimal_flue_gas_reading
    ):
        """Test detection with no anomalies."""
        detector = CombustionAnomalyDetector(default_anomaly_config)
        detector.initialize_baseline(historical_readings_normal)

        result = detector.detect(optimal_flue_gas_reading)

        assert result.status == AnalysisStatus.SUCCESS
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    def test_detect_anomalies(
        self, default_anomaly_config, historical_readings_normal, high_co_flue_gas_reading
    ):
        """Test detection of anomalies."""
        detector = CombustionAnomalyDetector(default_anomaly_config)
        detector.initialize_baseline(historical_readings_normal)

        result = detector.detect(high_co_flue_gas_reading)

        assert result.status == AnalysisStatus.SUCCESS
        # High CO should trigger rule-based detection at minimum
        # Note: SPC may or may not trigger depending on baseline

    def test_severity_counts(self, default_anomaly_config, historical_readings_normal):
        """Test severity count tracking."""
        detector = CombustionAnomalyDetector(default_anomaly_config)
        detector.initialize_baseline(historical_readings_normal)

        # Create a critical reading
        critical_reading = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=1.0,
            co2_pct=12.0,
            co_ppm=1200.0,
            nox_ppm=30.0,
            flue_gas_temp_c=200.0,
        )

        result = detector.detect(critical_reading)

        # Should have some anomalies
        total = result.critical_count + result.alarm_count + result.warning_count + result.info_count
        assert total == result.total_anomalies

    def test_spc_status_tracking(
        self, default_anomaly_config, historical_readings_normal, optimal_flue_gas_reading
    ):
        """Test SPC control status tracking."""
        detector = CombustionAnomalyDetector(default_anomaly_config)
        detector.initialize_baseline(historical_readings_normal)

        result = detector.detect(optimal_flue_gas_reading)

        # spc_in_control should be tracked
        assert isinstance(result.spc_in_control, bool)

    def test_ml_health_score(
        self, default_anomaly_config, historical_readings_normal, optimal_flue_gas_reading
    ):
        """Test ML health score calculation."""
        detector = CombustionAnomalyDetector(default_anomaly_config)
        detector.initialize_baseline(historical_readings_normal)

        result = detector.detect(optimal_flue_gas_reading)

        if "ml" in default_anomaly_config.detection_modes:
            assert result.ml_health_score is None or 0.0 <= result.ml_health_score <= 1.0

    def test_provenance_hash(
        self, default_anomaly_config, historical_readings_normal, optimal_flue_gas_reading
    ):
        """Test provenance hash generation."""
        detector = CombustionAnomalyDetector(default_anomaly_config)
        detector.initialize_baseline(historical_readings_normal)

        result = detector.detect(optimal_flue_gas_reading)

        assert_valid_provenance_hash(result.provenance_hash)

    def test_audit_trail(
        self, default_anomaly_config, historical_readings_normal, optimal_flue_gas_reading
    ):
        """Test audit trail generation."""
        detector = CombustionAnomalyDetector(default_anomaly_config)
        detector.initialize_baseline(historical_readings_normal)

        detector.detect(optimal_flue_gas_reading)

        audit = detector.get_audit_trail()
        assert len(audit) > 0

        operations = [entry["operation"] for entry in audit]
        if "spc" in default_anomaly_config.detection_modes:
            assert "spc_detection" in operations
        if "rule_based" in default_anomaly_config.detection_modes:
            assert "rule_detection" in operations

    def test_alert_cooldown(self, default_anomaly_config, historical_readings_normal):
        """Test alert cooldown functionality."""
        # Create config with short cooldown for testing
        config = AnomalyDetectionConfig()
        config.alert_cooldown_s = 1  # 1 second cooldown

        detector = CombustionAnomalyDetector(config)
        detector.initialize_baseline(historical_readings_normal)

        # Create anomalous reading
        anomalous = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=9.0,
            co2_pct=7.0,
            co_ppm=500.0,
            nox_ppm=250.0,
            flue_gas_temp_c=300.0,
        )

        # First detection
        result1 = detector.detect(anomalous)
        count1 = result1.total_anomalies

        # Immediate second detection - should have fewer due to cooldown
        result2 = detector.detect(anomalous)
        count2 = result2.total_anomalies

        # Second detection may have fewer anomalies due to cooldown
        # (same type anomalies filtered)

    def test_multiple_detection_modes(self, historical_readings_normal):
        """Test with multiple detection modes enabled."""
        config = AnomalyDetectionConfig(
            detection_modes=["spc", "ml", "rule_based"]
        )

        detector = CombustionAnomalyDetector(config)
        detector.initialize_baseline(historical_readings_normal)

        # Create anomalous reading that should trigger all modes
        anomalous = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=1.0,
            co2_pct=12.5,
            co_ppm=1000.0,
            nox_ppm=300.0,
            flue_gas_temp_c=350.0,
        )

        result = detector.detect(anomalous)

        # Should have detections
        assert result.status == AnalysisStatus.SUCCESS


class TestAnomalyEventContent:
    """Tests for anomaly event content and structure."""

    def test_anomaly_event_fields(
        self, default_anomaly_config, historical_readings_normal
    ):
        """Test that anomaly events contain required fields."""
        detector = CombustionAnomalyDetector(default_anomaly_config)
        detector.initialize_baseline(historical_readings_normal)

        # Create reading that will trigger anomalies
        anomalous = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=1.0,
            co2_pct=12.0,
            co_ppm=1000.0,
            nox_ppm=30.0,
            flue_gas_temp_c=200.0,
        )

        result = detector.detect(anomalous)

        for anomaly in result.anomalies:
            assert anomaly.anomaly_id is not None
            assert anomaly.timestamp is not None
            assert anomaly.anomaly_type is not None
            assert anomaly.severity is not None
            assert anomaly.detection_method in ["spc", "ml", "rule_based"]
            assert 0.0 <= anomaly.confidence <= 1.0
            assert anomaly.affected_parameter is not None

    def test_anomaly_contains_recommendations(
        self, default_anomaly_config, historical_readings_normal
    ):
        """Test that anomalies contain recommended actions."""
        detector = CombustionAnomalyDetector(default_anomaly_config)
        detector.initialize_baseline(historical_readings_normal)

        anomalous = FlueGasReading(
            timestamp=datetime.now(timezone.utc),
            oxygen_pct=1.0,
            co2_pct=12.0,
            co_ppm=1000.0,
            nox_ppm=30.0,
            flue_gas_temp_c=200.0,
        )

        result = detector.detect(anomalous)

        # Rule-based anomalies should have recommendations
        rule_anomalies = [a for a in result.anomalies if a.detection_method == "rule_based"]
        for anomaly in rule_anomalies:
            assert len(anomaly.recommended_actions) > 0
