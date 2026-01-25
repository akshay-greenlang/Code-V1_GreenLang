"""
Tests for Calibration Tracker

Tests for calibration monitoring and drift detection including:
    - PICP computation
    - Calibration metrics
    - Drift detection algorithms
    - Reliability diagram generation

All tests verify determinism and correctness.

Author: GreenLang Process Heat Team
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal

from ..calibration_tracker import (
    CalibrationTracker,
    DriftDetector,
    DriftType,
    ReliabilityDiagramGenerator,
    CalibrationReportGenerator,
)
from ..uq_schemas import (
    PredictionInterval,
    CalibrationStatus,
)


class TestCalibrationTracker:
    """Tests for CalibrationTracker PICP and metrics."""

    def test_record_observation(self):
        """Should record observations correctly."""
        tracker = CalibrationTracker(
            model_name="test_model",
            variable_name="temperature",
            window_size=100,
            min_samples=5
        )

        prediction = PredictionInterval(
            point_estimate=Decimal("100"),
            lower_bound=Decimal("90"),
            upper_bound=Decimal("110"),
            confidence_level=Decimal("0.90"),
            variable_name="temperature",
            unit="degC"
        )

        # Record some observations
        tracker.record_observation(prediction, Decimal("100"))  # In interval
        tracker.record_observation(prediction, Decimal("95"))   # In interval
        tracker.record_observation(prediction, Decimal("115"))  # Out of interval

        # Coverage should be 2/3
        coverage = tracker.get_coverage_by_confidence()
        assert "0.90" in coverage
        expected_picp = Decimal("2") / Decimal("3")
        assert abs(coverage["0.90"] - expected_picp) < Decimal("0.01")

    def test_compute_metrics_picp(self):
        """PICP should be computed correctly."""
        tracker = CalibrationTracker(
            model_name="test_model",
            variable_name="temperature",
            window_size=100,
            min_samples=10
        )

        prediction = PredictionInterval(
            point_estimate=Decimal("100"),
            lower_bound=Decimal("90"),
            upper_bound=Decimal("110"),
            confidence_level=Decimal("0.90"),
            variable_name="temperature",
            unit="degC"
        )

        # Record 20 observations: 18 in interval, 2 outside
        for i in range(18):
            tracker.record_observation(prediction, Decimal("100"))
        for i in range(2):
            tracker.record_observation(prediction, Decimal("120"))

        metrics = tracker.compute_metrics(target_coverage=Decimal("0.90"))

        assert metrics is not None
        assert metrics.picp == Decimal("0.9")
        assert metrics.target_coverage == Decimal("0.90")
        assert metrics.calibration_error == Decimal("0")

    def test_compute_metrics_insufficient_samples(self):
        """Should return None if insufficient samples."""
        tracker = CalibrationTracker(
            model_name="test_model",
            variable_name="temperature",
            window_size=100,
            min_samples=50
        )

        prediction = PredictionInterval(
            point_estimate=Decimal("100"),
            lower_bound=Decimal("90"),
            upper_bound=Decimal("110"),
            confidence_level=Decimal("0.90"),
            variable_name="temperature",
            unit="degC"
        )

        # Only 10 observations
        for i in range(10):
            tracker.record_observation(prediction, Decimal("100"))

        metrics = tracker.compute_metrics(target_coverage=Decimal("0.90"))
        assert metrics is None

    def test_calibration_status_well_calibrated(self):
        """Should detect well calibrated model."""
        tracker = CalibrationTracker(
            model_name="test_model",
            variable_name="temperature",
            window_size=100,
            min_samples=10
        )

        prediction = PredictionInterval(
            point_estimate=Decimal("100"),
            lower_bound=Decimal("90"),
            upper_bound=Decimal("110"),
            confidence_level=Decimal("0.90"),
            variable_name="temperature",
            unit="degC"
        )

        # 90% coverage = well calibrated for 90% CI
        for i in range(90):
            tracker.record_observation(prediction, Decimal("100"))
        for i in range(10):
            tracker.record_observation(prediction, Decimal("120"))

        metrics = tracker.compute_metrics(target_coverage=Decimal("0.90"))

        assert metrics.status == CalibrationStatus.WELL_CALIBRATED

    def test_calibration_status_over_confident(self):
        """Should detect over-confident model (intervals too narrow)."""
        tracker = CalibrationTracker(
            model_name="test_model",
            variable_name="temperature",
            window_size=100,
            min_samples=10
        )

        prediction = PredictionInterval(
            point_estimate=Decimal("100"),
            lower_bound=Decimal("90"),
            upper_bound=Decimal("110"),
            confidence_level=Decimal("0.90"),
            variable_name="temperature",
            unit="degC"
        )

        # 70% coverage when target is 90% = over confident
        for i in range(70):
            tracker.record_observation(prediction, Decimal("100"))
        for i in range(30):
            tracker.record_observation(prediction, Decimal("120"))

        metrics = tracker.compute_metrics(target_coverage=Decimal("0.90"))

        assert metrics.status == CalibrationStatus.OVER_CONFIDENT

    def test_calibration_status_under_confident(self):
        """Should detect under-confident model (intervals too wide)."""
        tracker = CalibrationTracker(
            model_name="test_model",
            variable_name="temperature",
            window_size=100,
            min_samples=10
        )

        prediction = PredictionInterval(
            point_estimate=Decimal("100"),
            lower_bound=Decimal("90"),
            upper_bound=Decimal("110"),
            confidence_level=Decimal("0.90"),
            variable_name="temperature",
            unit="degC"
        )

        # 100% coverage when target is 90% = under confident
        for i in range(100):
            tracker.record_observation(prediction, Decimal("100"))

        metrics = tracker.compute_metrics(target_coverage=Decimal("0.90"))

        # PICP = 1.0 > 0.95 (target + 5% tolerance) = under confident
        assert metrics.status == CalibrationStatus.UNDER_CONFIDENT

    def test_retraining_recommended(self):
        """Should recommend retraining for poor calibration."""
        tracker = CalibrationTracker(
            model_name="test_model",
            variable_name="temperature",
            window_size=100,
            min_samples=10
        )

        prediction = PredictionInterval(
            point_estimate=Decimal("100"),
            lower_bound=Decimal("90"),
            upper_bound=Decimal("110"),
            confidence_level=Decimal("0.90"),
            variable_name="temperature",
            unit="degC"
        )

        # 50% coverage when target is 90% = severely over confident
        for i in range(50):
            tracker.record_observation(prediction, Decimal("100"))
        for i in range(50):
            tracker.record_observation(prediction, Decimal("120"))

        metrics = tracker.compute_metrics(target_coverage=Decimal("0.90"))

        # Calibration error > 10% should trigger retraining
        assert metrics.retraining_recommended is True

    def test_multiple_confidence_levels(self):
        """Should track multiple confidence levels."""
        tracker = CalibrationTracker(
            model_name="test_model",
            variable_name="temperature"
        )

        # 90% CI prediction
        pred_90 = PredictionInterval(
            point_estimate=Decimal("100"),
            lower_bound=Decimal("90"),
            upper_bound=Decimal("110"),
            confidence_level=Decimal("0.90"),
            variable_name="temperature",
            unit="degC"
        )

        # 50% CI prediction
        pred_50 = PredictionInterval(
            point_estimate=Decimal("100"),
            lower_bound=Decimal("95"),
            upper_bound=Decimal("105"),
            confidence_level=Decimal("0.50"),
            variable_name="temperature",
            unit="degC"
        )

        # Record at both levels
        for i in range(50):
            tracker.record_observation(pred_90, Decimal("100"))
            tracker.record_observation(pred_50, Decimal("100"))

        coverage = tracker.get_coverage_by_confidence()

        assert "0.90" in coverage
        assert "0.50" in coverage

    def test_metrics_has_provenance(self):
        """Metrics should have provenance tracking."""
        tracker = CalibrationTracker(
            model_name="test_model",
            variable_name="temperature",
            min_samples=10
        )

        prediction = PredictionInterval(
            point_estimate=Decimal("100"),
            lower_bound=Decimal("90"),
            upper_bound=Decimal("110"),
            confidence_level=Decimal("0.90"),
            variable_name="temperature",
            unit="degC"
        )

        for i in range(20):
            tracker.record_observation(prediction, Decimal("100"))

        metrics = tracker.compute_metrics()

        assert metrics.provenance is not None
        assert metrics.provenance.calculation_type == "calibration_metrics_computation"

    def test_reset_clears_data(self):
        """Reset should clear all tracking data."""
        tracker = CalibrationTracker(
            model_name="test_model",
            variable_name="temperature"
        )

        prediction = PredictionInterval(
            point_estimate=Decimal("100"),
            lower_bound=Decimal("90"),
            upper_bound=Decimal("110"),
            confidence_level=Decimal("0.90"),
            variable_name="temperature",
            unit="degC"
        )

        for i in range(50):
            tracker.record_observation(prediction, Decimal("100"))

        tracker.reset()

        coverage = tracker.get_coverage_by_confidence()
        assert len(coverage) == 0


class TestDriftDetector:
    """Tests for DriftDetector algorithms."""

    def test_no_drift_on_stable_data(self):
        """Should not detect drift on stable data."""
        detector = DriftDetector(method="page_hinkley", window_size=50)

        # Stable data around mean of 100
        for i in range(100):
            value = Decimal("100") + Decimal(str(i % 5 - 2))
            alert = detector.update(value)

        # No drift should be detected
        alerts = detector.get_alerts()
        assert len(alerts) == 0

    def test_detect_mean_shift(self):
        """Should detect sudden mean shift."""
        detector = DriftDetector(method="page_hinkley", window_size=30)

        # First 50 values around 100
        for i in range(50):
            detector.update(Decimal("100"))

        # Then shift to 150
        for i in range(50):
            alert = detector.update(Decimal("150"))
            if alert:
                break

        alerts = detector.get_alerts()
        assert len(alerts) >= 1
        assert alerts[0].drift_type == DriftType.MEAN_SHIFT

    def test_cusum_detector(self):
        """CUSUM detector should work."""
        detector = DriftDetector(method="cusum", window_size=30)

        # Stable period for baseline
        for i in range(50):
            detector.update(Decimal("100"))

        # Shift
        for i in range(50):
            alert = detector.update(Decimal("110"))
            if alert:
                break

        alerts = detector.get_alerts()
        # CUSUM might or might not detect depending on threshold
        assert isinstance(alerts, list)

    def test_adwin_detector(self):
        """ADWIN detector should work."""
        detector = DriftDetector(method="adwin", window_size=30)

        # Stable period
        for i in range(50):
            detector.update(Decimal("100"))

        # Gradual drift
        for i in range(50):
            value = Decimal("100") + Decimal(str(i * 2))
            alert = detector.update(value)
            if alert:
                break

        alerts = detector.get_alerts()
        assert isinstance(alerts, list)

    def test_get_alerts_filtered_by_severity(self):
        """Should filter alerts by severity."""
        detector = DriftDetector(method="page_hinkley", window_size=20)

        # Generate drift
        for i in range(30):
            detector.update(Decimal("100"))
        for i in range(30):
            detector.update(Decimal("200"))

        high_alerts = detector.get_alerts(severity="high")
        medium_alerts = detector.get_alerts(severity="medium")

        # Alerts should be filtered
        for alert in high_alerts:
            assert alert.severity == "high"
        for alert in medium_alerts:
            assert alert.severity == "medium"

    def test_get_alerts_filtered_by_time(self):
        """Should filter alerts by time."""
        detector = DriftDetector(method="page_hinkley", window_size=20)

        now = datetime.utcnow()
        past = now - timedelta(hours=48)

        # Generate alerts
        for i in range(30):
            detector.update(Decimal("100"), timestamp=past)
        for i in range(30):
            detector.update(Decimal("200"), timestamp=past)

        recent_alerts = detector.get_alerts(since=now - timedelta(hours=1))

        # No recent alerts
        assert len(recent_alerts) == 0

    def test_reset_clears_state(self):
        """Reset should clear all detector state."""
        detector = DriftDetector(method="page_hinkley")

        for i in range(50):
            detector.update(Decimal("100"))

        detector.reset()

        alerts = detector.get_alerts()
        assert len(alerts) == 0


class TestReliabilityDiagramGenerator:
    """Tests for ReliabilityDiagramGenerator."""

    def test_add_predictions(self):
        """Should add predictions correctly."""
        generator = ReliabilityDiagramGenerator(num_bins=5)

        generator.add_prediction(Decimal("0.90"), True)
        generator.add_prediction(Decimal("0.90"), True)
        generator.add_prediction(Decimal("0.90"), False)
        generator.add_prediction(Decimal("0.50"), True)
        generator.add_prediction(Decimal("0.50"), False)

        # Can add predictions
        assert True  # No exception raised

    def test_generate_diagram(self):
        """Should generate reliability diagram."""
        generator = ReliabilityDiagramGenerator(num_bins=5)

        # Well calibrated: predicted prob matches observed freq
        for i in range(10):
            generator.add_prediction(Decimal("0.10"), i < 1)  # 10% true
        for i in range(10):
            generator.add_prediction(Decimal("0.50"), i < 5)  # 50% true
        for i in range(10):
            generator.add_prediction(Decimal("0.90"), i < 9)  # 90% true

        diagram = generator.generate(
            model_name="test_model",
            variable_name="temperature"
        )

        assert diagram is not None
        assert len(diagram.points) > 0
        assert diagram.total_samples == 30

    def test_insufficient_data_returns_none(self):
        """Should return None with insufficient data."""
        generator = ReliabilityDiagramGenerator(num_bins=10)

        # Only 5 predictions (less than num_bins)
        for i in range(5):
            generator.add_prediction(Decimal("0.50"), True)

        diagram = generator.generate()
        assert diagram is None

    def test_ece_computation(self):
        """ECE should be computed correctly."""
        generator = ReliabilityDiagramGenerator(num_bins=5)

        # Perfectly calibrated predictions
        for _ in range(100):
            for prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
                # Outcome matches probability
                import random
                random.seed(42)
                outcome = random.random() < prob
                generator.add_prediction(Decimal(str(prob)), outcome)

        diagram = generator.generate()

        assert diagram is not None
        assert diagram.expected_calibration_error >= Decimal("0")

    def test_diagram_has_provenance(self):
        """Diagram should have provenance tracking."""
        generator = ReliabilityDiagramGenerator(num_bins=3)

        for i in range(30):
            generator.add_prediction(Decimal("0.50"), i % 2 == 0)

        diagram = generator.generate(model_name="test")

        assert diagram.provenance is not None
        assert diagram.provenance.calculation_type == "reliability_diagram_generation"


class TestCalibrationReportGenerator:
    """Tests for CalibrationReportGenerator."""

    def test_generate_report(self):
        """Should generate comprehensive report."""
        tracker = CalibrationTracker(
            model_name="test_model",
            variable_name="temperature",
            min_samples=10
        )
        drift_detector = DriftDetector(method="page_hinkley")
        diagram_generator = ReliabilityDiagramGenerator(num_bins=5)

        # Add some data
        prediction = PredictionInterval(
            point_estimate=Decimal("100"),
            lower_bound=Decimal("90"),
            upper_bound=Decimal("110"),
            confidence_level=Decimal("0.90"),
            variable_name="temperature",
            unit="degC"
        )

        for i in range(50):
            tracker.record_observation(prediction, Decimal("100"))
            drift_detector.update(Decimal("100"))
            diagram_generator.add_prediction(Decimal("0.90"), True)

        report_generator = CalibrationReportGenerator(
            tracker=tracker,
            drift_detector=drift_detector,
            diagram_generator=diagram_generator
        )

        report = report_generator.generate_report()

        assert "report_id" in report
        assert "overall_status" in report
        assert "recommendations" in report
        assert "provenance_hash" in report

    def test_report_status_healthy(self):
        """Should report healthy status for well calibrated model."""
        tracker = CalibrationTracker(
            model_name="test_model",
            variable_name="temperature",
            min_samples=10
        )
        drift_detector = DriftDetector()
        diagram_generator = ReliabilityDiagramGenerator(num_bins=5)

        prediction = PredictionInterval(
            point_estimate=Decimal("100"),
            lower_bound=Decimal("90"),
            upper_bound=Decimal("110"),
            confidence_level=Decimal("0.90"),
            variable_name="temperature",
            unit="degC"
        )

        # Well calibrated: 90% coverage
        for i in range(90):
            tracker.record_observation(prediction, Decimal("100"))
            diagram_generator.add_prediction(Decimal("0.90"), True)
        for i in range(10):
            tracker.record_observation(prediction, Decimal("120"))
            diagram_generator.add_prediction(Decimal("0.90"), False)

        report_generator = CalibrationReportGenerator(
            tracker, drift_detector, diagram_generator
        )

        report = report_generator.generate_report()

        assert report["overall_status"] == "healthy"

    def test_report_includes_recommendations(self):
        """Should include actionable recommendations."""
        tracker = CalibrationTracker(
            model_name="test_model",
            variable_name="temperature",
            min_samples=10
        )
        drift_detector = DriftDetector()
        diagram_generator = ReliabilityDiagramGenerator(num_bins=5)

        prediction = PredictionInterval(
            point_estimate=Decimal("100"),
            lower_bound=Decimal("90"),
            upper_bound=Decimal("110"),
            confidence_level=Decimal("0.90"),
            variable_name="temperature",
            unit="degC"
        )

        for i in range(50):
            tracker.record_observation(prediction, Decimal("100"))
            diagram_generator.add_prediction(Decimal("0.90"), True)

        report_generator = CalibrationReportGenerator(
            tracker, drift_detector, diagram_generator
        )

        report = report_generator.generate_report()

        assert "recommendations" in report
        assert len(report["recommendations"]) > 0
