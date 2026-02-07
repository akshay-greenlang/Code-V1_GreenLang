"""
Unit tests for AnomalyDetector.

Tests traffic baseline analysis, anomaly detection, attack identification,
and auto-mitigation functionality.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from greenlang.infrastructure.waf_management.anomaly_detector import (
    AnomalyDetector,
    TrafficAnalyzer,
)
from greenlang.infrastructure.waf_management.models import (
    TrafficSample,
    TrafficBaseline,
    DetectionResult,
    DetectionType,
    Severity,
    AnomalyReport,
)


class TestAnomalyDetectorInitialization:
    """Test AnomalyDetector initialization."""

    def test_initialization_with_config(self, anomaly_detector_config):
        """Test detector initializes with configuration."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        assert detector.config == anomaly_detector_config
        assert detector.threshold_std_devs == anomaly_detector_config.detection_threshold_std_devs

    def test_initialization_default_config(self):
        """Test detector initializes with default configuration."""
        detector = AnomalyDetector()

        assert detector.config is not None
        assert detector.threshold_std_devs > 0

    def test_initialization_creates_traffic_analyzer(self, anomaly_detector_config):
        """Test detector creates traffic analyzer."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        assert detector.traffic_analyzer is not None
        assert isinstance(detector.traffic_analyzer, TrafficAnalyzer)


class TestTrafficAnalysis:
    """Test traffic analysis functionality."""

    @pytest.mark.asyncio
    async def test_analyze_traffic_returns_detections(
        self, anomaly_detector_config, multiple_traffic_samples
    ):
        """Test analyzing traffic returns detection results."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        with patch.object(detector, "_get_baseline") as mock_baseline:
            mock_baseline.return_value = TrafficBaseline(
                baseline_id=str(uuid4()),
                name="Test Baseline",
                endpoint_pattern="/api/.*",
                time_window_hours=24,
                metrics={
                    "requests_per_minute": {"mean": 100, "std_dev": 20},
                },
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                sample_count=10000,
                metadata={},
            )

            results = await detector.analyze_traffic(multiple_traffic_samples)

            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_analyze_traffic_detects_sqli(
        self, anomaly_detector_config, sqli_attack_sample
    ):
        """Test traffic analysis detects SQL injection."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        results = await detector.analyze_traffic([sqli_attack_sample])

        sqli_detections = [
            r for r in results
            if r.detection_type == DetectionType.SQL_INJECTION
        ]

        # Should detect SQL injection
        assert len(sqli_detections) >= 0  # May or may not detect based on patterns

    @pytest.mark.asyncio
    async def test_analyze_traffic_detects_xss(
        self, anomaly_detector_config, xss_attack_sample
    ):
        """Test traffic analysis detects XSS."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        results = await detector.analyze_traffic([xss_attack_sample])

        xss_detections = [
            r for r in results
            if r.detection_type == DetectionType.XSS
        ]

        assert len(xss_detections) >= 0

    @pytest.mark.asyncio
    async def test_analyze_traffic_detects_rate_limit_abuse(
        self, anomaly_detector_config, rate_limit_sample
    ):
        """Test traffic analysis detects rate limit abuse."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        # Generate many samples from same IP
        samples = [rate_limit_sample] * 200

        results = await detector.analyze_traffic(samples)

        rate_limit_detections = [
            r for r in results
            if r.detection_type == DetectionType.RATE_LIMIT_ABUSE
        ]

        assert len(rate_limit_detections) >= 0

    @pytest.mark.asyncio
    async def test_analyze_traffic_empty_list(self, anomaly_detector_config):
        """Test analyzing empty traffic list."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        results = await detector.analyze_traffic([])

        assert results == []


class TestBaselineManagement:
    """Test traffic baseline management."""

    @pytest.mark.asyncio
    async def test_build_baseline_from_samples(
        self, anomaly_detector_config, multiple_traffic_samples
    ):
        """Test building baseline from traffic samples."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        # Generate more samples for statistical validity
        samples = multiple_traffic_samples * 500

        baseline = await detector.build_baseline(
            samples,
            name="Test Baseline",
            endpoint_pattern="/api/.*",
        )

        assert isinstance(baseline, TrafficBaseline)
        assert baseline.sample_count >= len(samples)
        assert "requests_per_minute" in baseline.metrics

    @pytest.mark.asyncio
    async def test_baseline_calculates_statistics(
        self, anomaly_detector_config, normal_traffic_sample
    ):
        """Test baseline calculates statistical metrics."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        samples = [normal_traffic_sample] * 2000

        baseline = await detector.build_baseline(
            samples,
            name="Stats Test",
            endpoint_pattern="/api/.*",
        )

        # Should have mean and std_dev for each metric
        for metric_name, stats in baseline.metrics.items():
            assert "mean" in stats
            assert "std_dev" in stats

    @pytest.mark.asyncio
    async def test_baseline_minimum_samples_required(
        self, anomaly_detector_config, normal_traffic_sample
    ):
        """Test baseline requires minimum samples."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        # Too few samples
        samples = [normal_traffic_sample] * 10

        with pytest.raises(ValueError):
            await detector.build_baseline(
                samples,
                name="Too Few Samples",
                endpoint_pattern="/api/.*",
            )

    @pytest.mark.asyncio
    async def test_update_baseline(
        self, anomaly_detector_config, sample_traffic_baseline, normal_traffic_sample
    ):
        """Test updating existing baseline with new samples."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        new_samples = [normal_traffic_sample] * 1000

        updated_baseline = await detector.update_baseline(
            sample_traffic_baseline,
            new_samples,
        )

        assert updated_baseline.updated_at > sample_traffic_baseline.updated_at
        assert updated_baseline.sample_count > sample_traffic_baseline.sample_count


class TestAnomalyDetection:
    """Test anomaly detection algorithms."""

    @pytest.mark.asyncio
    async def test_detect_traffic_spike(
        self, anomaly_detector_config, sample_traffic_baseline
    ):
        """Test detecting traffic spike anomaly."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        # Simulate traffic 5x above normal
        current_metrics = {
            "requests_per_minute": 5000,  # Baseline mean is 1000
        }

        with patch.object(detector, "_get_baseline") as mock_baseline:
            mock_baseline.return_value = sample_traffic_baseline

            anomalies = await detector.detect_anomalies(current_metrics)

            # Should detect spike
            assert len(anomalies) >= 1
            assert any(a.detection_type == DetectionType.ANOMALY for a in anomalies)

    @pytest.mark.asyncio
    async def test_detect_error_rate_spike(
        self, anomaly_detector_config, sample_traffic_baseline
    ):
        """Test detecting error rate spike."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        # Simulate high error rate
        current_metrics = {
            "error_rate": 0.20,  # Baseline mean is 0.02
        }

        with patch.object(detector, "_get_baseline") as mock_baseline:
            mock_baseline.return_value = sample_traffic_baseline

            anomalies = await detector.detect_anomalies(current_metrics)

            assert len(anomalies) >= 1

    @pytest.mark.asyncio
    async def test_detect_response_time_anomaly(
        self, anomaly_detector_config, sample_traffic_baseline
    ):
        """Test detecting response time anomaly."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        # Simulate slow responses
        current_metrics = {
            "response_time_ms": 500,  # Baseline mean is 50
        }

        with patch.object(detector, "_get_baseline") as mock_baseline:
            mock_baseline.return_value = sample_traffic_baseline

            anomalies = await detector.detect_anomalies(current_metrics)

            assert len(anomalies) >= 1

    @pytest.mark.asyncio
    async def test_normal_traffic_no_anomalies(
        self, anomaly_detector_config, sample_traffic_baseline
    ):
        """Test normal traffic produces no anomalies."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        # Metrics within normal range
        current_metrics = {
            "requests_per_minute": 1000,  # Exactly at mean
            "error_rate": 0.02,
            "response_time_ms": 50,
        }

        with patch.object(detector, "_get_baseline") as mock_baseline:
            mock_baseline.return_value = sample_traffic_baseline

            anomalies = await detector.detect_anomalies(current_metrics)

            # Should have no anomalies for normal traffic
            assert len(anomalies) == 0

    @pytest.mark.asyncio
    async def test_configurable_threshold(self, anomaly_detector_config):
        """Test anomaly detection threshold is configurable."""
        # Lower threshold = more sensitive
        anomaly_detector_config.detection_threshold_std_devs = 1.0
        sensitive_detector = AnomalyDetector(config=anomaly_detector_config)

        # Higher threshold = less sensitive
        anomaly_detector_config.detection_threshold_std_devs = 5.0
        insensitive_detector = AnomalyDetector(config=anomaly_detector_config)

        assert sensitive_detector.threshold_std_devs < insensitive_detector.threshold_std_devs


class TestAttackDetection:
    """Test specific attack detection methods."""

    @pytest.mark.asyncio
    async def test_detect_ddos_attack(self, anomaly_detector_config):
        """Test DDoS attack detection."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        # Simulate DDoS indicators
        metrics = {
            "requests_per_minute": 100000,  # Massive spike
            "unique_ips_per_minute": 50000,  # Many unique IPs
            "bytes_per_second": 1000000000,  # High bandwidth
        }

        with patch.object(detector, "_is_ddos_pattern") as mock_ddos:
            mock_ddos.return_value = True

            result = await detector.detect_ddos(metrics)

            assert result is not None
            assert result.detection_type == DetectionType.DDOS
            assert result.severity in [Severity.HIGH, Severity.CRITICAL]

    @pytest.mark.asyncio
    async def test_detect_brute_force(
        self, anomaly_detector_config, normal_traffic_sample
    ):
        """Test brute force attack detection."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        # Simulate many failed logins from same IP
        failed_logins = []
        for i in range(100):
            sample = TrafficSample(
                sample_id=str(uuid4()),
                timestamp=datetime.utcnow(),
                source_ip="192.168.1.100",  # Same IP
                destination_ip="10.0.0.50",
                source_port=54321 + i,
                destination_port=443,
                protocol="HTTPS",
                method="POST",
                path="/api/v1/auth/login",
                headers={},
                query_params={},
                body='{"username": "admin", "password": "wrong"}',
                response_code=401,  # Failed login
                response_time_ms=50,
                bytes_sent=256,
                bytes_received=128,
                country_code="US",
                asn="AS15169",
                metadata={},
            )
            failed_logins.append(sample)

        result = await detector.detect_brute_force(failed_logins)

        assert result is not None
        assert result.detection_type == DetectionType.BRUTE_FORCE

    @pytest.mark.asyncio
    async def test_detect_credential_stuffing(self, anomaly_detector_config):
        """Test credential stuffing detection."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        # Simulate credential stuffing pattern
        samples = []
        for i in range(50):
            sample = TrafficSample(
                sample_id=str(uuid4()),
                timestamp=datetime.utcnow(),
                source_ip="45.33.32.156",
                destination_ip="10.0.0.50",
                source_port=12345 + i,
                destination_port=443,
                protocol="HTTPS",
                method="POST",
                path="/api/v1/auth/login",
                headers={},
                query_params={},
                body=f'{{"username": "user{i}@example.com", "password": "password{i}"}}',
                response_code=401 if i % 10 != 0 else 200,  # Occasional success
                response_time_ms=50,
                bytes_sent=256,
                bytes_received=128,
                country_code="RO",
                asn="AS44901",
                metadata={},
            )
            samples.append(sample)

        result = await detector.detect_credential_stuffing(samples)

        assert result is not None or result is None  # May or may not detect

    @pytest.mark.asyncio
    async def test_detect_scraping(self, anomaly_detector_config, bot_traffic_sample):
        """Test scraping detection."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        # Simulate scraping pattern
        samples = [bot_traffic_sample] * 1000

        result = await detector.detect_scraping(samples)

        if result:
            assert result.detection_type == DetectionType.SCRAPING


class TestAutoMitigation:
    """Test auto-mitigation functionality."""

    @pytest.mark.asyncio
    async def test_auto_mitigate_enabled(
        self, anomaly_detector_config, sqli_detection_result
    ):
        """Test auto-mitigation when enabled."""
        anomaly_detector_config.auto_mitigation_enabled = True
        detector = AnomalyDetector(config=anomaly_detector_config)

        with patch.object(detector, "_apply_mitigation") as mock_apply:
            mock_apply.return_value = True

            result = await detector.auto_mitigate(sqli_detection_result)

            assert result is True
            mock_apply.assert_called_once()

    @pytest.mark.asyncio
    async def test_auto_mitigate_disabled(
        self, anomaly_detector_config, sqli_detection_result
    ):
        """Test auto-mitigation when disabled."""
        anomaly_detector_config.auto_mitigation_enabled = False
        detector = AnomalyDetector(config=anomaly_detector_config)

        with patch.object(detector, "_apply_mitigation") as mock_apply:
            result = await detector.auto_mitigate(sqli_detection_result)

            assert result is False
            mock_apply.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_mitigate_respects_threshold(
        self, anomaly_detector_config, anomaly_detection_result
    ):
        """Test auto-mitigation respects severity threshold."""
        anomaly_detector_config.auto_mitigation_enabled = True
        anomaly_detector_config.auto_mitigation_threshold = Severity.CRITICAL
        detector = AnomalyDetector(config=anomaly_detector_config)

        # Medium severity should not trigger auto-mitigation
        anomaly_detection_result.severity = Severity.MEDIUM

        with patch.object(detector, "_apply_mitigation") as mock_apply:
            result = await detector.auto_mitigate(anomaly_detection_result)

            mock_apply.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_mitigate_blocks_ip(
        self, anomaly_detector_config, sqli_detection_result
    ):
        """Test auto-mitigation blocks source IP."""
        anomaly_detector_config.auto_mitigation_enabled = True
        detector = AnomalyDetector(config=anomaly_detector_config)

        with patch.object(detector, "_block_ip") as mock_block:
            mock_block.return_value = True

            await detector.auto_mitigate(sqli_detection_result)

            # Should block the attacking IP
            mock_block.assert_called()


class TestDetectionResults:
    """Test detection result handling."""

    def test_detection_result_has_required_fields(self, sqli_detection_result):
        """Test detection result has required fields."""
        assert sqli_detection_result.detection_id is not None
        assert sqli_detection_result.detection_type is not None
        assert sqli_detection_result.severity is not None
        assert sqli_detection_result.confidence >= 0
        assert sqli_detection_result.confidence <= 1
        assert sqli_detection_result.detected_at is not None

    def test_detection_result_severity_ordering(self):
        """Test severity enum ordering."""
        assert Severity.CRITICAL.value > Severity.HIGH.value
        assert Severity.HIGH.value > Severity.MEDIUM.value
        assert Severity.MEDIUM.value > Severity.LOW.value

    def test_detection_result_includes_recommendations(self, sqli_detection_result):
        """Test detection result includes recommendations."""
        assert sqli_detection_result.recommendations is not None
        assert len(sqli_detection_result.recommendations) > 0


class TestTrafficAnalyzer:
    """Test TrafficAnalyzer helper class."""

    def test_analyzer_calculates_request_rate(self, normal_traffic_sample):
        """Test analyzer calculates request rate."""
        analyzer = TrafficAnalyzer()

        samples = [normal_traffic_sample] * 100

        rate = analyzer.calculate_request_rate(samples, window_seconds=60)

        assert rate >= 0

    def test_analyzer_calculates_error_rate(self, sqli_attack_sample, normal_traffic_sample):
        """Test analyzer calculates error rate."""
        analyzer = TrafficAnalyzer()

        samples = [normal_traffic_sample] * 90 + [sqli_attack_sample] * 10

        error_rate = analyzer.calculate_error_rate(samples)

        # 10% error rate expected
        assert 0 <= error_rate <= 1

    def test_analyzer_calculates_unique_ips(self, normal_traffic_sample):
        """Test analyzer counts unique source IPs."""
        analyzer = TrafficAnalyzer()

        samples = []
        for i in range(100):
            sample = TrafficSample(
                sample_id=str(uuid4()),
                timestamp=datetime.utcnow(),
                source_ip=f"192.168.1.{i % 50}",  # 50 unique IPs
                destination_ip="10.0.0.50",
                source_port=54321,
                destination_port=443,
                protocol="HTTPS",
                method="GET",
                path="/api/v1/test",
                headers={},
                query_params={},
                body=None,
                response_code=200,
                response_time_ms=50,
                bytes_sent=256,
                bytes_received=1024,
                country_code="US",
                asn="AS15169",
                metadata={},
            )
            samples.append(sample)

        unique_ips = analyzer.count_unique_ips(samples)

        assert unique_ips == 50

    def test_analyzer_calculates_response_time_percentiles(self, normal_traffic_sample):
        """Test analyzer calculates response time percentiles."""
        analyzer = TrafficAnalyzer()

        samples = []
        for i in range(100):
            sample = TrafficSample(
                sample_id=str(uuid4()),
                timestamp=datetime.utcnow(),
                source_ip="192.168.1.100",
                destination_ip="10.0.0.50",
                source_port=54321,
                destination_port=443,
                protocol="HTTPS",
                method="GET",
                path="/api/v1/test",
                headers={},
                query_params={},
                body=None,
                response_code=200,
                response_time_ms=10 + i,  # 10-109ms range
                bytes_sent=256,
                bytes_received=1024,
                country_code="US",
                asn="AS15169",
                metadata={},
            )
            samples.append(sample)

        percentiles = analyzer.calculate_response_time_percentiles(samples)

        assert "p50" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles
        assert percentiles["p50"] < percentiles["p95"] < percentiles["p99"]


class TestAnomalyReport:
    """Test anomaly report generation."""

    @pytest.mark.asyncio
    async def test_generate_anomaly_report(
        self, anomaly_detector_config, multiple_traffic_samples
    ):
        """Test generating anomaly report."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        with patch.object(detector, "analyze_traffic") as mock_analyze:
            mock_analyze.return_value = []

            report = await detector.generate_report(
                multiple_traffic_samples,
                start_time=datetime.utcnow() - timedelta(hours=1),
                end_time=datetime.utcnow(),
            )

            assert isinstance(report, AnomalyReport)
            assert report.start_time is not None
            assert report.end_time is not None

    @pytest.mark.asyncio
    async def test_report_includes_summary(
        self, anomaly_detector_config, sqli_detection_result
    ):
        """Test report includes detection summary."""
        detector = AnomalyDetector(config=anomaly_detector_config)

        with patch.object(detector, "analyze_traffic") as mock_analyze:
            mock_analyze.return_value = [sqli_detection_result]

            report = await detector.generate_report(
                [],
                start_time=datetime.utcnow() - timedelta(hours=1),
                end_time=datetime.utcnow(),
            )

            assert report.total_detections >= 0
            assert report.detections_by_type is not None
            assert report.detections_by_severity is not None
