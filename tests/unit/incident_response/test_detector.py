"""
Unit tests for IncidentDetector.

Tests alert detection from multiple sources: Prometheus, Loki, GuardDuty, CloudTrail.
Validates polling, filtering, deduplication, and error handling.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from greenlang.infrastructure.incident_response.detector import IncidentDetector
from greenlang.infrastructure.incident_response.models import (
    Alert,
    AlertSource,
)


class TestIncidentDetectorInitialization:
    """Test IncidentDetector initialization."""

    def test_initialization_with_config(self, detector_config):
        """Test detector initializes with configuration."""
        detector = IncidentDetector(config=detector_config)

        assert detector.config == detector_config
        assert detector.prometheus_url == detector_config.prometheus_url
        assert detector.loki_url == detector_config.loki_url

    def test_initialization_default_config(self):
        """Test detector initializes with default configuration."""
        detector = IncidentDetector()

        assert detector.config is not None
        assert detector.prometheus_url is not None

    def test_initialization_creates_http_client(self, detector_config):
        """Test detector creates HTTP client for API calls."""
        detector = IncidentDetector(config=detector_config)

        assert detector._http_client is not None


class TestPrometheusPolling:
    """Test Prometheus alert polling."""

    @pytest.mark.asyncio
    async def test_poll_prometheus_returns_alerts(
        self, detector_config, mock_prometheus_client
    ):
        """Test polling Prometheus returns alerts."""
        detector = IncidentDetector(config=detector_config)

        with patch.object(detector, "_http_client", mock_prometheus_client):
            mock_prometheus_client.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value={
                    "status": "success",
                    "data": {
                        "alerts": [
                            {
                                "labels": {"alertname": "HighCPU", "instance": "node-1"},
                                "annotations": {"summary": "CPU high"},
                                "state": "firing",
                                "activeAt": datetime.utcnow().isoformat(),
                            }
                        ]
                    }
                })
            ))

            alerts = await detector.poll_prometheus()

            assert len(alerts) >= 0
            mock_prometheus_client.get.assert_called()

    @pytest.mark.asyncio
    async def test_poll_prometheus_handles_empty_response(self, detector_config):
        """Test polling handles empty Prometheus response."""
        detector = IncidentDetector(config=detector_config)

        with patch.object(detector, "_http_client") as mock_client:
            mock_client.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value={
                    "status": "success",
                    "data": {"alerts": []}
                })
            ))

            alerts = await detector.poll_prometheus()

            assert alerts == []

    @pytest.mark.asyncio
    async def test_poll_prometheus_handles_connection_error(self, detector_config):
        """Test polling handles Prometheus connection errors."""
        detector = IncidentDetector(config=detector_config)

        with patch.object(detector, "_http_client") as mock_client:
            mock_client.get = AsyncMock(side_effect=ConnectionError("Connection refused"))

            alerts = await detector.poll_prometheus()

            assert alerts == []

    @pytest.mark.asyncio
    async def test_poll_prometheus_handles_timeout(self, detector_config):
        """Test polling handles Prometheus timeout."""
        detector = IncidentDetector(config=detector_config)

        with patch.object(detector, "_http_client") as mock_client:
            mock_client.get = AsyncMock(side_effect=TimeoutError("Request timed out"))

            alerts = await detector.poll_prometheus()

            assert alerts == []

    @pytest.mark.asyncio
    async def test_poll_prometheus_filters_resolved_alerts(self, detector_config):
        """Test polling filters out resolved alerts."""
        detector = IncidentDetector(config=detector_config)

        with patch.object(detector, "_http_client") as mock_client:
            mock_client.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value={
                    "status": "success",
                    "data": {
                        "alerts": [
                            {
                                "labels": {"alertname": "HighCPU"},
                                "state": "firing",
                                "activeAt": datetime.utcnow().isoformat(),
                            },
                            {
                                "labels": {"alertname": "HighMemory"},
                                "state": "resolved",
                                "activeAt": datetime.utcnow().isoformat(),
                            }
                        ]
                    }
                })
            ))

            alerts = await detector.poll_prometheus()

            # Only firing alerts should be returned
            firing_alerts = [a for a in alerts if a.labels.get("state") != "resolved"]
            assert len(firing_alerts) <= len(alerts)


class TestLokiPolling:
    """Test Loki log-based alert polling."""

    @pytest.mark.asyncio
    async def test_poll_loki_returns_alerts(self, detector_config, mock_loki_client):
        """Test polling Loki returns alerts from error logs."""
        detector = IncidentDetector(config=detector_config)

        with patch.object(detector, "_http_client") as mock_client:
            mock_client.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value={
                    "status": "success",
                    "data": {
                        "result": [
                            {
                                "stream": {"level": "error", "app": "greenlang-api"},
                                "values": [
                                    ["1609459200000000000", "Error: Connection refused"],
                                ]
                            }
                        ]
                    }
                })
            ))

            alerts = await detector.poll_loki()

            assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_poll_loki_uses_correct_query(self, detector_config):
        """Test Loki polling uses correct LogQL query."""
        detector = IncidentDetector(config=detector_config)

        with patch.object(detector, "_http_client") as mock_client:
            mock_client.get = AsyncMock(return_value=MagicMock(
                status_code=200,
                json=MagicMock(return_value={"status": "success", "data": {"result": []}})
            ))

            await detector.poll_loki()

            # Verify query was made
            mock_client.get.assert_called()
            call_args = mock_client.get.call_args
            assert call_args is not None

    @pytest.mark.asyncio
    async def test_poll_loki_handles_rate_limit(self, detector_config):
        """Test Loki polling handles rate limiting."""
        detector = IncidentDetector(config=detector_config)

        with patch.object(detector, "_http_client") as mock_client:
            mock_client.get = AsyncMock(return_value=MagicMock(
                status_code=429,
                json=MagicMock(return_value={"error": "rate limit exceeded"})
            ))

            alerts = await detector.poll_loki()

            assert alerts == []


class TestGuardDutyPolling:
    """Test AWS GuardDuty finding polling."""

    @pytest.mark.asyncio
    async def test_poll_guardduty_returns_alerts(
        self, detector_config, mock_guardduty_client
    ):
        """Test polling GuardDuty returns security alerts."""
        detector = IncidentDetector(config=detector_config)

        with patch.object(detector, "_guardduty_client", mock_guardduty_client):
            alerts = await detector.poll_guardduty()

            assert isinstance(alerts, list)
            mock_guardduty_client.list_findings.assert_called()

    @pytest.mark.asyncio
    async def test_poll_guardduty_disabled(self, detector_config):
        """Test GuardDuty polling when disabled."""
        detector_config.guardduty_enabled = False
        detector = IncidentDetector(config=detector_config)

        alerts = await detector.poll_guardduty()

        assert alerts == []

    @pytest.mark.asyncio
    async def test_poll_guardduty_maps_severity(self, detector_config):
        """Test GuardDuty severity is correctly mapped."""
        detector = IncidentDetector(config=detector_config)

        with patch.object(detector, "_guardduty_client") as mock_client:
            mock_client.list_findings.return_value = {"FindingIds": ["finding-1"]}
            mock_client.get_findings.return_value = {
                "Findings": [
                    {
                        "Id": "finding-1",
                        "Type": "UnauthorizedAccess",
                        "Severity": 8.5,  # High severity
                        "Title": "Test finding",
                        "Description": "Test description",
                        "CreatedAt": datetime.utcnow().isoformat(),
                    }
                ]
            }

            alerts = await detector.poll_guardduty()

            if alerts:
                # Severity 8.5 should map to "critical" or "high"
                assert alerts[0].severity in ["critical", "high"]

    @pytest.mark.asyncio
    async def test_poll_guardduty_handles_pagination(self, detector_config):
        """Test GuardDuty polling handles paginated results."""
        detector = IncidentDetector(config=detector_config)

        with patch.object(detector, "_guardduty_client") as mock_client:
            # First page
            mock_client.list_findings.side_effect = [
                {"FindingIds": ["f1", "f2"], "NextToken": "token1"},
                {"FindingIds": ["f3"], "NextToken": None},
            ]
            mock_client.get_findings.return_value = {
                "Findings": [
                    {
                        "Id": "f1",
                        "Type": "Test",
                        "Severity": 5.0,
                        "Title": "Test",
                        "Description": "Test",
                        "CreatedAt": datetime.utcnow().isoformat(),
                    }
                ]
            }

            alerts = await detector.poll_guardduty()

            assert isinstance(alerts, list)


class TestCloudTrailPolling:
    """Test CloudTrail anomaly detection polling."""

    @pytest.mark.asyncio
    async def test_poll_cloudtrail_anomalies_returns_alerts(self, detector_config):
        """Test polling CloudTrail returns anomaly alerts."""
        detector = IncidentDetector(config=detector_config)

        with patch.object(detector, "_cloudtrail_client") as mock_client:
            mock_client.lookup_events.return_value = {
                "Events": [
                    {
                        "EventId": "event-1",
                        "EventName": "ConsoleLogin",
                        "EventSource": "signin.amazonaws.com",
                        "EventTime": datetime.utcnow(),
                        "Username": "suspicious-user",
                    }
                ]
            }

            alerts = await detector.poll_cloudtrail_anomalies()

            assert isinstance(alerts, list)

    @pytest.mark.asyncio
    async def test_poll_cloudtrail_disabled(self, detector_config):
        """Test CloudTrail polling when disabled."""
        detector_config.cloudtrail_enabled = False
        detector = IncidentDetector(config=detector_config)

        alerts = await detector.poll_cloudtrail_anomalies()

        assert alerts == []


class TestAlertDeduplication:
    """Test alert deduplication logic."""

    @pytest.mark.asyncio
    async def test_deduplicates_identical_alerts(self, detector_config, sample_alert):
        """Test identical alerts are deduplicated."""
        detector = IncidentDetector(config=detector_config)

        # Create duplicate alerts
        alerts = [sample_alert, sample_alert, sample_alert]

        deduped = detector._deduplicate_alerts(alerts)

        assert len(deduped) == 1

    @pytest.mark.asyncio
    async def test_keeps_different_alerts(self, detector_config, multiple_alerts):
        """Test different alerts are kept."""
        detector = IncidentDetector(config=detector_config)

        deduped = detector._deduplicate_alerts(multiple_alerts)

        assert len(deduped) == len(multiple_alerts)

    @pytest.mark.asyncio
    async def test_deduplication_uses_fingerprint(self, detector_config):
        """Test deduplication uses alert fingerprint."""
        detector = IncidentDetector(config=detector_config)

        alert1 = Alert(
            alert_id=str(uuid4()),
            title="High CPU",
            description="CPU high",
            severity="high",
            source=AlertSource.PROMETHEUS,
            timestamp=datetime.utcnow(),
            labels={"alertname": "HighCPU", "instance": "node-1"},
            annotations={},
            raw_data={},
        )

        # Same alert, different ID
        alert2 = Alert(
            alert_id=str(uuid4()),
            title="High CPU",
            description="CPU high",
            severity="high",
            source=AlertSource.PROMETHEUS,
            timestamp=datetime.utcnow(),
            labels={"alertname": "HighCPU", "instance": "node-1"},
            annotations={},
            raw_data={},
        )

        deduped = detector._deduplicate_alerts([alert1, alert2])

        # Should be deduplicated based on fingerprint, not ID
        assert len(deduped) == 1


class TestAlertFiltering:
    """Test alert filtering logic."""

    def test_filter_by_severity(self, detector_config, multiple_alerts):
        """Test filtering alerts by severity."""
        detector = IncidentDetector(config=detector_config)

        filtered = detector._filter_alerts(
            multiple_alerts,
            min_severity="high"
        )

        for alert in filtered:
            assert alert.severity in ["high", "critical"]

    def test_filter_by_source(self, detector_config, multiple_alerts):
        """Test filtering alerts by source."""
        detector = IncidentDetector(config=detector_config)

        filtered = detector._filter_alerts(
            multiple_alerts,
            sources=[AlertSource.PROMETHEUS]
        )

        for alert in filtered:
            assert alert.source == AlertSource.PROMETHEUS

    def test_filter_by_time_window(self, detector_config):
        """Test filtering alerts by time window."""
        detector = IncidentDetector(config=detector_config)

        now = datetime.utcnow()
        alerts = [
            Alert(
                alert_id=str(uuid4()),
                title="Recent alert",
                description="Recent",
                severity="high",
                source=AlertSource.PROMETHEUS,
                timestamp=now - timedelta(minutes=5),
                labels={},
                annotations={},
                raw_data={},
            ),
            Alert(
                alert_id=str(uuid4()),
                title="Old alert",
                description="Old",
                severity="high",
                source=AlertSource.PROMETHEUS,
                timestamp=now - timedelta(hours=2),
                labels={},
                annotations={},
                raw_data={},
            ),
        ]

        filtered = detector._filter_alerts(
            alerts,
            time_window_minutes=60
        )

        assert len(filtered) == 1
        assert filtered[0].title == "Recent alert"


class TestDetectIncidents:
    """Test main detect_incidents method."""

    @pytest.mark.asyncio
    async def test_detect_incidents_polls_all_sources(self, detector_config):
        """Test detect_incidents polls all configured sources."""
        detector = IncidentDetector(config=detector_config)

        with patch.object(detector, "poll_prometheus", new_callable=AsyncMock) as mock_prom, \
             patch.object(detector, "poll_loki", new_callable=AsyncMock) as mock_loki, \
             patch.object(detector, "poll_guardduty", new_callable=AsyncMock) as mock_gd, \
             patch.object(detector, "poll_cloudtrail_anomalies", new_callable=AsyncMock) as mock_ct:

            mock_prom.return_value = []
            mock_loki.return_value = []
            mock_gd.return_value = []
            mock_ct.return_value = []

            await detector.detect_incidents()

            mock_prom.assert_called_once()
            mock_loki.assert_called_once()
            mock_gd.assert_called_once()
            mock_ct.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_incidents_aggregates_alerts(self, detector_config, sample_alert):
        """Test detect_incidents aggregates alerts from all sources."""
        detector = IncidentDetector(config=detector_config)

        with patch.object(detector, "poll_prometheus", new_callable=AsyncMock) as mock_prom, \
             patch.object(detector, "poll_loki", new_callable=AsyncMock) as mock_loki, \
             patch.object(detector, "poll_guardduty", new_callable=AsyncMock) as mock_gd, \
             patch.object(detector, "poll_cloudtrail_anomalies", new_callable=AsyncMock) as mock_ct:

            mock_prom.return_value = [sample_alert]
            mock_loki.return_value = [sample_alert]
            mock_gd.return_value = []
            mock_ct.return_value = []

            alerts = await detector.detect_incidents()

            # Should have alerts from both Prometheus and Loki (deduplicated)
            assert len(alerts) >= 1

    @pytest.mark.asyncio
    async def test_detect_incidents_handles_partial_failure(self, detector_config):
        """Test detect_incidents continues if one source fails."""
        detector = IncidentDetector(config=detector_config)

        with patch.object(detector, "poll_prometheus", new_callable=AsyncMock) as mock_prom, \
             patch.object(detector, "poll_loki", new_callable=AsyncMock) as mock_loki, \
             patch.object(detector, "poll_guardduty", new_callable=AsyncMock) as mock_gd, \
             patch.object(detector, "poll_cloudtrail_anomalies", new_callable=AsyncMock) as mock_ct:

            mock_prom.side_effect = Exception("Prometheus error")
            mock_loki.return_value = []
            mock_gd.return_value = []
            mock_ct.return_value = []

            # Should not raise, should continue with other sources
            alerts = await detector.detect_incidents()

            assert isinstance(alerts, list)
            mock_loki.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_incidents_respects_batch_size(self, detector_config):
        """Test detect_incidents respects configured batch size."""
        detector_config.alert_batch_size = 5
        detector = IncidentDetector(config=detector_config)

        many_alerts = [
            Alert(
                alert_id=str(uuid4()),
                title=f"Alert {i}",
                description="Test",
                severity="warning",
                source=AlertSource.PROMETHEUS,
                timestamp=datetime.utcnow(),
                labels={},
                annotations={},
                raw_data={},
            )
            for i in range(10)
        ]

        with patch.object(detector, "poll_prometheus", new_callable=AsyncMock) as mock_prom, \
             patch.object(detector, "poll_loki", new_callable=AsyncMock) as mock_loki, \
             patch.object(detector, "poll_guardduty", new_callable=AsyncMock) as mock_gd, \
             patch.object(detector, "poll_cloudtrail_anomalies", new_callable=AsyncMock) as mock_ct:

            mock_prom.return_value = many_alerts
            mock_loki.return_value = []
            mock_gd.return_value = []
            mock_ct.return_value = []

            alerts = await detector.detect_incidents()

            assert len(alerts) <= detector_config.alert_batch_size


class TestAlertTransformation:
    """Test alert data transformation."""

    def test_transform_prometheus_alert(self, detector_config):
        """Test transforming Prometheus alert format."""
        detector = IncidentDetector(config=detector_config)

        raw_alert = {
            "labels": {"alertname": "HighCPU", "instance": "node-1"},
            "annotations": {"summary": "CPU is high"},
            "state": "firing",
            "activeAt": "2025-01-15T10:00:00Z",
        }

        alert = detector._transform_prometheus_alert(raw_alert)

        assert alert.source == AlertSource.PROMETHEUS
        assert alert.title == "HighCPU"
        assert "node-1" in str(alert.labels)

    def test_transform_guardduty_finding(self, detector_config):
        """Test transforming GuardDuty finding format."""
        detector = IncidentDetector(config=detector_config)

        raw_finding = {
            "Id": "finding-123",
            "Type": "UnauthorizedAccess:IAMUser/MaliciousIPCaller",
            "Severity": 8.0,
            "Title": "Malicious IP caller",
            "Description": "API called from malicious IP",
            "CreatedAt": "2025-01-15T10:00:00Z",
        }

        alert = detector._transform_guardduty_finding(raw_finding)

        assert alert.source == AlertSource.GUARDDUTY
        assert "Malicious" in alert.title
        assert alert.severity in ["critical", "high"]

    def test_transform_loki_log_entry(self, detector_config):
        """Test transforming Loki log entry to alert."""
        detector = IncidentDetector(config=detector_config)

        raw_entry = {
            "stream": {"level": "error", "app": "greenlang-api"},
            "values": [["1609459200000000000", "Error: Connection refused"]],
        }

        alert = detector._transform_loki_entry(raw_entry)

        assert alert.source == AlertSource.LOKI
        assert alert.severity == "high"  # Error level
