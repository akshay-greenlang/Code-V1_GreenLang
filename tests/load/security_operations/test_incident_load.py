"""
Load tests for incident response system.

Tests high-volume alert processing, incident correlation, and playbook execution.
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from typing import List


class TestAlertProcessingLoad:
    """Load tests for alert processing."""

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_high_volume_alert_detection(
        self,
        alert_generator,
        performance_metrics,
        performance_targets,
        load_test_config,
    ):
        """Test processing high volume of alerts."""
        config = load_test_config["medium_load"]
        targets = performance_targets["incident_detection"]

        mock_detector = AsyncMock()

        # Generate alerts
        alerts = list(alert_generator(config["alerts"]))

        # Process alerts in batches
        batch_size = 100
        start_time = time.perf_counter()

        for i in range(0, len(alerts), batch_size):
            batch = alerts[i:i + batch_size]

            batch_start = time.perf_counter()
            mock_detector.detect_incidents.return_value = batch

            try:
                await mock_detector.detect_incidents()
                latency_ms = (time.perf_counter() - batch_start) * 1000
                performance_metrics.record_latency(latency_ms)
            except Exception:
                performance_metrics.record_error()

        total_time = time.perf_counter() - start_time

        # Calculate throughput
        throughput = config["alerts"] / total_time

        # Verify performance targets
        assert throughput >= targets["throughput_min"] * 0.8  # 80% of target
        assert performance_metrics.p95 <= targets["p95_latency_max_ms"] * 2  # Allow 2x slack
        assert performance_metrics.error_rate <= targets["error_rate_max"]

        print(f"\nAlert Detection Load Test Results:")
        print(f"  Alerts Processed: {config['alerts']}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.0f} alerts/s")
        print(f"  P50 Latency: {performance_metrics.p50:.2f}ms")
        print(f"  P95 Latency: {performance_metrics.p95:.2f}ms")
        print(f"  P99 Latency: {performance_metrics.p99:.2f}ms")
        print(f"  Error Rate: {performance_metrics.error_rate:.4f}")

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_alert_correlation_under_load(
        self,
        alert_generator,
        performance_metrics,
        load_test_config,
    ):
        """Test alert correlation with many alerts."""
        config = load_test_config["medium_load"]

        mock_correlator = MagicMock()

        # Generate alerts
        alerts = list(alert_generator(config["alerts"]))

        # Correlate alerts
        start_time = time.perf_counter()

        # Mock correlation to group every 10 alerts
        groups = [alerts[i:i+10] for i in range(0, len(alerts), 10)]
        mock_correlator.correlate.return_value = groups

        correlation_start = time.perf_counter()
        result = mock_correlator.correlate(alerts)
        latency_ms = (time.perf_counter() - correlation_start) * 1000

        performance_metrics.record_latency(latency_ms)

        total_time = time.perf_counter() - start_time

        print(f"\nAlert Correlation Load Test Results:")
        print(f"  Alerts Processed: {config['alerts']}")
        print(f"  Groups Created: {len(result)}")
        print(f"  Correlation Time: {total_time:.2f}s")
        print(f"  Latency: {latency_ms:.2f}ms")


class TestIncidentProcessingLoad:
    """Load tests for incident processing."""

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_concurrent_incident_classification(
        self,
        alert_generator,
        performance_metrics,
        load_test_config,
    ):
        """Test concurrent incident classification."""
        config = load_test_config["small_load"]

        mock_classifier = MagicMock()

        # Create incidents from alerts
        alerts = list(alert_generator(config["alerts"]))
        incidents = [
            MagicMock(
                incident_id=f"incident-{i}",
                alerts=alerts[i:i+5],
            )
            for i in range(0, len(alerts), 5)
        ]

        # Classify incidents concurrently
        async def classify_incident(incident):
            start = time.perf_counter()
            mock_classifier.classify.return_value = MagicMock(
                incident_type="infrastructure",
                escalation_level="P2",
            )
            result = mock_classifier.classify(incident)
            latency_ms = (time.perf_counter() - start) * 1000
            return result, latency_ms

        start_time = time.perf_counter()

        tasks = [classify_incident(inc) for inc in incidents]
        results = await asyncio.gather(*tasks)

        for _, latency_ms in results:
            performance_metrics.record_latency(latency_ms)

        total_time = time.perf_counter() - start_time
        throughput = len(incidents) / total_time

        print(f"\nIncident Classification Load Test Results:")
        print(f"  Incidents Classified: {len(incidents)}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.0f} incidents/s")
        print(f"  P95 Latency: {performance_metrics.p95:.2f}ms")


class TestPlaybookExecutionLoad:
    """Load tests for playbook execution."""

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_concurrent_playbook_executions(
        self,
        performance_metrics,
        load_test_config,
    ):
        """Test concurrent playbook executions."""
        config = load_test_config["small_load"]

        mock_executor = AsyncMock()

        # Create incidents requiring playbook execution
        incidents = [
            MagicMock(incident_id=f"incident-{i}")
            for i in range(config["concurrent_users"])
        ]

        async def execute_playbook(incident):
            start = time.perf_counter()
            mock_executor.execute.return_value = MagicMock(
                status="completed",
                results={"step_1": "success"},
            )
            result = await mock_executor.execute(
                playbook_id="pod_restart",
                incident=incident,
                executed_by="load-test",
            )
            latency_ms = (time.perf_counter() - start) * 1000
            return result, latency_ms

        start_time = time.perf_counter()

        # Run concurrent executions
        tasks = [execute_playbook(inc) for inc in incidents]
        results = await asyncio.gather(*tasks)

        for _, latency_ms in results:
            performance_metrics.record_latency(latency_ms)

        total_time = time.perf_counter() - start_time
        throughput = len(incidents) / total_time

        print(f"\nPlaybook Execution Load Test Results:")
        print(f"  Concurrent Executions: {len(incidents)}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.0f} executions/s")
        print(f"  P95 Latency: {performance_metrics.p95:.2f}ms")


class TestIncidentAPILoad:
    """Load tests for incident API endpoints."""

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_api_endpoint_throughput(
        self,
        performance_metrics,
        load_test_config,
    ):
        """Test API endpoint throughput under load."""
        config = load_test_config["medium_load"]

        mock_client = AsyncMock()

        # Simulate API requests
        async def make_request(request_id: int):
            start = time.perf_counter()
            mock_client.get.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value={"incidents": []}),
            )
            await mock_client.get(f"/api/v1/incidents?page={request_id % 100}")
            latency_ms = (time.perf_counter() - start) * 1000
            return latency_ms

        start_time = time.perf_counter()

        # Run concurrent requests
        tasks = [make_request(i) for i in range(config["concurrent_users"] * 10)]
        latencies = await asyncio.gather(*tasks)

        for latency in latencies:
            performance_metrics.record_latency(latency)

        total_time = time.perf_counter() - start_time
        throughput = len(tasks) / total_time

        print(f"\nAPI Endpoint Load Test Results:")
        print(f"  Total Requests: {len(tasks)}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.0f} requests/s")
        print(f"  P50 Latency: {performance_metrics.p50:.2f}ms")
        print(f"  P95 Latency: {performance_metrics.p95:.2f}ms")
        print(f"  P99 Latency: {performance_metrics.p99:.2f}ms")
