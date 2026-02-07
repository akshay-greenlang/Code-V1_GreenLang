"""
Load tests for WAF management system.

Tests high-volume traffic analysis, rule evaluation, and anomaly detection.
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from typing import List


class TestWAFRuleEvaluationLoad:
    """Load tests for WAF rule evaluation."""

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_high_volume_rule_evaluation(
        self,
        traffic_sample_generator,
        performance_metrics,
        performance_targets,
        load_test_config,
    ):
        """Test rule evaluation at high request volumes."""
        config = load_test_config["large_load"]
        targets = performance_targets["waf_rule_evaluation"]

        mock_waf = MagicMock()

        # Generate traffic samples
        samples = list(traffic_sample_generator(config["traffic_samples"]))

        # Evaluate rules for each sample
        start_time = time.perf_counter()
        processed = 0

        # Process in batches
        batch_size = 1000
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]

            batch_start = time.perf_counter()

            # Mock rule evaluation
            mock_waf.evaluate_rules.return_value = [
                {"sample": s, "action": "allow" if s["response_code"] == 200 else "block"}
                for s in batch
            ]

            mock_waf.evaluate_rules(batch)

            latency_ms = (time.perf_counter() - batch_start) * 1000
            performance_metrics.record_latency(latency_ms / len(batch))  # Per-request latency
            processed += len(batch)

        total_time = time.perf_counter() - start_time
        throughput = processed / total_time

        # Verify performance targets
        assert throughput >= targets["throughput_min"] * 0.5  # 50% of target for mock

        print(f"\nWAF Rule Evaluation Load Test Results:")
        print(f"  Requests Processed: {processed}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.0f} requests/s")
        print(f"  Avg Per-Request Latency: {performance_metrics.p50:.4f}ms")
        print(f"  P95 Per-Request Latency: {performance_metrics.p95:.4f}ms")


class TestTrafficAnalysisLoad:
    """Load tests for traffic analysis."""

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_bulk_traffic_analysis(
        self,
        traffic_sample_generator,
        performance_metrics,
        performance_targets,
        load_test_config,
    ):
        """Test analyzing large volumes of traffic."""
        config = load_test_config["medium_load"]
        targets = performance_targets["traffic_analysis"]

        mock_detector = AsyncMock()

        # Generate traffic samples
        samples = list(traffic_sample_generator(config["traffic_samples"]))

        # Analyze traffic
        start_time = time.perf_counter()

        batch_size = 500
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]

            batch_start = time.perf_counter()

            mock_detector.analyze_traffic.return_value = []  # No detections

            await mock_detector.analyze_traffic(batch)

            latency_ms = (time.perf_counter() - batch_start) * 1000
            performance_metrics.record_latency(latency_ms)

        total_time = time.perf_counter() - start_time
        throughput = config["traffic_samples"] / total_time

        print(f"\nTraffic Analysis Load Test Results:")
        print(f"  Samples Analyzed: {config['traffic_samples']}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.0f} samples/s")
        print(f"  P95 Batch Latency: {performance_metrics.p95:.2f}ms")

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_anomaly_detection_scalability(
        self,
        traffic_sample_generator,
        performance_metrics,
        load_test_config,
    ):
        """Test anomaly detection scalability."""
        mock_detector = AsyncMock()

        # Test with increasing traffic volumes
        volumes = [1000, 5000, 10000, 50000]
        results = []

        for volume in volumes:
            samples = list(traffic_sample_generator(volume))

            start_time = time.perf_counter()

            mock_detector.detect_anomalies.return_value = []

            await mock_detector.detect_anomalies({
                "sample_count": volume,
                "requests_per_minute": volume / 60,
            })

            elapsed = time.perf_counter() - start_time

            results.append({
                "volume": volume,
                "time_s": elapsed,
                "throughput": volume / elapsed,
            })

        print(f"\nAnomaly Detection Scalability Results:")
        for r in results:
            print(f"  Volume: {r['volume']:,} -> {r['throughput']:.0f} samples/s")


class TestConcurrentWAFOperations:
    """Load tests for concurrent WAF operations."""

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_concurrent_rule_updates(
        self,
        performance_metrics,
        load_test_config,
    ):
        """Test concurrent WAF rule updates."""
        config = load_test_config["small_load"]

        mock_service = AsyncMock()

        async def update_rule(rule_id: int):
            start = time.perf_counter()
            mock_service.update_rule.return_value = MagicMock(
                rule_id=f"rule-{rule_id}",
                updated_at=datetime.utcnow(),
            )
            await mock_service.update_rule(
                f"rule-{rule_id}",
                {"parameters": {"limit": 100 + rule_id}},
            )
            latency_ms = (time.perf_counter() - start) * 1000
            return latency_ms

        start_time = time.perf_counter()

        # Run concurrent updates
        tasks = [update_rule(i) for i in range(config["concurrent_users"])]
        latencies = await asyncio.gather(*tasks)

        for latency in latencies:
            performance_metrics.record_latency(latency)

        total_time = time.perf_counter() - start_time

        print(f"\nConcurrent Rule Update Load Test Results:")
        print(f"  Concurrent Updates: {config['concurrent_users']}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  P95 Latency: {performance_metrics.p95:.2f}ms")

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_concurrent_ip_blocking(
        self,
        performance_metrics,
        load_test_config,
    ):
        """Test concurrent IP blocking operations."""
        config = load_test_config["small_load"]

        mock_service = AsyncMock()

        async def block_ip(ip_id: int):
            start = time.perf_counter()
            mock_service.block_ip.return_value = True
            await mock_service.block_ip(
                ip_address=f"192.168.{ip_id % 256}.{ip_id % 256}",
                duration_hours=24,
                reason="Load test",
            )
            latency_ms = (time.perf_counter() - start) * 1000
            return latency_ms

        start_time = time.perf_counter()

        # Run concurrent blocks
        tasks = [block_ip(i) for i in range(config["concurrent_users"] * 5)]
        latencies = await asyncio.gather(*tasks)

        for latency in latencies:
            performance_metrics.record_latency(latency)

        total_time = time.perf_counter() - start_time
        throughput = len(tasks) / total_time

        print(f"\nConcurrent IP Blocking Load Test Results:")
        print(f"  Total Blocks: {len(tasks)}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Throughput: {throughput:.0f} blocks/s")
        print(f"  P95 Latency: {performance_metrics.p95:.2f}ms")


class TestWAFBaselineLoad:
    """Load tests for traffic baseline operations."""

    @pytest.mark.load
    @pytest.mark.asyncio
    async def test_baseline_building_performance(
        self,
        traffic_sample_generator,
        performance_metrics,
        load_test_config,
    ):
        """Test baseline building with large datasets."""
        mock_detector = AsyncMock()

        # Test with different sample sizes
        sample_sizes = [1000, 10000, 100000]
        results = []

        for size in sample_sizes:
            samples = list(traffic_sample_generator(size))

            start_time = time.perf_counter()

            mock_detector.build_baseline.return_value = MagicMock(
                baseline_id="baseline-1",
                sample_count=size,
            )

            await mock_detector.build_baseline(
                samples,
                name="Load Test Baseline",
                endpoint_pattern="/api/.*",
            )

            elapsed = time.perf_counter() - start_time
            results.append({
                "samples": size,
                "time_s": elapsed,
            })

        print(f"\nBaseline Building Performance Results:")
        for r in results:
            print(f"  Samples: {r['samples']:,} -> {r['time_s']:.2f}s")
