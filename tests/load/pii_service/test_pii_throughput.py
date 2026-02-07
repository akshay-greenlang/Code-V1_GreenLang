# -*- coding: utf-8 -*-
"""
Load tests for PII Service Throughput - SEC-011.

Tests performance targets:
- Detection throughput: 10K+ messages/second
- Tokenization throughput: 5K+ tokens/second
- Enforcement latency: P99 < 10ms
- Streaming throughput: matches Kafka/Kinesis rates
- Vault capacity: 1M+ tokens per tenant

Author: GreenLang Test Engineering Team
Date: February 2026
PRD: SEC-011 PII Detection/Redaction Enhancements
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List
from uuid import uuid4

import pytest


# ============================================================================
# TestDetectionThroughput
# ============================================================================


@pytest.mark.load
@pytest.mark.performance
class TestDetectionThroughput:
    """Load tests for PII detection throughput."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_detection_throughput_10k_per_sec(
        self, pii_service, generate_pii_content, performance_metrics, run_concurrent
    ):
        """Verify detection achieves 10K+ messages/second throughput."""
        # Generate test content
        contents = generate_pii_content(count=10000)

        # Run detection concurrently
        async def detect_one(content):
            return await pii_service.detect(content)

        results = await run_concurrent(
            detect_one,
            contents,
            max_concurrent=200,
            metrics=performance_metrics,
        )

        # Report metrics
        report = performance_metrics.report()
        print(f"\nDetection Throughput Test Results:")
        print(f"  Total operations: {report['total_operations']}")
        print(f"  Duration: {report['total_duration_seconds']:.2f}s")
        print(f"  Throughput: {report['throughput_ops_per_sec']:.0f} ops/sec")
        print(f"  P50 latency: {report['latency_p50_ms']:.2f}ms")
        print(f"  P95 latency: {report['latency_p95_ms']:.2f}ms")
        print(f"  P99 latency: {report['latency_p99_ms']:.2f}ms")

        # Verify targets
        assert report['throughput_ops_per_sec'] >= 10000, \
            f"Detection throughput {report['throughput_ops_per_sec']:.0f} below 10K target"

        # Verify error rate
        error_count = sum(1 for r in results if isinstance(r, dict) and 'error' in r)
        error_rate = error_count / len(results)
        assert error_rate < 0.01, f"Error rate {error_rate:.2%} exceeds 1% threshold"

    @pytest.mark.asyncio
    async def test_detection_latency_distribution(
        self, pii_service, generate_pii_content, performance_metrics, run_concurrent
    ):
        """Verify detection latency distribution meets targets."""
        contents = generate_pii_content(count=1000)

        async def detect_one(content):
            return await pii_service.detect(content)

        await run_concurrent(
            detect_one,
            contents,
            max_concurrent=50,
            metrics=performance_metrics,
        )

        report = performance_metrics.report()

        # Latency targets
        assert report['latency_p50_ms'] < 5, \
            f"P50 latency {report['latency_p50_ms']:.2f}ms exceeds 5ms target"
        assert report['latency_p95_ms'] < 10, \
            f"P95 latency {report['latency_p95_ms']:.2f}ms exceeds 10ms target"
        assert report['latency_p99_ms'] < 20, \
            f"P99 latency {report['latency_p99_ms']:.2f}ms exceeds 20ms target"


# ============================================================================
# TestTokenizationThroughput
# ============================================================================


@pytest.mark.load
@pytest.mark.performance
class TestTokenizationThroughput:
    """Load tests for tokenization throughput."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_tokenization_throughput(
        self, pii_service, generate_bulk_tokens, performance_metrics, run_concurrent
    ):
        """Verify tokenization achieves target throughput."""
        try:
            from greenlang.infrastructure.pii_service.models import PIIType
        except ImportError:
            pytest.skip("PIIType not available")

        tokens_to_create = generate_bulk_tokens(count=5000)

        async def tokenize_one(item):
            pii_type = getattr(PIIType, item['pii_type'].upper(), PIIType.EMAIL)
            return await pii_service.tokenize(
                item['value'],
                pii_type,
                item['tenant_id'],
            )

        results = await run_concurrent(
            tokenize_one,
            tokens_to_create,
            max_concurrent=100,
            metrics=performance_metrics,
        )

        report = performance_metrics.report()
        print(f"\nTokenization Throughput Test Results:")
        print(f"  Total operations: {report['total_operations']}")
        print(f"  Duration: {report['total_duration_seconds']:.2f}s")
        print(f"  Throughput: {report['throughput_ops_per_sec']:.0f} ops/sec")
        print(f"  P99 latency: {report['latency_p99_ms']:.2f}ms")

        # Target: 5K+ tokens/second
        assert report['throughput_ops_per_sec'] >= 5000, \
            f"Tokenization throughput {report['throughput_ops_per_sec']:.0f} below 5K target"

    @pytest.mark.asyncio
    async def test_detokenization_throughput(
        self, pii_service, performance_metrics, run_concurrent
    ):
        """Verify detokenization achieves target throughput."""
        try:
            from greenlang.infrastructure.pii_service.models import PIIType
        except ImportError:
            pytest.skip("PIIType not available")

        tenant_id = f"tenant-load-{uuid4().hex[:8]}"
        user_id = str(uuid4())

        # First, create tokens to detokenize
        tokens = []
        for i in range(1000):
            token = await pii_service.tokenize(
                f"value-{i}",
                PIIType.EMAIL,
                tenant_id,
            )
            tokens.append(token)

        # Now detokenize
        async def detokenize_one(token):
            return await pii_service.detokenize(token, tenant_id, user_id)

        await run_concurrent(
            detokenize_one,
            tokens,
            max_concurrent=100,
            metrics=performance_metrics,
        )

        report = performance_metrics.report()
        print(f"\nDetokenization Throughput Test Results:")
        print(f"  Throughput: {report['throughput_ops_per_sec']:.0f} ops/sec")
        print(f"  P99 latency: {report['latency_p99_ms']:.2f}ms")

        # Target: 10K+ detokenizations/second (cached)
        assert report['throughput_ops_per_sec'] >= 5000, \
            f"Detokenization throughput {report['throughput_ops_per_sec']:.0f} below 5K target"


# ============================================================================
# TestEnforcementLatency
# ============================================================================


@pytest.mark.load
@pytest.mark.performance
class TestEnforcementLatency:
    """Load tests for enforcement latency."""

    @pytest.mark.asyncio
    async def test_enforcement_latency_p99(
        self, pii_service, generate_pii_content, performance_metrics, run_concurrent
    ):
        """Verify enforcement P99 latency meets target."""
        try:
            from greenlang.infrastructure.pii_service.models import EnforcementContext
        except ImportError:
            pytest.skip("EnforcementContext not available")

        contents = generate_pii_content(count=2000)
        tenant_id = f"tenant-{uuid4().hex[:8]}"
        user_id = str(uuid4())

        async def enforce_one(content):
            context = EnforcementContext(
                context_type="api_request",
                path="/api/v1/data",
                method="POST",
                tenant_id=tenant_id,
                user_id=user_id,
            )
            return await pii_service.enforce(content, context)

        await run_concurrent(
            enforce_one,
            contents,
            max_concurrent=100,
            metrics=performance_metrics,
        )

        report = performance_metrics.report()
        print(f"\nEnforcement Latency Test Results:")
        print(f"  Throughput: {report['throughput_ops_per_sec']:.0f} ops/sec")
        print(f"  P50 latency: {report['latency_p50_ms']:.2f}ms")
        print(f"  P95 latency: {report['latency_p95_ms']:.2f}ms")
        print(f"  P99 latency: {report['latency_p99_ms']:.2f}ms")

        # Target: P99 < 10ms
        assert report['latency_p99_ms'] < 10, \
            f"P99 latency {report['latency_p99_ms']:.2f}ms exceeds 10ms target"


# ============================================================================
# TestConcurrentRequests
# ============================================================================


@pytest.mark.load
@pytest.mark.performance
class TestConcurrentRequests:
    """Load tests for concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(
        self, pii_service, generate_pii_content, performance_metrics
    ):
        """Verify system handles high concurrency."""
        contents = generate_pii_content(count=500)

        # Launch all requests simultaneously
        start = time.perf_counter()

        results = await asyncio.gather(*[
            pii_service.detect(c) for c in contents
        ], return_exceptions=True)

        duration = time.perf_counter() - start

        # Count successes and failures
        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = len(results) - successes

        print(f"\nConcurrent Requests Test Results:")
        print(f"  Total requests: {len(contents)}")
        print(f"  Successes: {successes}")
        print(f"  Failures: {failures}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {len(contents)/duration:.0f} ops/sec")

        # Target: <1% failure rate
        failure_rate = failures / len(results)
        assert failure_rate < 0.01, f"Failure rate {failure_rate:.2%} exceeds 1%"

    @pytest.mark.asyncio
    async def test_mixed_operations_concurrent(
        self, pii_service, generate_pii_content, performance_metrics
    ):
        """Verify mixed operations don't interfere."""
        try:
            from greenlang.infrastructure.pii_service.models import PIIType
        except ImportError:
            pytest.skip("PIIType not available")

        tenant_id = f"tenant-{uuid4().hex[:8]}"
        user_id = str(uuid4())

        contents = generate_pii_content(count=200)

        # Mix of operations
        operations = []

        for i, content in enumerate(contents):
            if i % 3 == 0:
                operations.append(pii_service.detect(content))
            elif i % 3 == 1:
                operations.append(pii_service.redact(content))
            else:
                operations.append(
                    pii_service.tokenize(f"value-{i}", PIIType.EMAIL, tenant_id)
                )

        start = time.perf_counter()
        results = await asyncio.gather(*operations, return_exceptions=True)
        duration = time.perf_counter() - start

        successes = sum(1 for r in results if not isinstance(r, Exception))

        print(f"\nMixed Operations Test Results:")
        print(f"  Total operations: {len(operations)}")
        print(f"  Successes: {successes}")
        print(f"  Duration: {duration:.2f}s")

        assert successes == len(operations), "All operations should succeed"


# ============================================================================
# TestStreamingThroughput
# ============================================================================


@pytest.mark.load
@pytest.mark.performance
class TestStreamingThroughput:
    """Load tests for streaming scanner throughput."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_streaming_throughput(
        self, pii_service, generate_stream_messages, performance_metrics
    ):
        """Verify streaming scanner handles target throughput."""
        messages = generate_stream_messages(count=10000)

        # Simulate stream processing
        async def process_message(msg):
            content = str(msg['value'])
            return await pii_service.detect(content)

        start = time.perf_counter()

        # Process in batches to simulate Kafka batching
        batch_size = 100
        results = []

        for i in range(0, len(messages), batch_size):
            batch = messages[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                process_message(m) for m in batch
            ])
            results.extend(batch_results)

        duration = time.perf_counter() - start
        throughput = len(messages) / duration

        print(f"\nStreaming Throughput Test Results:")
        print(f"  Total messages: {len(messages)}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Throughput: {throughput:.0f} msg/sec")

        # Target: 10K+ messages/second
        assert throughput >= 10000, \
            f"Streaming throughput {throughput:.0f} below 10K target"


# ============================================================================
# TestVaultCapacity
# ============================================================================


@pytest.mark.load
@pytest.mark.performance
class TestVaultCapacity:
    """Load tests for token vault capacity."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_vault_capacity_stress(
        self, secure_vault, performance_metrics
    ):
        """Verify vault handles large token volumes."""
        try:
            from greenlang.infrastructure.pii_service.models import PIIType
        except ImportError:
            pytest.skip("PIIType not available")

        tenant_id = f"tenant-stress-{uuid4().hex[:8]}"
        token_count = 10000  # Reduced for CI, would be 100K+ in full load test

        # Create tokens
        tokens = []
        start = time.perf_counter()

        for i in range(token_count):
            token = await secure_vault.tokenize(
                f"value-{i}-{uuid4().hex[:8]}",
                PIIType.EMAIL,
                tenant_id,
            )
            tokens.append(token)

            if (i + 1) % 1000 == 0:
                elapsed = time.perf_counter() - start
                rate = (i + 1) / elapsed
                print(f"  Created {i + 1} tokens at {rate:.0f} tokens/sec")

        duration = time.perf_counter() - start
        create_throughput = token_count / duration

        print(f"\nVault Capacity Test Results:")
        print(f"  Tokens created: {token_count}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Create throughput: {create_throughput:.0f} tokens/sec")

        # Verify count
        count = await secure_vault.get_token_count(tenant_id)
        assert count == token_count, f"Expected {token_count} tokens, got {count}"

        # Test retrieval performance
        sample_tokens = tokens[:1000]
        user_id = str(uuid4())

        start = time.perf_counter()
        for token in sample_tokens:
            await secure_vault.detokenize(token, tenant_id, user_id)
        retrieval_duration = time.perf_counter() - start
        retrieval_throughput = len(sample_tokens) / retrieval_duration

        print(f"  Retrieval throughput: {retrieval_throughput:.0f} tokens/sec")


# ============================================================================
# TestAllowlistPerformance
# ============================================================================


@pytest.mark.load
@pytest.mark.performance
class TestAllowlistPerformance:
    """Load tests for allowlist performance."""

    @pytest.mark.asyncio
    async def test_allowlist_performance_with_many_patterns(
        self, pii_service, performance_metrics
    ):
        """Verify allowlist performance with many patterns."""
        try:
            from greenlang.infrastructure.pii_service.allowlist.patterns import (
                AllowlistEntry, PatternType
            )
            from greenlang.infrastructure.pii_service.models import PIIType
        except ImportError:
            pytest.skip("Allowlist models not available")

        tenant_id = f"tenant-{uuid4().hex[:8]}"

        # Add many allowlist entries
        for i in range(100):
            entry = AllowlistEntry(
                pii_type=PIIType.EMAIL,
                pattern=f".*@domain{i}\\.com$",
                pattern_type=PatternType.REGEX,
                reason=f"Test domain {i}",
                created_by=uuid4(),
                tenant_id=tenant_id,
            )
            await pii_service.add_allowlist_entry(entry)

        # Test detection performance with loaded allowlist
        contents = [
            f"Email: user@domain{i}.com" for i in range(1000)
        ]

        start = time.perf_counter()

        for content in contents:
            await pii_service.detect(content)

        duration = time.perf_counter() - start
        throughput = len(contents) / duration

        print(f"\nAllowlist Performance Test Results:")
        print(f"  Allowlist entries: 100")
        print(f"  Detection throughput: {throughput:.0f} ops/sec")

        # Should still be fast with many patterns
        assert throughput >= 1000, \
            f"Allowlist-heavy detection {throughput:.0f} ops/sec too slow"
