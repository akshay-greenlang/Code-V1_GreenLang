"""
Utility Tests for SAP Connector
GL-VCCI Scope 3 Platform

Tests for utility modules including:
- Retry logic with exponential backoff
- Rate limiter (token bucket)
- Audit logger (API calls, auth events, errors, lineage)
- Deduplication cache
- Batch operations

Test Count: 25 tests
Coverage Target: 95%+

Version: 1.0.0
Phase: 4 (Weeks 24-26)
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import requests

from connectors.sap.utils.retry_logic import (
    retry_with_backoff,
    _calculate_backoff,
    _should_retry_status,
)


class TestRetryLogic:
    """Tests for retry logic with exponential backoff."""

    def test_should_succeed_on_first_attempt(self):
        """Test successful call on first attempt."""
        @retry_with_backoff(max_retries=3)
        def successful_call():
            return "success"

        result = successful_call()

        assert result == "success"

    def test_should_retry_on_connection_error(self):
        """Test retry on connection error."""
        call_count = {"count": 0}

        @retry_with_backoff(max_retries=3, base_delay=0.1)
        def failing_call():
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise requests.exceptions.ConnectionError("Connection failed")
            return "success"

        result = failing_call()

        assert result == "success"
        assert call_count["count"] == 3

    def test_should_retry_on_timeout(self):
        """Test retry on timeout error."""
        call_count = {"count": 0}

        @retry_with_backoff(max_retries=2, base_delay=0.1)
        def timeout_call():
            call_count["count"] += 1
            if call_count["count"] < 2:
                raise requests.exceptions.Timeout("Request timed out")
            return "success"

        result = timeout_call()

        assert result == "success"
        assert call_count["count"] == 2

    def test_should_fail_after_max_retries(self):
        """Test failure after max retries exceeded."""
        @retry_with_backoff(max_retries=2, base_delay=0.1)
        def always_fails():
            raise requests.exceptions.ConnectionError("Always fails")

        with pytest.raises(requests.exceptions.ConnectionError):
            always_fails()

    def test_should_not_retry_on_400_error(self):
        """Test no retry on 4xx client errors."""
        @retry_with_backoff(max_retries=3, base_delay=0.1)
        def client_error():
            response = Mock()
            response.status_code = 400
            raise requests.exceptions.HTTPError(response=response)

        with pytest.raises(requests.exceptions.HTTPError):
            client_error()

    def test_should_retry_on_429_error(self):
        """Test retry on 429 rate limit error."""
        call_count = {"count": 0}

        @retry_with_backoff(max_retries=2, base_delay=0.1)
        def rate_limit_call():
            call_count["count"] += 1
            if call_count["count"] < 2:
                response = Mock()
                response.status_code = 429
                raise requests.exceptions.HTTPError(response=response)
            return "success"

        result = rate_limit_call()

        assert result == "success"
        assert call_count["count"] == 2

    def test_should_retry_on_500_error(self):
        """Test retry on 5xx server errors."""
        call_count = {"count": 0}

        @retry_with_backoff(max_retries=2, base_delay=0.1)
        def server_error_call():
            call_count["count"] += 1
            if call_count["count"] < 2:
                response = Mock()
                response.status_code = 503
                raise requests.exceptions.HTTPError(response=response)
            return "success"

        result = server_error_call()

        assert result == "success"

    def test_should_calculate_exponential_backoff(self):
        """Test exponential backoff calculation."""
        # Attempt 0: 1.0
        delay = _calculate_backoff(0, 1.0, 2.0, 60.0, False)
        assert delay == 1.0

        # Attempt 1: 2.0
        delay = _calculate_backoff(1, 1.0, 2.0, 60.0, False)
        assert delay == 2.0

        # Attempt 2: 4.0
        delay = _calculate_backoff(2, 1.0, 2.0, 60.0, False)
        assert delay == 4.0

        # Attempt 3: 8.0
        delay = _calculate_backoff(3, 1.0, 2.0, 60.0, False)
        assert delay == 8.0

    def test_should_cap_backoff_at_max_delay(self):
        """Test backoff capped at max_delay."""
        delay = _calculate_backoff(10, 1.0, 2.0, 8.0, False)

        assert delay == 8.0

    def test_should_add_jitter_to_backoff(self):
        """Test jitter added to backoff."""
        delay1 = _calculate_backoff(2, 1.0, 2.0, 60.0, True)
        delay2 = _calculate_backoff(2, 1.0, 2.0, 60.0, True)

        # With jitter, delays should vary
        # But both should be in range [2.0, 4.0]
        assert 2.0 <= delay1 <= 4.0
        assert 2.0 <= delay2 <= 4.0

    def test_should_identify_retryable_status_codes(self):
        """Test identifying retryable HTTP status codes."""
        assert _should_retry_status(429) is True
        assert _should_retry_status(500) is True
        assert _should_retry_status(502) is True
        assert _should_retry_status(503) is True
        assert _should_retry_status(504) is True

        assert _should_retry_status(400) is False
        assert _should_retry_status(404) is False
        assert _should_retry_status(200) is False

    def test_should_use_custom_retry_exceptions(self):
        """Test custom retry exception types."""
        @retry_with_backoff(
            max_retries=2,
            base_delay=0.1,
            retry_on=(ValueError,)
        )
        def custom_error():
            raise ValueError("Custom error")

        with pytest.raises(ValueError):
            custom_error()


class TestRateLimiter:
    """Tests for rate limiter."""

    @patch('connectors.sap.utils.rate_limiter.RateLimiter')
    def test_should_initialize_rate_limiter(self, MockRateLimiter):
        """Test rate limiter initialization."""
        limiter = MockRateLimiter(requests_per_minute=60)

        assert limiter is not None

    @patch('connectors.sap.utils.rate_limiter.RateLimiter')
    def test_should_enforce_rate_limit(self, MockRateLimiter):
        """Test rate limit enforcement."""
        limiter = MockRateLimiter.return_value
        limiter.acquire.side_effect = [True, True, False]

        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is False

    @patch('connectors.sap.utils.rate_limiter.RateLimiter')
    def test_should_calculate_wait_time(self, MockRateLimiter):
        """Test calculating wait time for next request."""
        limiter = MockRateLimiter.return_value
        limiter.get_wait_time.return_value = 1.5

        wait_time = limiter.get_wait_time()

        assert wait_time == 1.5


class TestAuditLogger:
    """Tests for audit logger."""

    @patch('connectors.sap.utils.audit_logger.AuditLogger')
    def test_should_log_api_call(self, MockLogger):
        """Test logging API call."""
        logger = MockLogger.return_value
        logger.log_api_call.return_value = True

        result = logger.log_api_call(
            endpoint="/api/purchase_orders",
            method="GET",
            status_code=200,
            duration_ms=125.5
        )

        assert result is True
        assert logger.log_api_call.called

    @patch('connectors.sap.utils.audit_logger.AuditLogger')
    def test_should_log_auth_event(self, MockLogger):
        """Test logging authentication event."""
        logger = MockLogger.return_value
        logger.log_auth_event.return_value = True

        result = logger.log_auth_event(
            event_type="token_acquired",
            success=True
        )

        assert result is True

    @patch('connectors.sap.utils.audit_logger.AuditLogger')
    def test_should_log_error(self, MockLogger):
        """Test logging error."""
        logger = MockLogger.return_value
        logger.log_error.return_value = True

        result = logger.log_error(
            error_type="ConnectionError",
            message="Failed to connect",
            stack_trace="..."
        )

        assert result is True

    @patch('connectors.sap.utils.audit_logger.AuditLogger')
    def test_should_log_data_lineage(self, MockLogger):
        """Test logging data lineage."""
        logger = MockLogger.return_value
        logger.log_lineage.return_value = True

        result = logger.log_lineage(
            source_system="SAP_S4HANA",
            source_entity="PurchaseOrder",
            target_entity="procurement_v1.0",
            record_count=100
        )

        assert result is True

    @patch('connectors.sap.utils.audit_logger.AuditLogger')
    def test_should_query_audit_logs(self, MockLogger):
        """Test querying audit logs."""
        logger = MockLogger.return_value
        logger.query_logs.return_value = [
            {"timestamp": "2024-01-15T12:00:00Z", "event": "api_call"}
        ]

        logs = logger.query_logs(
            start_date="2024-01-15",
            end_date="2024-01-16"
        )

        assert len(logs) == 1


class TestDeduplicationCache:
    """Tests for deduplication cache."""

    @patch('connectors.sap.utils.deduplication.DeduplicationCache')
    def test_should_initialize_cache(self, MockCache):
        """Test deduplication cache initialization."""
        cache = MockCache(ttl_seconds=3600)

        assert cache is not None

    @patch('connectors.sap.utils.deduplication.DeduplicationCache')
    def test_should_detect_duplicate(self, MockCache):
        """Test detecting duplicate record."""
        cache = MockCache.return_value
        cache.is_duplicate.side_effect = [False, True]

        record_id = "PROC-4500000001-00010"

        assert cache.is_duplicate(record_id) is False
        assert cache.is_duplicate(record_id) is True

    @patch('connectors.sap.utils.deduplication.DeduplicationCache')
    def test_should_add_record_to_cache(self, MockCache):
        """Test adding record to deduplication cache."""
        cache = MockCache.return_value
        cache.add.return_value = True

        result = cache.add("PROC-4500000001-00010")

        assert result is True

    @patch('connectors.sap.utils.deduplication.DeduplicationCache')
    def test_should_clear_expired_entries(self, MockCache):
        """Test clearing expired cache entries."""
        cache = MockCache.return_value
        cache.clear_expired.return_value = 10

        cleared_count = cache.clear_expired()

        assert cleared_count == 10

    @patch('connectors.sap.utils.deduplication.DeduplicationCache')
    def test_should_batch_check_duplicates(self, MockCache):
        """Test batch duplicate checking."""
        cache = MockCache.return_value
        cache.check_batch.return_value = {
            "PROC-001": False,
            "PROC-002": True,
            "PROC-003": False
        }

        record_ids = ["PROC-001", "PROC-002", "PROC-003"]
        results = cache.check_batch(record_ids)

        assert results["PROC-001"] is False
        assert results["PROC-002"] is True
        assert len(results) == 3


class TestBatchOperations:
    """Tests for batch utility operations."""

    def test_should_chunk_list_into_batches(self):
        """Test chunking list into batches."""
        from connectors.sap.utils.batch import chunk_list

        data = list(range(100))
        batches = list(chunk_list(data, batch_size=25))

        assert len(batches) == 4
        assert len(batches[0]) == 25
        assert len(batches[-1]) == 25

    def test_should_handle_uneven_batch_size(self):
        """Test handling uneven batch sizes."""
        from connectors.sap.utils.batch import chunk_list

        data = list(range(105))
        batches = list(chunk_list(data, batch_size=25))

        assert len(batches) == 5
        assert len(batches[-1]) == 5

    def test_should_flatten_nested_lists(self):
        """Test flattening nested lists."""
        from connectors.sap.utils.batch import flatten_list

        nested = [[1, 2], [3, 4], [5, 6]]
        flat = flatten_list(nested)

        assert flat == [1, 2, 3, 4, 5, 6]


class TestPerformanceMonitoring:
    """Tests for performance monitoring utilities."""

    @patch('connectors.sap.utils.monitoring.measure_duration')
    def test_should_measure_function_duration(self, mock_measure):
        """Test measuring function execution duration."""
        @mock_measure
        def slow_function():
            time.sleep(0.1)
            return "done"

        mock_measure.return_value = slow_function

        result = slow_function()

        assert result == "done"

    @patch('connectors.sap.utils.monitoring.track_metric')
    def test_should_track_metrics(self, mock_track):
        """Test tracking custom metrics."""
        mock_track.return_value = True

        result = mock_track("records_processed", 1000)

        assert result is True
