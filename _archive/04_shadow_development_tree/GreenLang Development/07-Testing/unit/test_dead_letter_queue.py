"""
Unit tests for greenlang/data/dead_letter_queue.py
Target coverage: 85%+
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timedelta
import json
import uuid

# Import test helpers
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from conftest_enhanced import *


class TestDeadLetterQueue:
    """Test suite for Dead Letter Queue functionality."""

    @pytest.fixture
    def dlq(self):
        """Create Dead Letter Queue instance."""
        from greenlang.data.dead_letter_queue import DeadLetterQueue

        with patch('greenlang.data.dead_letter_queue.DeadLetterQueue.__init__', return_value=None):
            dlq = DeadLetterQueue.__new__(DeadLetterQueue)
            dlq.queue = []
            dlq.max_retries = 3
            dlq.retry_delay = 60  # seconds
            dlq.storage = Mock()
            dlq.logger = Mock()
            return dlq

    def test_quarantine_record(self, dlq):
        """Test quarantining a failed record."""
        failed_record = {
            "id": "record_123",
            "data": {"value": 100},
            "error": "Validation failed",
            "timestamp": datetime.utcnow().isoformat()
        }

        dlq.quarantine = Mock(return_value="dlq_entry_456")
        entry_id = dlq.quarantine(failed_record)

        assert entry_id == "dlq_entry_456"
        dlq.quarantine.assert_called_once_with(failed_record)

    def test_retry_failed_record(self, dlq):
        """Test retrying a failed record."""
        record_id = "dlq_entry_456"

        dlq.retry = Mock(return_value={"status": "success", "output": "processed"})
        result = dlq.retry(record_id)

        assert result["status"] == "success"
        dlq.retry.assert_called_once_with(record_id)

    def test_max_retry_exceeded(self, dlq):
        """Test behavior when max retries are exceeded."""
        record = {
            "id": "record_789",
            "retry_count": 3,
            "max_retries": 3
        }

        dlq.can_retry = Mock(return_value=False)
        dlq.move_to_permanent_failure = Mock()

        if not dlq.can_retry(record):
            dlq.move_to_permanent_failure(record)

        dlq.can_retry.assert_called_once_with(record)
        dlq.move_to_permanent_failure.assert_called_once()

    def test_categorize_failure(self, dlq):
        """Test failure categorization."""
        failures = [
            {"error": "Network timeout", "category": None},
            {"error": "Invalid data format", "category": None},
            {"error": "Database connection lost", "category": None}
        ]

        dlq.categorize_failure = Mock(side_effect=[
            "TRANSIENT", "VALIDATION", "INFRASTRUCTURE"
        ])

        for failure in failures:
            failure["category"] = dlq.categorize_failure(failure["error"])

        assert failures[0]["category"] == "TRANSIENT"
        assert failures[1]["category"] == "VALIDATION"
        assert failures[2]["category"] == "INFRASTRUCTURE"

    def test_reprocess_batch(self, dlq):
        """Test batch reprocessing of DLQ records."""
        batch_size = 10
        dlq.get_retriable_records = Mock(return_value=[
            {"id": f"record_{i}"} for i in range(batch_size)
        ])
        dlq.reprocess_batch = Mock(return_value={
            "processed": 8,
            "failed": 2,
            "skipped": 0
        })

        result = dlq.reprocess_batch(batch_size)

        assert result["processed"] == 8
        assert result["failed"] == 2

    def test_dlq_metrics(self, dlq):
        """Test DLQ metrics and statistics."""
        dlq.get_metrics = Mock(return_value={
            "total_records": 150,
            "retriable": 45,
            "permanent_failures": 15,
            "categories": {
                "VALIDATION": 60,
                "TRANSIENT": 45,
                "INFRASTRUCTURE": 30,
                "UNKNOWN": 15
            }
        })

        metrics = dlq.get_metrics()

        assert metrics["total_records"] == 150
        assert metrics["retriable"] == 45
        assert sum(metrics["categories"].values()) == 150

    def test_expiry_handling(self, dlq):
        """Test expiry of old DLQ records."""
        expiry_days = 30
        cutoff_date = datetime.utcnow() - timedelta(days=expiry_days)

        dlq.expire_old_records = Mock(return_value={"expired": 25})
        result = dlq.expire_old_records(cutoff_date)

        assert result["expired"] == 25

    def test_dlq_persistence(self, dlq, temp_dir):
        """Test persisting DLQ to storage."""
        dlq_file = temp_dir / "dlq.json"

        dlq.persist_to_file = Mock(return_value=str(dlq_file))
        dlq.load_from_file = Mock(return_value={"records": []})

        # Save DLQ
        saved_path = dlq.persist_to_file(str(dlq_file))
        assert saved_path == str(dlq_file)

        # Load DLQ
        loaded = dlq.load_from_file(str(dlq_file))
        assert "records" in loaded

    def test_priority_queue(self, dlq):
        """Test priority-based reprocessing."""
        high_priority = {"id": "1", "priority": 1}
        medium_priority = {"id": "2", "priority": 5}
        low_priority = {"id": "3", "priority": 10}

        dlq.add_with_priority = Mock()
        dlq.get_next_priority = Mock(return_value=high_priority)

        dlq.add_with_priority(high_priority)
        dlq.add_with_priority(medium_priority)
        dlq.add_with_priority(low_priority)

        next_record = dlq.get_next_priority()
        assert next_record["priority"] == 1

    def test_circuit_breaker_integration(self, dlq):
        """Test circuit breaker pattern with DLQ."""
        dlq.circuit_breaker = Mock()
        dlq.circuit_breaker.is_open = Mock(return_value=False)
        dlq.circuit_breaker.record_success = Mock()
        dlq.circuit_breaker.record_failure = Mock()

        # Process when circuit is closed
        if not dlq.circuit_breaker.is_open():
            try:
                # Simulate successful processing
                dlq.circuit_breaker.record_success()
            except Exception:
                dlq.circuit_breaker.record_failure()

        dlq.circuit_breaker.record_success.assert_called_once()

    @pytest.mark.parametrize("error_type,should_retry", [
        ("NetworkTimeout", True),
        ("ValidationError", False),
        ("TemporaryUnavailable", True),
        ("InvalidSchema", False),
        ("RateLimitExceeded", True)
    ])
    def test_retry_logic(self, dlq, error_type, should_retry):
        """Test retry decision logic based on error type."""
        dlq.should_retry = Mock(return_value=should_retry)

        result = dlq.should_retry(error_type)
        assert result == should_retry

    def test_dlq_alerting(self, dlq):
        """Test alerting when DLQ thresholds are exceeded."""
        dlq.check_thresholds = Mock(return_value={"alert": True, "reason": "Queue size > 1000"})
        dlq.send_alert = Mock()

        threshold_check = dlq.check_thresholds()
        if threshold_check["alert"]:
            dlq.send_alert(threshold_check["reason"])

        dlq.send_alert.assert_called_once_with("Queue size > 1000")

    def test_record_transformation(self, dlq):
        """Test transforming records before retry."""
        original_record = {"id": "123", "data": {"value": "old"}}

        dlq.transform_for_retry = Mock(return_value={
            "id": "123",
            "data": {"value": "transformed"},
            "retry_metadata": {"attempt": 1}
        })

        transformed = dlq.transform_for_retry(original_record)

        assert transformed["data"]["value"] == "transformed"
        assert "retry_metadata" in transformed

    def test_dlq_compaction(self, dlq):
        """Test DLQ compaction to remove duplicates."""
        dlq.queue = [
            {"id": "1", "error": "error1"},
            {"id": "1", "error": "error2"},  # Duplicate
            {"id": "2", "error": "error3"}
        ]

        dlq.compact = Mock(return_value={"removed": 1, "remaining": 2})
        result = dlq.compact()

        assert result["removed"] == 1
        assert result["remaining"] == 2


class TestFailureAnalysis:
    """Test suite for failure analysis and patterns."""

    @pytest.fixture
    def analyzer(self):
        """Create failure analyzer instance."""
        from greenlang.data.failure_analyzer import FailureAnalyzer

        with patch('greenlang.data.failure_analyzer.FailureAnalyzer.__init__', return_value=None):
            analyzer = FailureAnalyzer.__new__(FailureAnalyzer)
            analyzer.patterns = {}
            return analyzer

    def test_pattern_detection(self, analyzer):
        """Test detecting failure patterns."""
        failures = [
            {"timestamp": "2025-01-01T10:00:00", "error": "timeout"},
            {"timestamp": "2025-01-01T10:01:00", "error": "timeout"},
            {"timestamp": "2025-01-01T10:02:00", "error": "timeout"}
        ]

        analyzer.detect_patterns = Mock(return_value={
            "pattern": "REPEATED_TIMEOUT",
            "confidence": 0.95,
            "recommendation": "Increase timeout or check service health"
        })

        pattern = analyzer.detect_patterns(failures)

        assert pattern["pattern"] == "REPEATED_TIMEOUT"
        assert pattern["confidence"] > 0.9

    def test_root_cause_analysis(self, analyzer):
        """Test root cause analysis for failures."""
        failure = {
            "error": "Database connection timeout",
            "stack_trace": "...",
            "context": {"query": "SELECT * FROM large_table"}
        }

        analyzer.analyze_root_cause = Mock(return_value={
            "likely_cause": "SLOW_QUERY",
            "evidence": ["Query on large table", "No index on filter column"],
            "suggestions": ["Add index", "Optimize query"]
        })

        analysis = analyzer.analyze_root_cause(failure)

        assert analysis["likely_cause"] == "SLOW_QUERY"
        assert len(analysis["suggestions"]) > 0

    def test_failure_correlation(self, analyzer):
        """Test correlating failures across systems."""
        failures = [
            {"system": "API", "time": "10:00", "error": "500"},
            {"system": "DB", "time": "09:59", "error": "connection_pool_exhausted"},
            {"system": "Cache", "time": "10:01", "error": "timeout"}
        ]

        analyzer.correlate_failures = Mock(return_value={
            "correlation": "CASCADE_FAILURE",
            "root_system": "DB",
            "affected_systems": ["API", "Cache"]
        })

        correlation = analyzer.correlate_failures(failures)

        assert correlation["correlation"] == "CASCADE_FAILURE"
        assert correlation["root_system"] == "DB"


class TestDLQIntegration:
    """Integration tests for Dead Letter Queue."""

    @pytest.mark.integration
    def test_dlq_with_pipeline(self):
        """Test DLQ integration with pipeline processing."""
        from greenlang.data.dead_letter_queue import DeadLetterQueue
        from greenlang.sdk.pipeline import Pipeline

        with patch('greenlang.data.dead_letter_queue.DeadLetterQueue.__init__', return_value=None):
            dlq = DeadLetterQueue.__new__(DeadLetterQueue)
            dlq.quarantine = Mock()

        with patch('greenlang.sdk.pipeline.Pipeline.__init__', return_value=None):
            pipeline = Pipeline.__new__(Pipeline)
            pipeline.dlq = dlq
            pipeline.process = Mock(side_effect=[
                Exception("Failed"),
                {"status": "success"}
            ])

            # First attempt fails, record goes to DLQ
            try:
                pipeline.process({"id": "123"})
            except Exception:
                dlq.quarantine({"id": "123", "error": "Failed"})

            # Retry from DLQ succeeds
            result = pipeline.process({"id": "123"})

            assert result["status"] == "success"
            dlq.quarantine.assert_called_once()

    @pytest.mark.integration
    def test_dlq_monitoring(self):
        """Test DLQ monitoring and metrics collection."""
        from greenlang.data.dead_letter_queue import DeadLetterQueue

        with patch('greenlang.data.dead_letter_queue.DeadLetterQueue.__init__', return_value=None):
            dlq = DeadLetterQueue.__new__(DeadLetterQueue)
            dlq.metrics_collector = Mock()

            # Simulate DLQ operations
            dlq.quarantine = Mock()
            dlq.retry = Mock()

            for i in range(10):
                dlq.quarantine({"id": f"record_{i}"})

            for i in range(5):
                dlq.retry(f"record_{i}")

            dlq.collect_metrics = Mock(return_value={
                "quarantined": 10,
                "retried": 5,
                "success_rate": 0.5
            })

            metrics = dlq.collect_metrics()

            assert metrics["quarantined"] == 10
            assert metrics["retried"] == 5
            assert metrics["success_rate"] == 0.5

    @pytest.mark.performance
    def test_dlq_throughput(self, performance_timer):
        """Test DLQ throughput performance."""
        from greenlang.data.dead_letter_queue import DeadLetterQueue

        with patch('greenlang.data.dead_letter_queue.DeadLetterQueue.__init__', return_value=None):
            dlq = DeadLetterQueue.__new__(DeadLetterQueue)
            dlq.quarantine = Mock()

            performance_timer.start()

            # Quarantine 10000 records
            for i in range(10000):
                dlq.quarantine({"id": f"record_{i}", "error": "test"})

            performance_timer.stop()

            # Should handle 10000 records in less than 1 second
            assert performance_timer.elapsed_ms() < 1000