# -*- coding: utf-8 -*-
"""
Unit Tests for PushGateway SDK (OBS-001 Phase 3)

Comprehensive tests for BatchJobMetrics, PushGatewayConfig, and factory functions.

Author: GreenLang Platform Team
Date: February 2026
"""

import time
import threading
from unittest.mock import MagicMock, patch, call
import pytest

from greenlang.monitoring.pushgateway import (
    BatchJobMetrics,
    PushGatewayConfig,
    PushGatewayError,
    get_pushgateway_client,
    clear_pushgateway_clients,
    create_batch_job_metrics,
    STATUS_VALUES,
    PROMETHEUS_CLIENT_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_push_to_gateway():
    """Mock push_to_gateway function."""
    with patch(
        "greenlang.monitoring.pushgateway.push_to_gateway"
    ) as mock:
        yield mock


@pytest.fixture
def mock_delete_from_gateway():
    """Mock delete_from_gateway function."""
    with patch(
        "greenlang.monitoring.pushgateway.delete_from_gateway"
    ) as mock:
        yield mock


@pytest.fixture
def metrics():
    """Create a fresh BatchJobMetrics instance."""
    # Clear singleton cache
    clear_pushgateway_clients()
    return BatchJobMetrics(
        job_name="test-job",
        pushgateway_url="http://test-pushgateway:9091",
        timeout=5.0,
        max_retries=2,
        fail_silently=True,
    )


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up singleton cache after each test."""
    yield
    clear_pushgateway_clients()


# ---------------------------------------------------------------------------
# PushGatewayConfig Tests
# ---------------------------------------------------------------------------


class TestPushGatewayConfig:
    """Tests for PushGatewayConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PushGatewayConfig()
        assert config.url == "http://pushgateway.monitoring.svc:9091"
        assert config.job_name == ""
        assert config.grouping_key == {}
        assert config.timeout == 10.0
        assert config.max_retries == 3
        assert config.retry_backoff == 1.0
        assert config.retry_jitter == 0.3
        assert config.fail_silently is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PushGatewayConfig(
            url="http://custom:9091",
            job_name="my-job",
            grouping_key={"instance": "worker-1"},
            timeout=5.0,
            max_retries=5,
            retry_backoff=2.0,
            retry_jitter=0.5,
            fail_silently=False,
        )
        assert config.url == "http://custom:9091"
        assert config.job_name == "my-job"
        assert config.grouping_key == {"instance": "worker-1"}
        assert config.timeout == 5.0
        assert config.max_retries == 5
        assert config.retry_backoff == 2.0
        assert config.retry_jitter == 0.5
        assert config.fail_silently is False

    def test_validation_timeout(self):
        """Test timeout validation."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            PushGatewayConfig(timeout=0)

        with pytest.raises(ValueError, match="timeout must be positive"):
            PushGatewayConfig(timeout=-1)

    def test_validation_max_retries(self):
        """Test max_retries validation."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            PushGatewayConfig(max_retries=-1)

    def test_validation_retry_backoff(self):
        """Test retry_backoff validation."""
        with pytest.raises(ValueError, match="retry_backoff must be positive"):
            PushGatewayConfig(retry_backoff=0)

    def test_validation_retry_jitter(self):
        """Test retry_jitter validation."""
        with pytest.raises(ValueError, match="retry_jitter must be between"):
            PushGatewayConfig(retry_jitter=-0.1)

        with pytest.raises(ValueError, match="retry_jitter must be between"):
            PushGatewayConfig(retry_jitter=1.1)

    def test_frozen(self):
        """Test config is immutable."""
        config = PushGatewayConfig()
        with pytest.raises(Exception):  # FrozenInstanceError
            config.url = "http://new:9091"


# ---------------------------------------------------------------------------
# BatchJobMetrics Initialization Tests
# ---------------------------------------------------------------------------


class TestBatchJobMetricsInit:
    """Tests for BatchJobMetrics initialization."""

    def test_basic_initialization(self):
        """Test basic initialization."""
        metrics = BatchJobMetrics("my-job")
        assert metrics.job_name == "my-job"
        assert metrics.pushgateway_url == "http://pushgateway.monitoring.svc:9091"
        assert metrics.grouping_key == {}
        assert metrics.timeout == 10.0
        assert metrics.max_retries == 3
        assert metrics.fail_silently is True

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        metrics = BatchJobMetrics(
            job_name="custom-job",
            pushgateway_url="http://custom:9091",
            grouping_key={"region": "us-east-1"},
            timeout=5.0,
            max_retries=5,
            retry_backoff=2.0,
            fail_silently=False,
        )
        assert metrics.job_name == "custom-job"
        assert metrics.pushgateway_url == "http://custom:9091"
        assert metrics.grouping_key == {"region": "us-east-1"}
        assert metrics.timeout == 5.0
        assert metrics.max_retries == 5
        assert metrics.fail_silently is False

    def test_empty_job_name_raises(self):
        """Test empty job name raises ValueError."""
        with pytest.raises(ValueError, match="job_name is required"):
            BatchJobMetrics("")

    def test_none_url_uses_default(self):
        """Test None URL uses default."""
        metrics = BatchJobMetrics("job", pushgateway_url=None)
        assert metrics.pushgateway_url == "http://pushgateway.monitoring.svc:9091"

    def test_initial_status(self):
        """Test initial status is idle."""
        metrics = BatchJobMetrics("job")
        assert metrics.get_status() == "idle"


# ---------------------------------------------------------------------------
# BatchJobMetrics Push Tests
# ---------------------------------------------------------------------------


class TestBatchJobMetricsPush:
    """Tests for push() method."""

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_push_success(self, metrics, mock_push_to_gateway):
        """Test successful push."""
        metrics.push()

        mock_push_to_gateway.assert_called_once_with(
            metrics.pushgateway_url,
            job=metrics.job_name,
            registry=metrics._registry,
            grouping_key=metrics.grouping_key,
            timeout=metrics.timeout,
        )

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_push_retry_on_failure(self, metrics, mock_push_to_gateway):
        """Test push retries on failure."""
        mock_push_to_gateway.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            None,  # Success on third attempt
        ]

        # Should not raise because fail_silently=True
        metrics.push()

        assert mock_push_to_gateway.call_count == 3

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_push_max_retries_exceeded(self, metrics, mock_push_to_gateway):
        """Test push fails after max retries."""
        mock_push_to_gateway.side_effect = Exception("Always fails")

        # Should not raise because fail_silently=True
        metrics.push()

        # max_retries=2, so 3 total attempts
        assert mock_push_to_gateway.call_count == 3

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_push_raises_when_not_silent(self, mock_push_to_gateway):
        """Test push raises PushGatewayError when fail_silently=False."""
        metrics = BatchJobMetrics(
            "test-job",
            fail_silently=False,
            max_retries=1,
        )
        mock_push_to_gateway.side_effect = Exception("Always fails")

        with pytest.raises(PushGatewayError) as exc_info:
            metrics.push()

        assert "test-job" in str(exc_info.value)
        assert exc_info.value.attempts == 2  # max_retries + 1


# ---------------------------------------------------------------------------
# BatchJobMetrics Delete Tests
# ---------------------------------------------------------------------------


class TestBatchJobMetricsDelete:
    """Tests for delete() method."""

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_delete_success(self, metrics, mock_delete_from_gateway):
        """Test successful delete."""
        metrics.delete()

        mock_delete_from_gateway.assert_called_once_with(
            metrics.pushgateway_url,
            job=metrics.job_name,
            grouping_key=metrics.grouping_key,
            timeout=metrics.timeout,
        )

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_delete_failure_silent(self, metrics, mock_delete_from_gateway):
        """Test delete failure is silent."""
        mock_delete_from_gateway.side_effect = Exception("Delete failed")

        # Should not raise because fail_silently=True
        metrics.delete()


# ---------------------------------------------------------------------------
# BatchJobMetrics Track Duration Tests
# ---------------------------------------------------------------------------


class TestBatchJobMetricsTrackDuration:
    """Tests for track_duration() context manager."""

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_track_duration_success(self, metrics, mock_push_to_gateway):
        """Test duration tracking on success."""
        with metrics.track_duration():
            time.sleep(0.01)  # Small delay

        assert metrics.get_status() == "success"

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_track_duration_failure(self, metrics, mock_push_to_gateway):
        """Test duration tracking on failure."""
        with pytest.raises(ValueError):
            with metrics.track_duration():
                raise ValueError("Test error")

        assert metrics.get_status() == "failed"

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_track_duration_custom_status(self, metrics, mock_push_to_gateway):
        """Test custom status on success."""
        with metrics.track_duration(status_on_success="completed"):
            pass

        # The final status should still be success since we only allow
        # idle, running, success, failed
        assert metrics.get_status() == "completed"


# ---------------------------------------------------------------------------
# BatchJobMetrics Record Methods Tests
# ---------------------------------------------------------------------------


class TestBatchJobMetricsRecordMethods:
    """Tests for record methods."""

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_record_success(self, metrics):
        """Test record_success."""
        metrics.record_success()
        assert metrics.get_status() == "success"

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_record_failure(self, metrics):
        """Test record_failure."""
        metrics.record_failure("TestError")
        assert metrics.get_status() == "failed"

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_record_records(self, metrics):
        """Test record_records."""
        # Should not raise
        metrics.record_records(100, "processed")
        metrics.record_records(50, "imported")

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_record_records_zero_ignored(self, metrics):
        """Test zero count is ignored."""
        # Should not raise and should not record
        metrics.record_records(0, "processed")
        metrics.record_records(-1, "processed")

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_record_failed_records(self, metrics):
        """Test record_failed_records."""
        metrics.record_failed_records(5, "validation")

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_record_error(self, metrics):
        """Test record_error."""
        metrics.record_error("ValidationError")
        # Status should not change (non-fatal error)
        assert metrics.get_status() == "idle"

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_record_retry(self, metrics):
        """Test record_retry."""
        metrics.record_retry()
        metrics.record_retry()


# ---------------------------------------------------------------------------
# BatchJobMetrics Status Tests
# ---------------------------------------------------------------------------


class TestBatchJobMetricsStatus:
    """Tests for status management."""

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_set_status_valid(self, metrics):
        """Test setting valid status."""
        for status in ["idle", "running", "success", "failed"]:
            metrics.set_status(status)
            assert metrics.get_status() == status

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_set_status_invalid(self, metrics):
        """Test setting invalid status raises."""
        with pytest.raises(ValueError, match="Invalid status"):
            metrics.set_status("unknown")

    def test_status_values(self):
        """Test STATUS_VALUES mapping."""
        assert STATUS_VALUES["idle"] == 0
        assert STATUS_VALUES["running"] == 1
        assert STATUS_VALUES["success"] == 2
        assert STATUS_VALUES["failed"] == 3


# ---------------------------------------------------------------------------
# BatchJobMetrics Utility Tests
# ---------------------------------------------------------------------------


class TestBatchJobMetricsUtility:
    """Tests for utility methods."""

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_reset(self, metrics):
        """Test reset method."""
        metrics.set_status("running")
        metrics.record_records(100, "processed")
        metrics.reset()

        assert metrics.get_status() == "idle"

    def test_get_info(self, metrics):
        """Test get_info method."""
        info = metrics.get_info()

        assert info["job_name"] == "test-job"
        assert info["pushgateway_url"] == "http://test-pushgateway:9091"
        assert info["grouping_key"] == {}
        assert info["current_status"] == "idle"
        assert "metrics_initialized" in info
        assert "prometheus_available" in info


# ---------------------------------------------------------------------------
# Thread Safety Tests
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Tests for thread safety."""

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_concurrent_record_records(self, metrics):
        """Test concurrent record_records calls."""
        def record():
            for _ in range(100):
                metrics.record_records(1, "processed")

        threads = [threading.Thread(target=record) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should complete without errors


# ---------------------------------------------------------------------------
# Factory Function Tests
# ---------------------------------------------------------------------------


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_get_pushgateway_client_creates_new(self):
        """Test get_pushgateway_client creates new instance."""
        client = get_pushgateway_client("test-job-1")
        assert client is not None
        assert client.job_name == "test-job-1"

    def test_get_pushgateway_client_returns_cached(self):
        """Test get_pushgateway_client returns cached instance."""
        client1 = get_pushgateway_client("test-job-2")
        client2 = get_pushgateway_client("test-job-2")
        assert client1 is client2

    def test_get_pushgateway_client_with_config(self):
        """Test get_pushgateway_client with config."""
        config = PushGatewayConfig(
            job_name="config-job",
            url="http://custom:9091",
            timeout=5.0,
        )
        client = get_pushgateway_client(config=config)
        assert client.job_name == "config-job"
        assert client.pushgateway_url == "http://custom:9091"
        assert client.timeout == 5.0

    def test_get_pushgateway_client_no_job_name_raises(self):
        """Test get_pushgateway_client with no job name raises."""
        with pytest.raises(ValueError, match="job_name is required"):
            get_pushgateway_client()

    def test_clear_pushgateway_clients(self):
        """Test clear_pushgateway_clients."""
        client1 = get_pushgateway_client("test-job-3")
        clear_pushgateway_clients()
        client2 = get_pushgateway_client("test-job-3")
        assert client1 is not client2

    def test_create_batch_job_metrics(self):
        """Test create_batch_job_metrics."""
        metrics = create_batch_job_metrics(
            "created-job",
            grouping_key={"region": "us-west-2"},
        )
        assert metrics.job_name == "created-job"
        assert metrics.grouping_key == {"region": "us-west-2"}

    def test_create_batch_job_metrics_env_override(self, monkeypatch):
        """Test create_batch_job_metrics uses environment variable."""
        monkeypatch.setenv("PUSHGATEWAY_URL", "http://env-pushgateway:9091")
        metrics = create_batch_job_metrics("env-job")
        assert metrics.pushgateway_url == "http://env-pushgateway:9091"


# ---------------------------------------------------------------------------
# PushGatewayError Tests
# ---------------------------------------------------------------------------


class TestPushGatewayError:
    """Tests for PushGatewayError exception."""

    def test_error_attributes(self):
        """Test error has correct attributes."""
        original = ValueError("Original error")
        error = PushGatewayError("Test message", attempts=3, last_error=original)

        assert "Test message" in str(error)
        assert error.attempts == 3
        assert error.last_error is original

    def test_error_without_last_error(self):
        """Test error without last_error."""
        error = PushGatewayError("Test message", attempts=1)
        assert error.last_error is None


# ---------------------------------------------------------------------------
# Integration-like Tests (no actual network)
# ---------------------------------------------------------------------------


class TestIntegrationPatterns:
    """Tests demonstrating integration patterns."""

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_basic_usage_pattern(self, mock_push_to_gateway):
        """Test basic usage pattern from docstring."""
        metrics = BatchJobMetrics("my-batch-job")

        with metrics.track_duration():
            # Simulate work
            time.sleep(0.01)
            metrics.record_records(100, "processed")

        metrics.push()

        assert metrics.get_status() == "success"
        mock_push_to_gateway.assert_called()

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_grouping_key_pattern(self, mock_push_to_gateway):
        """Test grouping key pattern from docstring."""
        metrics = BatchJobMetrics(
            "data-import",
            grouping_key={"instance": "worker-1", "region": "us-east-1"},
        )

        with metrics.track_duration():
            pass

        metrics.push()

        # Verify grouping key was passed
        call_args = mock_push_to_gateway.call_args
        assert call_args.kwargs["grouping_key"] == {
            "instance": "worker-1",
            "region": "us-east-1",
        }

    @pytest.mark.skipif(
        not PROMETHEUS_CLIENT_AVAILABLE,
        reason="prometheus_client not installed"
    )
    def test_error_handling_pattern(self, mock_push_to_gateway):
        """Test error handling pattern from docstring."""
        metrics = BatchJobMetrics("error-job")

        try:
            with metrics.track_duration():
                raise ValueError("Something went wrong")
        except ValueError:
            pass  # Expected

        assert metrics.get_status() == "failed"
