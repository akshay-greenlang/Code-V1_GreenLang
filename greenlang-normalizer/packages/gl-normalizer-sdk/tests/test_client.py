"""
Unit tests for NormalizerClient.
"""

import pytest
from typing import Any, Dict
from unittest.mock import patch, MagicMock

import httpx
from pytest_httpx import HTTPXMock

from gl_normalizer import (
    NormalizerClient,
    NormalizeRequest,
    BatchMode,
    PolicyMode,
    ClientConfig,
    NormalizeResult,
    BatchResult,
    Job,
    Vocabulary,
    EntityResult,
    ConfigurationError,
    ValidationError,
    ConversionError,
    RateLimitError,
    APIError,
)


class TestNormalizerClientInit:
    """Test NormalizerClient initialization."""

    def test_init_with_api_key(self, api_key: str) -> None:
        """Test initialization with valid API key."""
        client = NormalizerClient(api_key=api_key)
        assert client.api_key == api_key
        assert client.base_url == "https://api.greenlang.io"
        client.close()

    def test_init_with_custom_base_url(self, api_key: str, base_url: str) -> None:
        """Test initialization with custom base URL."""
        client = NormalizerClient(api_key=api_key, base_url=base_url)
        assert client.base_url == base_url
        client.close()

    def test_init_with_config(self, api_key: str) -> None:
        """Test initialization with custom config."""
        config = ClientConfig(timeout=60.0, max_retries=5)
        client = NormalizerClient(api_key=api_key, config=config)
        assert client.config.timeout == 60.0
        assert client.config.max_retries == 5
        client.close()

    def test_init_empty_api_key_raises(self) -> None:
        """Test that empty API key raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="API key is required"):
            NormalizerClient(api_key="")

    def test_init_whitespace_api_key_raises(self) -> None:
        """Test that whitespace-only API key raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="API key is required"):
            NormalizerClient(api_key="   ")

    def test_context_manager(self, api_key: str) -> None:
        """Test context manager protocol."""
        with NormalizerClient(api_key=api_key) as client:
            assert client.api_key == api_key


class TestNormalizerClientNormalize:
    """Test NormalizerClient.normalize method."""

    def test_normalize_basic(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        normalize_response: Dict[str, Any],
    ) -> None:
        """Test basic normalization."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.greenlang.io/v1/normalize",
            json=normalize_response,
        )

        with NormalizerClient(api_key=api_key) as client:
            result = client.normalize(100.0, "kWh", target_unit="MJ")

        assert isinstance(result, NormalizeResult)
        assert result.canonical_value == 360.0
        assert result.canonical_unit == "MJ"
        assert result.raw_value == 100.0
        assert result.raw_unit == "kWh"

    def test_normalize_with_expected_dimension(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        normalize_response: Dict[str, Any],
    ) -> None:
        """Test normalization with expected dimension."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.greenlang.io/v1/normalize",
            json=normalize_response,
        )

        with NormalizerClient(api_key=api_key) as client:
            result = client.normalize(
                100.0,
                "kWh",
                target_unit="MJ",
                expected_dimension="energy",
                field="energy_consumption",
            )

        assert result.dimension == "energy"
        assert result.field == "energy_consumption"

    def test_normalize_with_vocabulary_version(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        normalize_response: Dict[str, Any],
    ) -> None:
        """Test normalization with pinned vocabulary version."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.greenlang.io/v1/normalize",
            json=normalize_response,
        )

        with NormalizerClient(api_key=api_key) as client:
            result = client.normalize(
                100.0,
                "kWh",
                vocabulary_version="2026.01.0",
            )

        # Verify request included vocabulary_version
        request = httpx_mock.get_request()
        assert request is not None
        import json
        body = json.loads(request.content)
        assert body.get("vocabulary_version") == "2026.01.0"

    def test_normalize_validation_error(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        validation_error_response: Dict[str, Any],
    ) -> None:
        """Test normalization with validation error."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.greenlang.io/v1/normalize",
            json=validation_error_response,
            status_code=400,
        )

        with NormalizerClient(api_key=api_key) as client:
            with pytest.raises(ValidationError) as exc_info:
                client.normalize(100.0, "kg", expected_dimension="energy")

        assert exc_info.value.code == "GLNORM-E200"
        assert "Dimension mismatch" in exc_info.value.message

    def test_normalize_rate_limit_error(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        rate_limit_error_response: Dict[str, Any],
    ) -> None:
        """Test normalization with rate limit error."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.greenlang.io/v1/normalize",
            json=rate_limit_error_response,
            status_code=429,
            headers={"Retry-After": "60"},
        )

        with NormalizerClient(api_key=api_key) as client:
            with pytest.raises(RateLimitError) as exc_info:
                client.normalize(100.0, "kWh")

        assert exc_info.value.retry_after == 60.0


class TestNormalizerClientBatch:
    """Test NormalizerClient.normalize_batch method."""

    def test_normalize_batch_basic(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        batch_response: Dict[str, Any],
    ) -> None:
        """Test basic batch normalization."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.greenlang.io/v1/normalize/batch",
            json=batch_response,
        )

        requests = [
            NormalizeRequest(value=100.0, unit="kWh"),
            NormalizeRequest(value=50.0, unit="kg"),
        ]

        with NormalizerClient(api_key=api_key) as client:
            result = client.normalize_batch(requests)

        assert isinstance(result, BatchResult)
        assert result.summary.total == 2
        assert result.summary.success == 2
        assert result.summary.failed == 0
        assert len(result.results) == 2

    def test_normalize_batch_with_mode(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        batch_response: Dict[str, Any],
    ) -> None:
        """Test batch normalization with batch mode."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.greenlang.io/v1/normalize/batch",
            json=batch_response,
        )

        requests = [NormalizeRequest(value=100.0, unit="kWh")]

        with NormalizerClient(api_key=api_key) as client:
            result = client.normalize_batch(
                requests,
                mode=BatchMode.ALL_OR_NOTHING,
                policy_mode=PolicyMode.STRICT,
            )

        # Verify request included batch_mode
        request = httpx_mock.get_request()
        assert request is not None
        import json
        body = json.loads(request.content)
        assert body.get("batch_mode") == "ALL_OR_NOTHING"
        assert body.get("policy_mode") == "STRICT"

    def test_normalize_batch_exceeds_limit(self, api_key: str) -> None:
        """Test batch normalization exceeds limit."""
        requests = [NormalizeRequest(value=i, unit="kWh") for i in range(10001)]

        with NormalizerClient(api_key=api_key) as client:
            with pytest.raises(ConfigurationError, match="exceeds maximum"):
                client.normalize_batch(requests)


class TestNormalizerClientJobs:
    """Test NormalizerClient job methods."""

    def test_create_job(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        job_response: Dict[str, Any],
    ) -> None:
        """Test creating a job."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.greenlang.io/v1/jobs",
            json=job_response,
        )

        requests = [NormalizeRequest(value=i, unit="kWh") for i in range(1000)]

        with NormalizerClient(api_key=api_key) as client:
            job = client.create_job(requests)

        assert isinstance(job, Job)
        assert job.job_id == "job-abc123"
        assert job.status.value == "pending"
        assert job.total_items == 1000

    def test_get_job(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        job_response: Dict[str, Any],
    ) -> None:
        """Test getting job status."""
        job_response["status"] = "processing"
        job_response["progress"] = 50.0
        job_response["processed_items"] = 500

        httpx_mock.add_response(
            method="GET",
            url="https://api.greenlang.io/v1/jobs/job-abc123",
            json=job_response,
        )

        with NormalizerClient(api_key=api_key) as client:
            job = client.get_job("job-abc123")

        assert job.status.value == "processing"
        assert job.progress == 50.0
        assert job.processed_items == 500

    def test_cancel_job(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        job_response: Dict[str, Any],
    ) -> None:
        """Test cancelling a job."""
        job_response["status"] = "cancelled"

        httpx_mock.add_response(
            method="POST",
            url="https://api.greenlang.io/v1/jobs/job-abc123/cancel",
            json=job_response,
        )

        with NormalizerClient(api_key=api_key) as client:
            job = client.cancel_job("job-abc123")

        assert job.status.value == "cancelled"


class TestNormalizerClientEntities:
    """Test NormalizerClient entity resolution methods."""

    def test_resolve_entity(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        entity_response: Dict[str, Any],
    ) -> None:
        """Test entity resolution."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.greenlang.io/v1/resolve",
            json=entity_response,
        )

        with NormalizerClient(api_key=api_key) as client:
            result = client.resolve_entity("Nat Gas", entity_type="fuel")

        assert isinstance(result, EntityResult)
        assert result.reference_id == "GL-FUEL-NATGAS"
        assert result.canonical_name == "Natural gas"
        assert result.confidence == 1.0
        assert result.match_method.value == "alias"


class TestNormalizerClientVocabularies:
    """Test NormalizerClient vocabulary methods."""

    def test_list_vocabularies(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        vocabularies_response: Dict[str, Any],
    ) -> None:
        """Test listing vocabularies."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.greenlang.io/v1/vocabularies",
            json=vocabularies_response,
        )

        with NormalizerClient(api_key=api_key) as client:
            vocabs = client.list_vocabularies()

        assert len(vocabs) == 2
        assert isinstance(vocabs[0], Vocabulary)
        assert vocabs[0].vocabulary_id == "fuels"
        assert vocabs[0].entity_count == 150

    def test_list_vocabularies_filtered(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        vocabularies_response: Dict[str, Any],
    ) -> None:
        """Test listing vocabularies with filter."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.greenlang.io/v1/vocabularies",
            json=vocabularies_response,
        )

        with NormalizerClient(api_key=api_key) as client:
            vocabs = client.list_vocabularies(entity_type="fuel")

        # Verify query parameter
        request = httpx_mock.get_request()
        assert request is not None
        assert "entity_type=fuel" in str(request.url)


class TestNormalizerClientCaching:
    """Test NormalizerClient caching behavior."""

    def test_cache_enabled(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        normalize_response: Dict[str, Any],
    ) -> None:
        """Test that caching works when enabled."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.greenlang.io/v1/normalize",
            json=normalize_response,
        )

        config = ClientConfig(enable_cache=True, cache_ttl=300)

        with NormalizerClient(api_key=api_key, config=config) as client:
            # First call
            result1 = client.normalize(100.0, "kWh")
            # Second call should use cache
            result2 = client.normalize(100.0, "kWh")

        assert result1.canonical_value == result2.canonical_value
        # Should only have made one request
        assert len(httpx_mock.get_requests()) == 1

    def test_cache_disabled(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        normalize_response: Dict[str, Any],
    ) -> None:
        """Test that caching is disabled when configured."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.greenlang.io/v1/normalize",
            json=normalize_response,
        )
        httpx_mock.add_response(
            method="POST",
            url="https://api.greenlang.io/v1/normalize",
            json=normalize_response,
        )

        config = ClientConfig(enable_cache=False)

        with NormalizerClient(api_key=api_key, config=config) as client:
            client.normalize(100.0, "kWh")
            client.normalize(100.0, "kWh")

        # Should have made two requests
        assert len(httpx_mock.get_requests()) == 2

    def test_clear_cache(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        normalize_response: Dict[str, Any],
    ) -> None:
        """Test clearing cache."""
        httpx_mock.add_response(
            method="POST",
            url="https://api.greenlang.io/v1/normalize",
            json=normalize_response,
        )
        httpx_mock.add_response(
            method="POST",
            url="https://api.greenlang.io/v1/normalize",
            json=normalize_response,
        )

        config = ClientConfig(enable_cache=True)

        with NormalizerClient(api_key=api_key, config=config) as client:
            client.normalize(100.0, "kWh")
            client.clear_cache()
            client.normalize(100.0, "kWh")

        # Should have made two requests after cache clear
        assert len(httpx_mock.get_requests()) == 2


class TestNormalizerClientHealthCheck:
    """Test NormalizerClient health check."""

    def test_health_check(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
    ) -> None:
        """Test health check endpoint."""
        httpx_mock.add_response(
            method="GET",
            url="https://api.greenlang.io/v1/health",
            json={"status": "healthy", "version": "1.0.0"},
        )

        with NormalizerClient(api_key=api_key) as client:
            health = client.health_check()

        assert health["status"] == "healthy"
