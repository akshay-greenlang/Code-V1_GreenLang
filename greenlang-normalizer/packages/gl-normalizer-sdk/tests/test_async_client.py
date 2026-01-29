"""
Unit tests for AsyncNormalizerClient.
"""

import pytest
from typing import Any, Dict

from pytest_httpx import HTTPXMock

from gl_normalizer import (
    AsyncNormalizerClient,
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
    RateLimitError,
)


class TestAsyncNormalizerClientInit:
    """Test AsyncNormalizerClient initialization."""

    def test_init_with_api_key(self, api_key: str) -> None:
        """Test initialization with valid API key."""
        client = AsyncNormalizerClient(api_key=api_key)
        assert client.api_key == api_key
        assert client.base_url == "https://api.greenlang.io"

    def test_init_with_custom_base_url(self, api_key: str, base_url: str) -> None:
        """Test initialization with custom base URL."""
        client = AsyncNormalizerClient(api_key=api_key, base_url=base_url)
        assert client.base_url == base_url

    def test_init_with_config(self, api_key: str) -> None:
        """Test initialization with custom config."""
        config = ClientConfig(timeout=60.0, max_retries=5)
        client = AsyncNormalizerClient(api_key=api_key, config=config)
        assert client.config.timeout == 60.0
        assert client.config.max_retries == 5

    def test_init_empty_api_key_raises(self) -> None:
        """Test that empty API key raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="API key is required"):
            AsyncNormalizerClient(api_key="")


class TestAsyncNormalizerClientNormalize:
    """Test AsyncNormalizerClient.normalize method."""

    @pytest.mark.asyncio
    async def test_normalize_basic(
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

        async with AsyncNormalizerClient(api_key=api_key) as client:
            result = await client.normalize(100.0, "kWh", target_unit="MJ")

        assert isinstance(result, NormalizeResult)
        assert result.canonical_value == 360.0
        assert result.canonical_unit == "MJ"
        assert result.raw_value == 100.0
        assert result.raw_unit == "kWh"

    @pytest.mark.asyncio
    async def test_normalize_with_expected_dimension(
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

        async with AsyncNormalizerClient(api_key=api_key) as client:
            result = await client.normalize(
                100.0,
                "kWh",
                target_unit="MJ",
                expected_dimension="energy",
                field="energy_consumption",
            )

        assert result.dimension == "energy"
        assert result.field == "energy_consumption"

    @pytest.mark.asyncio
    async def test_normalize_validation_error(
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

        async with AsyncNormalizerClient(api_key=api_key) as client:
            with pytest.raises(ValidationError) as exc_info:
                await client.normalize(100.0, "kg", expected_dimension="energy")

        assert exc_info.value.code == "GLNORM-E200"

    @pytest.mark.asyncio
    async def test_normalize_rate_limit_error(
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

        async with AsyncNormalizerClient(api_key=api_key) as client:
            with pytest.raises(RateLimitError) as exc_info:
                await client.normalize(100.0, "kWh")

        assert exc_info.value.retry_after == 60.0


class TestAsyncNormalizerClientBatch:
    """Test AsyncNormalizerClient.normalize_batch method."""

    @pytest.mark.asyncio
    async def test_normalize_batch_basic(
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

        async with AsyncNormalizerClient(api_key=api_key) as client:
            result = await client.normalize_batch(requests)

        assert isinstance(result, BatchResult)
        assert result.summary.total == 2
        assert result.summary.success == 2

    @pytest.mark.asyncio
    async def test_normalize_batch_exceeds_limit(self, api_key: str) -> None:
        """Test batch normalization exceeds limit."""
        requests = [NormalizeRequest(value=i, unit="kWh") for i in range(10001)]

        async with AsyncNormalizerClient(api_key=api_key) as client:
            with pytest.raises(ConfigurationError, match="exceeds maximum"):
                await client.normalize_batch(requests)


class TestAsyncNormalizerClientConcurrent:
    """Test AsyncNormalizerClient.normalize_concurrent method."""

    @pytest.mark.asyncio
    async def test_normalize_concurrent(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        normalize_response: Dict[str, Any],
    ) -> None:
        """Test concurrent normalization."""
        # Add multiple responses
        for _ in range(5):
            httpx_mock.add_response(
                method="POST",
                url="https://api.greenlang.io/v1/normalize",
                json=normalize_response,
            )

        requests = [NormalizeRequest(value=i * 100, unit="kWh") for i in range(5)]

        async with AsyncNormalizerClient(api_key=api_key) as client:
            results = await client.normalize_concurrent(requests, max_concurrency=3)

        assert len(results) == 5
        assert all(isinstance(r, NormalizeResult) for r in results)


class TestAsyncNormalizerClientJobs:
    """Test AsyncNormalizerClient job methods."""

    @pytest.mark.asyncio
    async def test_create_job(
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

        async with AsyncNormalizerClient(api_key=api_key) as client:
            job = await client.create_job(requests)

        assert isinstance(job, Job)
        assert job.job_id == "job-abc123"
        assert job.status.value == "pending"

    @pytest.mark.asyncio
    async def test_get_job(
        self,
        httpx_mock: HTTPXMock,
        api_key: str,
        job_response: Dict[str, Any],
    ) -> None:
        """Test getting job status."""
        job_response["status"] = "processing"
        job_response["progress"] = 50.0

        httpx_mock.add_response(
            method="GET",
            url="https://api.greenlang.io/v1/jobs/job-abc123",
            json=job_response,
        )

        async with AsyncNormalizerClient(api_key=api_key) as client:
            job = await client.get_job("job-abc123")

        assert job.status.value == "processing"
        assert job.progress == 50.0

    @pytest.mark.asyncio
    async def test_cancel_job(
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

        async with AsyncNormalizerClient(api_key=api_key) as client:
            job = await client.cancel_job("job-abc123")

        assert job.status.value == "cancelled"


class TestAsyncNormalizerClientEntities:
    """Test AsyncNormalizerClient entity resolution methods."""

    @pytest.mark.asyncio
    async def test_resolve_entity(
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

        async with AsyncNormalizerClient(api_key=api_key) as client:
            result = await client.resolve_entity("Nat Gas", entity_type="fuel")

        assert isinstance(result, EntityResult)
        assert result.reference_id == "GL-FUEL-NATGAS"
        assert result.canonical_name == "Natural gas"


class TestAsyncNormalizerClientVocabularies:
    """Test AsyncNormalizerClient vocabulary methods."""

    @pytest.mark.asyncio
    async def test_list_vocabularies(
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

        async with AsyncNormalizerClient(api_key=api_key) as client:
            vocabs = await client.list_vocabularies()

        assert len(vocabs) == 2
        assert isinstance(vocabs[0], Vocabulary)


class TestAsyncNormalizerClientCaching:
    """Test AsyncNormalizerClient caching behavior."""

    @pytest.mark.asyncio
    async def test_cache_enabled(
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

        async with AsyncNormalizerClient(api_key=api_key, config=config) as client:
            result1 = await client.normalize(100.0, "kWh")
            result2 = await client.normalize(100.0, "kWh")

        assert result1.canonical_value == result2.canonical_value
        # Should only have made one request
        assert len(httpx_mock.get_requests()) == 1

    @pytest.mark.asyncio
    async def test_clear_cache(
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

        async with AsyncNormalizerClient(api_key=api_key, config=config) as client:
            await client.normalize(100.0, "kWh")
            await client.clear_cache()
            await client.normalize(100.0, "kWh")

        # Should have made two requests after cache clear
        assert len(httpx_mock.get_requests()) == 2


class TestAsyncNormalizerClientHealthCheck:
    """Test AsyncNormalizerClient health check."""

    @pytest.mark.asyncio
    async def test_health_check(
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

        async with AsyncNormalizerClient(api_key=api_key) as client:
            health = await client.health_check()

        assert health["status"] == "healthy"
