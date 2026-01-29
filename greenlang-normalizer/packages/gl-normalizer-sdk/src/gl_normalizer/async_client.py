"""
GL-FOUND-X-003: GreenLang Normalizer SDK - Async Client

This module provides the asynchronous AsyncNormalizerClient for interacting with
the GreenLang Unit & Reference Normalizer API using async/await patterns.

Example:
    >>> import asyncio
    >>> from gl_normalizer import AsyncNormalizerClient
    >>>
    >>> async def main():
    ...     async with AsyncNormalizerClient(api_key="your-api-key") as client:
    ...         result = await client.normalize(100, "kWh", target_unit="MJ")
    ...         print(result.canonical_value)  # 360.0
    >>>
    >>> asyncio.run(main())
"""

import asyncio
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Union

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from gl_normalizer.exceptions import (
    APIError,
    ConfigurationError,
    ConnectionError,
    NormalizerError,
    RateLimitError,
    ServiceUnavailableError,
    TimeoutError,
    raise_for_error_response,
)
from gl_normalizer.models import (
    BatchMode,
    BatchResult,
    ClientConfig,
    EntityResult,
    EntityType,
    Job,
    JobStatus,
    NormalizeMetadata,
    NormalizeRequest,
    NormalizeResult,
    PolicyMode,
    Vocabulary,
)

logger = logging.getLogger(__name__)

# Default API base URL
DEFAULT_BASE_URL = "https://api.greenlang.io"

# API version
API_VERSION = "v1"

# Maximum batch size for synchronous batch requests
MAX_BATCH_SIZE = 10_000

# Maximum items for async jobs
MAX_JOB_SIZE = 1_000_000


class AsyncNormalizerClient:
    """
    Asynchronous client for the GreenLang Normalizer API.

    This client provides async methods for normalizing units, resolving entities,
    and managing async jobs. It includes automatic retry with exponential
    backoff, connection pooling, and optional response caching.

    Attributes:
        api_key: API key for authentication.
        base_url: Base URL for the API.
        config: Client configuration options.

    Example:
        >>> async with AsyncNormalizerClient(api_key="your-api-key") as client:
        ...     result = await client.normalize(100, "kWh", target_unit="MJ")
        ...     print(result.canonical_value)  # 360.0

        >>> # Or manage lifecycle manually
        >>> client = AsyncNormalizerClient(api_key="your-api-key")
        >>> try:
        ...     result = await client.normalize(100, "kWh")
        ... finally:
        ...     await client.close()
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        config: Optional[ClientConfig] = None,
    ) -> None:
        """
        Initialize AsyncNormalizerClient.

        Args:
            api_key: API key for authentication. Required.
            base_url: Base URL for the API. Defaults to production.
            config: Optional client configuration.

        Raises:
            ConfigurationError: If api_key is empty or invalid.
        """
        if not api_key or not api_key.strip():
            raise ConfigurationError("API key is required")

        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.config = config or ClientConfig()

        # Initialize async HTTP client with connection pooling
        self._client = httpx.AsyncClient(
            base_url=f"{self.base_url}/{API_VERSION}",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "gl-normalizer-sdk/0.1.0 (async)",
            },
            timeout=httpx.Timeout(self.config.timeout),
            limits=httpx.Limits(
                max_connections=self.config.pool_maxsize,
                max_keepalive_connections=self.config.pool_connections,
            ),
        )

        # Simple in-memory cache with asyncio lock
        self._cache: Dict[str, tuple[float, Any]] = {}
        self._cache_lock = asyncio.Lock()

        logger.info(
            "AsyncNormalizerClient initialized",
            extra={"base_url": self.base_url, "timeout": self.config.timeout},
        )

    async def __aenter__(self) -> "AsyncNormalizerClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        await self._client.aclose()
        self._cache.clear()
        logger.debug("AsyncNormalizerClient closed")

    def _get_cache_key(self, endpoint: str, data: Dict[str, Any]) -> str:
        """Generate cache key from endpoint and request data."""
        content = f"{endpoint}:{hashlib.sha256(str(data).encode()).hexdigest()}"
        return content

    async def _get_cached(self, cache_key: str) -> Optional[Any]:
        """Get cached response if valid."""
        if not self.config.enable_cache:
            return None

        async with self._cache_lock:
            if cache_key in self._cache:
                timestamp, data = self._cache[cache_key]
                if time.time() - timestamp < self.config.cache_ttl:
                    logger.debug(f"Cache hit for {cache_key[:32]}...")
                    return data
                else:
                    del self._cache[cache_key]
        return None

    async def _set_cached(self, cache_key: str, data: Any) -> None:
        """Store response in cache."""
        if self.config.enable_cache:
            async with self._cache_lock:
                self._cache[cache_key] = (time.time(), data)

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        """
        Make async HTTP request to the API with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.).
            endpoint: API endpoint path.
            data: Request body data.
            params: Query parameters.
            use_cache: Whether to use caching for this request.

        Returns:
            Parsed JSON response.

        Raises:
            ConnectionError: If connection fails.
            TimeoutError: If request times out.
            RateLimitError: If rate limit exceeded.
            ServiceUnavailableError: If service unavailable.
            APIError: For other HTTP errors.
        """
        # Check cache
        cache_key: Optional[str] = None
        if use_cache:
            if method.upper() == "GET":
                cache_key = self._get_cache_key(endpoint, params or {})
            elif method.upper() == "POST" and data:
                cache_key = self._get_cache_key(endpoint, data)

            if cache_key:
                cached = await self._get_cached(cache_key)
                if cached is not None:
                    return cached

        response: Optional[httpx.Response] = None

        try:
            async for attempt in AsyncRetrying(
                retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
                stop=stop_after_attempt(self.config.max_retries + 1),
                wait=wait_exponential(
                    multiplier=self.config.retry_delay,
                    max=self.config.retry_max_delay,
                ),
                reraise=True,
            ):
                with attempt:
                    response = await self._client.request(
                        method=method,
                        url=endpoint,
                        json=data,
                        params=params,
                    )
        except httpx.ConnectError as e:
            logger.error(f"Connection error: {e}")
            raise ConnectionError(f"Failed to connect to API: {e}") from e
        except httpx.ReadTimeout as e:
            logger.error(f"Request timeout: {e}")
            raise TimeoutError(
                f"Request timed out after {self.config.timeout}s",
                timeout=self.config.timeout,
            ) from e
        except httpx.HTTPError as e:
            logger.error(f"HTTP error: {e}")
            raise ConnectionError(f"HTTP error: {e}") from e

        if response is None:
            raise ConnectionError("No response received")

        # Handle response
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                "Rate limit exceeded",
                retry_after=float(retry_after) if retry_after else None,
            )

        if response.status_code == 503:
            raise ServiceUnavailableError("Service temporarily unavailable")

        if response.status_code >= 400:
            try:
                error_data = response.json()
                raise_for_error_response(error_data, response.status_code)
            except ValueError:
                raise APIError(
                    f"HTTP {response.status_code}: {response.text}",
                    status_code=response.status_code,
                    response_body=response.text,
                )

        result = response.json()

        # Cache successful response
        if use_cache and cache_key:
            await self._set_cached(cache_key, result)

        return result

    async def normalize(
        self,
        value: float,
        unit: str,
        target_unit: Optional[str] = None,
        expected_dimension: Optional[str] = None,
        field: Optional[str] = None,
        policy_mode: PolicyMode = PolicyMode.STRICT,
        vocabulary_version: Optional[str] = None,
        metadata: Optional[NormalizeMetadata] = None,
        **context: Any,
    ) -> NormalizeResult:
        """
        Normalize a single measurement value asynchronously.

        Args:
            value: Numeric value to normalize.
            unit: Unit string (may be messy, e.g., "kWh", "kilowatt-hours").
            target_unit: Optional target unit for conversion.
            expected_dimension: Expected dimension for validation (e.g., "energy").
            field: Field name for audit trail.
            policy_mode: Policy mode (STRICT or LENIENT).
            vocabulary_version: Pin to specific vocabulary version.
            metadata: Additional metadata (locale, reference conditions, etc.).
            **context: Additional context passed to the API.

        Returns:
            NormalizeResult with canonical value, unit, and audit info.

        Raises:
            ValidationError: If unit parsing or dimension validation fails.
            ConversionError: If conversion is not possible.
            APIError: For other API errors.

        Example:
            >>> async with AsyncNormalizerClient(api_key="...") as client:
            ...     result = await client.normalize(100, "kWh", target_unit="MJ")
            ...     print(result.canonical_value)  # 360.0
        """
        request_data: Dict[str, Any] = {
            "measurements": [
                {
                    "value": value,
                    "unit": unit,
                    "target_unit": target_unit,
                    "expected_dimension": expected_dimension,
                    "field": field,
                    "metadata": metadata.model_dump(by_alias=True, exclude_none=True)
                    if metadata
                    else None,
                }
            ],
            "policy_mode": policy_mode.value,
        }

        if vocabulary_version:
            request_data["vocabulary_version"] = vocabulary_version

        for key, val in context.items():
            if key not in request_data:
                request_data[key] = val

        logger.debug(f"Normalizing: {value} {unit} -> {target_unit or 'canonical'}")

        response = await self._make_request(
            "POST",
            "/normalize",
            data=request_data,
            use_cache=self.config.enable_cache,
        )

        measurements = response.get("canonical_measurements", [])
        if not measurements:
            raise APIError("No measurements in response", status_code=500)

        return NormalizeResult.model_validate(measurements[0])

    async def normalize_batch(
        self,
        items: List[Union[NormalizeRequest, Dict[str, Any]]],
        mode: BatchMode = BatchMode.PARTIAL,
        policy_mode: PolicyMode = PolicyMode.STRICT,
        vocabulary_version: Optional[str] = None,
    ) -> BatchResult:
        """
        Normalize a batch of measurements asynchronously (up to 10K items).

        Args:
            items: List of NormalizeRequest objects or dicts.
            mode: Batch processing mode (PARTIAL or ALL_OR_NOTHING).
            policy_mode: Policy mode for all items.
            vocabulary_version: Pin all items to specific vocabulary version.

        Returns:
            BatchResult with summary and per-item results.

        Raises:
            ConfigurationError: If batch exceeds MAX_BATCH_SIZE.
            ValidationError: For validation failures (in ALL_OR_NOTHING mode).
            APIError: For other API errors.

        Example:
            >>> async with AsyncNormalizerClient(api_key="...") as client:
            ...     requests = [
            ...         NormalizeRequest(value=100, unit="kWh"),
            ...         NormalizeRequest(value=50, unit="kg"),
            ...     ]
            ...     result = await client.normalize_batch(requests)
            ...     print(f"Success: {result.summary.success}")
        """
        if len(items) > MAX_BATCH_SIZE:
            raise ConfigurationError(
                f"Batch size {len(items)} exceeds maximum {MAX_BATCH_SIZE}. "
                "Use create_job() for larger batches."
            )

        records = []
        for i, item in enumerate(items):
            if isinstance(item, NormalizeRequest):
                record = {
                    "source_record_id": f"batch-{i}",
                    "measurements": [
                        item.model_dump(by_alias=True, exclude_none=True)
                    ],
                }
            else:
                record = {
                    "source_record_id": item.get("source_record_id", f"batch-{i}"),
                    "measurements": [item],
                }
            records.append(record)

        request_data: Dict[str, Any] = {
            "policy_mode": policy_mode.value,
            "batch_mode": mode.value,
            "records": records,
        }

        if vocabulary_version:
            request_data["vocabulary_version"] = vocabulary_version

        logger.info(f"Batch normalizing {len(items)} items")

        response = await self._make_request("POST", "/normalize/batch", data=request_data)

        return BatchResult.model_validate(response)

    async def normalize_concurrent(
        self,
        items: List[Union[NormalizeRequest, Dict[str, Any]]],
        max_concurrency: int = 10,
        policy_mode: PolicyMode = PolicyMode.STRICT,
        vocabulary_version: Optional[str] = None,
    ) -> List[NormalizeResult]:
        """
        Normalize items concurrently with controlled concurrency.

        This method normalizes items individually but runs multiple
        requests concurrently for improved throughput.

        Args:
            items: List of NormalizeRequest objects or dicts.
            max_concurrency: Maximum concurrent requests.
            policy_mode: Policy mode for all items.
            vocabulary_version: Pin all items to specific vocabulary version.

        Returns:
            List of NormalizeResult objects in same order as input.

        Example:
            >>> async with AsyncNormalizerClient(api_key="...") as client:
            ...     requests = [
            ...         NormalizeRequest(value=i, unit="kWh")
            ...         for i in range(100)
            ...     ]
            ...     results = await client.normalize_concurrent(requests, max_concurrency=20)
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def normalize_one(item: Union[NormalizeRequest, Dict[str, Any]]) -> NormalizeResult:
            async with semaphore:
                if isinstance(item, NormalizeRequest):
                    return await self.normalize(
                        value=item.value,
                        unit=item.unit,
                        target_unit=item.target_unit,
                        expected_dimension=item.expected_dimension,
                        field=item.field,
                        policy_mode=policy_mode,
                        vocabulary_version=vocabulary_version,
                        metadata=item.metadata,
                    )
                else:
                    return await self.normalize(
                        value=item["value"],
                        unit=item["unit"],
                        target_unit=item.get("target_unit"),
                        expected_dimension=item.get("expected_dimension"),
                        field=item.get("field"),
                        policy_mode=policy_mode,
                        vocabulary_version=vocabulary_version,
                    )

        tasks = [normalize_one(item) for item in items]
        return await asyncio.gather(*tasks)

    async def create_job(
        self,
        items: List[Union[NormalizeRequest, Dict[str, Any]]],
        policy_mode: PolicyMode = PolicyMode.STRICT,
        vocabulary_version: Optional[str] = None,
        callback_url: Optional[str] = None,
    ) -> Job:
        """
        Create an async job for processing large batches (100K+ items).

        Args:
            items: List of NormalizeRequest objects or dicts.
            policy_mode: Policy mode for all items.
            vocabulary_version: Pin all items to specific vocabulary version.
            callback_url: Optional webhook URL for completion notification.

        Returns:
            Job object with job_id and initial status.

        Raises:
            ConfigurationError: If batch exceeds MAX_JOB_SIZE.
            APIError: For API errors.

        Example:
            >>> async with AsyncNormalizerClient(api_key="...") as client:
            ...     requests = [NormalizeRequest(value=i, unit="kWh") for i in range(100000)]
            ...     job = await client.create_job(requests)
            ...     print(f"Job created: {job.job_id}")
        """
        if len(items) > MAX_JOB_SIZE:
            raise ConfigurationError(
                f"Job size {len(items)} exceeds maximum {MAX_JOB_SIZE}"
            )

        records = []
        for i, item in enumerate(items):
            if isinstance(item, NormalizeRequest):
                record = {
                    "source_record_id": f"job-{i}",
                    "measurements": [
                        item.model_dump(by_alias=True, exclude_none=True)
                    ],
                }
            else:
                record = {
                    "source_record_id": item.get("source_record_id", f"job-{i}"),
                    "measurements": [item],
                }
            records.append(record)

        request_data: Dict[str, Any] = {
            "policy_mode": policy_mode.value,
            "records": records,
        }

        if vocabulary_version:
            request_data["vocabulary_version"] = vocabulary_version
        if callback_url:
            request_data["callback_url"] = callback_url

        logger.info(f"Creating job with {len(items)} items")

        response = await self._make_request("POST", "/jobs", data=request_data)

        return Job.model_validate(response)

    async def get_job(self, job_id: str) -> Job:
        """
        Get the status and details of an async job.

        Args:
            job_id: The job identifier returned from create_job().

        Returns:
            Job object with current status and progress.

        Raises:
            APIError: If job not found or other API error.

        Example:
            >>> async with AsyncNormalizerClient(api_key="...") as client:
            ...     job = await client.get_job("job-abc123")
            ...     print(f"Status: {job.status}, Progress: {job.progress}%")
        """
        logger.debug(f"Getting job status: {job_id}")

        response = await self._make_request("GET", f"/jobs/{job_id}")

        return Job.model_validate(response)

    async def cancel_job(self, job_id: str) -> Job:
        """
        Cancel a pending or processing job.

        Args:
            job_id: The job identifier to cancel.

        Returns:
            Job object with updated status.

        Raises:
            APIError: If job cannot be cancelled or other API error.

        Example:
            >>> async with AsyncNormalizerClient(api_key="...") as client:
            ...     job = await client.cancel_job("job-abc123")
            ...     print(f"Status: {job.status}")  # "cancelled"
        """
        logger.info(f"Cancelling job: {job_id}")

        response = await self._make_request("POST", f"/jobs/{job_id}/cancel")

        return Job.model_validate(response)

    async def wait_for_job(
        self,
        job_id: str,
        poll_interval: float = 5.0,
        timeout: Optional[float] = None,
    ) -> Job:
        """
        Wait for a job to complete, polling at specified interval.

        Args:
            job_id: The job identifier to wait for.
            poll_interval: Seconds between status checks (default: 5.0).
            timeout: Maximum seconds to wait (default: None = unlimited).

        Returns:
            Job object with final status.

        Raises:
            TimeoutError: If timeout exceeded.
            JobError: If job fails.

        Example:
            >>> async with AsyncNormalizerClient(api_key="...") as client:
            ...     job = await client.create_job(requests)
            ...     completed = await client.wait_for_job(job.job_id, timeout=3600)
            ...     print(f"Final status: {completed.status}")
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            job = await self.get_job(job_id)

            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return job

            if timeout and (asyncio.get_event_loop().time() - start_time) > timeout:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout}s",
                    timeout=timeout,
                )

            logger.debug(f"Job {job_id} progress: {job.progress}%")
            await asyncio.sleep(poll_interval)

    async def resolve_entity(
        self,
        raw_name: str,
        entity_type: Union[EntityType, str],
        raw_code: Optional[str] = None,
        vocabulary_version: Optional[str] = None,
        policy_mode: PolicyMode = PolicyMode.STRICT,
        return_candidates: bool = False,
        max_candidates: int = 5,
    ) -> EntityResult:
        """
        Resolve a raw entity name to a reference ID asynchronously.

        Args:
            raw_name: Raw entity name to resolve.
            entity_type: Type of entity (fuel, material, process).
            raw_code: Optional raw code/identifier.
            vocabulary_version: Pin to specific vocabulary version.
            policy_mode: Policy mode (STRICT or LENIENT).
            return_candidates: Whether to return candidate matches.
            max_candidates: Maximum candidates to return.

        Returns:
            EntityResult with reference_id, canonical_name, and confidence.

        Raises:
            ResolutionError: If entity cannot be resolved.
            APIError: For other API errors.

        Example:
            >>> async with AsyncNormalizerClient(api_key="...") as client:
            ...     result = await client.resolve_entity("Nat Gas", entity_type="fuel")
            ...     print(result.reference_id)  # "GL-FUEL-NATGAS"
        """
        if isinstance(entity_type, str):
            entity_type = EntityType(entity_type)

        request_data: Dict[str, Any] = {
            "entity_type": entity_type.value,
            "raw_name": raw_name,
            "policy_mode": policy_mode.value,
            "options": {
                "return_candidates": return_candidates,
                "max_candidates": max_candidates,
            },
        }

        if raw_code:
            request_data["raw_code"] = raw_code
        if vocabulary_version:
            request_data["vocabulary_version"] = vocabulary_version

        logger.debug(f"Resolving entity: {raw_name} ({entity_type.value})")

        response = await self._make_request(
            "POST",
            "/resolve",
            data=request_data,
            use_cache=self.config.enable_cache,
        )

        best_match = response.get("best_match", {})
        if not best_match:
            raise APIError("No match in response", status_code=500)

        return EntityResult.model_validate({
            "entity_type": entity_type.value,
            "raw_name": raw_name,
            **best_match,
        })

    async def list_vocabularies(
        self,
        entity_type: Optional[Union[EntityType, str]] = None,
    ) -> List[Vocabulary]:
        """
        List available vocabularies asynchronously.

        Args:
            entity_type: Optional filter by entity type.

        Returns:
            List of Vocabulary objects.

        Example:
            >>> async with AsyncNormalizerClient(api_key="...") as client:
            ...     vocabs = await client.list_vocabularies()
            ...     for vocab in vocabs:
            ...         print(f"{vocab.name} v{vocab.version}")
        """
        params: Dict[str, Any] = {}
        if entity_type:
            if isinstance(entity_type, str):
                entity_type = EntityType(entity_type)
            params["entity_type"] = entity_type.value

        logger.debug("Listing vocabularies")

        response = await self._make_request(
            "GET",
            "/vocabularies",
            params=params,
            use_cache=True,
        )

        vocabularies = response.get("vocabularies", [])
        return [Vocabulary.model_validate(v) for v in vocabularies]

    async def get_vocabulary(
        self,
        vocabulary_id: str,
        version: Optional[str] = None,
    ) -> Vocabulary:
        """
        Get details of a specific vocabulary asynchronously.

        Args:
            vocabulary_id: Vocabulary identifier.
            version: Optional specific version.

        Returns:
            Vocabulary object with metadata.

        Raises:
            VocabularyError: If vocabulary not found.

        Example:
            >>> async with AsyncNormalizerClient(api_key="...") as client:
            ...     vocab = await client.get_vocabulary("fuels", version="2026.01.0")
            ...     print(vocab.entity_count)
        """
        params: Dict[str, Any] = {}
        if version:
            params["version"] = version

        response = await self._make_request(
            "GET",
            f"/vocabularies/{vocabulary_id}",
            params=params,
            use_cache=True,
        )

        return Vocabulary.model_validate(response)

    async def health_check(self) -> Dict[str, Any]:
        """
        Check API health status asynchronously.

        Returns:
            Dict with health status information.

        Example:
            >>> async with AsyncNormalizerClient(api_key="...") as client:
            ...     health = await client.health_check()
            ...     print(health["status"])  # "healthy"
        """
        response = await self._make_request("GET", "/health")
        return response

    async def clear_cache(self) -> None:
        """Clear the local response cache."""
        async with self._cache_lock:
            self._cache.clear()
        logger.debug("Cache cleared")
