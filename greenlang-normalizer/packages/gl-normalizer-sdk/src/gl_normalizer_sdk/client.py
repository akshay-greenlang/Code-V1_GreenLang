"""
HTTP client for the GreenLang Normalizer Service.

This module provides synchronous and asynchronous clients for
interacting with the Normalizer API.

Example:
    >>> from gl_normalizer_sdk import NormalizerClient
    >>> client = NormalizerClient("http://localhost:8000")
    >>> result = client.convert("100 kg", "t")
    >>> print(result.value, result.unit)
    0.1 t
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

import httpx
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


class ConversionRequest(BaseModel):
    """Request model for unit conversion."""

    value: float = Field(..., description="Numeric value to convert")
    source_unit: str = Field(..., description="Source unit")
    target_unit: str = Field(..., description="Target unit")
    policy_id: Optional[str] = Field(None, description="Compliance policy ID")


class ConversionResponse(BaseModel):
    """Response model for unit conversion."""

    value: float = Field(..., description="Converted value")
    unit: str = Field(..., description="Target unit")
    conversion_factor: float = Field(..., description="Factor applied")
    provenance_hash: str = Field(..., description="Audit hash")
    warnings: List[str] = Field(default_factory=list)


class ResolutionRequest(BaseModel):
    """Request model for reference resolution."""

    query: str = Field(..., description="Query string to resolve")
    vocabulary: str = Field(..., description="Vocabulary to search")
    min_confidence: Optional[float] = Field(None, ge=0, le=100)


class ResolutionResponse(BaseModel):
    """Response model for reference resolution."""

    resolved_id: str = Field(..., description="Resolved reference ID")
    resolved_name: str = Field(..., description="Resolved reference name")
    confidence: float = Field(..., description="Match confidence 0-100")
    vocabulary: str = Field(..., description="Vocabulary used")
    provenance_hash: str = Field(..., description="Audit hash")
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)


class ParseRequest(BaseModel):
    """Request model for unit parsing."""

    input_string: str = Field(..., description="String to parse")


class ParseResponse(BaseModel):
    """Response model for unit parsing."""

    success: bool = Field(..., description="Whether parsing succeeded")
    magnitude: Optional[float] = Field(None, description="Parsed magnitude")
    unit: Optional[str] = Field(None, description="Parsed unit")
    warnings: List[str] = Field(default_factory=list)


@dataclass
class ClientConfig:
    """Configuration for the Normalizer client."""

    base_url: str
    api_key: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3
    verify_ssl: bool = True
    headers: Dict[str, str] = field(default_factory=dict)

    def get_headers(self) -> Dict[str, str]:
        """Get headers including API key if configured."""
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class NormalizerClient:
    """
    Synchronous client for the GreenLang Normalizer Service.

    This client provides methods for unit conversion, reference resolution,
    and vocabulary queries.

    Example:
        >>> config = ClientConfig(base_url="http://localhost:8000")
        >>> client = NormalizerClient(config)
        >>> result = client.convert(100, "kg", "t")
        >>> print(result.value)
        0.1
    """

    def __init__(
        self,
        config: Union[ClientConfig, str],
    ) -> None:
        """
        Initialize the client.

        Args:
            config: ClientConfig or base URL string
        """
        if isinstance(config, str):
            config = ClientConfig(base_url=config)
        self.config = config
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
                headers=self.config.get_headers(),
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "NormalizerClient":
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def convert(
        self,
        value: float,
        source_unit: str,
        target_unit: str,
        policy_id: Optional[str] = None,
    ) -> ConversionResponse:
        """
        Convert a value from one unit to another.

        Args:
            value: Numeric value to convert
            source_unit: Source unit string
            target_unit: Target unit string
            policy_id: Optional policy ID for compliance

        Returns:
            ConversionResponse with converted value

        Raises:
            httpx.HTTPError: If request fails
        """
        request = ConversionRequest(
            value=value,
            source_unit=source_unit,
            target_unit=target_unit,
            policy_id=policy_id,
        )
        response = self._get_client().post(
            "/api/v1/convert",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return ConversionResponse.model_validate(response.json())

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def resolve(
        self,
        query: str,
        vocabulary: str,
        min_confidence: Optional[float] = None,
    ) -> ResolutionResponse:
        """
        Resolve a query string to a vocabulary entry.

        Args:
            query: Query string to resolve
            vocabulary: Vocabulary to search
            min_confidence: Minimum confidence threshold

        Returns:
            ResolutionResponse with resolved reference

        Raises:
            httpx.HTTPError: If request fails
        """
        request = ResolutionRequest(
            query=query,
            vocabulary=vocabulary,
            min_confidence=min_confidence,
        )
        response = self._get_client().post(
            "/api/v1/resolve",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        return ResolutionResponse.model_validate(response.json())

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def parse(self, input_string: str) -> ParseResponse:
        """
        Parse a quantity string.

        Args:
            input_string: String to parse

        Returns:
            ParseResponse with parsed components

        Raises:
            httpx.HTTPError: If request fails
        """
        request = ParseRequest(input_string=input_string)
        response = self._get_client().post(
            "/api/v1/parse",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return ParseResponse.model_validate(response.json())

    def health(self) -> Dict[str, Any]:
        """Check service health."""
        response = self._get_client().get("/health")
        response.raise_for_status()
        return response.json()


class AsyncNormalizerClient:
    """
    Asynchronous client for the GreenLang Normalizer Service.

    Example:
        >>> async with AsyncNormalizerClient("http://localhost:8000") as client:
        ...     result = await client.convert(100, "kg", "t")
        ...     print(result.value)
    """

    def __init__(
        self,
        config: Union[ClientConfig, str],
    ) -> None:
        """Initialize async client."""
        if isinstance(config, str):
            config = ClientConfig(base_url=config)
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                verify=self.config.verify_ssl,
                headers=self.config.get_headers(),
            )
        return self._client

    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "AsyncNormalizerClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    async def convert(
        self,
        value: float,
        source_unit: str,
        target_unit: str,
        policy_id: Optional[str] = None,
    ) -> ConversionResponse:
        """Convert a value asynchronously."""
        request = ConversionRequest(
            value=value,
            source_unit=source_unit,
            target_unit=target_unit,
            policy_id=policy_id,
        )
        client = await self._get_client()
        response = await client.post(
            "/api/v1/convert",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return ConversionResponse.model_validate(response.json())

    async def resolve(
        self,
        query: str,
        vocabulary: str,
        min_confidence: Optional[float] = None,
    ) -> ResolutionResponse:
        """Resolve a query asynchronously."""
        request = ResolutionRequest(
            query=query,
            vocabulary=vocabulary,
            min_confidence=min_confidence,
        )
        client = await self._get_client()
        response = await client.post(
            "/api/v1/resolve",
            json=request.model_dump(exclude_none=True),
        )
        response.raise_for_status()
        return ResolutionResponse.model_validate(response.json())

    async def parse(self, input_string: str) -> ParseResponse:
        """Parse a quantity string asynchronously."""
        request = ParseRequest(input_string=input_string)
        client = await self._get_client()
        response = await client.post(
            "/api/v1/parse",
            json=request.model_dump(),
        )
        response.raise_for_status()
        return ParseResponse.model_validate(response.json())

    async def health(self) -> Dict[str, Any]:
        """Check service health asynchronously."""
        client = await self._get_client()
        response = await client.get("/health")
        response.raise_for_status()
        return response.json()


__all__ = [
    "NormalizerClient",
    "AsyncNormalizerClient",
    "ClientConfig",
    "ConversionRequest",
    "ConversionResponse",
    "ResolutionRequest",
    "ResolutionResponse",
    "ParseRequest",
    "ParseResponse",
]
