# -*- coding: utf-8 -*-
"""
Ecoinvent Factor Source
GL-VCCI Scope 3 Platform

Integration with ecoinvent database (v3.10) for high-quality emission factors.
Implements license compliance (no bulk export, 24h caching limit).

Version: 1.0.0
License: Commercial - ecoinvent license terms apply
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import asyncio
import aiohttp
from greenlang.utilities.determinism import DeterministicClock
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from .base import FactorSource
from ..models import (
    FactorRequest,
    FactorResponse,
    FactorMetadata,
    DataQualityIndicator,
    SourceType,
    GWPStandard,
    UnitType
)
from ..exceptions import (
    FactorNotFoundError,
    SourceUnavailableError,
    RateLimitExceededError,
    LicenseViolationError,
    ValidationError
)
from ..config import SourceConfig


logger = logging.getLogger(__name__)


class EcoinventSource(FactorSource):
    """
    Ecoinvent emission factor data source.

    Provides access to ecoinvent v3.10 database via REST API.
    Implements license compliance including:
    - No bulk redistribution
    - Runtime API access only
    - Caching limited to 24 hours
    - Attribution in reports

    Attributes:
        api_endpoint: Ecoinvent API endpoint URL
        api_key: API authentication key
        session: aiohttp session for API calls
        request_count: Count of API requests (for rate limiting)
    """

    def __init__(self, config: SourceConfig):
        """
        Initialize Ecoinvent source.

        Args:
            config: Source configuration

        Raises:
            ValidationError: If API endpoint or key is missing
        """
        super().__init__(
            source_type=SourceType.ECOINVENT,
            config=config,
            version="3.10"
        )

        if not config.api_endpoint:
            raise ValidationError(
                field="api_endpoint",
                value=None,
                reason="Ecoinvent API endpoint is required"
            )

        if not config.api_key:
            raise ValidationError(
                field="api_key",
                value=None,
                reason="Ecoinvent API key is required"
            )

        self.api_endpoint = config.api_endpoint
        self.api_key = config.api_key
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_count = 0
        self.last_request_time = DeterministicClock.utcnow()

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create aiohttp session.

        Returns:
            aiohttp.ClientSession instance
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "GL-VCCI-FactorBroker/1.0.0"
                },
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            )

        return self.session

    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    def _check_rate_limit(self):
        """
        Check if rate limit would be exceeded.

        Raises:
            RateLimitExceededError: If rate limit is exceeded
        """
        # Simple rate limiting: 1000 requests per minute
        now = DeterministicClock.utcnow()
        time_diff = (now - self.last_request_time).total_seconds()

        if time_diff < 60:  # Within same minute
            if self.request_count >= self.config.rate_limit:
                retry_after = int(60 - time_diff) + 1
                raise RateLimitExceededError(
                    source=self.name,
                    limit=self.config.rate_limit,
                    retry_after_seconds=retry_after
                )
        else:
            # Reset counter for new minute
            self.request_count = 0
            self.last_request_time = now

        self.request_count += 1

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to ecoinvent API with retry logic.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            Response data as dictionary

        Raises:
            SourceUnavailableError: If API is unavailable
            RateLimitExceededError: If rate limit is exceeded
        """
        self._check_rate_limit()

        session = await self._get_session()
        url = f"{self.api_endpoint}/{endpoint}"

        try:
            async with session.get(url, params=params) as response:
                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", "60"))
                    raise RateLimitExceededError(
                        source=self.name,
                        limit=self.config.rate_limit,
                        retry_after_seconds=retry_after
                    )

                if response.status == 404:
                    return {}  # Factor not found

                if response.status >= 500:
                    raise SourceUnavailableError(
                        source=self.name,
                        reason=f"Server error: {response.status}"
                    )

                response.raise_for_status()
                return await response.json()

        except aiohttp.ClientError as e:
            raise SourceUnavailableError(
                source=self.name,
                reason="Connection error",
                original_exception=e
            )

    def _map_gwp_standard(self, gwp_standard: GWPStandard) -> str:
        """
        Map our GWP standard to ecoinvent parameter.

        Args:
            gwp_standard: GWP standard

        Returns:
            Ecoinvent GWP parameter
        """
        mapping = {
            GWPStandard.AR5: "IPCC2013",  # AR5
            GWPStandard.AR6: "IPCC2021"   # AR6
        }
        return mapping.get(gwp_standard, "IPCC2021")

    def _calculate_data_quality(
        self,
        dataset_metadata: Dict[str, Any]
    ) -> DataQualityIndicator:
        """
        Calculate data quality from ecoinvent metadata.

        Ecoinvent provides pedigree matrix scores which we map to our DQI.

        Args:
            dataset_metadata: Ecoinvent dataset metadata

        Returns:
            DataQualityIndicator instance
        """
        # Ecoinvent has high quality data - typically 4-5 scores
        pedigree = dataset_metadata.get("pedigree_matrix", {})

        return self.create_data_quality_indicator(
            reliability=pedigree.get("reliability", 5),
            completeness=pedigree.get("completeness", 5),
            temporal=pedigree.get("temporal_correlation", 5),
            geographical=pedigree.get("geographical_correlation", 5),
            technological=pedigree.get("technological_correlation", 5)
        )

    async def fetch_factor(
        self,
        request: FactorRequest
    ) -> Optional[FactorResponse]:
        """
        Fetch emission factor from ecoinvent database.

        Args:
            request: Factor request

        Returns:
            FactorResponse if found, None otherwise

        Raises:
            SourceUnavailableError: If ecoinvent API is unavailable
            RateLimitExceededError: If rate limit is exceeded
        """
        start_time = DeterministicClock.utcnow()

        try:
            self.validate_request(request)

            # Normalize inputs
            product = self.normalize_product_name(request.product)
            region = self.normalize_region(request.region)

            # Build query parameters
            params = {
                "name": product,
                "geography": region,
                "gwp_method": self._map_gwp_standard(request.gwp_standard),
                "version": "3.10"
            }

            if request.year:
                params["reference_year"] = request.year

            # Make API request
            data = await self._make_request("activities", params)

            if not data or not data.get("results"):
                # Factor not found in ecoinvent
                latency_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000
                self.log_lookup(request, success=False, latency_ms=latency_ms)
                return None

            # Parse first result
            result = data["results"][0]
            factor_data = result.get("exchanges", [{}])[0]

            # Extract factor value
            value = factor_data.get("amount", 0.0)
            if value <= 0:
                return None

            # Determine unit
            unit = factor_data.get("unit", "kg CO2e")
            if request.unit:
                # TODO: Unit conversion if requested unit differs
                unit = request.unit

            # Create factor ID
            factor_id = self.create_factor_id(
                product=request.product,
                region=region,
                gwp_standard=request.gwp_standard,
                unit=unit
            )

            # Calculate data quality
            data_quality = self._calculate_data_quality(result)

            # Create metadata
            metadata = FactorMetadata(
                source=SourceType.ECOINVENT,
                source_version=self.version,
                source_dataset_id=result.get("activity_id"),
                gwp_standard=request.gwp_standard,
                reference_year=result.get("reference_year", 2024),
                last_updated=datetime.fromisoformat(
                    result.get("last_updated", DeterministicClock.utcnow().isoformat())
                ),
                geographic_scope=region,
                technology_scope=result.get("technology", "Average mix"),
                data_quality=data_quality,
                citation=(
                    f"ecoinvent v{self.version}, "
                    f"{result.get('activity_name', request.product)}, "
                    f"{region}, {request.gwp_standard.value}"
                ),
                license_info=(
                    "ecoinvent commercial license - "
                    "attribution required, no bulk redistribution"
                )
            )

            # Create provenance
            provenance = self.create_provenance_info(
                is_proxy=False,
                fallback_chain=[self.name]
            )

            # Uncertainty from ecoinvent pedigree
            uncertainty = result.get("uncertainty", {}).get("coefficient_of_variation", 0.10)

            # Create response
            response = FactorResponse(
                factor_id=factor_id,
                value=value,
                unit=unit,
                uncertainty=uncertainty,
                metadata=metadata,
                provenance=provenance
            )

            # Calculate and set provenance hash
            response.provenance.calculation_hash = self.calculate_provenance_hash(response)

            # Log successful lookup
            latency_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000
            self.log_lookup(request, success=True, latency_ms=latency_ms)

            return response

        except (SourceUnavailableError, RateLimitExceededError):
            raise
        except Exception as e:
            latency_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000
            self.log_lookup(
                request,
                success=False,
                latency_ms=latency_ms,
                error=str(e)
            )
            logger.error(f"Error fetching factor from ecoinvent: {e}", exc_info=True)
            return None

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of ecoinvent API.

        Returns:
            Health status dictionary
        """
        start_time = DeterministicClock.utcnow()

        try:
            # Try to fetch a known factor (e.g., "steel")
            data = await self._make_request(
                "activities",
                params={"name": "steel", "limit": 1}
            )

            latency_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000

            return {
                "status": "healthy",
                "latency_ms": latency_ms,
                "last_check": DeterministicClock.utcnow(),
                "error": None,
                "request_count": self.request_count,
                "rate_limit": self.config.rate_limit
            }

        except Exception as e:
            latency_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000

            return {
                "status": "unhealthy",
                "latency_ms": latency_ms,
                "last_check": DeterministicClock.utcnow(),
                "error": str(e)
            }

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
