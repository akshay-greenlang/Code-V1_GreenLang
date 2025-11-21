# -*- coding: utf-8 -*-
"""
EPA US Factor Source
GL-VCCI Scope 3 Platform

Integration with US EPA Emission Factors Hub for US-specific emissions data.

Version: 1.0.0
License: Public Domain (US Government work)
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import asyncio
import aiohttp
from tenacity import (
from greenlang.determinism import DeterministicClock
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
    GWPStandard
)
from ..exceptions import (
    SourceUnavailableError,
    ValidationError
)
from ..config import SourceConfig


logger = logging.getLogger(__name__)


class EPASource(FactorSource):
    """
    EPA US emission factor data source.

    Provides access to US EPA Emission Factors Hub.
    Public domain data under US Government work.

    Particularly strong for:
    - US electricity grid (eGRID factors)
    - US industrial processes
    - US transportation

    Attributes:
        api_endpoint: EPA API endpoint URL
        session: aiohttp session for API calls
        source_url: Official source URL for attribution
    """

    SUPPORTED_REGIONS = {"US"}  # EPA primarily covers US

    def __init__(self, config: SourceConfig):
        """
        Initialize EPA source.

        Args:
            config: Source configuration
        """
        super().__init__(
            source_type=SourceType.EPA_US,
            config=config,
            version="2024"
        )

        self.api_endpoint = config.api_endpoint or (
            "https://api.epa.gov/easey/emission-factors"
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self.source_url = (
            "https://www.epa.gov/climateleadership/ghg-emission-factors-hub"
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """
        Get or create aiohttp session.

        Returns:
            aiohttp.ClientSession instance
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(
                headers={
                    "Accept": "application/json",
                    "User-Agent": "GL-VCCI-FactorBroker/1.0.0"
                },
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            )

        return self.session

    async def close(self):
        """Close aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to EPA API with retry logic.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            Response data as dictionary

        Raises:
            SourceUnavailableError: If API is unavailable
        """
        session = await self._get_session()
        url = f"{self.api_endpoint}/{endpoint}"

        try:
            async with session.get(url, params=params) as response:
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

    def _is_region_supported(self, region: str) -> bool:
        """
        Check if region is supported by EPA.

        Args:
            region: Region code

        Returns:
            True if supported, False otherwise
        """
        return region.upper() in self.SUPPORTED_REGIONS

    def _map_gwp_standard(self, gwp_standard: GWPStandard) -> str:
        """
        Map GWP standard to EPA parameter.

        Args:
            gwp_standard: GWP standard

        Returns:
            EPA GWP parameter
        """
        # EPA supports both AR5 and AR6
        mapping = {
            GWPStandard.AR5: "AR5",
            GWPStandard.AR6: "AR6"
        }
        return mapping.get(gwp_standard, "AR6")

    def _calculate_data_quality(
        self,
        region: str,
        reference_year: int,
        data_source: str
    ) -> DataQualityIndicator:
        """
        Calculate data quality for EPA factor.

        EPA provides high quality government data for US.

        Args:
            region: Region code
            reference_year: Reference year
            data_source: Specific EPA data source (e.g., eGRID, GHGRP)

        Returns:
            DataQualityIndicator instance
        """
        current_year = DeterministicClock.utcnow().year

        # Reliability: 5 (government data)
        reliability = 5

        # Completeness: 4-5 depending on source
        completeness = 5 if data_source == "eGRID" else 4

        # Temporal: Based on reference year
        year_diff = current_year - reference_year
        if year_diff == 0:
            temporal = 5
        elif year_diff == 1:
            temporal = 4
        else:
            temporal = max(1, 5 - year_diff)

        # Geographical: 5 for US
        geographical = 5

        # Technological: 4 (standard technologies)
        technological = 4

        return self.create_data_quality_indicator(
            reliability=reliability,
            completeness=completeness,
            temporal=temporal,
            geographical=geographical,
            technological=technological
        )

    def _map_product_to_category(self, product: str) -> str:
        """
        Map product name to EPA category.

        Args:
            product: Product name

        Returns:
            EPA category
        """
        product_lower = product.lower()

        if "electric" in product_lower or "power" in product_lower:
            return "electricity"
        elif any(word in product_lower for word in ["gas", "fuel", "coal", "oil"]):
            return "stationary_combustion"
        elif any(word in product_lower for word in ["truck", "car", "vehicle"]):
            return "mobile_combustion"
        elif "refriger" in product_lower or "hvac" in product_lower:
            return "fugitive_emissions"
        else:
            return "industrial_process"

    async def fetch_factor(
        self,
        request: FactorRequest
    ) -> Optional[FactorResponse]:
        """
        Fetch emission factor from EPA database.

        Args:
            request: Factor request

        Returns:
            FactorResponse if found, None otherwise

        Raises:
            SourceUnavailableError: If EPA API is unavailable
        """
        start_time = DeterministicClock.utcnow()

        try:
            self.validate_request(request)

            # Check if region is supported
            region = self.normalize_region(request.region)
            if not self._is_region_supported(region):
                self.log_lookup(
                    request,
                    success=False,
                    error=f"Region {region} not supported by EPA"
                )
                return None

            # Normalize product
            product = self.normalize_product_name(request.product)

            # Build query parameters
            category = self._map_product_to_category(product)
            params = {
                "category": category,
                "fuel_type": product,
                "year": request.year or 2024,
                "gwp_standard": self._map_gwp_standard(request.gwp_standard)
            }

            # Make API request
            data = await self._make_request("factors", params)

            if not data or not data.get("emission_factors"):
                # Factor not found
                latency_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000
                self.log_lookup(request, success=False, latency_ms=latency_ms)
                return None

            # Parse result
            result = data["emission_factors"][0]

            # Extract factor value
            value = result.get("co2e_factor", 0.0)
            if value <= 0:
                return None

            # Determine unit
            unit = result.get("unit", "kgCO2e")
            if request.unit:
                unit = request.unit

            # Create factor ID
            factor_id = self.create_factor_id(
                product=request.product,
                region=region,
                gwp_standard=request.gwp_standard,
                unit=unit
            )

            # Calculate data quality
            reference_year = result.get("year", 2024)
            data_source = result.get("source", "EPA")
            data_quality = self._calculate_data_quality(
                region,
                reference_year,
                data_source
            )

            # Create metadata
            metadata = FactorMetadata(
                source=SourceType.EPA_US,
                source_version=self.version,
                source_dataset_id=result.get("id"),
                gwp_standard=request.gwp_standard,
                reference_year=reference_year,
                geographic_scope=region,
                technology_scope=result.get("description", "Average"),
                data_quality=data_quality,
                citation=(
                    f"US EPA Emission Factors Hub {self.version}, "
                    f"{request.product}, {region}"
                ),
                license_info="Public Domain (US Government work)"
            )

            # Create provenance
            provenance = self.create_provenance_info(
                is_proxy=False,
                fallback_chain=[self.name]
            )

            # Uncertainty (EPA typically has moderate uncertainty)
            uncertainty = result.get("uncertainty", 0.10)  # ±10%

            # Create response
            response = FactorResponse(
                factor_id=factor_id,
                value=value,
                unit=unit,
                uncertainty=uncertainty,
                metadata=metadata,
                provenance=provenance
            )

            # Calculate provenance hash
            response.provenance.calculation_hash = self.calculate_provenance_hash(response)

            # Log successful lookup
            latency_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000
            self.log_lookup(request, success=True, latency_ms=latency_ms)

            return response

        except SourceUnavailableError:
            raise
        except Exception as e:
            latency_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000
            self.log_lookup(
                request,
                success=False,
                latency_ms=latency_ms,
                error=str(e)
            )
            logger.error(f"Error fetching factor from EPA: {e}", exc_info=True)
            return None

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of EPA API.

        Returns:
            Health status dictionary
        """
        start_time = DeterministicClock.utcnow()

        try:
            # Try to fetch electricity factor
            data = await self._make_request(
                "factors",
                params={"category": "electricity", "year": 2024, "limit": 1}
            )

            latency_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000

            return {
                "status": "healthy",
                "latency_ms": latency_ms,
                "last_check": DeterministicClock.utcnow(),
                "error": None
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
