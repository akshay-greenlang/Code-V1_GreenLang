"""
DESNZ UK Factor Source
GL-VCCI Scope 3 Platform

Integration with UK DESNZ (Department for Energy Security and Net Zero)
greenhouse gas conversion factors for UK/EU emissions.

Version: 1.0.0
License: Open Government License v3.0
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import asyncio
import aiohttp
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
    GWPStandard
)
from ..exceptions import (
    SourceUnavailableError,
    ValidationError
)
from ..config import SourceConfig


logger = logging.getLogger(__name__)


class DESNZSource(FactorSource):
    """
    DESNZ UK emission factor data source.

    Provides access to UK Government greenhouse gas conversion factors.
    Free and open source under Open Government License v3.0.

    Particularly strong for:
    - UK electricity grid factors
    - UK transportation
    - EU-specific factors

    Attributes:
        api_endpoint: DESNZ API endpoint URL
        session: aiohttp session for API calls
        source_url: Official source URL for attribution
    """

    SUPPORTED_REGIONS = {"GB", "UK", "EU", "DE", "FR", "ES", "IT", "NL"}

    def __init__(self, config: SourceConfig):
        """
        Initialize DESNZ source.

        Args:
            config: Source configuration
        """
        super().__init__(
            source_type=SourceType.DESNZ_UK,
            config=config,
            version="2024"
        )

        self.api_endpoint = config.api_endpoint or (
            "https://api.gov.uk/desnz/emission-factors"
        )
        self.session: Optional[aiohttp.ClientSession] = None
        self.source_url = (
            "https://www.gov.uk/government/publications/"
            "greenhouse-gas-reporting-conversion-factors-2024"
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
        Make HTTP request to DESNZ API with retry logic.

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
        Check if region is supported by DESNZ.

        Args:
            region: Region code

        Returns:
            True if supported, False otherwise
        """
        return region.upper() in self.SUPPORTED_REGIONS

    def _map_gwp_standard(self, gwp_standard: GWPStandard) -> str:
        """
        Map GWP standard to DESNZ parameter.

        Args:
            gwp_standard: GWP standard

        Returns:
            DESNZ GWP parameter
        """
        # DESNZ 2024 uses AR6 by default
        mapping = {
            GWPStandard.AR5: "AR5",
            GWPStandard.AR6: "AR6"
        }
        return mapping.get(gwp_standard, "AR6")

    def _calculate_data_quality(
        self,
        region: str,
        reference_year: int
    ) -> DataQualityIndicator:
        """
        Calculate data quality for DESNZ factor.

        DESNZ provides high quality government data for UK/EU.

        Args:
            region: Region code
            reference_year: Reference year

        Returns:
            DataQualityIndicator instance
        """
        # DESNZ has high quality data for UK
        current_year = datetime.utcnow().year

        # Reliability: 5 (government data)
        reliability = 5

        # Completeness: 4-5 (comprehensive for UK/EU)
        completeness = 5 if region in {"GB", "UK"} else 4

        # Temporal: Based on reference year
        year_diff = current_year - reference_year
        if year_diff == 0:
            temporal = 5
        elif year_diff == 1:
            temporal = 4
        else:
            temporal = max(1, 5 - year_diff)

        # Geographical: 5 for UK, 4 for EU
        geographical = 5 if region in {"GB", "UK"} else 4

        # Technological: 4 (standard technologies)
        technological = 4

        return self.create_data_quality_indicator(
            reliability=reliability,
            completeness=completeness,
            temporal=temporal,
            geographical=geographical,
            technological=technological
        )

    async def fetch_factor(
        self,
        request: FactorRequest
    ) -> Optional[FactorResponse]:
        """
        Fetch emission factor from DESNZ database.

        Args:
            request: Factor request

        Returns:
            FactorResponse if found, None otherwise

        Raises:
            SourceUnavailableError: If DESNZ API is unavailable
        """
        start_time = datetime.utcnow()

        try:
            self.validate_request(request)

            # Check if region is supported
            region = self.normalize_region(request.region)
            if not self._is_region_supported(region):
                self.log_lookup(
                    request,
                    success=False,
                    error=f"Region {region} not supported by DESNZ"
                )
                return None

            # Normalize product
            product = self.normalize_product_name(request.product)

            # Build query parameters
            params = {
                "category": self._map_product_to_category(product),
                "name": product,
                "year": request.year or 2024,
                "gwp": self._map_gwp_standard(request.gwp_standard)
            }

            # Make API request
            data = await self._make_request("conversion-factors", params)

            if not data or not data.get("data"):
                # Factor not found
                latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                self.log_lookup(request, success=False, latency_ms=latency_ms)
                return None

            # Parse result
            result = data["data"][0]

            # Extract factor value
            value = result.get("ghg_conversion_factor", 0.0)
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
            data_quality = self._calculate_data_quality(region, reference_year)

            # Create metadata
            metadata = FactorMetadata(
                source=SourceType.DESNZ_UK,
                source_version=self.version,
                source_dataset_id=result.get("id"),
                gwp_standard=request.gwp_standard,
                reference_year=reference_year,
                geographic_scope=region,
                technology_scope=result.get("scope", "Average"),
                data_quality=data_quality,
                citation=(
                    f"UK DESNZ Conversion Factors {self.version}, "
                    f"{request.product}, {region}"
                ),
                license_info="Open Government License v3.0"
            )

            # Create provenance
            provenance = self.create_provenance_info(
                is_proxy=False,
                fallback_chain=[self.name]
            )

            # Uncertainty (DESNZ typically has low uncertainty)
            uncertainty = result.get("uncertainty", 0.05)  # ±5%

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
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.log_lookup(request, success=True, latency_ms=latency_ms)

            return response

        except SourceUnavailableError:
            raise
        except Exception as e:
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.log_lookup(
                request,
                success=False,
                latency_ms=latency_ms,
                error=str(e)
            )
            logger.error(f"Error fetching factor from DESNZ: {e}", exc_info=True)
            return None

    def _map_product_to_category(self, product: str) -> str:
        """
        Map product name to DESNZ category.

        Args:
            product: Product name

        Returns:
            DESNZ category
        """
        # Simple mapping - in production, use more sophisticated lookup
        product_lower = product.lower()

        if "electric" in product_lower or "power" in product_lower:
            return "electricity"
        elif "gas" in product_lower or "natural gas" in product_lower:
            return "fuels"
        elif any(word in product_lower for word in ["truck", "car", "vehicle", "transport"]):
            return "transport"
        elif any(word in product_lower for word in ["flight", "air", "aviation"]):
            return "business_travel"
        else:
            return "materials"

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of DESNZ API.

        Returns:
            Health status dictionary
        """
        start_time = datetime.utcnow()

        try:
            # Try to fetch electricity factor for UK
            data = await self._make_request(
                "conversion-factors",
                params={"category": "electricity", "year": 2024, "limit": 1}
            )

            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return {
                "status": "healthy",
                "latency_ms": latency_ms,
                "last_check": datetime.utcnow(),
                "error": None
            }

        except Exception as e:
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return {
                "status": "unhealthy",
                "latency_ms": latency_ms,
                "last_check": datetime.utcnow(),
                "error": str(e)
            }

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
