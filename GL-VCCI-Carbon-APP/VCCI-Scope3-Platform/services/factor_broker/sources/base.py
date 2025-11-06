"""
Base Factor Source
GL-VCCI Scope 3 Platform

Abstract base class for all emission factor data sources.
Defines common interface and shared functionality.

Version: 1.0.0
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from datetime import datetime
import hashlib

from ..models import (
    FactorRequest,
    FactorResponse,
    FactorMetadata,
    DataQualityIndicator,
    ProvenanceInfo,
    SourceType,
    GWPStandard
)
from ..exceptions import (
    FactorNotFoundError,
    SourceUnavailableError,
    ValidationError
)
from ..config import SourceConfig


logger = logging.getLogger(__name__)


class FactorSource(ABC):
    """
    Abstract base class for emission factor data sources.

    All data source implementations (ecoinvent, DESNZ, EPA, proxy)
    must inherit from this class and implement the required methods.

    Attributes:
        source_type: Type of data source
        config: Source-specific configuration
        name: Human-readable source name
        version: Source database version
    """

    def __init__(
        self,
        source_type: SourceType,
        config: SourceConfig,
        version: str = "1.0.0"
    ):
        """
        Initialize factor source.

        Args:
            source_type: Type of data source
            config: Source-specific configuration
            version: Source database version
        """
        self.source_type = source_type
        self.config = config
        self.name = source_type.value
        self.version = version
        self._logger = logging.getLogger(f"{__name__}.{self.name}")

    @abstractmethod
    async def fetch_factor(
        self,
        request: FactorRequest
    ) -> Optional[FactorResponse]:
        """
        Fetch emission factor from this source.

        Args:
            request: Factor request with product, region, GWP standard, etc.

        Returns:
            FactorResponse if found, None otherwise

        Raises:
            SourceUnavailableError: If source is unavailable
            ValidationError: If request is invalid for this source
        """
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of this data source.

        Returns:
            Dictionary with health status information:
            {
                "status": "healthy" | "degraded" | "unhealthy",
                "latency_ms": float,
                "last_check": datetime,
                "error": Optional[str]
            }
        """
        pass

    def validate_request(self, request: FactorRequest) -> None:
        """
        Validate factor request for this source.

        Args:
            request: Factor request to validate

        Raises:
            ValidationError: If request is invalid
        """
        # Basic validation
        if not request.product:
            raise ValidationError(
                field="product",
                value=request.product,
                reason="Product name is required"
            )

        if not request.region:
            raise ValidationError(
                field="region",
                value=request.region,
                reason="Region code is required"
            )

        if len(request.region) != 2:
            raise ValidationError(
                field="region",
                value=request.region,
                reason="Region code must be 2 characters (ISO 3166-1 alpha-2)"
            )

    def normalize_product_name(self, product: str) -> str:
        """
        Normalize product name for consistent matching.

        Args:
            product: Product name

        Returns:
            Normalized product name
        """
        # Convert to lowercase and strip whitespace
        normalized = product.lower().strip()

        # Remove special characters
        normalized = normalized.replace("_", " ")
        normalized = normalized.replace("-", " ")

        # Collapse multiple spaces
        normalized = " ".join(normalized.split())

        return normalized

    def normalize_region(self, region: str) -> str:
        """
        Normalize region code.

        Args:
            region: Region code

        Returns:
            Normalized region code (uppercase)
        """
        return region.upper().strip()

    def create_factor_id(
        self,
        product: str,
        region: str,
        gwp_standard: GWPStandard,
        unit: str
    ) -> str:
        """
        Create unique factor identifier.

        Format: {source}_{version}_{product}_{region}_{gwp}_{unit}

        Args:
            product: Product name
            region: Region code
            gwp_standard: GWP standard
            unit: Unit of measurement

        Returns:
            Unique factor identifier
        """
        # Normalize components
        product_normalized = self.normalize_product_name(product).replace(" ", "_")
        region_normalized = self.normalize_region(region)
        unit_normalized = unit.lower().replace("/", "_per_").replace(" ", "_")

        factor_id = (
            f"{self.name}_{self.version}_{product_normalized}_"
            f"{region_normalized}_{gwp_standard.value.lower()}_{unit_normalized}"
        )

        return factor_id

    def create_data_quality_indicator(
        self,
        reliability: int = 3,
        completeness: int = 3,
        temporal: int = 3,
        geographical: int = 3,
        technological: int = 3
    ) -> DataQualityIndicator:
        """
        Create data quality indicator.

        Args:
            reliability: Data reliability score (0-5)
            completeness: Dataset completeness (0-5)
            temporal: Temporal correlation (0-5)
            geographical: Geographical correlation (0-5)
            technological: Technological correlation (0-5)

        Returns:
            DataQualityIndicator instance
        """
        dqi = DataQualityIndicator(
            reliability=reliability,
            completeness=completeness,
            temporal_correlation=temporal,
            geographical_correlation=geographical,
            technological_correlation=technological
        )

        # Calculate overall score
        dqi.overall_score = dqi.calculate_overall_score()

        return dqi

    def create_provenance_info(
        self,
        is_proxy: bool = False,
        fallback_chain: Optional[list] = None,
        proxy_method: Optional[str] = None
    ) -> ProvenanceInfo:
        """
        Create provenance information.

        Args:
            is_proxy: Whether factor is a proxy
            fallback_chain: Sources tried in cascade
            proxy_method: Proxy calculation method if applicable

        Returns:
            ProvenanceInfo instance
        """
        return ProvenanceInfo(
            lookup_timestamp=datetime.utcnow(),
            cache_hit=False,  # Will be updated by cache layer
            is_proxy=is_proxy,
            fallback_chain=fallback_chain or [self.name],
            proxy_method=proxy_method
        )

    def calculate_provenance_hash(self, factor_response: FactorResponse) -> str:
        """
        Calculate SHA256 hash for provenance chain.

        Args:
            factor_response: Factor response

        Returns:
            SHA256 hash as hex string
        """
        # Create hash input from key factor attributes
        hash_input = (
            f"{factor_response.factor_id}"
            f"{factor_response.value}"
            f"{factor_response.unit}"
            f"{factor_response.metadata.source}"
            f"{factor_response.metadata.source_version}"
            f"{factor_response.metadata.gwp_standard}"
            f"{factor_response.provenance.lookup_timestamp.isoformat()}"
        )

        return hashlib.sha256(hash_input.encode()).hexdigest()

    def log_lookup(
        self,
        request: FactorRequest,
        success: bool,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log factor lookup for monitoring.

        Args:
            request: Factor request
            success: Whether lookup was successful
            latency_ms: Lookup latency in milliseconds
            error: Error message if lookup failed
        """
        log_data = {
            "source": self.name,
            "product": request.product,
            "region": request.region,
            "gwp_standard": request.gwp_standard.value,
            "success": success,
            "latency_ms": latency_ms
        }

        if error:
            log_data["error"] = error

        if success:
            self._logger.info(f"Factor lookup successful: {log_data}")
        else:
            self._logger.warning(f"Factor lookup failed: {log_data}")

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(source={self.name}, version={self.version})"
