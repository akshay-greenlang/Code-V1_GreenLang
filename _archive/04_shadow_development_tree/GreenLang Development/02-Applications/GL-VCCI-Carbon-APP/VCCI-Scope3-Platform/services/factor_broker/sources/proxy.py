# -*- coding: utf-8 -*-
"""
Proxy Factor Source
GL-VCCI Scope 3 Platform

Fallback proxy factor calculation when exact factors are not available.
Uses category averages and industry estimates with appropriate data quality degradation.

Version: 1.0.0
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import statistics

from .base import FactorSource
from greenlang.determinism import DeterministicClock
from ..models import (
    FactorRequest,
    FactorResponse,
    FactorMetadata,
    DataQualityIndicator,
    SourceType,
    GWPStandard
)
from ..exceptions import (
    ProxyCalculationError,
    DataQualityError
)
from ..config import SourceConfig, ProxyConfig


logger = logging.getLogger(__name__)


class ProxySource(FactorSource):
    """
    Proxy emission factor calculator.

    When exact emission factors are not available from primary sources,
    this source calculates proxy factors using:
    - Category averages
    - Industry estimates
    - Similar product interpolation

    Proxy factors are flagged and have lower data quality scores.

    Attributes:
        proxy_config: Proxy calculation configuration
        category_factors: Pre-loaded category average factors
    """

    # Category average factors (kgCO2e per unit)
    # These are rough estimates for major categories
    CATEGORY_AVERAGES = {
        "metals": {
            "factor": 2.5,
            "unit": "kgCO2e/kg",
            "uncertainty": 0.50,
            "examples": ["steel", "aluminum", "copper", "iron"]
        },
        "plastics": {
            "factor": 3.0,
            "unit": "kgCO2e/kg",
            "uncertainty": 0.50,
            "examples": ["polyethylene", "polypropylene", "pvc", "pet"]
        },
        "electronics": {
            "factor": 150.0,
            "unit": "kgCO2e/unit",
            "uncertainty": 0.60,
            "examples": ["computer", "phone", "tablet", "monitor"]
        },
        "textiles": {
            "factor": 15.0,
            "unit": "kgCO2e/kg",
            "uncertainty": 0.50,
            "examples": ["cotton", "polyester", "wool", "nylon"]
        },
        "chemicals": {
            "factor": 2.0,
            "unit": "kgCO2e/kg",
            "uncertainty": 0.60,
            "examples": ["solvent", "acid", "base", "reagent"]
        },
        "energy": {
            "factor": 0.40,
            "unit": "kgCO2e/kWh",
            "uncertainty": 0.30,
            "examples": ["electricity", "power", "energy"]
        },
        "fuels": {
            "factor": 2.5,
            "unit": "kgCO2e/litre",
            "uncertainty": 0.20,
            "examples": ["gasoline", "diesel", "jet fuel", "natural gas"]
        },
        "transportation": {
            "factor": 0.10,
            "unit": "kgCO2e/km",
            "uncertainty": 0.40,
            "examples": ["truck", "ship", "rail", "air freight"]
        },
        "construction": {
            "factor": 0.50,
            "unit": "kgCO2e/kg",
            "uncertainty": 0.50,
            "examples": ["concrete", "cement", "brick", "lumber"]
        },
        "food": {
            "factor": 5.0,
            "unit": "kgCO2e/kg",
            "uncertainty": 0.60,
            "examples": ["meat", "dairy", "vegetables", "grains"]
        },
        "packaging": {
            "factor": 2.0,
            "unit": "kgCO2e/kg",
            "uncertainty": 0.50,
            "examples": ["cardboard", "paper", "plastic film", "glass"]
        },
        "services": {
            "factor": 50.0,
            "unit": "kgCO2e/hour",
            "uncertainty": 0.70,
            "examples": ["consulting", "maintenance", "support"]
        }
    }

    def __init__(self, config: SourceConfig, proxy_config: Optional[ProxyConfig] = None):
        """
        Initialize proxy source.

        Args:
            config: Source configuration
            proxy_config: Proxy-specific configuration
        """
        super().__init__(
            source_type=SourceType.PROXY,
            config=config,
            version="1.0.0"
        )

        self.proxy_config = proxy_config or ProxyConfig()
        self.category_factors = self.CATEGORY_AVERAGES

    def _identify_category(self, product: str) -> Optional[str]:
        """
        Identify product category from product name.

        Args:
            product: Product name

        Returns:
            Category name if identified, None otherwise
        """
        product_lower = product.lower()

        # Check each category's examples
        for category, data in self.category_factors.items():
            examples = data.get("examples", [])
            for example in examples:
                if example in product_lower or product_lower in example:
                    return category

        # Fallback heuristics
        if any(word in product_lower for word in ["metal", "steel", "aluminum", "copper"]):
            return "metals"
        elif any(word in product_lower for word in ["plastic", "polymer", "resin"]):
            return "plastics"
        elif any(word in product_lower for word in ["electric", "power", "energy"]):
            return "energy"
        elif any(word in product_lower for word in ["fuel", "gas", "oil", "diesel"]):
            return "fuels"
        elif any(word in product_lower for word in ["truck", "transport", "freight", "ship"]):
            return "transportation"
        elif any(word in product_lower for word in ["concrete", "cement", "building"]):
            return "construction"
        elif any(word in product_lower for word in ["food", "meat", "dairy"]):
            return "food"

        return None

    def _calculate_category_average(
        self,
        category: str,
        region: str
    ) -> Dict[str, Any]:
        """
        Calculate category average factor.

        Args:
            category: Product category
            region: Region code

        Returns:
            Dictionary with factor value, unit, and uncertainty
        """
        if category not in self.category_factors:
            raise ProxyCalculationError(
                product="unknown",
                category=category,
                reason=f"Category '{category}' not recognized"
            )

        category_data = self.category_factors[category]

        # Base factor from category
        base_factor = category_data["factor"]
        unit = category_data["unit"]
        base_uncertainty = category_data["uncertainty"]

        # Regional adjustment (simplified)
        regional_multiplier = self._get_regional_multiplier(region)
        adjusted_factor = base_factor * regional_multiplier

        # Increase uncertainty for regional adjustment
        adjusted_uncertainty = min(
            base_uncertainty * 1.2,
            0.80  # Cap at 80%
        )

        return {
            "value": adjusted_factor,
            "unit": unit,
            "uncertainty": adjusted_uncertainty,
            "category": category
        }

    def _get_regional_multiplier(self, region: str) -> float:
        """
        Get regional multiplier for carbon intensity.

        Args:
            region: Region code

        Returns:
            Regional multiplier (1.0 = global average)
        """
        # Simplified regional multipliers based on grid carbon intensity
        # and industrial practices
        regional_multipliers = {
            "US": 1.0,   # Average
            "GB": 0.85,  # Lower (cleaner grid)
            "DE": 1.1,   # Slightly higher (coal dependency)
            "CN": 1.3,   # Higher (coal-heavy)
            "FR": 0.7,   # Lower (nuclear)
            "IN": 1.2,   # Higher
            "BR": 0.9,   # Lower (hydroelectric)
            "AU": 1.1,   # Slightly higher
        }

        return regional_multipliers.get(region.upper(), 1.0)

    def _create_proxy_data_quality(self, category: str) -> DataQualityIndicator:
        """
        Create data quality indicator for proxy factor.

        Proxy factors have degraded data quality.

        Args:
            category: Product category

        Returns:
            DataQualityIndicator instance
        """
        # Proxy factors have lower quality scores
        return self.create_data_quality_indicator(
            reliability=2,      # Low reliability (estimated)
            completeness=2,     # Low completeness (category average)
            temporal=3,         # Moderate temporal (current estimates)
            geographical=2,     # Low geographical (regional adjustment)
            technological=2     # Low technological (mixed technologies)
        )

    async def fetch_factor(
        self,
        request: FactorRequest
    ) -> Optional[FactorResponse]:
        """
        Calculate proxy emission factor.

        Args:
            request: Factor request

        Returns:
            FactorResponse with proxy factor

        Raises:
            ProxyCalculationError: If proxy calculation fails
        """
        start_time = DeterministicClock.utcnow()

        try:
            self.validate_request(request)

            # Normalize inputs
            product = self.normalize_product_name(request.product)
            region = self.normalize_region(request.region)

            # Identify category
            category = request.category or self._identify_category(product)

            if not category:
                raise ProxyCalculationError(
                    product=request.product,
                    category=None,
                    reason="Could not identify product category for proxy calculation"
                )

            # Calculate category average
            proxy_data = self._calculate_category_average(category, region)

            # Create factor ID
            factor_id = self.create_factor_id(
                product=f"proxy_{category}",
                region=region,
                gwp_standard=request.gwp_standard,
                unit=proxy_data["unit"]
            )

            # Create data quality indicator
            data_quality = self._create_proxy_data_quality(category)

            # Create metadata
            metadata = FactorMetadata(
                source=SourceType.PROXY,
                source_version=self.version,
                source_dataset_id=f"proxy_{category}_{region}",
                gwp_standard=request.gwp_standard,
                reference_year=DeterministicClock.utcnow().year,
                geographic_scope=region,
                technology_scope="Category average",
                data_quality=data_quality,
                citation=(
                    f"Proxy calculation (category average), "
                    f"category: {category}, region: {region}"
                ),
                license_info="GL-VCCI internal calculation"
            )

            # Create provenance
            provenance = self.create_provenance_info(
                is_proxy=True,
                fallback_chain=["ecoinvent", "desnz_uk", "epa_us", "proxy"],
                proxy_method=self.proxy_config.method
            )

            # Create warning message
            warning = (
                f"This is a proxy factor (category average for '{category}'). "
                f"Data quality score: {data_quality.overall_score}/100. "
                f"Consider supplier engagement for Tier 1 data collection."
            )

            # Create response
            response = FactorResponse(
                factor_id=factor_id,
                value=proxy_data["value"],
                unit=proxy_data["unit"],
                uncertainty=proxy_data["uncertainty"],
                metadata=metadata,
                provenance=provenance,
                warning=warning if self.proxy_config.flag_in_response else None
            )

            # Calculate provenance hash
            response.provenance.calculation_hash = self.calculate_provenance_hash(response)

            # Log proxy calculation
            latency_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000
            self.log_lookup(request, success=True, latency_ms=latency_ms)

            logger.warning(
                f"Using proxy factor for '{request.product}' "
                f"(category: {category}, region: {region})"
            )

            return response

        except ProxyCalculationError:
            raise
        except Exception as e:
            latency_ms = (DeterministicClock.utcnow() - start_time).total_seconds() * 1000
            self.log_lookup(
                request,
                success=False,
                latency_ms=latency_ms,
                error=str(e)
            )
            logger.error(f"Error calculating proxy factor: {e}", exc_info=True)
            raise ProxyCalculationError(
                product=request.product,
                category=request.category,
                reason=str(e)
            )

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of proxy calculator.

        Returns:
            Health status dictionary
        """
        # Proxy calculator is always "healthy" as it's local
        return {
            "status": "healthy",
            "latency_ms": 0.0,
            "last_check": DeterministicClock.utcnow(),
            "error": None,
            "categories_available": len(self.category_factors)
        }

    def get_available_categories(self) -> List[str]:
        """
        Get list of available categories for proxy calculation.

        Returns:
            List of category names
        """
        return list(self.category_factors.keys())

    def get_category_info(self, category: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific category.

        Args:
            category: Category name

        Returns:
            Category information dictionary if found, None otherwise
        """
        return self.category_factors.get(category)
