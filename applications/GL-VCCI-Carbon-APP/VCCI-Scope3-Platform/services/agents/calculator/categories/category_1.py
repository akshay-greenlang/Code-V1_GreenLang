# -*- coding: utf-8 -*-
"""
Category 1: Purchased Goods & Services Calculator
GL-VCCI Scope 3 Platform

3-Tier Calculation Waterfall:
- Tier 1: Supplier-specific Product Carbon Footprint (PCF)
- Tier 2: Average-data (product emission factors)
- Tier 3: Spend-based (economic intensity)

Version: 1.0.0
Date: 2025-10-30
"""

import logging
import asyncio
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

from greenlang.determinism import DeterministicClock
from ..models import (
    Category1Input,
    CalculationResult,
    DataQualityInfo,
    EmissionFactorInfo,
    ProvenanceChain,
    UncertaintyResult,
)
from ..config import TierType, get_config
from ..exceptions import (
    DataValidationError,
    EmissionFactorNotFoundError,
    TierFallbackError,
    CalculationError,
)

logger = logging.getLogger(__name__)


class Category1Calculator:
    """
    Category 1 (Purchased Goods & Services) calculator.

    Implements 3-tier calculation waterfall with automatic fallback:
    1. Tier 1: Supplier-specific PCF (highest quality)
    2. Tier 2: Product emission factors from databases
    3. Tier 3: Spend-based using economic intensity factors

    Features:
    - Automatic tier selection based on data availability
    - Data quality scoring (DQI)
    - Uncertainty propagation
    - Product categorization support
    - Complete provenance tracking
    """

    def __init__(
        self,
        factor_broker: Any,
        industry_mapper: Any,
        uncertainty_engine: Any,
        provenance_builder: Any,
        config: Optional[Any] = None
    ):
        """
        Initialize Category 1 calculator.

        Args:
            factor_broker: FactorBroker instance for emission factors
            industry_mapper: IndustryMapper instance for product categorization
            uncertainty_engine: UncertaintyEngine for Monte Carlo
            provenance_builder: ProvenanceChainBuilder for tracking
            config: Calculator configuration
        """
        self.factor_broker = factor_broker
        self.industry_mapper = industry_mapper
        self.uncertainty_engine = uncertainty_engine
        self.provenance_builder = provenance_builder
        self.config = config or get_config()

        logger.info("Initialized Category1Calculator")

    async def calculate(self, input_data: Category1Input) -> CalculationResult:
        """
        Calculate Category 1 emissions with 3-tier fallback.

        Args:
            input_data: Category 1 input data

        Returns:
            CalculationResult with emissions and provenance

        Raises:
            DataValidationError: If input data is invalid
            TierFallbackError: If all tiers fail
        """
        start_time = DeterministicClock.utcnow()

        # Validate input
        self._validate_input(input_data)

        # Attempt tier cascade
        attempted_tiers = []
        result = None

        try:
            # Tier 1: Supplier-specific PCF
            if self.config.category_1_prefer_supplier_pcf and input_data.supplier_pcf:
                logger.info(f"Attempting Tier 1 calculation for {input_data.product_name}")
                result = await self._calculate_tier_1(input_data)
                attempted_tiers.append(1)

                if result:
                    logger.info(f"Tier 1 successful for {input_data.product_name}")
                    return result

            # Tier 2: Product emission factors
            if self.config.category_1_enable_tier_fallback:
                logger.info(f"Attempting Tier 2 calculation for {input_data.product_name}")
                result = await self._calculate_tier_2(input_data)
                attempted_tiers.append(2)

                if result and result.data_quality.dqi_score >= self.config.category_1_min_dqi_score:
                    logger.info(f"Tier 2 successful for {input_data.product_name}")
                    return result

            # Tier 3: Spend-based
            if input_data.spend_usd and input_data.spend_usd > 0:
                logger.info(f"Attempting Tier 3 calculation for {input_data.product_name}")
                result = await self._calculate_tier_3(input_data)
                attempted_tiers.append(3)

                if result:
                    logger.info(f"Tier 3 successful for {input_data.product_name}")
                    return result

            # If we get here, all tiers failed
            raise TierFallbackError(
                attempted_tiers=attempted_tiers,
                reason="No suitable emission factor found and no spend data available",
                category=1
            )

        except TierFallbackError:
            raise
        except Exception as e:
            logger.error(f"Category 1 calculation failed: {e}", exc_info=True)
            raise CalculationError(
                calculation_type="category_1",
                reason=str(e),
                category=1,
                input_data=input_data.dict()
            )

    async def _calculate_tier_1(
        self, input_data: Category1Input
    ) -> Optional[CalculationResult]:
        """
        Tier 1: Supplier-specific PCF calculation.

        Formula: emissions = quantity × supplier_pcf

        Args:
            input_data: Category 1 input

        Returns:
            CalculationResult or None if data unavailable
        """
        if not input_data.supplier_pcf or input_data.supplier_pcf <= 0:
            return None

        # Calculate emissions
        emissions_kgco2e = input_data.quantity * input_data.supplier_pcf

        # Uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo and input_data.supplier_pcf_uncertainty:
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=input_data.quantity,
                quantity_uncertainty=0.05,  # Assume 5% for measured quantity
                emission_factor=input_data.supplier_pcf,
                factor_uncertainty=input_data.supplier_pcf_uncertainty,
                iterations=self.config.monte_carlo_iterations
            )

        # Emission factor info (supplier-provided)
        ef_info = EmissionFactorInfo(
            factor_id=f"supplier_pcf_{input_data.supplier_name or 'unknown'}",
            value=input_data.supplier_pcf,
            unit=f"kgCO2e/{input_data.quantity_unit}",
            source="supplier_specific",
            source_version="2024",
            gwp_standard="AR6",
            uncertainty=input_data.supplier_pcf_uncertainty or 0.10,
            data_quality_score=self.config.tier_1_dqi_score,
            reference_year=2024,
            geographic_scope=input_data.region,
            hash=self.provenance_builder.hash_factor_info(
                input_data.supplier_pcf,
                input_data.supplier_name or "unknown"
            )
        )

        # Data quality
        data_quality = DataQualityInfo(
            dqi_score=self.config.tier_1_dqi_score,
            tier=TierType.TIER_1,
            rating="excellent",
            pedigree_score=4.8,
            warnings=[]
        )

        # Provenance chain
        provenance = await self.provenance_builder.build(
            category=1,
            tier=TierType.TIER_1,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": "quantity × supplier_pcf",
                "quantity": input_data.quantity,
                "quantity_unit": input_data.quantity_unit,
                "supplier_pcf": input_data.supplier_pcf,
                "result_kgco2e": emissions_kgco2e,
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=1,
            tier=TierType.TIER_1,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_1_supplier_pcf",
            warnings=[],
            metadata={
                "supplier_name": input_data.supplier_name,
                "product_name": input_data.product_name,
            }
        )

    async def _calculate_tier_2(
        self, input_data: Category1Input
    ) -> Optional[CalculationResult]:
        """
        Tier 2: Average product emission factor calculation.

        Formula: emissions = quantity × product_ef

        Uses Factor Broker to find appropriate emission factor
        based on product categorization.

        Args:
            input_data: Category 1 input

        Returns:
            CalculationResult or None if factor not found
        """
        # Get emission factor from Factor Broker
        emission_factor = await self._get_product_emission_factor(input_data)

        if not emission_factor:
            logger.warning(
                f"No emission factor found for {input_data.product_name} "
                f"in {input_data.region}"
            )
            return None

        # Calculate emissions
        emissions_kgco2e = input_data.quantity * emission_factor.value

        # Uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=input_data.quantity,
                quantity_uncertainty=0.10,  # 10% for typical procurement data
                emission_factor=emission_factor.value,
                factor_uncertainty=emission_factor.uncertainty,
                iterations=self.config.monte_carlo_iterations
            )

        # Emission factor info
        ef_info = EmissionFactorInfo(
            factor_id=emission_factor.factor_id,
            value=emission_factor.value,
            unit=emission_factor.unit,
            source=emission_factor.source,
            source_version=emission_factor.metadata.source_version,
            gwp_standard=emission_factor.metadata.gwp_standard.value,
            uncertainty=emission_factor.uncertainty,
            data_quality_score=emission_factor.data_quality_score,
            reference_year=emission_factor.metadata.reference_year,
            geographic_scope=emission_factor.metadata.geographic_scope,
            hash=emission_factor.provenance.calculation_hash or "unknown"
        )

        # Data quality (blend of factor quality and tier score)
        base_dqi = self.config.tier_2_dqi_score
        adjusted_dqi = (base_dqi + emission_factor.data_quality_score) / 2

        warnings = []
        if adjusted_dqi < self.config.low_dqi_threshold:
            warnings.append(
                f"Low data quality score ({adjusted_dqi:.1f}/100). "
                "Consider obtaining supplier-specific data."
            )

        data_quality = DataQualityInfo(
            dqi_score=adjusted_dqi,
            tier=TierType.TIER_2,
            rating=self._get_quality_rating(adjusted_dqi),
            pedigree_score=emission_factor.data_quality_score / 20.0,  # Convert to 0-5 scale
            warnings=warnings
        )

        # Provenance chain
        provenance = await self.provenance_builder.build(
            category=1,
            tier=TierType.TIER_2,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": "quantity × product_ef",
                "quantity": input_data.quantity,
                "quantity_unit": input_data.quantity_unit,
                "product_ef": emission_factor.value,
                "result_kgco2e": emissions_kgco2e,
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=1,
            tier=TierType.TIER_2,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_2_average_data",
            warnings=warnings,
            metadata={
                "product_name": input_data.product_name,
                "emission_factor_source": emission_factor.source,
            }
        )

    async def _calculate_tier_3(
        self, input_data: Category1Input
    ) -> Optional[CalculationResult]:
        """
        Tier 3: Spend-based calculation.

        Formula: emissions = spend_usd × economic_intensity_ef

        Uses economic input-output factors based on sector.

        Args:
            input_data: Category 1 input

        Returns:
            CalculationResult or None if spend data unavailable
        """
        if not input_data.spend_usd or input_data.spend_usd <= 0:
            return None

        # Get economic intensity factor
        economic_sector = input_data.economic_sector or "average"
        intensity_factor = await self._get_economic_intensity_factor(
            economic_sector, input_data.region
        )

        if not intensity_factor:
            logger.error(f"No economic intensity factor for sector {economic_sector}")
            return None

        # Calculate emissions
        emissions_kgco2e = input_data.spend_usd * intensity_factor.value

        # Uncertainty propagation (higher uncertainty for spend-based)
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=input_data.spend_usd,
                quantity_uncertainty=0.15,  # 15% for spend data
                emission_factor=intensity_factor.value,
                factor_uncertainty=intensity_factor.uncertainty,
                iterations=self.config.monte_carlo_iterations
            )

        # Emission factor info
        ef_info = EmissionFactorInfo(
            factor_id=intensity_factor.factor_id,
            value=intensity_factor.value,
            unit="kgCO2e/USD",
            source=intensity_factor.source,
            source_version=intensity_factor.metadata.source_version,
            gwp_standard=intensity_factor.metadata.gwp_standard.value,
            uncertainty=intensity_factor.uncertainty,
            data_quality_score=intensity_factor.data_quality_score,
            reference_year=intensity_factor.metadata.reference_year,
            geographic_scope=intensity_factor.metadata.geographic_scope,
            hash=intensity_factor.provenance.calculation_hash or "unknown"
        )

        # Data quality (lowest tier)
        warnings = [
            "Spend-based calculation has lower accuracy than product-specific data",
            "Consider upgrading to Tier 1 or Tier 2 data for better accuracy"
        ]

        data_quality = DataQualityInfo(
            dqi_score=self.config.tier_3_dqi_score,
            tier=TierType.TIER_3,
            rating="fair",
            pedigree_score=2.5,
            warnings=warnings
        )

        # Provenance chain
        provenance = await self.provenance_builder.build(
            category=1,
            tier=TierType.TIER_3,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": "spend_usd × economic_intensity_ef",
                "spend_usd": input_data.spend_usd,
                "economic_intensity_ef": intensity_factor.value,
                "economic_sector": economic_sector,
                "result_kgco2e": emissions_kgco2e,
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=1,
            tier=TierType.TIER_3,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_3_spend_based",
            warnings=warnings,
            metadata={
                "product_name": input_data.product_name,
                "spend_usd": input_data.spend_usd,
                "economic_sector": economic_sector,
            }
        )

    async def _get_product_emission_factor(
        self, input_data: Category1Input
    ) -> Optional[Any]:
        """
        Get product emission factor from Factor Broker.

        Strategy:
        1. Use product_code if provided
        2. Categorize product name using IndustryMapper
        3. Look up emission factor

        Args:
            input_data: Category 1 input

        Returns:
            FactorResponse or None
        """
        from ...factor_broker.models import FactorRequest

        product_category = input_data.product_category

        # If no category provided, try to categorize
        if not product_category and input_data.product_code:
            product_category = input_data.product_code

        if not product_category:
            # Use IndustryMapper to categorize
            mapping = self.industry_mapper.map(input_data.product_name)
            if mapping.matched:
                product_category = mapping.matched_title

        if not product_category:
            logger.warning(f"Could not categorize product: {input_data.product_name}")
            return None

        # Request factor from broker
        request = FactorRequest(
            product=product_category,
            region=input_data.region,
            gwp_standard="AR6",
            unit=input_data.quantity_unit,
            category=input_data.product_category
        )

        try:
            response = await self.factor_broker.resolve(request)
            return response
        except Exception as e:
            logger.error(f"Factor lookup failed: {e}")
            return None

    async def _get_economic_intensity_factor(
        self, sector: str, region: str
    ) -> Optional[Any]:
        """
        Get economic intensity factor for spend-based calculation.

        Args:
            sector: Economic sector
            region: Region code

        Returns:
            FactorResponse or None
        """
        from ...factor_broker.models import FactorRequest

        request = FactorRequest(
            product=f"{sector}_economic_intensity",
            region=region,
            gwp_standard="AR6",
            unit="usd",
            category="economic_intensity"
        )

        try:
            response = await self.factor_broker.resolve(request)
            return response
        except Exception as e:
            logger.warning(f"Economic intensity factor not found: {e}")
            # Return default if not found
            return self._get_default_economic_intensity(sector, region)

    def _get_default_economic_intensity(self, sector: str, region: str) -> Any:
        """Get default economic intensity factor."""
        # Default economic intensities (kgCO2e/USD) - approximations
        defaults = {
            "manufacturing": 0.45,
            "services": 0.22,
            "agriculture": 0.38,
            "construction": 0.42,
            "average": 0.35,
        }

        value = defaults.get(sector.lower(), defaults["average"])

        # Create mock response
        from ...factor_broker.models import (
            FactorResponse,
            FactorMetadata,
            ProvenanceInfo,
            SourceType,
            DataQualityIndicator,
        )

        return FactorResponse(
            factor_id=f"default_eio_{sector}_{region}",
            value=value,
            unit="kgCO2e/USD",
            uncertainty=0.50,  # High uncertainty for defaults
            metadata=FactorMetadata(
                source=SourceType.PROXY,
                source_version="default_v1",
                gwp_standard="AR6",
                reference_year=2024,
                geographic_scope=region,
                data_quality=DataQualityIndicator(
                    reliability=2,
                    completeness=2,
                    temporal_correlation=3,
                    geographical_correlation=2,
                    technological_correlation=2,
                    overall_score=44
                )
            ),
            provenance=ProvenanceInfo(
                is_proxy=True,
                proxy_method="default_economic_intensity"
            )
        )

    def _validate_input(self, input_data: Category1Input):
        """
        Validate Category 1 input data.

        Args:
            input_data: Input to validate

        Raises:
            DataValidationError: If validation fails
        """
        if input_data.quantity <= 0:
            raise DataValidationError(
                field="quantity",
                value=input_data.quantity,
                reason="Quantity must be positive",
                category=1
            )

        if not input_data.product_name or not input_data.product_name.strip():
            raise DataValidationError(
                field="product_name",
                value=input_data.product_name,
                reason="Product name cannot be empty",
                category=1
            )

        # Check if at least one tier's data is available
        has_tier_1 = input_data.supplier_pcf and input_data.supplier_pcf > 0
        has_tier_2 = bool(input_data.product_code or input_data.product_category)
        has_tier_3 = input_data.spend_usd and input_data.spend_usd > 0

        if not (has_tier_1 or has_tier_2 or has_tier_3):
            raise DataValidationError(
                field="tier_data",
                value=None,
                reason=(
                    "At least one tier's data must be provided: "
                    "supplier_pcf (Tier 1), product_code/category (Tier 2), "
                    "or spend_usd (Tier 3)"
                ),
                category=1
            )

    def _get_quality_rating(self, dqi_score: float) -> str:
        """Convert DQI score to quality rating."""
        if dqi_score >= 80:
            return "excellent"
        elif dqi_score >= 60:
            return "good"
        elif dqi_score >= 40:
            return "fair"
        else:
            return "poor"


__all__ = ["Category1Calculator"]
