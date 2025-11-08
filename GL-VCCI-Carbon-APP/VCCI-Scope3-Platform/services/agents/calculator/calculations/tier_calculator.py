"""
Tier Calculator - 3-Tier Calculation Waterfall Utilities
GL-VCCI Scope 3 Platform

Provides reusable tier-based calculation logic for categories that implement
the GHG Protocol 3-tier data quality hierarchy:
- Tier 1: Supplier-specific data (highest quality)
- Tier 2: Average/secondary data (medium quality)
- Tier 3: Spend-based/proxy data (lowest quality)

Version: 1.0.0
Date: 2025-11-08
"""

import logging
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime

from ..config import TierType
from ..models import CalculationResult, DataQualityInfo
from ..exceptions import TierFallbackError, CalculationError

logger = logging.getLogger(__name__)


class TierCalculator:
    """
    Tier-based calculation engine with automatic fallback.

    Implements the GHG Protocol 3-tier hierarchy:
    1. Tier 1: Primary/supplier-specific data
    2. Tier 2: Secondary/average data
    3. Tier 3: Spend-based/proxy data

    Features:
    - Automatic tier fallback
    - Data quality threshold enforcement
    - Tier attempt tracking
    - Flexible calculation functions
    """

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize tier calculator.

        Args:
            config: Calculator configuration
        """
        self.config = config
        logger.info("Initialized TierCalculator")

    async def calculate_with_fallback(
        self,
        category: int,
        tier_1_func: Optional[Callable] = None,
        tier_2_func: Optional[Callable] = None,
        tier_3_func: Optional[Callable] = None,
        min_dqi_score: float = 40.0,
        enable_fallback: bool = True,
    ) -> CalculationResult:
        """
        Execute 3-tier calculation with automatic fallback.

        Attempts tiers in order (1 -> 2 -> 3) until successful calculation
        that meets minimum DQI score requirement.

        Args:
            category: Scope 3 category number
            tier_1_func: Async function for Tier 1 calculation
            tier_2_func: Async function for Tier 2 calculation
            tier_3_func: Async function for Tier 3 calculation
            min_dqi_score: Minimum acceptable DQI score
            enable_fallback: Enable automatic tier fallback

        Returns:
            CalculationResult from first successful tier

        Raises:
            TierFallbackError: If all tiers fail
        """
        attempted_tiers = []
        result = None

        # Tier 1: Supplier-specific/primary data
        if tier_1_func:
            logger.info(f"Attempting Tier 1 calculation for category {category}")
            try:
                result = await tier_1_func()
                attempted_tiers.append(1)

                if result and result.data_quality.dqi_score >= min_dqi_score:
                    logger.info(f"Tier 1 successful for category {category}")
                    return result
                elif result:
                    logger.warning(
                        f"Tier 1 DQI score ({result.data_quality.dqi_score:.1f}) "
                        f"below threshold ({min_dqi_score})"
                    )
            except Exception as e:
                logger.warning(f"Tier 1 calculation failed: {e}")

        # Tier 2: Average/secondary data
        if enable_fallback and tier_2_func:
            logger.info(f"Attempting Tier 2 calculation for category {category}")
            try:
                result = await tier_2_func()
                attempted_tiers.append(2)

                if result and result.data_quality.dqi_score >= min_dqi_score:
                    logger.info(f"Tier 2 successful for category {category}")
                    return result
                elif result:
                    logger.warning(
                        f"Tier 2 DQI score ({result.data_quality.dqi_score:.1f}) "
                        f"below threshold ({min_dqi_score})"
                    )
            except Exception as e:
                logger.warning(f"Tier 2 calculation failed: {e}")

        # Tier 3: Spend-based/proxy data
        if enable_fallback and tier_3_func:
            logger.info(f"Attempting Tier 3 calculation for category {category}")
            try:
                result = await tier_3_func()
                attempted_tiers.append(3)

                if result:
                    logger.info(f"Tier 3 successful for category {category}")
                    return result
            except Exception as e:
                logger.warning(f"Tier 3 calculation failed: {e}")

        # All tiers failed
        raise TierFallbackError(
            attempted_tiers=attempted_tiers,
            reason="No tier produced valid result meeting quality requirements",
            category=category
        )

    async def calculate_tier_1(
        self,
        quantity: float,
        supplier_pcf: float,
        category: int,
        uncertainty_engine: Any = None,
        provenance_builder: Any = None,
        input_data: Dict[str, Any] = None,
        **kwargs
    ) -> Optional[CalculationResult]:
        """
        Generic Tier 1 calculation: quantity × supplier_pcf.

        Args:
            quantity: Activity quantity
            supplier_pcf: Supplier-specific product carbon footprint
            category: Scope 3 category
            uncertainty_engine: Optional uncertainty engine
            provenance_builder: Optional provenance builder
            input_data: Original input data
            **kwargs: Additional parameters

        Returns:
            CalculationResult or None if data unavailable
        """
        if not supplier_pcf or supplier_pcf <= 0:
            return None

        # Calculate emissions
        emissions_kgco2e = quantity * supplier_pcf

        logger.debug(
            f"Tier 1 calculation: {quantity} × {supplier_pcf} = {emissions_kgco2e} kgCO2e"
        )

        # Build minimal result (caller should enhance with full provenance)
        return self._build_result(
            emissions_kgco2e=emissions_kgco2e,
            category=category,
            tier=TierType.TIER_1,
            dqi_score=kwargs.get('tier_1_dqi_score', 85.0),
            metadata=kwargs.get('metadata', {})
        )

    async def calculate_tier_2(
        self,
        quantity: float,
        emission_factor: float,
        category: int,
        uncertainty_engine: Any = None,
        provenance_builder: Any = None,
        input_data: Dict[str, Any] = None,
        **kwargs
    ) -> Optional[CalculationResult]:
        """
        Generic Tier 2 calculation: quantity × emission_factor.

        Args:
            quantity: Activity quantity
            emission_factor: Average emission factor from database
            category: Scope 3 category
            uncertainty_engine: Optional uncertainty engine
            provenance_builder: Optional provenance builder
            input_data: Original input data
            **kwargs: Additional parameters

        Returns:
            CalculationResult or None if factor unavailable
        """
        if not emission_factor or emission_factor <= 0:
            return None

        # Calculate emissions
        emissions_kgco2e = quantity * emission_factor

        logger.debug(
            f"Tier 2 calculation: {quantity} × {emission_factor} = {emissions_kgco2e} kgCO2e"
        )

        return self._build_result(
            emissions_kgco2e=emissions_kgco2e,
            category=category,
            tier=TierType.TIER_2,
            dqi_score=kwargs.get('tier_2_dqi_score', 65.0),
            metadata=kwargs.get('metadata', {})
        )

    async def calculate_tier_3(
        self,
        spend_usd: float,
        economic_intensity: float,
        category: int,
        uncertainty_engine: Any = None,
        provenance_builder: Any = None,
        input_data: Dict[str, Any] = None,
        **kwargs
    ) -> Optional[CalculationResult]:
        """
        Generic Tier 3 calculation: spend_usd × economic_intensity.

        Args:
            spend_usd: Spend amount in USD
            economic_intensity: Economic intensity factor (kgCO2e/USD)
            category: Scope 3 category
            uncertainty_engine: Optional uncertainty engine
            provenance_builder: Optional provenance builder
            input_data: Original input data
            **kwargs: Additional parameters

        Returns:
            CalculationResult or None if spend data unavailable
        """
        if not spend_usd or spend_usd <= 0:
            return None

        if not economic_intensity or economic_intensity <= 0:
            return None

        # Calculate emissions
        emissions_kgco2e = spend_usd * economic_intensity

        logger.debug(
            f"Tier 3 calculation: ${spend_usd} × {economic_intensity} = {emissions_kgco2e} kgCO2e"
        )

        warnings = [
            "Spend-based calculation has lower accuracy than product-specific data",
            "Consider upgrading to Tier 1 or Tier 2 data for better accuracy"
        ]

        return self._build_result(
            emissions_kgco2e=emissions_kgco2e,
            category=category,
            tier=TierType.TIER_3,
            dqi_score=kwargs.get('tier_3_dqi_score', 45.0),
            warnings=warnings,
            metadata=kwargs.get('metadata', {})
        )

    def _build_result(
        self,
        emissions_kgco2e: float,
        category: int,
        tier: TierType,
        dqi_score: float,
        warnings: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> CalculationResult:
        """
        Build minimal calculation result.

        Args:
            emissions_kgco2e: Emissions in kgCO2e
            category: Category number
            tier: Tier type
            dqi_score: Data quality score
            warnings: Optional warnings
            metadata: Optional metadata

        Returns:
            CalculationResult
        """
        from ..models import EmissionFactorInfo, ProvenanceChain

        data_quality = DataQualityInfo(
            dqi_score=dqi_score,
            tier=tier,
            rating=self._get_quality_rating(dqi_score),
            pedigree_score=dqi_score / 20.0,
            warnings=warnings or []
        )

        # Minimal provenance (caller should enhance)
        provenance = ProvenanceChain(
            calculation_timestamp=datetime.utcnow(),
            calculation_hash="placeholder",
            data_lineage=[]
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=category,
            tier=tier,
            uncertainty=None,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method=f"{tier.value}_calculation",
            warnings=warnings or [],
            metadata=metadata or {}
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

    def get_tier_priority(self, preferred_tier: Optional[TierType] = None) -> List[TierType]:
        """
        Get tier calculation priority order.

        Args:
            preferred_tier: Optionally prefer specific tier first

        Returns:
            List of tiers in priority order
        """
        if preferred_tier == TierType.TIER_1:
            return [TierType.TIER_1, TierType.TIER_2, TierType.TIER_3]
        elif preferred_tier == TierType.TIER_2:
            return [TierType.TIER_2, TierType.TIER_1, TierType.TIER_3]
        elif preferred_tier == TierType.TIER_3:
            return [TierType.TIER_3, TierType.TIER_2, TierType.TIER_1]
        else:
            # Default: highest quality first
            return [TierType.TIER_1, TierType.TIER_2, TierType.TIER_3]


__all__ = ["TierCalculator"]
