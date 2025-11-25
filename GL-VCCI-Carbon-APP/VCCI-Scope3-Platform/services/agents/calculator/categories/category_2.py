# -*- coding: utf-8 -*-
"""
Category 2: Capital Goods Calculator
GL-VCCI Scope 3 Platform

Intelligent capital asset emission calculator with LLM-powered classification.

3-Tier Calculation Waterfall:
- Tier 1: Supplier-specific Product Carbon Footprint (PCF)
- Tier 2: Asset-specific emission factors with amortization
- Tier 3: Spend-based using economic intensity

LLM Intelligence Features:
- Asset classification (buildings, machinery, vehicles, IT, equipment)
- Useful life estimation based on asset type and industry
- Data quality assessment

Version: 1.0.0
Date: 2025-11-08
"""

import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime

from greenlang.determinism import DeterministicClock
from ..models import (
    Category2Input,
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


# ============================================================================
# Asset Classification Data
# ============================================================================

ASSET_CATEGORIES = {
    "buildings": {
        "useful_life_range": (20, 50),
        "default_useful_life": 30,
        "uncertainty": 0.25,
        "keywords": ["building", "facility", "construction", "warehouse", "office", "plant"]
    },
    "machinery": {
        "useful_life_range": (5, 20),
        "default_useful_life": 10,
        "uncertainty": 0.20,
        "keywords": ["machine", "equipment", "production", "manufacturing", "assembly"]
    },
    "vehicles": {
        "useful_life_range": (3, 15),
        "default_useful_life": 8,
        "uncertainty": 0.18,
        "keywords": ["vehicle", "truck", "car", "fleet", "transport", "delivery"]
    },
    "it_equipment": {
        "useful_life_range": (3, 7),
        "default_useful_life": 4,
        "uncertainty": 0.15,
        "keywords": ["computer", "server", "laptop", "IT", "software", "hardware", "data center"]
    },
    "other_equipment": {
        "useful_life_range": (5, 15),
        "default_useful_life": 10,
        "uncertainty": 0.30,
        "keywords": ["furniture", "fixture", "tool", "instrument"]
    }
}

# Default emission factors (kgCO2e per USD) by asset category
ASSET_EMISSION_FACTORS = {
    "buildings": 0.42,
    "machinery": 0.48,
    "vehicles": 0.52,
    "it_equipment": 0.38,
    "other_equipment": 0.45,
    "default": 0.45
}


class Category2Calculator:
    """
    Category 2 (Capital Goods) calculator with LLM intelligence.

    Implements 3-tier calculation waterfall with automatic fallback:
    1. Tier 1: Supplier-specific PCF (highest quality)
    2. Tier 2: Asset-specific emission factors with amortization
    3. Tier 3: Spend-based using economic intensity factors

    Features:
    - LLM asset classification from description
    - LLM useful life estimation based on asset type and industry
    - Capex amortization over useful life
    - Data quality scoring (DQI)
    - Uncertainty propagation
    - Complete provenance tracking
    """

    def __init__(
        self,
        factor_broker: Any,
        llm_client: Any,
        uncertainty_engine: Any,
        provenance_builder: Any,
        config: Optional[Any] = None
    ):
        """
        Initialize Category 2 calculator.

        Args:
            factor_broker: FactorBroker instance for emission factors
            llm_client: LLMClient instance for intelligent classification
            uncertainty_engine: UncertaintyEngine for Monte Carlo
            provenance_builder: ProvenanceChainBuilder for tracking
            config: Calculator configuration
        """
        self.factor_broker = factor_broker
        self.llm_client = llm_client
        self.uncertainty_engine = uncertainty_engine
        self.provenance_builder = provenance_builder
        self.config = config or get_config()

        logger.info("Initialized Category2Calculator with LLM intelligence")

    async def calculate(self, input_data: Category2Input) -> CalculationResult:
        """
        Calculate Category 2 emissions with 3-tier fallback.

        Args:
            input_data: Category 2 input data

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
            if input_data.supplier_pcf:
                logger.info(f"Attempting Tier 1 calculation for {input_data.asset_description}")
                result = await self._calculate_tier_1(input_data)
                attempted_tiers.append(1)

                if result:
                    logger.info(f"Tier 1 successful for {input_data.asset_description}")
                    return result

            # Tier 2: Asset-specific with amortization (with LLM classification)
            logger.info(f"Attempting Tier 2 calculation for {input_data.asset_description}")
            result = await self._calculate_tier_2(input_data)
            attempted_tiers.append(2)

            if result and result.data_quality.dqi_score >= 40.0:
                logger.info(f"Tier 2 successful for {input_data.asset_description}")
                return result

            # Tier 3: Spend-based
            logger.info(f"Attempting Tier 3 calculation for {input_data.asset_description}")
            result = await self._calculate_tier_3(input_data)
            attempted_tiers.append(3)

            if result:
                logger.info(f"Tier 3 successful for {input_data.asset_description}")
                return result

            # If we get here, all tiers failed
            raise TierFallbackError(
                attempted_tiers=attempted_tiers,
                reason="No suitable emission factor found for capital asset",
                category=2
            )

        except TierFallbackError:
            raise
        except Exception as e:
            logger.error(f"Category 2 calculation failed: {e}", exc_info=True)
            raise CalculationError(
                calculation_type="category_2",
                reason=str(e),
                category=2,
                input_data=input_data.dict()
            )

    async def _calculate_tier_1(
        self, input_data: Category2Input
    ) -> Optional[CalculationResult]:
        """
        Tier 1: Supplier-specific PCF calculation.

        Formula: emissions = supplier_pcf (no amortization, already full lifecycle)

        Args:
            input_data: Category 2 input

        Returns:
            CalculationResult or None if data unavailable
        """
        if not input_data.supplier_pcf or input_data.supplier_pcf <= 0:
            return None

        # Direct emissions from supplier PCF
        emissions_kgco2e = input_data.supplier_pcf

        # Uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo and input_data.supplier_pcf_uncertainty:
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=1.0,  # Single asset
                quantity_uncertainty=0.0,
                emission_factor=input_data.supplier_pcf,
                factor_uncertainty=input_data.supplier_pcf_uncertainty,
                iterations=self.config.monte_carlo_iterations
            )

        # Emission factor info
        ef_info = EmissionFactorInfo(
            factor_id=f"supplier_pcf_cat2_{input_data.supplier_name or 'unknown'}",
            value=input_data.supplier_pcf,
            unit="kgCO2e",
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
            category=2,
            tier=TierType.TIER_1,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": "supplier_pcf",
                "supplier_pcf": input_data.supplier_pcf,
                "result_kgco2e": emissions_kgco2e,
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=2,
            tier=TierType.TIER_1,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_1_supplier_pcf",
            warnings=[],
            metadata={
                "supplier_name": input_data.supplier_name,
                "asset_description": input_data.asset_description,
                "capex_amount": input_data.capex_amount,
            }
        )

    async def _calculate_tier_2(
        self, input_data: Category2Input
    ) -> Optional[CalculationResult]:
        """
        Tier 2: Asset-specific emission factor with amortization and LLM classification.

        Formula: emissions = (capex_amount × emission_factor) / useful_life_years

        Uses LLM to classify asset and estimate useful life if not provided.

        Args:
            input_data: Category 2 input

        Returns:
            CalculationResult or None if classification fails
        """
        # Use LLM to classify asset and estimate useful life
        asset_info = await self._classify_asset_with_llm(input_data)

        if not asset_info:
            logger.warning(f"LLM asset classification failed for {input_data.asset_description}")
            # Fallback to keyword-based classification
            asset_info = self._classify_asset_keyword(input_data)

        # Get emission factor
        emission_factor = input_data.emission_factor_kgco2e_per_usd
        if not emission_factor:
            emission_factor = ASSET_EMISSION_FACTORS.get(
                asset_info["category"],
                ASSET_EMISSION_FACTORS["default"]
            )

        # Get useful life
        useful_life = input_data.useful_life_years or asset_info["useful_life_years"]

        # Calculate amortized emissions
        total_lifecycle_emissions = input_data.capex_amount * emission_factor
        annual_emissions = total_lifecycle_emissions / useful_life
        emissions_kgco2e = annual_emissions  # Report annual amortized emissions

        # Uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=input_data.capex_amount / useful_life,
                quantity_uncertainty=0.15,  # 15% for capex/amortization
                emission_factor=emission_factor,
                factor_uncertainty=asset_info["uncertainty"],
                iterations=self.config.monte_carlo_iterations
            )

        # Emission factor info
        ef_info = EmissionFactorInfo(
            factor_id=f"asset_ef_{asset_info['category']}_{input_data.region}",
            value=emission_factor,
            unit="kgCO2e/USD",
            source="asset_category_database",
            source_version="2024",
            gwp_standard="AR6",
            uncertainty=asset_info["uncertainty"],
            data_quality_score=70.0,
            reference_year=2024,
            geographic_scope=input_data.region,
            hash=self.provenance_builder.hash_factor_info(emission_factor, asset_info["category"])
        )

        # Data quality
        warnings = []
        if asset_info.get("llm_classified", False):
            warnings.append(f"Asset classified by LLM as '{asset_info['category']}' with {asset_info['confidence']*100:.0f}% confidence")
        if asset_info.get("llm_estimated_life", False):
            warnings.append(f"Useful life estimated by LLM at {useful_life} years")

        data_quality = DataQualityInfo(
            dqi_score=self.config.tier_2_dqi_score,
            tier=TierType.TIER_2,
            rating=self._get_quality_rating(self.config.tier_2_dqi_score),
            pedigree_score=3.5,
            warnings=warnings
        )

        # Provenance chain
        provenance = await self.provenance_builder.build(
            category=2,
            tier=TierType.TIER_2,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": "(capex_amount × emission_factor) / useful_life_years",
                "capex_amount": input_data.capex_amount,
                "emission_factor": emission_factor,
                "useful_life_years": useful_life,
                "total_lifecycle_emissions_kgco2e": total_lifecycle_emissions,
                "annual_amortized_emissions_kgco2e": emissions_kgco2e,
                "asset_category": asset_info["category"],
                "llm_classified": asset_info.get("llm_classified", False),
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=2,
            tier=TierType.TIER_2,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_2_asset_amortization",
            warnings=warnings,
            metadata={
                "asset_description": input_data.asset_description,
                "asset_category": asset_info["category"],
                "capex_amount": input_data.capex_amount,
                "useful_life_years": useful_life,
                "total_lifecycle_emissions_kgco2e": total_lifecycle_emissions,
                "llm_classification": asset_info.get("llm_reasoning", "N/A"),
            }
        )

    async def _calculate_tier_3(
        self, input_data: Category2Input
    ) -> Optional[CalculationResult]:
        """
        Tier 3: Spend-based calculation using economic intensity.

        Formula: emissions = capex_amount × economic_intensity_ef

        Args:
            input_data: Category 2 input

        Returns:
            CalculationResult
        """
        # Get economic intensity factor
        economic_sector = input_data.economic_sector or "capital_goods_average"
        intensity_factor = await self._get_economic_intensity_factor(
            economic_sector, input_data.region
        )

        if not intensity_factor:
            logger.error(f"No economic intensity factor for sector {economic_sector}")
            return None

        # Calculate emissions (no amortization in Tier 3, report full amount)
        emissions_kgco2e = input_data.capex_amount * intensity_factor.value

        # Uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=input_data.capex_amount,
                quantity_uncertainty=0.20,  # 20% for spend data
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

        # Data quality
        warnings = [
            "Spend-based calculation has lower accuracy than asset-specific data",
            "Consider obtaining supplier PCF or asset-specific emission factors",
            "No amortization applied in Tier 3 - reports full lifecycle emissions"
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
            category=2,
            tier=TierType.TIER_3,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": "capex_amount × economic_intensity_ef",
                "capex_amount": input_data.capex_amount,
                "economic_intensity_ef": intensity_factor.value,
                "economic_sector": economic_sector,
                "result_kgco2e": emissions_kgco2e,
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=2,
            tier=TierType.TIER_3,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_3_spend_based",
            warnings=warnings,
            metadata={
                "asset_description": input_data.asset_description,
                "capex_amount": input_data.capex_amount,
                "economic_sector": economic_sector,
            }
        )

    async def _classify_asset_with_llm(
        self, input_data: Category2Input
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to classify capital asset and estimate useful life.

        Args:
            input_data: Category 2 input

        Returns:
            Asset classification info or None if LLM fails
        """
        if not self.llm_client:
            return None

        try:
            prompt = f"""
Classify this capital asset purchase and estimate its useful life:

Description: {input_data.asset_description}
Value: ${input_data.capex_amount:,.2f}
Industry: {input_data.industry or 'Not specified'}

Classify into ONE of these categories:
- buildings: Commercial/industrial buildings and facilities
- machinery: Production machinery and manufacturing equipment
- vehicles: Fleet vehicles, trucks, cars for business use
- it_equipment: Computers, servers, IT infrastructure
- other_equipment: Furniture, fixtures, tools, instruments

Estimate useful life based on:
- Asset type and category
- Industry standards
- Typical depreciation schedules

Return JSON only:
{{
    "category": "category_name",
    "useful_life_years": X,
    "confidence": 0.XX,
    "reasoning": "Brief explanation"
}}
"""

            # Use the LLM complete method
            response_text = await self.llm_client.complete(prompt)

            # Parse JSON response
            result = json.loads(response_text)

            # Validate and enrich
            category = result.get("category", "other_equipment")
            if category not in ASSET_CATEGORIES:
                category = "other_equipment"

            useful_life = result.get("useful_life_years", ASSET_CATEGORIES[category]["default_useful_life"])

            # Clamp to reasonable range
            min_life, max_life = ASSET_CATEGORIES[category]["useful_life_range"]
            useful_life = max(min_life, min(max_life, useful_life))

            return {
                "category": category,
                "useful_life_years": useful_life,
                "confidence": result.get("confidence", 0.7),
                "uncertainty": ASSET_CATEGORIES[category]["uncertainty"],
                "llm_classified": True,
                "llm_estimated_life": True,
                "llm_reasoning": result.get("reasoning", "LLM classification")
            }

        except Exception as e:
            logger.warning(f"LLM asset classification failed: {e}")
            return None

    def _classify_asset_keyword(self, input_data: Category2Input) -> Dict[str, Any]:
        """
        Fallback keyword-based asset classification.

        Args:
            input_data: Category 2 input

        Returns:
            Asset classification info
        """
        description_lower = input_data.asset_description.lower()

        # Try provided category first
        if input_data.asset_category and input_data.asset_category in ASSET_CATEGORIES:
            category = input_data.asset_category
        else:
            # Keyword matching
            category = "other_equipment"
            max_matches = 0

            for cat, info in ASSET_CATEGORIES.items():
                matches = sum(1 for keyword in info["keywords"] if keyword in description_lower)
                if matches > max_matches:
                    max_matches = matches
                    category = cat

        return {
            "category": category,
            "useful_life_years": ASSET_CATEGORIES[category]["default_useful_life"],
            "confidence": 0.6,
            "uncertainty": ASSET_CATEGORIES[category]["uncertainty"],
            "llm_classified": False,
            "llm_estimated_life": False,
            "llm_reasoning": "Keyword-based classification"
        }

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
            product=f"{sector}_capital_goods",
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
            # Return default
            return self._get_default_economic_intensity(sector, region)

    def _get_default_economic_intensity(self, sector: str, region: str) -> Any:
        """Get default economic intensity factor for capital goods."""
        # Default economic intensities (kgCO2e/USD) - capital goods specific
        defaults = {
            "capital_goods_average": 0.45,
            "construction": 0.52,
            "machinery": 0.48,
            "vehicles": 0.55,
            "it_equipment": 0.38,
        }

        value = defaults.get(sector.lower(), defaults["capital_goods_average"])

        # Create mock response
        from ...factor_broker.models import (
            FactorResponse,
            FactorMetadata,
            ProvenanceInfo,
            SourceType,
            DataQualityIndicator,
        )

        return FactorResponse(
            factor_id=f"default_eio_capgoods_{sector}_{region}",
            value=value,
            unit="kgCO2e/USD",
            uncertainty=0.50,
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
                proxy_method="default_capital_goods_intensity"
            )
        )

    def _validate_input(self, input_data: Category2Input):
        """
        Validate Category 2 input data.

        Args:
            input_data: Input to validate

        Raises:
            DataValidationError: If validation fails
        """
        if input_data.capex_amount <= 0:
            raise DataValidationError(
                field="capex_amount",
                value=input_data.capex_amount,
                reason="Capex amount must be positive",
                category=2
            )

        if not input_data.asset_description or not input_data.asset_description.strip():
            raise DataValidationError(
                field="asset_description",
                value=input_data.asset_description,
                reason="Asset description cannot be empty",
                category=2
            )

        if input_data.useful_life_years is not None and input_data.useful_life_years <= 0:
            raise DataValidationError(
                field="useful_life_years",
                value=input_data.useful_life_years,
                reason="Useful life must be positive",
                category=2
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


__all__ = ["Category2Calculator"]
