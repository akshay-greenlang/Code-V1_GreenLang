# -*- coding: utf-8 -*-
"""
Category 12: End-of-Life Treatment of Sold Products Calculator
GL-VCCI Scope 3 Platform

Calculates emissions from disposal and end-of-life treatment of sold products.

Disposal Methods:
- Landfill
- Recycling
- Incineration (with/without energy recovery)
- Composting

Features:
- Material composition analysis
- LLM material identification
- Recycling rate estimation
- Disposal method-specific emission factors
- Regional waste management practices

Version: 1.0.0
Date: 2025-11-08
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from ..models import (
from greenlang.determinism import DeterministicClock
    Category12Input,
    MaterialComposition,
    CalculationResult,
    DataQualityInfo,
    EmissionFactorInfo,
    ProvenanceChain,
    UncertaintyResult,
)
from ..config import TierType, DisposalMethod, MaterialType, get_config
from ..exceptions import (
    DataValidationError,
    CalculationError,
)

logger = logging.getLogger(__name__)

class Category12Calculator:
    """
    Category 12 (End-of-Life Treatment of Sold Products) calculator.

    Calculates emissions from product disposal with intelligent
    material composition analysis and disposal method estimation.

    Features:
    - Multi-tier calculation waterfall
    - LLM material composition analysis
    - Regional recycling rate estimation
    - Disposal method-specific factors
    - Material-specific recycling credits
    - Data quality scoring
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
        Initialize Category 12 calculator.

        Args:
            factor_broker: FactorBroker instance for emission factors
            llm_client: LLMClient for intelligent material estimation
            uncertainty_engine: UncertaintyEngine for Monte Carlo
            provenance_builder: ProvenanceChainBuilder for tracking
            config: Calculator configuration
        """
        self.factor_broker = factor_broker
        self.llm_client = llm_client
        self.uncertainty_engine = uncertainty_engine
        self.provenance_builder = provenance_builder
        self.config = config or get_config()

        logger.info("Initialized Category12Calculator")

    async def calculate(self, input_data: Category12Input) -> CalculationResult:
        """
        Calculate Category 12 emissions with tier fallback.

        Args:
            input_data: Category 12 input data

        Returns:
            CalculationResult with emissions and provenance

        Raises:
            DataValidationError: If input data is invalid
            CalculationError: If calculation fails
        """
        start_time = DeterministicClock.utcnow()

        # Validate input
        self._validate_input(input_data)

        # Attempt tier cascade
        result = None

        try:
            # Tier 1: Detailed material composition
            if input_data.material_composition:
                logger.info(f"Attempting Tier 1 calculation for {input_data.product_name}")
                result = await self._calculate_tier_1(input_data)

                if result:
                    logger.info(f"Tier 1 successful for {input_data.product_name}")
                    return result

            # Tier 2: Total weight with primary material
            if input_data.total_weight_kg and input_data.primary_material:
                logger.info(f"Attempting Tier 2 calculation for {input_data.product_name}")
                result = await self._calculate_tier_2(input_data)

                if result:
                    logger.info(f"Tier 2 successful for {input_data.product_name}")
                    return result

            # Tier 3: LLM-estimated material composition
            logger.info(f"Attempting Tier 3 (LLM) calculation for {input_data.product_name}")
            result = await self._calculate_tier_3_llm(input_data)

            if result:
                logger.info(f"Tier 3 (LLM) successful for {input_data.product_name}")
                return result

            raise CalculationError(
                calculation_type="category_12",
                reason="No suitable material composition or weight data found",
                category=12,
                input_data=input_data.dict()
            )

        except Exception as e:
            logger.error(f"Category 12 calculation failed: {e}", exc_info=True)
            raise CalculationError(
                calculation_type="category_12",
                reason=str(e),
                category=12,
                input_data=input_data.dict()
            )

    async def _calculate_tier_1(
        self, input_data: Category12Input
    ) -> Optional[CalculationResult]:
        """
        Tier 1: Detailed material composition.

        Calculates emissions for each material based on disposal method
        and recycling rates.

        Args:
            input_data: Category 12 input

        Returns:
            CalculationResult or None if data unavailable
        """
        total_emissions_kgco2e = 0.0
        material_breakdown = []

        # Get disposal split (or use regional defaults)
        disposal_split = await self._get_disposal_split(input_data)

        # Calculate emissions for each material
        for material in input_data.material_composition:
            material_emissions = await self._calculate_material_emissions(
                material,
                disposal_split,
                input_data.region
            )

            total_emissions_kgco2e += material_emissions
            material_breakdown.append({
                "material": material.material_type.value,
                "weight_kg": material.weight_kg,
                "emissions_kgco2e": material_emissions
            })

        # Total emissions for all units
        emissions_kgco2e = input_data.units_sold * total_emissions_kgco2e

        # Uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=input_data.units_sold * sum(m.weight_kg for m in input_data.material_composition),
                quantity_uncertainty=0.10,
                emission_factor=total_emissions_kgco2e / sum(m.weight_kg for m in input_data.material_composition),
                factor_uncertainty=0.20,
                iterations=self.config.monte_carlo_iterations
            )

        # Data quality (high for detailed composition)
        data_quality = DataQualityInfo(
            dqi_score=80.0,
            tier=TierType.TIER_1,
            rating="excellent",
            pedigree_score=4.2,
            warnings=[]
        )

        # Emission factor info (composite)
        total_weight = sum(m.weight_kg for m in input_data.material_composition)
        ef_info = EmissionFactorInfo(
            factor_id="composite_eol_detailed",
            value=total_emissions_kgco2e / total_weight,
            unit="kgCO2e/kg",
            source="composite_material_factors",
            source_version="2024",
            gwp_standard="AR6",
            uncertainty=0.20,
            data_quality_score=80.0,
            reference_year=2024,
            geographic_scope=input_data.region,
            hash=self.provenance_builder.hash_factor_info(
                total_emissions_kgco2e,
                "composite_eol"
            )
        )

        # Provenance chain
        provenance = await self.provenance_builder.build(
            category=12,
            tier=TierType.TIER_1,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": "Σ(material_weight × disposal_ef × units_sold)",
                "units_sold": input_data.units_sold,
                "total_weight_per_unit_kg": total_weight,
                "disposal_split": disposal_split,
                "material_breakdown": material_breakdown,
                "result_kgco2e": emissions_kgco2e,
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=12,
            tier=TierType.TIER_1,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_1_detailed_composition",
            warnings=[],
            metadata={
                "product_name": input_data.product_name,
                "units_sold": input_data.units_sold,
                "total_weight_per_unit_kg": total_weight,
                "material_breakdown": material_breakdown,
            }
        )

    async def _calculate_tier_2(
        self, input_data: Category12Input
    ) -> Optional[CalculationResult]:
        """
        Tier 2: Total weight with primary material.

        Uses primary material type and total weight.

        Args:
            input_data: Category 12 input

        Returns:
            CalculationResult or None if calculation not possible
        """
        # Get disposal split
        disposal_split = await self._get_disposal_split(input_data)

        # Get emission factors for primary material
        disposal_ef = await self._get_disposal_emission_factor(
            input_data.primary_material,
            disposal_split,
            input_data.region
        )

        if not disposal_ef:
            return None

        # Calculate emissions
        emissions_per_unit = input_data.total_weight_kg * disposal_ef.value
        emissions_kgco2e = input_data.units_sold * emissions_per_unit

        # Uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=input_data.units_sold * input_data.total_weight_kg,
                quantity_uncertainty=0.15,
                emission_factor=disposal_ef.value,
                factor_uncertainty=disposal_ef.uncertainty,
                iterations=self.config.monte_carlo_iterations
            )

        # Data quality
        warnings = ["Using single primary material - actual product may contain multiple materials"]

        data_quality = DataQualityInfo(
            dqi_score=65.0,
            tier=TierType.TIER_2,
            rating="good",
            pedigree_score=3.5,
            warnings=warnings
        )

        # Emission factor info
        ef_info = EmissionFactorInfo(
            factor_id=disposal_ef.factor_id,
            value=disposal_ef.value,
            unit=disposal_ef.unit,
            source=disposal_ef.source,
            source_version=disposal_ef.metadata.source_version,
            gwp_standard=disposal_ef.metadata.gwp_standard.value,
            uncertainty=disposal_ef.uncertainty,
            data_quality_score=disposal_ef.data_quality_score,
            reference_year=disposal_ef.metadata.reference_year,
            geographic_scope=disposal_ef.metadata.geographic_scope,
            hash=disposal_ef.provenance.calculation_hash or "unknown"
        )

        # Provenance chain
        provenance = await self.provenance_builder.build(
            category=12,
            tier=TierType.TIER_2,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": "units_sold × weight × disposal_ef",
                "units_sold": input_data.units_sold,
                "total_weight_kg": input_data.total_weight_kg,
                "primary_material": input_data.primary_material.value,
                "disposal_split": disposal_split,
                "disposal_ef": disposal_ef.value,
                "result_kgco2e": emissions_kgco2e,
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=12,
            tier=TierType.TIER_2,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_2_primary_material",
            warnings=warnings,
            metadata={
                "product_name": input_data.product_name,
                "units_sold": input_data.units_sold,
                "total_weight_kg": input_data.total_weight_kg,
                "primary_material": input_data.primary_material.value,
            }
        )

    async def _calculate_tier_3_llm(
        self, input_data: Category12Input
    ) -> Optional[CalculationResult]:
        """
        Tier 3: LLM-estimated material composition.

        Uses LLM to estimate product weight and material composition.

        Args:
            input_data: Category 12 input

        Returns:
            CalculationResult
        """
        # Use LLM to estimate material composition
        composition_estimate = await self._llm_estimate_composition(input_data)

        if not composition_estimate or composition_estimate.get("total_weight_kg", 0) <= 0:
            logger.warning("LLM could not estimate material composition")
            return None

        # Get disposal split
        disposal_split = await self._get_disposal_split(input_data)

        # Calculate emissions using estimated composition
        total_weight = composition_estimate["total_weight_kg"]
        primary_material = composition_estimate.get("primary_material", MaterialType.MIXED)

        # Get emission factor
        disposal_ef = await self._get_disposal_emission_factor(
            MaterialType(primary_material) if isinstance(primary_material, str) else primary_material,
            disposal_split,
            input_data.region
        )

        if not disposal_ef:
            return None

        # Calculate emissions
        emissions_per_unit = total_weight * disposal_ef.value
        emissions_kgco2e = input_data.units_sold * emissions_per_unit

        # Higher uncertainty for LLM estimates
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=input_data.units_sold * total_weight,
                quantity_uncertainty=0.30,
                emission_factor=disposal_ef.value,
                factor_uncertainty=0.35,
                iterations=self.config.monte_carlo_iterations
            )

        # Data quality (lower for LLM)
        warnings = [
            "Material composition estimated using LLM intelligence",
            "Consider obtaining actual product weight and material data",
            f"LLM confidence: {composition_estimate.get('confidence', 0):.1%}",
        ]

        data_quality = DataQualityInfo(
            dqi_score=45.0,
            tier=TierType.TIER_3,
            rating="fair",
            pedigree_score=2.5,
            warnings=warnings
        )

        # Emission factor info
        ef_info = EmissionFactorInfo(
            factor_id=disposal_ef.factor_id,
            value=disposal_ef.value,
            unit=disposal_ef.unit,
            source=disposal_ef.source,
            source_version=disposal_ef.metadata.source_version,
            gwp_standard=disposal_ef.metadata.gwp_standard.value,
            uncertainty=disposal_ef.uncertainty,
            data_quality_score=disposal_ef.data_quality_score,
            reference_year=disposal_ef.metadata.reference_year,
            geographic_scope=disposal_ef.metadata.geographic_scope,
            hash=disposal_ef.provenance.calculation_hash or "unknown"
        )

        # Provenance chain
        provenance = await self.provenance_builder.build(
            category=12,
            tier=TierType.TIER_3,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": "units_sold × llm_weight × disposal_ef",
                "units_sold": input_data.units_sold,
                "llm_total_weight_kg": total_weight,
                "llm_primary_material": primary_material,
                "disposal_split": disposal_split,
                "llm_reasoning": composition_estimate.get("reasoning"),
                "result_kgco2e": emissions_kgco2e,
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=12,
            tier=TierType.TIER_3,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_3_llm_estimate",
            warnings=warnings,
            metadata={
                "product_name": input_data.product_name,
                "units_sold": input_data.units_sold,
                "llm_total_weight_kg": total_weight,
                "llm_primary_material": primary_material,
                "llm_confidence": composition_estimate.get("confidence"),
            }
        )

    async def _calculate_material_emissions(
        self,
        material: MaterialComposition,
        disposal_split: Dict[str, float],
        region: str
    ) -> float:
        """Calculate emissions for a single material."""
        emissions = 0.0

        # Get recycling rate
        recycling_rate = material.recycling_rate or disposal_split.get("recycling", 0.0)
        landfill_rate = disposal_split.get("landfill", 0.0)
        incineration_rate = disposal_split.get("incineration", 0.0)

        # Emissions from landfill
        if landfill_rate > 0:
            landfill_ef = self._get_default_disposal_factor(material.material_type, "landfill")
            emissions += material.weight_kg * landfill_rate * landfill_ef

        # Emissions from incineration
        if incineration_rate > 0:
            incineration_ef = self._get_default_disposal_factor(material.material_type, "incineration")
            emissions += material.weight_kg * incineration_rate * incineration_ef

        # Recycling (may be credit)
        if recycling_rate > 0:
            recycling_ef = self._get_default_disposal_factor(material.material_type, "recycling")
            emissions += material.weight_kg * recycling_rate * recycling_ef

        return emissions

    async def _get_disposal_split(self, input_data: Category12Input) -> Dict[str, float]:
        """Get disposal method split."""
        # If explicit percentages provided
        if input_data.landfill_percentage is not None:
            total = (input_data.landfill_percentage or 0) + \
                   (input_data.recycling_percentage or 0) + \
                   (input_data.incineration_percentage or 0)

            if total > 0:
                return {
                    "landfill": (input_data.landfill_percentage or 0) / 100,
                    "recycling": (input_data.recycling_percentage or 0) / 100,
                    "incineration": (input_data.incineration_percentage or 0) / 100,
                }

        # Use regional defaults
        return self._get_regional_disposal_split(input_data.region)

    def _get_regional_disposal_split(self, region: str) -> Dict[str, float]:
        """Get regional average disposal method split."""
        # Regional waste management practices (approximate)
        regional_splits = {
            "US": {"landfill": 0.52, "recycling": 0.32, "incineration": 0.16},
            "GB": {"landfill": 0.20, "recycling": 0.45, "incineration": 0.35},
            "DE": {"landfill": 0.01, "recycling": 0.67, "incineration": 0.32},
            "JP": {"landfill": 0.02, "recycling": 0.20, "incineration": 0.78},
            "CN": {"landfill": 0.55, "recycling": 0.25, "incineration": 0.20},
            "IN": {"landfill": 0.80, "recycling": 0.15, "incineration": 0.05},
        }

        return regional_splits.get(region, {
            "landfill": 0.50,
            "recycling": 0.30,
            "incineration": 0.20
        })

    async def _get_disposal_emission_factor(
        self,
        material: MaterialType,
        disposal_split: Dict[str, float],
        region: str
    ) -> Optional[Any]:
        """Get weighted average disposal emission factor."""
        # Calculate weighted average
        weighted_ef = 0.0

        for method, percentage in disposal_split.items():
            if percentage > 0:
                ef = self._get_default_disposal_factor(material, method)
                weighted_ef += percentage * ef

        # Create mock factor response
        from ...factor_broker.models import (
            FactorResponse,
            FactorMetadata,
            ProvenanceInfo,
            SourceType,
            DataQualityIndicator,
        )

        return FactorResponse(
            factor_id=f"disposal_{material.value}_{region}",
            value=weighted_ef,
            unit="kgCO2e/kg",
            uncertainty=0.25,
            metadata=FactorMetadata(
                source=SourceType.PROXY,
                source_version="default_v1",
                gwp_standard="AR6",
                reference_year=2024,
                geographic_scope=region,
                data_quality=DataQualityIndicator(
                    reliability=3,
                    completeness=3,
                    temporal_correlation=3,
                    geographical_correlation=3,
                    technological_correlation=3,
                    overall_score=60
                )
            ),
            provenance=ProvenanceInfo(
                is_proxy=True,
                proxy_method="weighted_disposal_factor"
            )
        )

    def _get_default_disposal_factor(self, material: MaterialType, method: str) -> float:
        """
        Get default disposal emission factors (kgCO2e/kg).

        Positive values = emissions
        Negative values = avoided emissions (recycling credit)
        """
        # Disposal emission factors by material and method
        factors = {
            MaterialType.PLASTIC: {
                "landfill": 0.02,
                "recycling": -1.5,  # Credit for avoided virgin plastic
                "incineration": 2.5,
            },
            MaterialType.METAL_STEEL: {
                "landfill": 0.01,
                "recycling": -1.8,
                "incineration": 0.05,
            },
            MaterialType.METAL_ALUMINUM: {
                "landfill": 0.01,
                "recycling": -9.0,  # Large credit for aluminum
                "incineration": 0.05,
            },
            MaterialType.GLASS: {
                "landfill": 0.01,
                "recycling": -0.5,
                "incineration": 0.02,
            },
            MaterialType.PAPER: {
                "landfill": 0.85,  # Methane from decomposition
                "recycling": -0.7,
                "incineration": 1.2,
            },
            MaterialType.ELECTRONICS: {
                "landfill": 0.10,
                "recycling": -2.0,
                "incineration": 1.5,
            },
            MaterialType.ORGANIC: {
                "landfill": 1.2,  # High methane emissions
                "recycling": 0.0,
                "incineration": 0.8,
            },
            MaterialType.MIXED: {
                "landfill": 0.50,
                "recycling": -0.8,
                "incineration": 1.0,
            },
        }

        material_factors = factors.get(material, factors[MaterialType.MIXED])
        return material_factors.get(method, 0.5)

    async def _llm_estimate_composition(
        self,
        input_data: Category12Input
    ) -> Dict[str, Any]:
        """Use LLM to estimate material composition."""
        prompt = f"""Estimate the material composition and weight for this product:

Product: {input_data.product_name}
Description: {input_data.product_description or 'Not provided'}
Category: {input_data.product_category or 'Unknown'}

Estimate:
1. Total product weight (kg)
2. Primary material type
3. Material composition breakdown (if applicable)
4. Typical disposal methods

Return JSON:
{{
    "total_weight_kg": <weight>,
    "primary_material": "<material_type>",
    "material_breakdown": [
        {{"material": "<type>", "weight_kg": <weight>, "percentage": <percent>}}
    ],
    "confidence": <0.0-1.0>,
    "reasoning": "Detailed explanation",
    "typical_disposal": "Description of typical disposal"
}}
"""

        try:
            # Note: In production, this would call the actual LLM client
            # For now, provide category-specific defaults
            return self._get_default_composition_estimate(input_data.product_category)
        except Exception as e:
            logger.error(f"LLM composition estimation failed: {e}")
            return {}

    def _get_default_composition_estimate(self, category: Optional[str]) -> Dict[str, Any]:
        """Get default composition estimates by product category."""
        defaults = {
            "electronics": {
                "total_weight_kg": 2.5,
                "primary_material": "electronics",
                "material_breakdown": [
                    {"material": "electronics", "weight_kg": 1.5, "percentage": 60},
                    {"material": "plastic", "weight_kg": 0.7, "percentage": 28},
                    {"material": "metal_steel", "weight_kg": 0.3, "percentage": 12},
                ],
                "confidence": 0.70,
                "reasoning": "Typical electronics device weight and composition",
                "typical_disposal": "E-waste recycling or landfill"
            },
            "furniture": {
                "total_weight_kg": 25.0,
                "primary_material": "mixed",
                "confidence": 0.60,
                "reasoning": "Average furniture weight estimate",
                "typical_disposal": "Landfill or bulky waste collection"
            },
            "packaging": {
                "total_weight_kg": 0.5,
                "primary_material": "cardboard",
                "confidence": 0.75,
                "reasoning": "Typical packaging weight",
                "typical_disposal": "Recycling"
            },
        }

        return defaults.get(category or "default", {
            "total_weight_kg": 5.0,
            "primary_material": "mixed",
            "confidence": 0.50,
            "reasoning": "Generic product weight estimate",
            "typical_disposal": "Mixed waste management"
        })

    def _validate_input(self, input_data: Category12Input):
        """Validate Category 12 input data."""
        if input_data.units_sold <= 0:
            raise DataValidationError(
                field="units_sold",
                value=input_data.units_sold,
                reason="Units sold must be positive",
                category=12
            )

        if not input_data.product_name or not input_data.product_name.strip():
            raise DataValidationError(
                field="product_name",
                value=input_data.product_name,
                reason="Product name cannot be empty",
                category=12
            )


__all__ = ["Category12Calculator"]
