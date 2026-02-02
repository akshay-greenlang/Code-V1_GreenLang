# -*- coding: utf-8 -*-
"""
Category 5: Waste Generated in Operations Calculator
GL-VCCI Scope 3 Platform

Intelligent waste disposal emissions calculator with LLM-powered classification.

Covers emissions from disposal and treatment of waste generated in operations:
- Landfilling
- Incineration (with/without energy recovery)
- Recycling
- Composting
- Wastewater treatment

3-Tier Calculation Waterfall:
- Tier 1: Supplier-specific disposal emission factors
- Tier 2: Waste-type and disposal-method specific factors with LLM classification
- Tier 3: Generic waste disposal factors

LLM Intelligence Features:
- Waste type categorization (hazardous, municipal, organic, construction, etc.)
- Disposal method identification
- Recycling rate estimation
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
    Category5Input,
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
# Waste Classification Data
# ============================================================================

WASTE_TYPES = {
    "municipal_solid_waste": {
        "description": "General office/facility waste",
        "uncertainty": 0.30,
        "keywords": ["municipal", "solid waste", "msw", "general", "office", "mixed"]
    },
    "hazardous_waste": {
        "description": "Hazardous and chemical waste",
        "uncertainty": 0.35,
        "keywords": ["hazardous", "chemical", "toxic", "dangerous", "regulated"]
    },
    "construction_waste": {
        "description": "Construction and demolition debris",
        "uncertainty": 0.25,
        "keywords": ["construction", "demolition", "debris", "concrete", "rubble"]
    },
    "organic_waste": {
        "description": "Food and organic waste",
        "uncertainty": 0.28,
        "keywords": ["organic", "food", "compost", "biodegradable", "kitchen"]
    },
    "plastic_waste": {
        "description": "Plastic materials",
        "uncertainty": 0.20,
        "keywords": ["plastic", "polymer", "packaging"]
    },
    "metal_waste": {
        "description": "Metal scrap and waste",
        "uncertainty": 0.18,
        "keywords": ["metal", "scrap", "aluminum", "steel", "copper"]
    },
    "paper_cardboard": {
        "description": "Paper and cardboard",
        "uncertainty": 0.22,
        "keywords": ["paper", "cardboard", "carton", "document"]
    },
    "electronic_waste": {
        "description": "Electronic waste (e-waste)",
        "uncertainty": 0.30,
        "keywords": ["electronic", "e-waste", "ewaste", "electronics", "circuit"]
    }
}

DISPOSAL_METHODS = {
    "landfill": {
        "description": "Landfilling without gas capture",
        "base_ef_range": (0.4, 1.2),  # kgCO2e per kg waste
        "default_ef": 0.8,
        "uncertainty": 0.40,
        "keywords": ["landfill", "dump", "bury"]
    },
    "landfill_with_gas_recovery": {
        "description": "Landfilling with methane gas capture",
        "base_ef_range": (0.1, 0.4),
        "default_ef": 0.2,
        "uncertainty": 0.35,
        "keywords": ["landfill gas", "lfg", "gas recovery", "methane capture"]
    },
    "incineration": {
        "description": "Incineration without energy recovery",
        "base_ef_range": (0.6, 1.0),
        "default_ef": 0.8,
        "uncertainty": 0.30,
        "keywords": ["incinerate", "burn", "combustion", "waste-to-energy"]
    },
    "incineration_with_energy_recovery": {
        "description": "Incineration with energy recovery",
        "base_ef_range": (0.0, 0.3),
        "default_ef": 0.1,
        "uncertainty": 0.25,
        "keywords": ["wte", "waste to energy", "energy recovery", "efw"]
    },
    "recycling": {
        "description": "Recycling and material recovery",
        "base_ef_range": (-0.5, 0.2),  # Negative = avoided emissions
        "default_ef": -0.2,
        "uncertainty": 0.35,
        "keywords": ["recycle", "recovery", "reuse", "reprocess"]
    },
    "composting": {
        "description": "Composting organic waste",
        "base_ef_range": (0.05, 0.2),
        "default_ef": 0.1,
        "uncertainty": 0.30,
        "keywords": ["compost", "organic treatment", "aerobic"]
    },
    "anaerobic_digestion": {
        "description": "Anaerobic digestion with biogas recovery",
        "base_ef_range": (-0.2, 0.1),
        "default_ef": -0.05,
        "uncertainty": 0.28,
        "keywords": ["anaerobic", "biogas", "digestion", "ad"]
    },
    "wastewater_treatment": {
        "description": "Wastewater treatment",
        "base_ef_range": (0.3, 0.7),
        "default_ef": 0.5,
        "uncertainty": 0.35,
        "keywords": ["wastewater", "sewage", "effluent", "treatment"]
    }
}


class Category5Calculator:
    """
    Category 5 (Waste Generated in Operations) calculator with LLM intelligence.

    Implements 3-tier calculation waterfall:
    1. Tier 1: Supplier-specific disposal emission factors (highest quality)
    2. Tier 2: Waste-type and disposal-method specific factors with LLM classification
    3. Tier 3: Generic waste disposal factors

    Features:
    - LLM waste categorization from description
    - LLM disposal method identification
    - Recycling rate adjustments
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
        Initialize Category 5 calculator.

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

        logger.info("Initialized Category5Calculator with LLM intelligence")

    async def calculate(self, input_data: Category5Input) -> CalculationResult:
        """
        Calculate Category 5 emissions with 3-tier fallback.

        Args:
            input_data: Category 5 input data

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
            # Tier 1: Supplier-specific disposal factors
            if input_data.supplier_disposal_ef:
                logger.info(f"Attempting Tier 1 calculation for {input_data.waste_description}")
                result = await self._calculate_tier_1(input_data)
                attempted_tiers.append(1)

                if result:
                    logger.info(f"Tier 1 successful for {input_data.waste_description}")
                    return result

            # Tier 2: Waste-specific factors with LLM classification
            logger.info(f"Attempting Tier 2 calculation for {input_data.waste_description}")
            result = await self._calculate_tier_2(input_data)
            attempted_tiers.append(2)

            if result and result.data_quality.dqi_score >= 40.0:
                logger.info(f"Tier 2 successful for {input_data.waste_description}")
                return result

            # Tier 3: Generic waste disposal factors
            logger.info(f"Attempting Tier 3 calculation for {input_data.waste_description}")
            result = await self._calculate_tier_3(input_data)
            attempted_tiers.append(3)

            if result:
                logger.info(f"Tier 3 successful for {input_data.waste_description}")
                return result

            # If we get here, all tiers failed
            raise TierFallbackError(
                attempted_tiers=attempted_tiers,
                reason="No suitable emission factor found for waste disposal",
                category=5
            )

        except TierFallbackError:
            raise
        except Exception as e:
            logger.error(f"Category 5 calculation failed: {e}", exc_info=True)
            raise CalculationError(
                calculation_type="category_5",
                reason=str(e),
                category=5,
                input_data=input_data.dict()
            )

    async def _calculate_tier_1(
        self, input_data: Category5Input
    ) -> Optional[CalculationResult]:
        """
        Tier 1: Supplier-specific disposal emission factors.

        Formula: emissions = waste_mass_kg × supplier_disposal_ef

        Args:
            input_data: Category 5 input

        Returns:
            CalculationResult or None if data unavailable
        """
        if not input_data.supplier_disposal_ef or input_data.supplier_disposal_ef < 0:
            return None

        # Calculate emissions
        emissions_kgco2e = input_data.waste_mass_kg * input_data.supplier_disposal_ef

        # Uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=input_data.waste_mass_kg,
                quantity_uncertainty=0.05,  # 5% for weighed waste
                emission_factor=input_data.supplier_disposal_ef,
                factor_uncertainty=0.15,
                iterations=self.config.monte_carlo_iterations
            )

        # Emission factor info
        ef_info = EmissionFactorInfo(
            factor_id=f"supplier_disposal_cat5_{input_data.waste_handler or 'unknown'}",
            value=input_data.supplier_disposal_ef,
            unit="kgCO2e/kg",
            source="supplier_specific",
            source_version="2024",
            gwp_standard="AR6",
            uncertainty=0.15,
            data_quality_score=self.config.tier_1_dqi_score,
            reference_year=2024,
            geographic_scope=input_data.region,
            hash=self.provenance_builder.hash_factor_info(
                input_data.supplier_disposal_ef,
                input_data.waste_handler or "unknown"
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
            category=5,
            tier=TierType.TIER_1,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": "waste_mass_kg × supplier_disposal_ef",
                "waste_mass_kg": input_data.waste_mass_kg,
                "supplier_disposal_ef": input_data.supplier_disposal_ef,
                "result_kgco2e": emissions_kgco2e,
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=5,
            tier=TierType.TIER_1,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_1_supplier_disposal",
            warnings=[],
            metadata={
                "waste_handler": input_data.waste_handler,
                "waste_description": input_data.waste_description,
                "waste_mass_kg": input_data.waste_mass_kg,
            }
        )

    async def _calculate_tier_2(
        self, input_data: Category5Input
    ) -> Optional[CalculationResult]:
        """
        Tier 2: Waste-type and disposal-method specific factors with LLM classification.

        Formula: emissions = waste_mass_kg × (disposal_ef × (1 - recycling_rate))

        Uses LLM to classify waste and identify disposal method.

        Args:
            input_data: Category 5 input

        Returns:
            CalculationResult or None if classification fails
        """
        # Use LLM to classify waste
        waste_info = await self._classify_waste_with_llm(input_data)

        if not waste_info:
            logger.warning(f"LLM waste classification failed for {input_data.waste_description}")
            # Fallback to keyword-based
            waste_info = self._classify_waste_keyword(input_data)

        # Get emission factor
        emission_factor = input_data.emission_factor_kgco2e_per_kg
        if not emission_factor:
            disposal_method = waste_info["disposal_method"]
            emission_factor = DISPOSAL_METHODS[disposal_method]["default_ef"]

        # Apply recycling rate adjustment
        recycling_rate = input_data.recycling_rate or 0.0
        effective_ef = emission_factor * (1 - recycling_rate)

        # Calculate emissions
        emissions_kgco2e = input_data.waste_mass_kg * effective_ef

        # Uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=input_data.waste_mass_kg,
                quantity_uncertainty=0.10,
                emission_factor=effective_ef,
                factor_uncertainty=waste_info["uncertainty"],
                iterations=self.config.monte_carlo_iterations
            )

        # Emission factor info
        ef_info = EmissionFactorInfo(
            factor_id=f"waste_{waste_info['waste_type']}_{waste_info['disposal_method']}_{input_data.region}",
            value=effective_ef,
            unit="kgCO2e/kg",
            source="waste_disposal_database",
            source_version="2024",
            gwp_standard="AR6",
            uncertainty=waste_info["uncertainty"],
            data_quality_score=70.0,
            reference_year=2024,
            geographic_scope=input_data.region,
            hash=self.provenance_builder.hash_factor_info(emission_factor, waste_info["disposal_method"])
        )

        # Data quality
        warnings = []
        if waste_info.get("llm_classified", False):
            warnings.append(
                f"Waste classified by LLM as '{waste_info['waste_type']}' "
                f"with disposal method '{waste_info['disposal_method']}' "
                f"({waste_info['confidence']*100:.0f}% confidence)"
            )
        if recycling_rate > 0:
            warnings.append(f"Recycling rate of {recycling_rate*100:.0f}% applied")

        data_quality = DataQualityInfo(
            dqi_score=self.config.tier_2_dqi_score,
            tier=TierType.TIER_2,
            rating=self._get_quality_rating(self.config.tier_2_dqi_score),
            pedigree_score=3.5,
            warnings=warnings
        )

        # Provenance chain
        provenance = await self.provenance_builder.build(
            category=5,
            tier=TierType.TIER_2,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": "waste_mass_kg × disposal_ef × (1 - recycling_rate)",
                "waste_mass_kg": input_data.waste_mass_kg,
                "disposal_ef": emission_factor,
                "recycling_rate": recycling_rate,
                "effective_ef": effective_ef,
                "waste_type": waste_info["waste_type"],
                "disposal_method": waste_info["disposal_method"],
                "llm_classified": waste_info.get("llm_classified", False),
                "result_kgco2e": emissions_kgco2e,
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=5,
            tier=TierType.TIER_2,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_2_waste_specific",
            warnings=warnings,
            metadata={
                "waste_description": input_data.waste_description,
                "waste_type": waste_info["waste_type"],
                "disposal_method": waste_info["disposal_method"],
                "waste_mass_kg": input_data.waste_mass_kg,
                "recycling_rate": recycling_rate,
                "llm_classification": waste_info.get("llm_reasoning", "N/A"),
            }
        )

    async def _calculate_tier_3(
        self, input_data: Category5Input
    ) -> Optional[CalculationResult]:
        """
        Tier 3: Generic waste disposal factors.

        Uses average disposal factors when specific data unavailable.

        Args:
            input_data: Category 5 input

        Returns:
            CalculationResult
        """
        # Generic disposal factor (average landfill)
        generic_ef = 0.7  # kgCO2e per kg waste

        # Calculate emissions
        emissions_kgco2e = input_data.waste_mass_kg * generic_ef

        # Uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=input_data.waste_mass_kg,
                quantity_uncertainty=0.15,
                emission_factor=generic_ef,
                factor_uncertainty=0.50,  # High uncertainty for generic
                iterations=self.config.monte_carlo_iterations
            )

        # Emission factor info
        ef_info = EmissionFactorInfo(
            factor_id=f"generic_waste_{input_data.region}",
            value=generic_ef,
            unit="kgCO2e/kg",
            source="proxy",
            source_version="default_v1",
            gwp_standard="AR6",
            uncertainty=0.50,
            data_quality_score=self.config.tier_3_dqi_score,
            reference_year=2024,
            geographic_scope=input_data.region,
            hash=self.provenance_builder.hash_factor_info(generic_ef, "generic")
        )

        # Data quality
        warnings = [
            "Generic waste disposal factor used (average landfill)",
            "Consider obtaining waste handler data for better accuracy",
            "No recycling rate applied"
        ]

        data_quality = DataQualityInfo(
            dqi_score=self.config.tier_3_dqi_score,
            tier=TierType.TIER_3,
            rating="fair",
            pedigree_score=2.0,
            warnings=warnings
        )

        # Provenance chain
        provenance = await self.provenance_builder.build(
            category=5,
            tier=TierType.TIER_3,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": "waste_mass_kg × generic_ef",
                "waste_mass_kg": input_data.waste_mass_kg,
                "generic_ef": generic_ef,
                "result_kgco2e": emissions_kgco2e,
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=5,
            tier=TierType.TIER_3,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_3_generic_waste",
            warnings=warnings,
            metadata={
                "waste_description": input_data.waste_description,
                "waste_mass_kg": input_data.waste_mass_kg,
            }
        )

    async def _classify_waste_with_llm(
        self, input_data: Category5Input
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to classify waste type and disposal method.

        Args:
            input_data: Category 5 input

        Returns:
            Waste classification info or None if LLM fails
        """
        if not self.llm_client:
            return None

        try:
            waste_types_list = "\n".join([f"- {wt}: {info['description']}" for wt, info in WASTE_TYPES.items()])
            disposal_methods_list = "\n".join([f"- {dm}: {info['description']}" for dm, info in DISPOSAL_METHODS.items()])

            prompt = f"""
Classify this waste and identify its disposal method:

Description: {input_data.waste_description}
Mass: {input_data.waste_mass_kg} kg
Waste Handler: {input_data.waste_handler or 'Not specified'}

Classify waste type into ONE of:
{waste_types_list}

Identify disposal method as ONE of:
{disposal_methods_list}

If recycling rate is mentioned, estimate it (0.0-1.0).

Return JSON only:
{{
    "waste_type": "waste_type_name",
    "disposal_method": "disposal_method_name",
    "recycling_rate": 0.XX,
    "confidence": 0.XX,
    "reasoning": "Brief explanation"
}}
"""

            # Use the LLM complete method
            response_text = await self.llm_client.complete(prompt)

            # Parse JSON response
            result = json.loads(response_text)

            # Validate
            waste_type = result.get("waste_type", "municipal_solid_waste")
            if waste_type not in WASTE_TYPES:
                waste_type = "municipal_solid_waste"

            disposal_method = result.get("disposal_method", "landfill")
            if disposal_method not in DISPOSAL_METHODS:
                disposal_method = "landfill"

            return {
                "waste_type": waste_type,
                "disposal_method": disposal_method,
                "recycling_rate": result.get("recycling_rate", 0.0),
                "confidence": result.get("confidence", 0.7),
                "uncertainty": DISPOSAL_METHODS[disposal_method]["uncertainty"],
                "llm_classified": True,
                "llm_reasoning": result.get("reasoning", "LLM classification")
            }

        except Exception as e:
            logger.warning(f"LLM waste classification failed: {e}")
            return None

    def _classify_waste_keyword(self, input_data: Category5Input) -> Dict[str, Any]:
        """
        Fallback keyword-based waste classification.

        Args:
            input_data: Category 5 input

        Returns:
            Waste classification info
        """
        description_lower = input_data.waste_description.lower()

        # Classify waste type
        waste_type = "municipal_solid_waste"  # Default
        if input_data.waste_type and input_data.waste_type in WASTE_TYPES:
            waste_type = input_data.waste_type
        else:
            max_matches = 0
            for wt, info in WASTE_TYPES.items():
                matches = sum(1 for keyword in info["keywords"] if keyword in description_lower)
                if matches > max_matches:
                    max_matches = matches
                    waste_type = wt

        # Classify disposal method
        disposal_method = "landfill"  # Default
        if input_data.disposal_method and input_data.disposal_method in DISPOSAL_METHODS:
            disposal_method = input_data.disposal_method
        else:
            max_matches = 0
            for dm, info in DISPOSAL_METHODS.items():
                matches = sum(1 for keyword in info["keywords"] if keyword in description_lower)
                if matches > max_matches:
                    max_matches = matches
                    disposal_method = dm

        return {
            "waste_type": waste_type,
            "disposal_method": disposal_method,
            "recycling_rate": 0.0,
            "confidence": 0.6,
            "uncertainty": DISPOSAL_METHODS[disposal_method]["uncertainty"],
            "llm_classified": False,
            "llm_reasoning": "Keyword-based classification"
        }

    def _validate_input(self, input_data: Category5Input):
        """
        Validate Category 5 input data.

        Args:
            input_data: Input to validate

        Raises:
            DataValidationError: If validation fails
        """
        if input_data.waste_mass_kg <= 0:
            raise DataValidationError(
                field="waste_mass_kg",
                value=input_data.waste_mass_kg,
                reason="Waste mass must be positive",
                category=5
            )

        if not input_data.waste_description or not input_data.waste_description.strip():
            raise DataValidationError(
                field="waste_description",
                value=input_data.waste_description,
                reason="Waste description cannot be empty",
                category=5
            )

        if input_data.recycling_rate is not None:
            if not (0 <= input_data.recycling_rate <= 1):
                raise DataValidationError(
                    field="recycling_rate",
                    value=input_data.recycling_rate,
                    reason="Recycling rate must be between 0 and 1",
                    category=5
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


__all__ = ["Category5Calculator"]
