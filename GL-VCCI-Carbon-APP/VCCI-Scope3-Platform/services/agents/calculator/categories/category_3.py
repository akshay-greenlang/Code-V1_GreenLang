"""
Category 3: Fuel and Energy-Related Activities Calculator
GL-VCCI Scope 3 Platform

Intelligent upstream emissions calculator for fuel and energy with LLM-powered classification.

Covers:
- Upstream emissions from fuel extraction, refining, and transportation (well-to-tank)
- Transmission and distribution (T&D) losses for electricity

3-Tier Calculation Waterfall:
- Tier 1: Supplier-specific upstream emission factors
- Tier 2: Well-to-tank factors from databases with LLM fuel identification
- Tier 3: Generic proxy factors

LLM Intelligence Features:
- Fuel type identification from text descriptions
- Energy source classification
- Data quality assessment

Version: 1.0.0
Date: 2025-11-08
"""

import logging
import json
from typing import Optional, Dict, Any
from datetime import datetime

from ..models import (
    Category3Input,
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
# Fuel Type Classification Data
# ============================================================================

FUEL_TYPES = {
    "electricity": {
        "td_loss_default": 0.065,  # 6.5% average T&D losses
        "wtt_factor": 0.0,  # Already included in grid factors
        "uncertainty": 0.15,
        "keywords": ["electric", "power", "grid", "kwh", "mwh", "utility"]
    },
    "natural_gas": {
        "td_loss_default": 0.02,  # 2% pipeline losses
        "wtt_factor": 0.18,  # kgCO2e per kgCO2e combustion (upstream ~18%)
        "uncertainty": 0.12,
        "keywords": ["natural gas", "gas", "methane", "ng", "pipeline"]
    },
    "diesel": {
        "td_loss_default": 0.0,
        "wtt_factor": 0.21,  # kgCO2e per kgCO2e combustion
        "uncertainty": 0.10,
        "keywords": ["diesel", "gasoil", "distillate"]
    },
    "gasoline": {
        "td_loss_default": 0.0,
        "wtt_factor": 0.24,  # kgCO2e per kgCO2e combustion
        "uncertainty": 0.10,
        "keywords": ["gasoline", "petrol", "gas", "fuel"]
    },
    "fuel_oil": {
        "td_loss_default": 0.0,
        "wtt_factor": 0.19,
        "uncertainty": 0.12,
        "keywords": ["fuel oil", "heating oil", "residual"]
    },
    "coal": {
        "td_loss_default": 0.0,
        "wtt_factor": 0.08,
        "uncertainty": 0.15,
        "keywords": ["coal", "anthracite", "bituminous"]
    },
    "lpg": {
        "td_loss_default": 0.0,
        "wtt_factor": 0.20,
        "uncertainty": 0.12,
        "keywords": ["lpg", "propane", "butane", "liquefied"]
    },
    "biomass": {
        "td_loss_default": 0.0,
        "wtt_factor": 0.15,
        "uncertainty": 0.25,
        "keywords": ["biomass", "wood", "pellet", "biofuel"]
    }
}


class Category3Calculator:
    """
    Category 3 (Fuel & Energy-Related Activities) calculator with LLM intelligence.

    Implements 3-tier calculation waterfall:
    1. Tier 1: Supplier-specific upstream emission factors (highest quality)
    2. Tier 2: Well-to-tank factors from databases with LLM fuel identification
    3. Tier 3: Generic proxy factors

    Features:
    - LLM fuel type identification from description
    - Automatic T&D loss calculations for electricity
    - Well-to-tank (WTT) emission factors
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
        Initialize Category 3 calculator.

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

        logger.info("Initialized Category3Calculator with LLM intelligence")

    async def calculate(self, input_data: Category3Input) -> CalculationResult:
        """
        Calculate Category 3 emissions with 3-tier fallback.

        Args:
            input_data: Category 3 input data

        Returns:
            CalculationResult with emissions and provenance

        Raises:
            DataValidationError: If input data is invalid
            TierFallbackError: If all tiers fail
        """
        start_time = datetime.utcnow()

        # Validate input
        self._validate_input(input_data)

        # Attempt tier cascade
        attempted_tiers = []
        result = None

        try:
            # Tier 1: Supplier-specific upstream factors
            if input_data.supplier_upstream_ef:
                logger.info(f"Attempting Tier 1 calculation for {input_data.fuel_or_energy_type}")
                result = await self._calculate_tier_1(input_data)
                attempted_tiers.append(1)

                if result:
                    logger.info(f"Tier 1 successful for {input_data.fuel_or_energy_type}")
                    return result

            # Tier 2: Database WTT factors with LLM identification
            logger.info(f"Attempting Tier 2 calculation for {input_data.fuel_or_energy_type}")
            result = await self._calculate_tier_2(input_data)
            attempted_tiers.append(2)

            if result and result.data_quality.dqi_score >= 40.0:
                logger.info(f"Tier 2 successful for {input_data.fuel_or_energy_type}")
                return result

            # Tier 3: Generic proxy factors
            logger.info(f"Attempting Tier 3 calculation for {input_data.fuel_or_energy_type}")
            result = await self._calculate_tier_3(input_data)
            attempted_tiers.append(3)

            if result:
                logger.info(f"Tier 3 successful for {input_data.fuel_or_energy_type}")
                return result

            # If we get here, all tiers failed
            raise TierFallbackError(
                attempted_tiers=attempted_tiers,
                reason="No suitable emission factor found for fuel/energy type",
                category=3
            )

        except TierFallbackError:
            raise
        except Exception as e:
            logger.error(f"Category 3 calculation failed: {e}", exc_info=True)
            raise CalculationError(
                calculation_type="category_3",
                reason=str(e),
                category=3,
                input_data=input_data.dict()
            )

    async def _calculate_tier_1(
        self, input_data: Category3Input
    ) -> Optional[CalculationResult]:
        """
        Tier 1: Supplier-specific upstream emission factors.

        Formula: emissions = quantity × (supplier_upstream_ef + supplier_td_losses_ef)

        Args:
            input_data: Category 3 input

        Returns:
            CalculationResult or None if data unavailable
        """
        if not input_data.supplier_upstream_ef:
            return None

        # Calculate upstream emissions
        upstream_emissions = input_data.quantity * input_data.supplier_upstream_ef

        # Calculate T&D losses if provided
        td_emissions = 0.0
        if input_data.supplier_td_losses_ef:
            td_emissions = input_data.quantity * input_data.supplier_td_losses_ef

        emissions_kgco2e = upstream_emissions + td_emissions

        # Uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo:
            total_ef = input_data.supplier_upstream_ef + (input_data.supplier_td_losses_ef or 0)
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=input_data.quantity,
                quantity_uncertainty=0.05,  # 5% for measured energy
                emission_factor=total_ef,
                factor_uncertainty=0.10,
                iterations=self.config.monte_carlo_iterations
            )

        # Emission factor info
        ef_info = EmissionFactorInfo(
            factor_id=f"supplier_upstream_cat3_{input_data.supplier_name or 'unknown'}",
            value=input_data.supplier_upstream_ef + (input_data.supplier_td_losses_ef or 0),
            unit=f"kgCO2e/{input_data.quantity_unit}",
            source="supplier_specific",
            source_version="2024",
            gwp_standard="AR6",
            uncertainty=0.10,
            data_quality_score=self.config.tier_1_dqi_score,
            reference_year=2024,
            geographic_scope=input_data.region,
            hash=self.provenance_builder.hash_factor_info(
                input_data.supplier_upstream_ef,
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
            category=3,
            tier=TierType.TIER_1,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": "quantity × (upstream_ef + td_losses_ef)",
                "quantity": input_data.quantity,
                "quantity_unit": input_data.quantity_unit,
                "supplier_upstream_ef": input_data.supplier_upstream_ef,
                "supplier_td_losses_ef": input_data.supplier_td_losses_ef,
                "upstream_emissions_kgco2e": upstream_emissions,
                "td_losses_emissions_kgco2e": td_emissions,
                "result_kgco2e": emissions_kgco2e,
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=3,
            tier=TierType.TIER_1,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_1_supplier_upstream",
            warnings=[],
            metadata={
                "supplier_name": input_data.supplier_name,
                "fuel_or_energy_type": input_data.fuel_or_energy_type,
                "upstream_emissions_kgco2e": upstream_emissions,
                "td_losses_emissions_kgco2e": td_emissions,
            }
        )

    async def _calculate_tier_2(
        self, input_data: Category3Input
    ) -> Optional[CalculationResult]:
        """
        Tier 2: Well-to-tank factors with LLM fuel identification.

        Formula (electricity): emissions = quantity × grid_ef × td_loss_percentage
        Formula (fuels): emissions = quantity × combustion_ef × wtt_factor

        Uses LLM to identify fuel type if ambiguous.

        Args:
            input_data: Category 3 input

        Returns:
            CalculationResult or None if factors not found
        """
        # Use LLM to identify fuel type
        fuel_info = await self._identify_fuel_with_llm(input_data)

        if not fuel_info:
            logger.warning(f"LLM fuel identification failed for {input_data.fuel_or_energy_type}")
            # Fallback to keyword-based
            fuel_info = self._identify_fuel_keyword(input_data)

        fuel_type = fuel_info["fuel_type"]
        is_electricity = fuel_type == "electricity"

        # Calculate emissions based on fuel type
        if is_electricity:
            # For electricity: T&D losses
            td_loss_pct = input_data.td_loss_percentage or FUEL_TYPES[fuel_type]["td_loss_default"]

            # Get grid emission factor for combustion (Category 2 Scope 2)
            grid_ef = await self._get_grid_emission_factor(input_data.region, input_data.grid_region)

            # T&D loss emissions
            emissions_kgco2e = input_data.quantity * grid_ef * td_loss_pct

            calculation_formula = "quantity × grid_ef × td_loss_percentage"
            metadata_extra = {
                "grid_emission_factor": grid_ef,
                "td_loss_percentage": td_loss_pct,
                "is_electricity": True
            }
        else:
            # For fuels: Well-to-tank (upstream)
            wtt_factor = input_data.well_to_tank_ef or FUEL_TYPES[fuel_type]["wtt_factor"]

            # Get combustion emission factor (this would be Scope 1 or 2)
            combustion_ef = await self._get_combustion_emission_factor(
                fuel_type, input_data.quantity_unit, input_data.region
            )

            # Upstream emissions
            emissions_kgco2e = input_data.quantity * combustion_ef * wtt_factor

            calculation_formula = "quantity × combustion_ef × wtt_factor"
            metadata_extra = {
                "combustion_emission_factor": combustion_ef,
                "wtt_factor": wtt_factor,
                "is_electricity": False
            }

        # Uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=input_data.quantity,
                quantity_uncertainty=0.10,
                emission_factor=emissions_kgco2e / input_data.quantity,
                factor_uncertainty=fuel_info["uncertainty"],
                iterations=self.config.monte_carlo_iterations
            )

        # Emission factor info
        ef_info = EmissionFactorInfo(
            factor_id=f"wtt_{fuel_type}_{input_data.region}",
            value=emissions_kgco2e / input_data.quantity,
            unit=f"kgCO2e/{input_data.quantity_unit}",
            source="database_wtt_factors",
            source_version="2024",
            gwp_standard="AR6",
            uncertainty=fuel_info["uncertainty"],
            data_quality_score=70.0,
            reference_year=2024,
            geographic_scope=input_data.region,
            hash=self.provenance_builder.hash_factor_info(wtt_factor if not is_electricity else td_loss_pct, fuel_type)
        )

        # Data quality
        warnings = []
        if fuel_info.get("llm_identified", False):
            warnings.append(f"Fuel type identified by LLM as '{fuel_type}' with {fuel_info['confidence']*100:.0f}% confidence")

        data_quality = DataQualityInfo(
            dqi_score=self.config.tier_2_dqi_score,
            tier=TierType.TIER_2,
            rating=self._get_quality_rating(self.config.tier_2_dqi_score),
            pedigree_score=3.5,
            warnings=warnings
        )

        # Provenance chain
        provenance = await self.provenance_builder.build(
            category=3,
            tier=TierType.TIER_2,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": calculation_formula,
                "quantity": input_data.quantity,
                "quantity_unit": input_data.quantity_unit,
                "fuel_type": fuel_type,
                "llm_identified": fuel_info.get("llm_identified", False),
                "result_kgco2e": emissions_kgco2e,
                **metadata_extra
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=3,
            tier=TierType.TIER_2,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_2_wtt_factors",
            warnings=warnings,
            metadata={
                "fuel_or_energy_type": input_data.fuel_or_energy_type,
                "identified_fuel_type": fuel_type,
                "llm_identification": fuel_info.get("llm_reasoning", "N/A"),
                **metadata_extra
            }
        )

    async def _calculate_tier_3(
        self, input_data: Category3Input
    ) -> Optional[CalculationResult]:
        """
        Tier 3: Generic proxy upstream factors.

        Uses average upstream factors when specific data unavailable.

        Args:
            input_data: Category 3 input

        Returns:
            CalculationResult
        """
        # Generic proxy: assume 15% upstream for all fuels/energy
        proxy_upstream_factor = 0.15

        # Get base combustion/grid factor
        fuel_info = self._identify_fuel_keyword(input_data)
        fuel_type = fuel_info["fuel_type"]

        if fuel_type == "electricity":
            base_ef = await self._get_grid_emission_factor(input_data.region, input_data.grid_region)
        else:
            base_ef = await self._get_combustion_emission_factor(
                fuel_type, input_data.quantity_unit, input_data.region
            )

        # Calculate proxy upstream emissions
        emissions_kgco2e = input_data.quantity * base_ef * proxy_upstream_factor

        # Uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=input_data.quantity,
                quantity_uncertainty=0.20,
                emission_factor=base_ef * proxy_upstream_factor,
                factor_uncertainty=0.50,  # High uncertainty for proxies
                iterations=self.config.monte_carlo_iterations
            )

        # Emission factor info
        ef_info = EmissionFactorInfo(
            factor_id=f"proxy_upstream_{fuel_type}_{input_data.region}",
            value=base_ef * proxy_upstream_factor,
            unit=f"kgCO2e/{input_data.quantity_unit}",
            source="proxy",
            source_version="default_v1",
            gwp_standard="AR6",
            uncertainty=0.50,
            data_quality_score=self.config.tier_3_dqi_score,
            reference_year=2024,
            geographic_scope=input_data.region,
            hash=self.provenance_builder.hash_factor_info(proxy_upstream_factor, "proxy")
        )

        # Data quality
        warnings = [
            "Generic proxy upstream factor used (15% of combustion emissions)",
            "Consider obtaining supplier-specific upstream data for better accuracy"
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
            category=3,
            tier=TierType.TIER_3,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": "quantity × base_ef × proxy_upstream_factor",
                "quantity": input_data.quantity,
                "quantity_unit": input_data.quantity_unit,
                "base_ef": base_ef,
                "proxy_upstream_factor": proxy_upstream_factor,
                "result_kgco2e": emissions_kgco2e,
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=3,
            tier=TierType.TIER_3,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_3_proxy_upstream",
            warnings=warnings,
            metadata={
                "fuel_or_energy_type": input_data.fuel_or_energy_type,
                "proxy_upstream_factor": proxy_upstream_factor,
            }
        )

    async def _identify_fuel_with_llm(
        self, input_data: Category3Input
    ) -> Optional[Dict[str, Any]]:
        """
        Use LLM to identify fuel/energy type from description.

        Args:
            input_data: Category 3 input

        Returns:
            Fuel identification info or None if LLM fails
        """
        if not self.llm_client:
            return None

        try:
            fuel_types_list = "\n".join([f"- {ft}: {info.get('keywords', [])}" for ft, info in FUEL_TYPES.items()])

            prompt = f"""
Identify the fuel or energy type from this description:

Description: {input_data.fuel_or_energy_type}
Quantity: {input_data.quantity} {input_data.quantity_unit}
Grid Region: {input_data.grid_region or 'Not specified'}

Classify into ONE of these fuel types:
{fuel_types_list}

Return JSON only:
{{
    "fuel_type": "fuel_type_name",
    "confidence": 0.XX,
    "reasoning": "Brief explanation"
}}
"""

            # Use the LLM complete method
            response_text = await self.llm_client.complete(prompt)

            # Parse JSON response
            result = json.loads(response_text)

            # Validate
            fuel_type = result.get("fuel_type", "electricity")
            if fuel_type not in FUEL_TYPES:
                fuel_type = "electricity"

            return {
                "fuel_type": fuel_type,
                "confidence": result.get("confidence", 0.7),
                "uncertainty": FUEL_TYPES[fuel_type]["uncertainty"],
                "llm_identified": True,
                "llm_reasoning": result.get("reasoning", "LLM identification")
            }

        except Exception as e:
            logger.warning(f"LLM fuel identification failed: {e}")
            return None

    def _identify_fuel_keyword(self, input_data: Category3Input) -> Dict[str, Any]:
        """
        Fallback keyword-based fuel identification.

        Args:
            input_data: Category 3 input

        Returns:
            Fuel identification info
        """
        description_lower = input_data.fuel_or_energy_type.lower()

        # Keyword matching
        fuel_type = "electricity"  # Default
        max_matches = 0

        for ft, info in FUEL_TYPES.items():
            matches = sum(1 for keyword in info["keywords"] if keyword in description_lower)
            if matches > max_matches:
                max_matches = matches
                fuel_type = ft

        return {
            "fuel_type": fuel_type,
            "confidence": 0.6,
            "uncertainty": FUEL_TYPES[fuel_type]["uncertainty"],
            "llm_identified": False,
            "llm_reasoning": "Keyword-based identification"
        }

    async def _get_grid_emission_factor(
        self, region: str, grid_region: Optional[str]
    ) -> float:
        """Get grid emission factor for electricity."""
        # Simplified: return regional average
        # In production, this would query factor_broker
        grid_factors = {
            "US": 0.417,  # kgCO2e/kWh
            "GB": 0.233,
            "DE": 0.366,
            "FR": 0.056,
            "CN": 0.555,
            "Global": 0.475
        }
        return grid_factors.get(region, grid_factors["Global"])

    async def _get_combustion_emission_factor(
        self, fuel_type: str, unit: str, region: str
    ) -> float:
        """Get combustion emission factor for fuel."""
        # Simplified combustion factors (kgCO2e per unit)
        # In production, this would query factor_broker with unit conversion
        combustion_factors = {
            "natural_gas": 2.0,  # kgCO2e/m3 or similar
            "diesel": 2.68,  # kgCO2e/liter
            "gasoline": 2.31,  # kgCO2e/liter
            "fuel_oil": 3.19,  # kgCO2e/liter
            "coal": 2.42,  # kgCO2e/kg
            "lpg": 1.51,  # kgCO2e/liter
            "biomass": 0.39,  # kgCO2e/kg (lower due to biogenic)
        }
        return combustion_factors.get(fuel_type, 2.0)

    def _validate_input(self, input_data: Category3Input):
        """
        Validate Category 3 input data.

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
                category=3
            )

        if not input_data.fuel_or_energy_type or not input_data.fuel_or_energy_type.strip():
            raise DataValidationError(
                field="fuel_or_energy_type",
                value=input_data.fuel_or_energy_type,
                reason="Fuel/energy type cannot be empty",
                category=3
            )

        if input_data.td_loss_percentage is not None:
            if not (0 <= input_data.td_loss_percentage <= 1):
                raise DataValidationError(
                    field="td_loss_percentage",
                    value=input_data.td_loss_percentage,
                    reason="T&D loss percentage must be between 0 and 1",
                    category=3
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


__all__ = ["Category3Calculator"]
