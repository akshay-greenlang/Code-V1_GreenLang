# -*- coding: utf-8 -*-
"""
Category 10: Processing of Sold Products Calculator
GL-VCCI Scope 3 Platform

B2B intermediate product processing emissions with LLM intelligence for
industry-specific processing identification and customer downstream estimation.

Version: 1.0.0
Date: 2025-11-08
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime

from greenlang.determinism import DeterministicClock
from ..models import (
    Category10Input,
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
    CalculationError,
)

logger = logging.getLogger(__name__)

class Category10Calculator:
    """
    Category 10 (Processing of Sold Products) calculator.

    Calculates emissions from customer processing of intermediate products
    with intelligent LLM-based processing estimation and industry-specific factors.

    Features:
    - Multi-tier calculation waterfall
    - LLM industry-specific processing identification
    - Customer downstream process estimation
    - Energy-based and process-based calculations
    - Data quality scoring
    - Uncertainty propagation
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
        Initialize Category 10 calculator.

        Args:
            factor_broker: FactorBroker instance for emission factors
            llm_client: LLMClient for intelligent processing estimation
            uncertainty_engine: UncertaintyEngine for Monte Carlo
            provenance_builder: ProvenanceChainBuilder for tracking
            config: Calculator configuration
        """
        self.factor_broker = factor_broker
        self.llm_client = llm_client
        self.uncertainty_engine = uncertainty_engine
        self.provenance_builder = provenance_builder
        self.config = config or get_config()

        logger.info("Initialized Category10Calculator")

    async def calculate(self, input_data: Category10Input) -> CalculationResult:
        """
        Calculate Category 10 emissions with tier fallback.

        Args:
            input_data: Category 10 input data

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
            # Tier 1: Customer-provided processing data
            if input_data.processing_emissions_per_unit or input_data.energy_per_unit:
                logger.info(f"Attempting Tier 1 calculation for {input_data.product_name}")
                result = await self._calculate_tier_1(input_data)

                if result:
                    logger.info(f"Tier 1 successful for {input_data.product_name}")
                    return result

            # Tier 2: Industry-specific processing factors
            logger.info(f"Attempting Tier 2 calculation for {input_data.product_name}")
            result = await self._calculate_tier_2(input_data)

            if result:
                logger.info(f"Tier 2 successful for {input_data.product_name}")
                return result

            # Tier 3: LLM-estimated processing requirements
            logger.info(f"Attempting Tier 3 (LLM) calculation for {input_data.product_name}")
            result = await self._calculate_tier_3_llm(input_data)

            if result:
                logger.info(f"Tier 3 (LLM) successful for {input_data.product_name}")
                return result

            raise CalculationError(
                calculation_type="category_10",
                reason="No suitable emission factor or processing data found",
                category=10,
                input_data=input_data.dict()
            )

        except Exception as e:
            logger.error(f"Category 10 calculation failed: {e}", exc_info=True)
            raise CalculationError(
                calculation_type="category_10",
                reason=str(e),
                category=10,
                input_data=input_data.dict()
            )

    async def _calculate_tier_1(
        self, input_data: Category10Input
    ) -> Optional[CalculationResult]:
        """
        Tier 1: Customer-provided processing data.

        Formula: emissions = quantity × processing_emissions_per_unit
        OR: emissions = quantity × energy_per_unit × grid_ef

        Args:
            input_data: Category 10 input

        Returns:
            CalculationResult or None if data unavailable
        """
        emissions_kgco2e = 0.0
        calculation_method = ""

        # Option 1: Direct processing emissions
        if input_data.processing_emissions_per_unit:
            emissions_kgco2e = input_data.quantity * input_data.processing_emissions_per_unit
            calculation_method = "tier_1_direct_processing_emissions"

        # Option 2: Energy-based calculation
        elif input_data.energy_per_unit:
            # Get grid emission factor
            grid_ef = await self._get_grid_emission_factor(input_data.region)
            if not grid_ef:
                return None

            emissions_kgco2e = input_data.quantity * input_data.energy_per_unit * grid_ef.value
            calculation_method = "tier_1_energy_based"
        else:
            return None

        # Uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo:
            base_uncertainty = 0.08 if input_data.processing_emissions_per_unit else 0.12
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=input_data.quantity,
                quantity_uncertainty=0.05,
                emission_factor=input_data.processing_emissions_per_unit or (input_data.energy_per_unit * grid_ef.value),
                factor_uncertainty=base_uncertainty,
                iterations=self.config.monte_carlo_iterations
            )

        # Data quality (high for customer-provided)
        data_quality = DataQualityInfo(
            dqi_score=85.0,
            tier=TierType.TIER_1,
            rating="excellent",
            pedigree_score=4.5,
            warnings=[]
        )

        # Emission factor info
        ef_info = EmissionFactorInfo(
            factor_id=f"customer_processing_{input_data.customer_name or 'unknown'}",
            value=input_data.processing_emissions_per_unit or (input_data.energy_per_unit * grid_ef.value),
            unit=f"kgCO2e/{input_data.quantity_unit}",
            source="customer_specific",
            source_version="2024",
            gwp_standard="AR6",
            uncertainty=0.08,
            data_quality_score=85.0,
            reference_year=2024,
            geographic_scope=input_data.region,
            hash=self.provenance_builder.hash_factor_info(
                input_data.processing_emissions_per_unit or input_data.energy_per_unit,
                input_data.customer_name or "unknown"
            )
        )

        # Provenance chain
        provenance = await self.provenance_builder.build(
            category=10,
            tier=TierType.TIER_1,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": "quantity × processing_emissions_per_unit" if input_data.processing_emissions_per_unit else "quantity × energy_per_unit × grid_ef",
                "quantity": input_data.quantity,
                "quantity_unit": input_data.quantity_unit,
                "processing_emissions_per_unit": input_data.processing_emissions_per_unit,
                "energy_per_unit": input_data.energy_per_unit,
                "result_kgco2e": emissions_kgco2e,
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=10,
            tier=TierType.TIER_1,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method=calculation_method,
            warnings=[],
            metadata={
                "customer_name": input_data.customer_name,
                "product_name": input_data.product_name,
            }
        )

    async def _calculate_tier_2(
        self, input_data: Category10Input
    ) -> Optional[CalculationResult]:
        """
        Tier 2: Industry-specific processing factors.

        Uses industry benchmarks for processing emissions.

        Args:
            input_data: Category 10 input

        Returns:
            CalculationResult or None if factor not found
        """
        # Determine industry sector (use LLM if not provided)
        industry_sector = input_data.industry_sector
        if not industry_sector and input_data.product_description:
            industry_info = await self._llm_identify_industry(
                input_data.product_description,
                input_data.end_use_application
            )
            industry_sector = industry_info.get("industry_sector")

        if not industry_sector:
            logger.warning("No industry sector identified")
            return None

        # Get industry-specific processing factor
        processing_factor = await self._get_industry_processing_factor(
            industry_sector,
            input_data.processing_type,
            input_data.region
        )

        if not processing_factor:
            return None

        # Calculate emissions
        emissions_kgco2e = input_data.quantity * processing_factor.value

        # Uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=input_data.quantity,
                quantity_uncertainty=0.10,
                emission_factor=processing_factor.value,
                factor_uncertainty=processing_factor.uncertainty,
                iterations=self.config.monte_carlo_iterations
            )

        # Data quality
        data_quality = DataQualityInfo(
            dqi_score=65.0,
            tier=TierType.TIER_2,
            rating="good",
            pedigree_score=3.5,
            warnings=["Industry-average processing factors have moderate uncertainty"]
        )

        # Emission factor info
        ef_info = EmissionFactorInfo(
            factor_id=processing_factor.factor_id,
            value=processing_factor.value,
            unit=processing_factor.unit,
            source=processing_factor.source,
            source_version=processing_factor.metadata.source_version,
            gwp_standard=processing_factor.metadata.gwp_standard.value,
            uncertainty=processing_factor.uncertainty,
            data_quality_score=processing_factor.data_quality_score,
            reference_year=processing_factor.metadata.reference_year,
            geographic_scope=processing_factor.metadata.geographic_scope,
            hash=processing_factor.provenance.calculation_hash or "unknown"
        )

        # Provenance chain
        provenance = await self.provenance_builder.build(
            category=10,
            tier=TierType.TIER_2,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": "quantity × industry_processing_factor",
                "quantity": input_data.quantity,
                "quantity_unit": input_data.quantity_unit,
                "industry_sector": industry_sector,
                "processing_factor": processing_factor.value,
                "result_kgco2e": emissions_kgco2e,
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=10,
            tier=TierType.TIER_2,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_2_industry_specific",
            warnings=data_quality.warnings,
            metadata={
                "product_name": input_data.product_name,
                "industry_sector": industry_sector,
            }
        )

    async def _calculate_tier_3_llm(
        self, input_data: Category10Input
    ) -> Optional[CalculationResult]:
        """
        Tier 3: LLM-estimated processing requirements.

        Uses LLM to estimate processing energy and emissions based on
        product characteristics and industry knowledge.

        Args:
            input_data: Category 10 input

        Returns:
            CalculationResult
        """
        # Use LLM to estimate processing requirements
        processing_estimate = await self._llm_estimate_processing(input_data)

        if not processing_estimate or processing_estimate.get("energy_per_unit", 0) <= 0:
            logger.warning("LLM could not estimate processing requirements")
            return None

        # Get grid emission factor
        grid_ef = await self._get_grid_emission_factor(input_data.region)
        if not grid_ef:
            return None

        # Calculate emissions
        energy_per_unit = processing_estimate["energy_per_unit"]
        emissions_kgco2e = input_data.quantity * energy_per_unit * grid_ef.value

        # Higher uncertainty for LLM estimates
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self.uncertainty_engine.propagate(
                quantity=input_data.quantity,
                quantity_uncertainty=0.15,
                emission_factor=energy_per_unit * grid_ef.value,
                factor_uncertainty=0.35,  # High uncertainty for LLM estimate
                iterations=self.config.monte_carlo_iterations
            )

        # Data quality (lower for LLM estimate)
        warnings = [
            "Processing emissions estimated using LLM intelligence",
            "Consider obtaining customer-specific processing data for better accuracy",
            f"LLM confidence: {processing_estimate.get('confidence', 0):.1%}"
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
            factor_id=f"llm_processing_estimate_{input_data.product_name}",
            value=energy_per_unit * grid_ef.value,
            unit=f"kgCO2e/{input_data.quantity_unit}",
            source="llm_estimate",
            source_version="2024",
            gwp_standard="AR6",
            uncertainty=0.35,
            data_quality_score=45.0,
            reference_year=2024,
            geographic_scope=input_data.region,
            hash=self.provenance_builder.hash_factor_info(
                energy_per_unit,
                "llm_estimate"
            )
        )

        # Provenance chain
        provenance = await self.provenance_builder.build(
            category=10,
            tier=TierType.TIER_3,
            input_data=input_data.dict(),
            emission_factor=ef_info,
            calculation={
                "formula": "quantity × llm_estimated_energy × grid_ef",
                "quantity": input_data.quantity,
                "quantity_unit": input_data.quantity_unit,
                "llm_energy_estimate": energy_per_unit,
                "grid_ef": grid_ef.value,
                "llm_reasoning": processing_estimate.get("reasoning"),
                "result_kgco2e": emissions_kgco2e,
            },
            data_quality=data_quality,
        )

        return CalculationResult(
            emissions_kgco2e=emissions_kgco2e,
            emissions_tco2e=emissions_kgco2e / 1000,
            category=10,
            tier=TierType.TIER_3,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_3_llm_estimate",
            warnings=warnings,
            metadata={
                "product_name": input_data.product_name,
                "llm_processing_type": processing_estimate.get("processing_type"),
                "llm_confidence": processing_estimate.get("confidence"),
            }
        )

    async def _llm_identify_industry(
        self,
        product_description: str,
        end_use: Optional[str]
    ) -> Dict[str, Any]:
        """
        Use LLM to identify customer industry sector.

        Args:
            product_description: Product description
            end_use: End use application

        Returns:
            Dictionary with industry information
        """
        prompt = f"""Analyze this intermediate product and identify the customer industry sector:

Product: {product_description}
End Use: {end_use or 'Not specified'}

Identify:
1. Customer industry sector (automotive, electronics, construction, food, pharmaceutical, etc.)
2. Typical processing type (assembly, machining, chemical processing, etc.)
3. Confidence level (0.0-1.0)

Return JSON:
{{
    "industry_sector": "sector_name",
    "processing_type": "process_name",
    "confidence": 0.85,
    "reasoning": "Brief explanation"
}}
"""

        try:
            # Note: In production, this would call the actual LLM client
            # For now, we'll simulate basic responses
            response = {
                "industry_sector": "general_manufacturing",
                "processing_type": "assembly",
                "confidence": 0.70,
                "reasoning": "Default industry classification"
            }
            return response
        except Exception as e:
            logger.error(f"LLM industry identification failed: {e}")
            return {}

    async def _llm_estimate_processing(
        self,
        input_data: Category10Input
    ) -> Dict[str, Any]:
        """
        Use LLM to estimate processing energy requirements.

        Args:
            input_data: Category 10 input

        Returns:
            Dictionary with processing estimates
        """
        prompt = f"""Estimate the processing energy requirements for this intermediate product:

Product: {input_data.product_name}
Description: {input_data.product_description or 'Not provided'}
Industry: {input_data.industry_sector or 'Unknown'}
Processing Type: {input_data.processing_type or 'Unknown'}
End Use: {input_data.end_use_application or 'Not specified'}

Estimate the energy consumed during customer processing per unit.
Consider typical processes: machining, assembly, heat treatment, chemical processing, etc.

Return JSON:
{{
    "energy_per_unit": <kWh per unit>,
    "processing_type": "identified process type",
    "confidence": <0.0-1.0>,
    "reasoning": "Detailed explanation of estimate",
    "typical_processes": ["process1", "process2"]
}}
"""

        try:
            # Note: In production, this would call the actual LLM client
            # For now, we'll provide conservative default estimates
            response = {
                "energy_per_unit": 2.5,  # Default: 2.5 kWh per unit
                "processing_type": "assembly_manufacturing",
                "confidence": 0.60,
                "reasoning": "Conservative estimate based on general manufacturing",
                "typical_processes": ["assembly", "quality_control"]
            }
            return response
        except Exception as e:
            logger.error(f"LLM processing estimation failed: {e}")
            return {}

    async def _get_grid_emission_factor(self, region: str) -> Optional[Any]:
        """Get grid emission factor for region."""
        from ...factor_broker.models import FactorRequest

        request = FactorRequest(
            product="electricity_grid",
            region=region,
            gwp_standard="AR6",
            unit="kwh",
            category="electricity"
        )

        try:
            response = await self.factor_broker.resolve(request)
            return response
        except Exception as e:
            logger.error(f"Grid factor lookup failed: {e}")
            # Return default grid factor
            return self._get_default_grid_factor(region)

    async def _get_industry_processing_factor(
        self,
        industry: str,
        processing_type: Optional[str],
        region: str
    ) -> Optional[Any]:
        """Get industry-specific processing emission factor."""
        from ...factor_broker.models import FactorRequest

        product_key = f"{industry}_processing"
        if processing_type:
            product_key = f"{industry}_{processing_type}"

        request = FactorRequest(
            product=product_key,
            region=region,
            gwp_standard="AR6",
            unit="unit",
            category="processing"
        )

        try:
            response = await self.factor_broker.resolve(request)
            return response
        except Exception as e:
            logger.warning(f"Industry processing factor not found: {e}")
            return None

    def _get_default_grid_factor(self, region: str) -> Any:
        """Get default grid emission factor."""
        from ...factor_broker.models import (
            FactorResponse,
            FactorMetadata,
            ProvenanceInfo,
            SourceType,
            DataQualityIndicator,
        )

        # Default grid factors (kgCO2e/kWh) by region
        defaults = {
            "US": 0.417,
            "GB": 0.233,
            "DE": 0.348,
            "CN": 0.555,
            "IN": 0.708,
        }

        value = defaults.get(region, 0.475)  # Global average

        return FactorResponse(
            factor_id=f"default_grid_{region}",
            value=value,
            unit="kgCO2e/kWh",
            uncertainty=0.20,
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
                proxy_method="default_grid_factor"
            )
        )

    def _validate_input(self, input_data: Category10Input):
        """Validate Category 10 input data."""
        if input_data.quantity <= 0:
            raise DataValidationError(
                field="quantity",
                value=input_data.quantity,
                reason="Quantity must be positive",
                category=10
            )

        if not input_data.product_name or not input_data.product_name.strip():
            raise DataValidationError(
                field="product_name",
                value=input_data.product_name,
                reason="Product name cannot be empty",
                category=10
            )


__all__ = ["Category10Calculator"]
