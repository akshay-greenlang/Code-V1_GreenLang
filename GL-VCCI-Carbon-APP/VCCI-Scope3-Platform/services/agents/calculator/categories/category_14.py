# -*- coding: utf-8 -*-
"""
Category 14: Franchises Calculator
GL-VCCI Scope 3 Platform

Emissions from the operation of franchises not included in Scope 1 or 2.

Key Features:
- Franchise location emissions calculator
- LLM operational control determination
- Energy use per franchise estimation
- Industry-specific franchise modeling
- Multiple calculation approaches

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
    Category14Input,
    CalculationResult,
    DataQualityInfo,
    EmissionFactorInfo,
    ProvenanceChain,
    UncertaintyResult,
)
from ..config import TierType, FranchiseType, OperationalControl, get_config
from ..exceptions import (
    DataValidationError,
    EmissionFactorNotFoundError,
    CalculationError,
)

logger = logging.getLogger(__name__)

class Category14Calculator:
    """
    Category 14 (Franchises) calculator with LLM intelligence.

    Calculation Hierarchy:
    1. Tier 1: Actual energy data from franchises (highest quality)
    2. Tier 2: Revenue-based or area-based estimation
    3. Tier 3: LLM-based estimation from industry benchmarks

    Features:
    - LLM franchise type classification
    - LLM operational control determination
    - Industry-specific benchmarks
    - Multi-location aggregation
    - Regional emission factor variation
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
        Initialize Category 14 calculator.

        Args:
            factor_broker: FactorBroker instance for emission factors
            llm_client: LLM client for intelligent classification
            uncertainty_engine: UncertaintyEngine for Monte Carlo
            provenance_builder: ProvenanceChainBuilder for tracking
            config: Calculator configuration
        """
        self.factor_broker = factor_broker
        self.llm_client = llm_client
        self.uncertainty_engine = uncertainty_engine
        self.provenance_builder = provenance_builder
        self.config = config or get_config()

        # Franchise energy intensity benchmarks (kWh/sqm/year)
        self.energy_intensities = {
            FranchiseType.QUICK_SERVICE_RESTAURANT: 600.0,
            FranchiseType.CASUAL_DINING: 500.0,
            FranchiseType.FAST_FOOD: 650.0,
            FranchiseType.COFFEE_SHOP: 450.0,
            FranchiseType.RETAIL_STORE: 200.0,
            FranchiseType.CONVENIENCE_STORE: 350.0,
            FranchiseType.GYM_FITNESS: 180.0,
            FranchiseType.HOTEL: 250.0,
            FranchiseType.AUTO_SERVICE: 150.0,
            FranchiseType.BEAUTY_SALON: 220.0,
            FranchiseType.CLEANING_SERVICE: 100.0,
            FranchiseType.EDUCATION: 160.0,
            FranchiseType.HEALTHCARE: 280.0,
            FranchiseType.OTHER: 200.0,
        }

        # Revenue-based intensity (kgCO2e per $1000 revenue)
        self.revenue_intensities = {
            FranchiseType.QUICK_SERVICE_RESTAURANT: 45.0,
            FranchiseType.CASUAL_DINING: 55.0,
            FranchiseType.FAST_FOOD: 50.0,
            FranchiseType.COFFEE_SHOP: 35.0,
            FranchiseType.RETAIL_STORE: 15.0,
            FranchiseType.CONVENIENCE_STORE: 25.0,
            FranchiseType.GYM_FITNESS: 20.0,
            FranchiseType.HOTEL: 30.0,
            FranchiseType.AUTO_SERVICE: 18.0,
            FranchiseType.BEAUTY_SALON: 22.0,
            FranchiseType.CLEANING_SERVICE: 12.0,
            FranchiseType.EDUCATION: 16.0,
            FranchiseType.HEALTHCARE: 28.0,
            FranchiseType.OTHER: 20.0,
        }

        # Typical floor areas by franchise type (sqm)
        self.typical_floor_areas = {
            FranchiseType.QUICK_SERVICE_RESTAURANT: 180.0,
            FranchiseType.CASUAL_DINING: 350.0,
            FranchiseType.FAST_FOOD: 150.0,
            FranchiseType.COFFEE_SHOP: 120.0,
            FranchiseType.RETAIL_STORE: 200.0,
            FranchiseType.CONVENIENCE_STORE: 130.0,
            FranchiseType.GYM_FITNESS: 600.0,
            FranchiseType.HOTEL: 2000.0,
            FranchiseType.AUTO_SERVICE: 400.0,
            FranchiseType.BEAUTY_SALON: 100.0,
            FranchiseType.CLEANING_SERVICE: 50.0,
            FranchiseType.EDUCATION: 300.0,
            FranchiseType.HEALTHCARE: 250.0,
            FranchiseType.OTHER: 200.0,
        }

        logger.info("Initialized Category14Calculator with LLM intelligence")

    async def calculate(self, input_data: Category14Input) -> CalculationResult:
        """
        Calculate Category 14 emissions with LLM-enhanced intelligence.

        Args:
            input_data: Category 14 input data

        Returns:
            CalculationResult with emissions and provenance

        Raises:
            DataValidationError: If input data is invalid
            CalculationError: If calculation fails
        """
        start_time = DeterministicClock.utcnow()

        # Validate input
        self._validate_input(input_data)

        # LLM Enhancement: Classify franchise type if needed
        if not input_data.franchise_type and input_data.franchise_description:
            input_data.franchise_type = await self._llm_classify_franchise_type(
                input_data.franchise_description
            )
            logger.info(
                f"LLM classified franchise type: {input_data.franchise_type.value}"
            )

        # LLM Enhancement: Determine operational control
        if not input_data.operational_control and input_data.franchise_description:
            input_data.operational_control = await self._llm_determine_operational_control(
                input_data.franchise_description,
                input_data.franchise_name
            )
            logger.info(
                f"LLM determined operational control: {input_data.operational_control.value}"
            )

        # Check if franchises should be in Category 14
        if input_data.operational_control == OperationalControl.FRANCHISOR_FULL:
            logger.warning(
                "Franchises with full franchisor control should be in Scope 1/2, not Category 14"
            )

        # Determine calculation tier and compute emissions
        try:
            if input_data.total_energy_kwh or input_data.avg_energy_per_location_kwh:
                # Tier 1: Actual energy data
                result = await self._calculate_tier_1(input_data)
                logger.info(
                    f"Tier 1 calculation successful for franchise {input_data.franchise_id}"
                )
            elif input_data.total_revenue_usd or input_data.avg_revenue_per_location_usd:
                # Tier 2: Revenue-based
                result = await self._calculate_tier_2_revenue(input_data)
                logger.info(
                    f"Tier 2 (revenue) calculation successful for franchise {input_data.franchise_id}"
                )
            elif input_data.avg_floor_area_sqm and input_data.franchise_type:
                # Tier 2: Area-based
                result = await self._calculate_tier_2_area(input_data)
                logger.info(
                    f"Tier 2 (area) calculation successful for franchise {input_data.franchise_id}"
                )
            else:
                # Tier 3: LLM-based estimation
                result = await self._calculate_tier_3(input_data)
                logger.info(
                    f"Tier 3 calculation successful for franchise {input_data.franchise_id}"
                )

            return result

        except Exception as e:
            logger.error(f"Category 14 calculation failed: {e}", exc_info=True)
            raise CalculationError(
                calculation_type="category_14",
                reason=str(e),
                category=14,
                input_data=input_data.model_dump()
            )

    async def _calculate_tier_1(
        self,
        input_data: Category14Input
    ) -> CalculationResult:
        """
        Tier 1: Calculate using actual franchise energy data.

        Args:
            input_data: Category 14 input data

        Returns:
            CalculationResult
        """
        # Calculate total energy
        if input_data.total_energy_kwh:
            total_energy = input_data.total_energy_kwh
        else:
            total_energy = input_data.avg_energy_per_location_kwh * input_data.num_locations

        # Get grid emission factors (weighted by regions if provided)
        if input_data.locations_by_region:
            total_emissions_kg = 0.0
            for region, num_locs in input_data.locations_by_region.items():
                regional_energy = total_energy * (num_locs / input_data.num_locations)
                grid_ef = await self._get_grid_emission_factor(region)
                total_emissions_kg += regional_energy * grid_ef.value

            # Use primary region EF for metadata
            grid_ef = await self._get_grid_emission_factor(input_data.region)
        else:
            grid_ef = await self._get_grid_emission_factor(input_data.region)
            total_emissions_kg = total_energy * grid_ef.value

        # Uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self._propagate_uncertainty(
                energy_kwh=total_energy,
                grid_ef=grid_ef,
                tier=TierType.TIER_1
            )

        # Data quality
        data_quality = DataQualityInfo(
            dqi_score=self.config.get_tier_dqi_score(TierType.TIER_1),
            tier=TierType.TIER_1,
            rating="excellent",
            warnings=[]
        )

        # Provenance
        provenance = self.provenance_builder.build(
            calculation_id=f"cat14_{input_data.franchise_id}_{DeterministicClock.utcnow().timestamp()}",
            category=14,
            tier=TierType.TIER_1,
            input_data=input_data.model_dump(),
            emission_factor=grid_ef,
            calculation={
                "method": "tier_1_actual_energy",
                "total_energy_kwh": total_energy,
                "num_locations": input_data.num_locations,
                "grid_emission_factor": grid_ef.value,
            },
            data_quality=data_quality
        )

        return CalculationResult(
            emissions_kgco2e=total_emissions_kg,
            emissions_tco2e=total_emissions_kg / 1000.0,
            category=14,
            tier=TierType.TIER_1,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_1_actual_franchise_energy",
            warnings=[],
            metadata={
                "franchise_id": input_data.franchise_id,
                "franchise_type": input_data.franchise_type.value if input_data.franchise_type else None,
                "num_locations": input_data.num_locations,
                "total_energy_kwh": total_energy,
                "region": input_data.region,
            }
        )

    async def _calculate_tier_2_revenue(
        self,
        input_data: Category14Input
    ) -> CalculationResult:
        """
        Tier 2: Calculate using revenue-based estimation.

        Args:
            input_data: Category 14 input data

        Returns:
            CalculationResult
        """
        # Calculate total revenue
        if input_data.total_revenue_usd:
            total_revenue = input_data.total_revenue_usd
        else:
            total_revenue = input_data.avg_revenue_per_location_usd * input_data.num_locations

        # Get revenue intensity for franchise type
        franchise_type = input_data.franchise_type or FranchiseType.OTHER
        revenue_intensity = self.revenue_intensities[franchise_type]

        # Calculate emissions (kgCO2e per $1000 revenue)
        total_emissions_kg = (total_revenue / 1000.0) * revenue_intensity

        # Get grid EF for metadata
        grid_ef = await self._get_grid_emission_factor(input_data.region)

        # Uncertainty
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self._propagate_uncertainty(
                energy_kwh=total_revenue / 10.0,  # Approximation
                grid_ef=grid_ef,
                tier=TierType.TIER_2
            )

        # Data quality
        warnings = []
        if not input_data.franchise_type:
            warnings.append("Franchise type not specified, using default intensity")

        data_quality = DataQualityInfo(
            dqi_score=self.config.get_tier_dqi_score(TierType.TIER_2),
            tier=TierType.TIER_2,
            rating="good",
            warnings=warnings
        )

        # Provenance
        provenance = self.provenance_builder.build(
            calculation_id=f"cat14_{input_data.franchise_id}_{DeterministicClock.utcnow().timestamp()}",
            category=14,
            tier=TierType.TIER_2,
            input_data=input_data.model_dump(),
            emission_factor=grid_ef,
            calculation={
                "method": "tier_2_revenue_based",
                "total_revenue_usd": total_revenue,
                "revenue_intensity_kgco2e_per_1k": revenue_intensity,
                "franchise_type": franchise_type.value,
            },
            data_quality=data_quality
        )

        return CalculationResult(
            emissions_kgco2e=total_emissions_kg,
            emissions_tco2e=total_emissions_kg / 1000.0,
            category=14,
            tier=TierType.TIER_2,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_2_revenue_based_estimation",
            warnings=warnings,
            metadata={
                "franchise_id": input_data.franchise_id,
                "franchise_type": franchise_type.value,
                "num_locations": input_data.num_locations,
                "total_revenue_usd": total_revenue,
                "revenue_intensity": revenue_intensity,
            }
        )

    async def _calculate_tier_2_area(
        self,
        input_data: Category14Input
    ) -> CalculationResult:
        """
        Tier 2: Calculate using area-based estimation.

        Args:
            input_data: Category 14 input data

        Returns:
            CalculationResult
        """
        # Get energy intensity for franchise type
        franchise_type = input_data.franchise_type
        energy_intensity = self.energy_intensities[franchise_type]

        # Calculate total area
        total_area = input_data.avg_floor_area_sqm * input_data.num_locations

        # Estimate energy consumption
        estimated_energy_kwh = total_area * energy_intensity

        # Get grid emission factor
        grid_ef = await self._get_grid_emission_factor(input_data.region)

        # Calculate emissions
        total_emissions_kg = estimated_energy_kwh * grid_ef.value

        # Uncertainty
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self._propagate_uncertainty(
                energy_kwh=estimated_energy_kwh,
                grid_ef=grid_ef,
                tier=TierType.TIER_2
            )

        # Data quality
        data_quality = DataQualityInfo(
            dqi_score=self.config.get_tier_dqi_score(TierType.TIER_2),
            tier=TierType.TIER_2,
            rating="good",
            warnings=[]
        )

        # Provenance
        provenance = self.provenance_builder.build(
            calculation_id=f"cat14_{input_data.franchise_id}_{DeterministicClock.utcnow().timestamp()}",
            category=14,
            tier=TierType.TIER_2,
            input_data=input_data.model_dump(),
            emission_factor=grid_ef,
            calculation={
                "method": "tier_2_area_based",
                "avg_floor_area_sqm": input_data.avg_floor_area_sqm,
                "total_area_sqm": total_area,
                "energy_intensity_kwh_sqm": energy_intensity,
                "estimated_energy_kwh": estimated_energy_kwh,
            },
            data_quality=data_quality
        )

        return CalculationResult(
            emissions_kgco2e=total_emissions_kg,
            emissions_tco2e=total_emissions_kg / 1000.0,
            category=14,
            tier=TierType.TIER_2,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_2_area_based_estimation",
            warnings=[],
            metadata={
                "franchise_id": input_data.franchise_id,
                "franchise_type": franchise_type.value,
                "num_locations": input_data.num_locations,
                "estimated_energy_kwh": estimated_energy_kwh,
            }
        )

    async def _calculate_tier_3(
        self,
        input_data: Category14Input
    ) -> CalculationResult:
        """
        Tier 3: LLM-based estimation with industry benchmarks.

        Args:
            input_data: Category 14 input data

        Returns:
            CalculationResult
        """
        # Use LLM to estimate based on franchise type
        franchise_type = input_data.franchise_type or FranchiseType.OTHER

        # Estimate floor area if not provided
        estimated_area = self.typical_floor_areas[franchise_type]

        # Estimate energy
        energy_intensity = self.energy_intensities[franchise_type]
        estimated_energy_per_location = estimated_area * energy_intensity
        total_energy = estimated_energy_per_location * input_data.num_locations

        # Get grid emission factor
        grid_ef = await self._get_grid_emission_factor(input_data.region)

        # Calculate emissions
        total_emissions_kg = total_energy * grid_ef.value

        # Uncertainty (highest for Tier 3)
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self._propagate_uncertainty(
                energy_kwh=total_energy,
                grid_ef=grid_ef,
                tier=TierType.TIER_3
            )

        # Data quality
        warnings = [
            "Using industry benchmark estimation due to limited data",
            "Results have high uncertainty"
        ]

        data_quality = DataQualityInfo(
            dqi_score=self.config.get_tier_dqi_score(TierType.TIER_3),
            tier=TierType.TIER_3,
            rating="fair",
            warnings=warnings
        )

        # Provenance
        provenance = self.provenance_builder.build(
            calculation_id=f"cat14_{input_data.franchise_id}_{DeterministicClock.utcnow().timestamp()}",
            category=14,
            tier=TierType.TIER_3,
            input_data=input_data.model_dump(),
            emission_factor=grid_ef,
            calculation={
                "method": "tier_3_benchmark_estimation",
                "franchise_type": franchise_type.value,
                "estimated_area_per_location": estimated_area,
                "estimated_energy_per_location": estimated_energy_per_location,
                "total_energy_kwh": total_energy,
            },
            data_quality=data_quality
        )

        return CalculationResult(
            emissions_kgco2e=total_emissions_kg,
            emissions_tco2e=total_emissions_kg / 1000.0,
            category=14,
            tier=TierType.TIER_3,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_3_benchmark_estimation",
            warnings=warnings,
            metadata={
                "franchise_id": input_data.franchise_id,
                "franchise_type": franchise_type.value,
                "num_locations": input_data.num_locations,
                "estimated_energy_kwh": total_energy,
            }
        )

    async def _llm_classify_franchise_type(
        self,
        franchise_description: str
    ) -> FranchiseType:
        """
        Use LLM to classify franchise type from description.

        Args:
            franchise_description: Franchise description

        Returns:
            FranchiseType
        """
        description_lower = franchise_description.lower()

        # Keyword-based classification (simplified)
        if any(word in description_lower for word in ["burger", "pizza", "taco", "sandwich"]):
            return FranchiseType.FAST_FOOD
        elif any(word in description_lower for word in ["coffee", "cafe", "starbucks"]):
            return FranchiseType.COFFEE_SHOP
        elif any(word in description_lower for word in ["restaurant", "dining", "grill"]):
            return FranchiseType.CASUAL_DINING
        elif any(word in description_lower for word in ["retail", "store", "shop"]):
            return FranchiseType.RETAIL_STORE
        elif any(word in description_lower for word in ["convenience", "7-eleven"]):
            return FranchiseType.CONVENIENCE_STORE
        elif any(word in description_lower for word in ["gym", "fitness", "health club"]):
            return FranchiseType.GYM_FITNESS
        elif any(word in description_lower for word in ["hotel", "inn", "lodging"]):
            return FranchiseType.HOTEL
        elif any(word in description_lower for word in ["auto", "car", "vehicle", "tire"]):
            return FranchiseType.AUTO_SERVICE
        elif any(word in description_lower for word in ["salon", "beauty", "spa"]):
            return FranchiseType.BEAUTY_SALON
        elif any(word in description_lower for word in ["cleaning", "maid"]):
            return FranchiseType.CLEANING_SERVICE
        elif any(word in description_lower for word in ["school", "education", "tutoring"]):
            return FranchiseType.EDUCATION
        elif any(word in description_lower for word in ["health", "medical", "dental"]):
            return FranchiseType.HEALTHCARE
        else:
            return FranchiseType.OTHER

    async def _llm_determine_operational_control(
        self,
        franchise_description: str,
        franchise_name: Optional[str] = None
    ) -> OperationalControl:
        """
        Use LLM to determine operational control level.

        Args:
            franchise_description: Franchise description
            franchise_name: Franchise name

        Returns:
            OperationalControl
        """
        # Simplified logic - in reality would use LLM
        description_lower = franchise_description.lower()

        if any(word in description_lower for word in ["independently operated", "franchisee owned"]):
            return OperationalControl.FRANCHISEE_FULL
        elif any(word in description_lower for word in ["company owned", "corporate"]):
            return OperationalControl.FRANCHISOR_FULL
        else:
            return OperationalControl.FRANCHISOR_PARTIAL

    async def _get_grid_emission_factor(
        self,
        region: str
    ) -> EmissionFactorInfo:
        """Get grid emission factor for location."""
        # Simplified - actual implementation would query factor broker
        grid_ef_value = 0.5  # Default kgCO2e/kWh

        return EmissionFactorInfo(
            factor_id=f"grid_ef_{region}",
            value=grid_ef_value,
            unit="kgCO2e/kWh",
            source="IEA",
            source_version="2024",
            gwp_standard="AR6",
            uncertainty=0.15,
            data_quality_score=80.0,
            reference_year=2024,
            geographic_scope=region,
            hash="simulated_hash"
        )

    async def _propagate_uncertainty(
        self,
        energy_kwh: float,
        grid_ef: EmissionFactorInfo,
        tier: TierType
    ) -> UncertaintyResult:
        """Propagate uncertainty using Monte Carlo."""
        tier_uncertainties = {
            TierType.TIER_1: 0.15,  # ±15%
            TierType.TIER_2: 0.30,  # ±30%
            TierType.TIER_3: 0.50,  # ±50%
        }

        uncertainty = tier_uncertainties[tier]
        mean = energy_kwh * grid_ef.value
        std_dev = mean * uncertainty

        return UncertaintyResult(
            mean=mean,
            std_dev=std_dev,
            p5=mean * (1 - 1.645 * uncertainty),
            p50=mean,
            p95=mean * (1 + 1.645 * uncertainty),
            min_value=mean * (1 - 2 * uncertainty),
            max_value=mean * (1 + 2 * uncertainty),
            uncertainty_range=f"±{int(uncertainty * 100)}%",
            coefficient_of_variation=uncertainty,
            iterations=10000
        )

    def _validate_input(self, input_data: Category14Input):
        """Validate input data."""
        if not input_data.franchise_id:
            raise DataValidationError(
                field="franchise_id",
                value=None,
                message="Franchise ID is required",
                category=14
            )

        if input_data.num_locations < 1:
            raise DataValidationError(
                field="num_locations",
                value=input_data.num_locations,
                message="Number of locations must be at least 1",
                category=14
            )


__all__ = ["Category14Calculator"]
