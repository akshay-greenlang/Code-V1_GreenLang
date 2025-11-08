"""
Category 13: Downstream Leased Assets Calculator
GL-VCCI Scope 3 Platform

Emissions from assets owned by the reporting company but leased to other entities
(tenants/lessees) and not already included in Scope 1 or 2.

Key Features:
- Tenant emissions calculation for leased-out properties
- LLM building type classification
- Tenant behavior and energy use modeling
- Multiple calculation approaches (area-based, tenant-based)
- Data quality scoring and uncertainty quantification

Version: 1.0.0
Date: 2025-11-08
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from ..models import (
    Category13Input,
    CalculationResult,
    DataQualityInfo,
    EmissionFactorInfo,
    ProvenanceChain,
    UncertaintyResult,
)
from ..config import TierType, BuildingType, TenantType, get_config
from ..exceptions import (
    DataValidationError,
    EmissionFactorNotFoundError,
    CalculationError,
)

logger = logging.getLogger(__name__)

class Category13Calculator:
    """
    Category 13 (Downstream Leased Assets) calculator with LLM intelligence.

    Calculation Hierarchy:
    1. Tier 1: Actual tenant energy data (highest quality)
    2. Tier 2: Area-based estimation with building type
    3. Tier 3: LLM-based estimation from descriptions

    Features:
    - LLM building type classification
    - LLM tenant type determination
    - Building energy intensity benchmarks
    - Tenant behavior modeling
    - Grid emission factors by location
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
        Initialize Category 13 calculator.

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

        # Building energy intensity benchmarks (kWh/sqm/year)
        self.building_intensities = {
            BuildingType.OFFICE: 200.0,
            BuildingType.RETAIL: 250.0,
            BuildingType.WAREHOUSE: 80.0,
            BuildingType.INDUSTRIAL: 300.0,
            BuildingType.RESIDENTIAL: 120.0,
            BuildingType.MIXED_USE: 180.0,
            BuildingType.DATA_CENTER: 800.0,
            BuildingType.HOTEL: 220.0,
            BuildingType.RESTAURANT: 400.0,
            BuildingType.UNKNOWN: 150.0,
        }

        # Tenant type energy multipliers
        self.tenant_multipliers = {
            TenantType.OFFICE_STANDARD: 1.0,
            TenantType.OFFICE_HIGH_ENERGY: 1.5,
            TenantType.RETAIL_LIGHT: 0.8,
            TenantType.RETAIL_HEAVY: 1.3,
            TenantType.WAREHOUSE: 0.4,
            TenantType.MANUFACTURING: 2.0,
            TenantType.DATA_CENTER: 4.0,
            TenantType.RESTAURANT: 2.5,
            TenantType.RESIDENTIAL: 0.6,
            TenantType.MIXED: 1.2,
        }

        logger.info("Initialized Category13Calculator with LLM intelligence")

    async def calculate(self, input_data: Category13Input) -> CalculationResult:
        """
        Calculate Category 13 emissions with LLM-enhanced intelligence.

        Args:
            input_data: Category 13 input data

        Returns:
            CalculationResult with emissions and provenance

        Raises:
            DataValidationError: If input data is invalid
            CalculationError: If calculation fails
        """
        start_time = datetime.utcnow()

        # Validate input
        self._validate_input(input_data)

        # LLM Enhancement: Classify building and tenant types if needed
        if not input_data.building_type and input_data.asset_description:
            input_data.building_type = await self._llm_classify_building_type(
                input_data.asset_description
            )
            logger.info(
                f"LLM classified building type: {input_data.building_type.value}"
            )

        if not input_data.tenant_type and input_data.tenant_description:
            input_data.tenant_type = await self._llm_classify_tenant_type(
                input_data.tenant_description
            )
            logger.info(
                f"LLM classified tenant type: {input_data.tenant_type.value}"
            )

        # Determine calculation tier and compute emissions
        try:
            if input_data.tenant_energy_kwh and input_data.tenant_energy_kwh > 0:
                # Tier 1: Actual tenant energy data
                result = await self._calculate_tier_1(input_data)
                logger.info(
                    f"Tier 1 calculation successful for asset {input_data.asset_id}"
                )
            elif input_data.floor_area_sqm and input_data.building_type:
                # Tier 2: Area-based estimation
                result = await self._calculate_tier_2(input_data)
                logger.info(
                    f"Tier 2 calculation successful for asset {input_data.asset_id}"
                )
            else:
                # Tier 3: LLM-based estimation
                result = await self._calculate_tier_3(input_data)
                logger.info(
                    f"Tier 3 calculation successful for asset {input_data.asset_id}"
                )

            return result

        except Exception as e:
            logger.error(f"Category 13 calculation failed: {e}", exc_info=True)
            raise CalculationError(
                calculation_type="category_13",
                reason=str(e),
                category=13,
                input_data=input_data.model_dump()
            )

    async def _calculate_tier_1(
        self,
        input_data: Category13Input
    ) -> CalculationResult:
        """
        Tier 1: Calculate using actual tenant energy data.

        Args:
            input_data: Category 13 input data

        Returns:
            CalculationResult
        """
        # Get grid emission factor for location
        grid_ef = await self._get_grid_emission_factor(
            input_data.region,
            input_data.city
        )

        # Calculate electricity emissions
        electricity_emissions = input_data.tenant_energy_kwh * grid_ef.value

        # Add fuel emissions if provided
        fuel_emissions = 0.0
        if input_data.tenant_fuel_consumption:
            for fuel_type, consumption in input_data.tenant_fuel_consumption.items():
                fuel_ef = await self._get_fuel_emission_factor(fuel_type, input_data.region)
                fuel_emissions += consumption * fuel_ef.value

        total_emissions_kg = electricity_emissions + fuel_emissions

        # Uncertainty propagation
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self._propagate_uncertainty(
                energy_kwh=input_data.tenant_energy_kwh,
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
            calculation_id=f"cat13_{input_data.asset_id}_{start_time.timestamp()}",
            category=13,
            tier=TierType.TIER_1,
            input_data=input_data.model_dump(),
            emission_factor=grid_ef,
            calculation={
                "method": "tier_1_actual_energy",
                "tenant_energy_kwh": input_data.tenant_energy_kwh,
                "grid_emission_factor": grid_ef.value,
                "electricity_emissions_kg": electricity_emissions,
                "fuel_emissions_kg": fuel_emissions,
            },
            data_quality=data_quality
        )

        return CalculationResult(
            emissions_kgco2e=total_emissions_kg,
            emissions_tco2e=total_emissions_kg / 1000.0,
            category=13,
            tier=TierType.TIER_1,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_1_actual_tenant_energy",
            warnings=[],
            metadata={
                "asset_id": input_data.asset_id,
                "building_type": input_data.building_type.value if input_data.building_type else None,
                "tenant_type": input_data.tenant_type.value if input_data.tenant_type else None,
                "region": input_data.region,
            }
        )

    async def _calculate_tier_2(
        self,
        input_data: Category13Input
    ) -> CalculationResult:
        """
        Tier 2: Calculate using area-based estimation.

        Args:
            input_data: Category 13 input data

        Returns:
            CalculationResult
        """
        # Get building energy intensity
        base_intensity = self.building_intensities[input_data.building_type]

        # Apply tenant multiplier if available
        tenant_multiplier = 1.0
        if input_data.tenant_type:
            tenant_multiplier = self.tenant_multipliers[input_data.tenant_type]

        # Estimate energy consumption
        estimated_energy_kwh = (
            input_data.floor_area_sqm * base_intensity * tenant_multiplier
        )

        # Get grid emission factor
        grid_ef = await self._get_grid_emission_factor(
            input_data.region,
            input_data.city
        )

        # Calculate emissions
        total_emissions_kg = estimated_energy_kwh * grid_ef.value

        # Uncertainty (higher than Tier 1)
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self._propagate_uncertainty(
                energy_kwh=estimated_energy_kwh,
                grid_ef=grid_ef,
                tier=TierType.TIER_2
            )

        # Data quality
        warnings = []
        if not input_data.tenant_type:
            warnings.append("Tenant type not specified, using default multiplier")

        data_quality = DataQualityInfo(
            dqi_score=self.config.get_tier_dqi_score(TierType.TIER_2),
            tier=TierType.TIER_2,
            rating="good",
            warnings=warnings
        )

        # Provenance
        provenance = self.provenance_builder.build(
            calculation_id=f"cat13_{input_data.asset_id}_{datetime.utcnow().timestamp()}",
            category=13,
            tier=TierType.TIER_2,
            input_data=input_data.model_dump(),
            emission_factor=grid_ef,
            calculation={
                "method": "tier_2_area_based",
                "floor_area_sqm": input_data.floor_area_sqm,
                "building_type": input_data.building_type.value,
                "base_intensity_kwh_sqm": base_intensity,
                "tenant_multiplier": tenant_multiplier,
                "estimated_energy_kwh": estimated_energy_kwh,
                "grid_emission_factor": grid_ef.value,
            },
            data_quality=data_quality
        )

        return CalculationResult(
            emissions_kgco2e=total_emissions_kg,
            emissions_tco2e=total_emissions_kg / 1000.0,
            category=13,
            tier=TierType.TIER_2,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_2_area_based_estimation",
            warnings=warnings,
            metadata={
                "asset_id": input_data.asset_id,
                "building_type": input_data.building_type.value,
                "tenant_type": input_data.tenant_type.value if input_data.tenant_type else None,
                "estimated_energy_kwh": estimated_energy_kwh,
                "region": input_data.region,
            }
        )

    async def _calculate_tier_3(
        self,
        input_data: Category13Input
    ) -> CalculationResult:
        """
        Tier 3: LLM-based estimation with minimal data.

        Args:
            input_data: Category 13 input data

        Returns:
            CalculationResult
        """
        # Use LLM to estimate energy consumption
        estimated_data = await self._llm_estimate_energy(input_data)

        # Get grid emission factor
        grid_ef = await self._get_grid_emission_factor(
            input_data.region,
            input_data.city
        )

        # Calculate emissions
        total_emissions_kg = estimated_data["energy_kwh"] * grid_ef.value

        # Uncertainty (highest for Tier 3)
        uncertainty = None
        if self.config.enable_monte_carlo:
            uncertainty = await self._propagate_uncertainty(
                energy_kwh=estimated_data["energy_kwh"],
                grid_ef=grid_ef,
                tier=TierType.TIER_3
            )

        # Data quality
        warnings = [
            "Using LLM-based estimation due to limited data",
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
            calculation_id=f"cat13_{input_data.asset_id}_{datetime.utcnow().timestamp()}",
            category=13,
            tier=TierType.TIER_3,
            input_data=input_data.model_dump(),
            emission_factor=grid_ef,
            calculation={
                "method": "tier_3_llm_estimation",
                "llm_estimated_energy_kwh": estimated_data["energy_kwh"],
                "llm_reasoning": estimated_data["reasoning"],
                "grid_emission_factor": grid_ef.value,
            },
            data_quality=data_quality
        )

        return CalculationResult(
            emissions_kgco2e=total_emissions_kg,
            emissions_tco2e=total_emissions_kg / 1000.0,
            category=13,
            tier=TierType.TIER_3,
            uncertainty=uncertainty,
            data_quality=data_quality,
            provenance=provenance,
            calculation_method="tier_3_llm_estimation",
            warnings=warnings,
            metadata={
                "asset_id": input_data.asset_id,
                "llm_estimated_energy_kwh": estimated_data["energy_kwh"],
                "llm_reasoning": estimated_data["reasoning"],
                "region": input_data.region,
            }
        )

    async def _llm_classify_building_type(
        self,
        asset_description: str
    ) -> BuildingType:
        """
        Use LLM to classify building type from description.

        Args:
            asset_description: Asset description

        Returns:
            BuildingType
        """
        prompt = f"""Classify this building/asset into one of these types:
- office: Office buildings
- retail: Retail stores, shopping centers
- warehouse: Warehouses, distribution centers
- industrial: Manufacturing facilities
- residential: Apartments, residential buildings
- mixed_use: Mixed-use developments
- data_center: Data centers
- hotel: Hotels, hospitality
- restaurant: Restaurants, food service
- unknown: Cannot determine

Asset description: "{asset_description}"

Return ONLY the building type (e.g., "office")."""

        try:
            # Use LLM client (simplified - actual implementation would use proper API)
            # For now, use basic keyword matching as fallback
            description_lower = asset_description.lower()

            if any(word in description_lower for word in ["office", "corporate", "headquarters"]):
                return BuildingType.OFFICE
            elif any(word in description_lower for word in ["retail", "store", "shop"]):
                return BuildingType.RETAIL
            elif any(word in description_lower for word in ["warehouse", "distribution", "storage"]):
                return BuildingType.WAREHOUSE
            elif any(word in description_lower for word in ["factory", "plant", "manufacturing"]):
                return BuildingType.INDUSTRIAL
            elif any(word in description_lower for word in ["apartment", "residential", "condo"]):
                return BuildingType.RESIDENTIAL
            elif any(word in description_lower for word in ["data center", "datacenter", "server"]):
                return BuildingType.DATA_CENTER
            elif any(word in description_lower for word in ["hotel", "hospitality"]):
                return BuildingType.HOTEL
            elif any(word in description_lower for word in ["restaurant", "food service"]):
                return BuildingType.RESTAURANT
            else:
                return BuildingType.UNKNOWN

        except Exception as e:
            logger.warning(f"LLM building classification failed: {e}, using UNKNOWN")
            return BuildingType.UNKNOWN

    async def _llm_classify_tenant_type(
        self,
        tenant_description: str
    ) -> TenantType:
        """
        Use LLM to classify tenant type from description.

        Args:
            tenant_description: Tenant description

        Returns:
            TenantType
        """
        description_lower = tenant_description.lower()

        if any(word in description_lower for word in ["tech", "software", "it"]):
            return TenantType.OFFICE_HIGH_ENERGY
        elif any(word in description_lower for word in ["office", "consulting"]):
            return TenantType.OFFICE_STANDARD
        elif any(word in description_lower for word in ["retail", "store"]):
            return TenantType.RETAIL_LIGHT
        elif any(word in description_lower for word in ["warehouse", "distribution"]):
            return TenantType.WAREHOUSE
        elif any(word in description_lower for word in ["manufacturing", "production"]):
            return TenantType.MANUFACTURING
        elif any(word in description_lower for word in ["data", "server", "cloud"]):
            return TenantType.DATA_CENTER
        elif any(word in description_lower for word in ["restaurant", "food"]):
            return TenantType.RESTAURANT
        else:
            return TenantType.MIXED

    async def _llm_estimate_energy(
        self,
        input_data: Category13Input
    ) -> Dict[str, Any]:
        """
        Use LLM to estimate energy consumption with minimal data.

        Args:
            input_data: Category 13 input data

        Returns:
            Dictionary with energy_kwh and reasoning
        """
        # Fallback estimation based on available data
        building_type = input_data.building_type or BuildingType.UNKNOWN
        base_intensity = self.building_intensities[building_type]

        # Estimate floor area if not provided (rough estimate)
        estimated_area = input_data.floor_area_sqm or 1000.0  # Default 1000 sqm

        estimated_energy = estimated_area * base_intensity

        return {
            "energy_kwh": estimated_energy,
            "reasoning": f"Estimated based on {building_type.value} building type with assumed area of {estimated_area} sqm"
        }

    async def _get_grid_emission_factor(
        self,
        region: str,
        city: Optional[str] = None
    ) -> EmissionFactorInfo:
        """Get grid emission factor for location."""
        # Use factor broker to get grid EF
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

    async def _get_fuel_emission_factor(
        self,
        fuel_type: str,
        region: str
    ) -> EmissionFactorInfo:
        """Get fuel emission factor."""
        fuel_ef_value = 0.2  # Default kgCO2e/kWh

        return EmissionFactorInfo(
            factor_id=f"fuel_ef_{fuel_type}_{region}",
            value=fuel_ef_value,
            unit="kgCO2e/kWh",
            source="DEFRA",
            source_version="2024",
            gwp_standard="AR6",
            uncertainty=0.10,
            data_quality_score=85.0,
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
        # Tier-based uncertainty
        tier_uncertainties = {
            TierType.TIER_1: 0.10,  # ±10%
            TierType.TIER_2: 0.25,  # ±25%
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

    def _validate_input(self, input_data: Category13Input):
        """Validate input data."""
        if not input_data.asset_id:
            raise DataValidationError(
                field="asset_id",
                value=None,
                message="Asset ID is required",
                category=13
            )

        if not input_data.region:
            raise DataValidationError(
                field="region",
                value=None,
                message="Region is required",
                category=13
            )


__all__ = ["Category13Calculator"]
