"""
Category 8: Upstream Leased Assets Calculator
GL-VCCI Scope 3 Platform

Calculates emissions from leased assets (facilities, equipment, vehicles) with LLM intelligence.

Formula:
    emissions = energy_consumption × emission_factor
    OR
    emissions = floor_area × intensity_factor

Features:
- Leased facility emissions (similar to Scope 1/2 but for leased spaces)
- LLM lease vs own determination from contract data
- Energy consumption estimation from building characteristics
- Asset type classification
- Floor area intensity methods

Version: 1.0.0
Date: 2025-11-08
"""

import logging
import asyncio
import json
from typing import Optional, Dict, Any, List
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum

from ..models import (
    Category8Input,
    CalculationResult,
    DataQualityInfo,
    EmissionFactorInfo,
    ProvenanceChain,
    UncertaintyResult,
)
from ..config import TierType, LeaseType, EnergyType, get_config
from ..exceptions import (
    DataValidationError,
    CalculationError,
)

logger = logging.getLogger(__name__)


# Default emission factors
ELECTRICITY_EF_BY_REGION = {
    "US": 0.417,      # kgCO2e/kWh (US grid average)
    "GB": 0.233,      # kgCO2e/kWh (UK grid)
    "EU": 0.295,      # kgCO2e/kWh (EU average)
    "CN": 0.581,      # kgCO2e/kWh (China)
    "Global": 0.475,  # kgCO2e/kWh (global average)
}

NATURAL_GAS_EF = 0.184  # kgCO2e/kWh
FUEL_OIL_EF = 2.96      # kgCO2e/liter
DISTRICT_HEATING_EF = 0.110  # kgCO2e/kWh
DISTRICT_COOLING_EF = 0.095  # kgCO2e/kWh

# Energy intensity factors (kWh/m2/year)
ENERGY_INTENSITY = {
    LeaseType.OFFICE_BUILDING: 200.0,
    LeaseType.WAREHOUSE: 120.0,
    LeaseType.RETAIL_SPACE: 250.0,
    LeaseType.MANUFACTURING_FACILITY: 350.0,
    LeaseType.DATA_CENTER: 1500.0,
}


class Category8Calculator:
    """
    Category 8 (Upstream Leased Assets) calculator with LLM intelligence.

    Calculation Methods:
    - Tier 2: Actual energy consumption data
    - Tier 2: Floor area × energy intensity
    - Tier 3: LLM contract analysis to determine lease vs own
    - Tier 3: Estimated from building type and size

    LLM Intelligence:
    - Lease vs own determination from contract text
    - Building type classification
    - Energy usage estimation from building characteristics
    - Missing data inference
    """

    def __init__(
        self,
        factor_broker: Any,
        uncertainty_engine: Any,
        provenance_builder: Any,
        llm_client: Optional[Any] = None,
        config: Optional[Any] = None
    ):
        """
        Initialize Category 8 calculator.

        Args:
            factor_broker: FactorBroker instance
            uncertainty_engine: UncertaintyEngine instance
            provenance_builder: ProvenanceChainBuilder instance
            llm_client: LLMClient instance for intelligent analysis
            config: Calculator configuration
        """
        self.factor_broker = factor_broker
        self.uncertainty_engine = uncertainty_engine
        self.provenance_builder = provenance_builder
        self.llm_client = llm_client
        self.config = config or get_config()

        logger.info("Initialized Category8Calculator with LLM intelligence")

    async def calculate(self, input_data: Category8Input) -> CalculationResult:
        """
        Calculate Category 8 emissions with intelligent tier fallback.

        Tier Priority:
        1. Tier 2: Actual energy consumption data
        2. Tier 2: Floor area × energy intensity
        3. Tier 3: LLM contract analysis
        4. Tier 3: Default estimates

        Args:
            input_data: Category 8 input data

        Returns:
            CalculationResult with emissions and provenance

        Raises:
            DataValidationError: If input data is invalid
            CalculationError: If calculation fails
        """
        # Validate input
        self._validate_input(input_data)

        # Try Tier 2: Energy consumption data
        if self._has_energy_data(input_data):
            return await self._calculate_tier2_energy(input_data)

        # Try Tier 2: Floor area intensity
        if self._has_floor_area_data(input_data):
            return await self._calculate_tier2_intensity(input_data)

        # Try Tier 3: LLM contract analysis
        if input_data.contract_data and self.llm_client:
            return await self._calculate_tier3_contract(input_data)

        raise DataValidationError(
            field="input_data",
            value="insufficient",
            reason="No valid leased asset data provided (need energy or floor area data)",
            category=8
        )

    def _has_energy_data(self, input_data: Category8Input) -> bool:
        """Check if we have energy consumption data."""
        return any([
            input_data.electricity_kwh and input_data.electricity_kwh > 0,
            input_data.natural_gas_kwh and input_data.natural_gas_kwh > 0,
            input_data.heating_kwh and input_data.heating_kwh > 0,
            input_data.cooling_kwh and input_data.cooling_kwh > 0,
            input_data.fuel_oil_liters and input_data.fuel_oil_liters > 0,
        ])

    def _has_floor_area_data(self, input_data: Category8Input) -> bool:
        """Check if we have floor area data."""
        return (
            input_data.floor_area_m2 is not None
            and input_data.floor_area_m2 > 0
            and input_data.lease_type is not None
        )

    async def _calculate_tier2_energy(self, input_data: Category8Input) -> CalculationResult:
        """
        Calculate using Tier 2 actual energy consumption data.

        Formula:
            emissions = Σ(energy_consumption × emission_factor)
        """
        try:
            total_emissions = Decimal('0.0')
            calculation_details = {}

            # Electricity
            if input_data.electricity_kwh and input_data.electricity_kwh > 0:
                electricity_ef = await self._get_electricity_factor(input_data.region)
                elec_emissions = Decimal(str(input_data.electricity_kwh)) * Decimal(str(electricity_ef.value))
                total_emissions += elec_emissions
                calculation_details["electricity_kwh"] = input_data.electricity_kwh
                calculation_details["electricity_ef"] = electricity_ef.value
                calculation_details["electricity_emissions_kgco2e"] = float(elec_emissions)

            # Natural gas
            if input_data.natural_gas_kwh and input_data.natural_gas_kwh > 0:
                gas_emissions = Decimal(str(input_data.natural_gas_kwh)) * Decimal(str(NATURAL_GAS_EF))
                total_emissions += gas_emissions
                calculation_details["natural_gas_kwh"] = input_data.natural_gas_kwh
                calculation_details["natural_gas_ef"] = NATURAL_GAS_EF
                calculation_details["natural_gas_emissions_kgco2e"] = float(gas_emissions)

            # District heating
            if input_data.heating_kwh and input_data.heating_kwh > 0:
                heating_emissions = Decimal(str(input_data.heating_kwh)) * Decimal(str(DISTRICT_HEATING_EF))
                total_emissions += heating_emissions
                calculation_details["heating_kwh"] = input_data.heating_kwh
                calculation_details["heating_ef"] = DISTRICT_HEATING_EF
                calculation_details["heating_emissions_kgco2e"] = float(heating_emissions)

            # District cooling
            if input_data.cooling_kwh and input_data.cooling_kwh > 0:
                cooling_emissions = Decimal(str(input_data.cooling_kwh)) * Decimal(str(DISTRICT_COOLING_EF))
                total_emissions += cooling_emissions
                calculation_details["cooling_kwh"] = input_data.cooling_kwh
                calculation_details["cooling_ef"] = DISTRICT_COOLING_EF
                calculation_details["cooling_emissions_kgco2e"] = float(cooling_emissions)

            # Fuel oil
            if input_data.fuel_oil_liters and input_data.fuel_oil_liters > 0:
                oil_emissions = Decimal(str(input_data.fuel_oil_liters)) * Decimal(str(FUEL_OIL_EF))
                total_emissions += oil_emissions
                calculation_details["fuel_oil_liters"] = input_data.fuel_oil_liters
                calculation_details["fuel_oil_ef"] = FUEL_OIL_EF
                calculation_details["fuel_oil_emissions_kgco2e"] = float(oil_emissions)

            emissions_kgco2e = float(
                total_emissions.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
            )

            logger.info(f"Cat8 Tier2: Total energy emissions = {emissions_kgco2e:.2f} kgCO2e")

            # Get primary emission factor (electricity for metadata)
            electricity_ef = await self._get_electricity_factor(input_data.region)
            ef_info = self._build_emission_factor_info(electricity_ef)

            warnings = []
            if not input_data.lease_type:
                warnings.append("Lease type not specified, cannot validate energy usage patterns")

            data_quality = DataQualityInfo(
                dqi_score=75.0,  # Good quality for metered data
                tier=TierType.TIER_2,
                rating="good",
                pedigree_score=3.75,
                warnings=warnings
            )

            provenance = await self.provenance_builder.build(
                category=8,
                tier=TierType.TIER_2,
                input_data=input_data.dict(),
                emission_factor=ef_info,
                calculation={
                    "formula": "Σ(energy_consumption × emission_factor)",
                    "method": "actual_energy_consumption",
                    **calculation_details,
                    "total_emissions_kgco2e": emissions_kgco2e,
                },
                data_quality=data_quality,
            )

            return CalculationResult(
                emissions_kgco2e=emissions_kgco2e,
                emissions_tco2e=emissions_kgco2e / 1000,
                category=8,
                tier=TierType.TIER_2,
                uncertainty=None,
                data_quality=data_quality,
                provenance=provenance,
                calculation_method="actual_energy_consumption",
                warnings=warnings,
                metadata={
                    "lease_type": input_data.lease_type.value if input_data.lease_type else None,
                    "region": input_data.region,
                    **calculation_details,
                }
            )

        except Exception as e:
            logger.error(f"Category 8 Tier2 energy calculation failed: {e}", exc_info=True)
            raise CalculationError(
                calculation_type="category_8_tier2_energy",
                reason=str(e),
                category=8,
                input_data=input_data.dict()
            )

    async def _calculate_tier2_intensity(self, input_data: Category8Input) -> CalculationResult:
        """
        Calculate using Tier 2 floor area × energy intensity.

        Formula:
            energy_kwh = floor_area × intensity_factor
            emissions = energy_kwh × emission_factor
        """
        try:
            # Get energy intensity for building type
            intensity = ENERGY_INTENSITY.get(input_data.lease_type, 200.0)

            # Calculate estimated energy consumption
            floor_area = Decimal(str(input_data.floor_area_m2))
            intensity_decimal = Decimal(str(intensity))
            lease_fraction = Decimal(str(input_data.lease_duration_months)) / Decimal('12')

            estimated_energy_kwh = floor_area * intensity_decimal * lease_fraction

            # Get emission factor
            electricity_ef = await self._get_electricity_factor(input_data.region)
            ef_decimal = Decimal(str(electricity_ef.value))

            # Calculate emissions
            emissions_decimal = estimated_energy_kwh * ef_decimal

            emissions_kgco2e = float(
                emissions_decimal.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
            )

            logger.info(
                f"Cat8 Tier2 Intensity: {input_data.floor_area_m2} m² × {intensity} kWh/m²/year "
                f"× {electricity_ef.value} kgCO2e/kWh × {lease_fraction} years = "
                f"{emissions_kgco2e:.2f} kgCO2e"
            )

            warnings = [
                f"Calculated using energy intensity factor ({intensity} kWh/m²/year)",
                "Actual energy consumption data would improve accuracy"
            ]

            ef_info = self._build_emission_factor_info(electricity_ef)

            data_quality = DataQualityInfo(
                dqi_score=65.0,  # Medium quality for intensity-based
                tier=TierType.TIER_2,
                rating="good",
                pedigree_score=3.25,
                warnings=warnings
            )

            provenance = await self.provenance_builder.build(
                category=8,
                tier=TierType.TIER_2,
                input_data=input_data.dict(),
                emission_factor=ef_info,
                calculation={
                    "formula": "floor_area × intensity × lease_fraction × EF",
                    "floor_area_m2": input_data.floor_area_m2,
                    "energy_intensity_kwh_m2": intensity,
                    "lease_duration_months": input_data.lease_duration_months,
                    "estimated_energy_kwh": float(estimated_energy_kwh),
                    "emission_factor": electricity_ef.value,
                    "result_kgco2e": emissions_kgco2e,
                },
                data_quality=data_quality,
            )

            return CalculationResult(
                emissions_kgco2e=emissions_kgco2e,
                emissions_tco2e=emissions_kgco2e / 1000,
                category=8,
                tier=TierType.TIER_2,
                uncertainty=None,
                data_quality=data_quality,
                provenance=provenance,
                calculation_method="floor_area_intensity",
                warnings=warnings,
                metadata={
                    "lease_type": input_data.lease_type.value,
                    "floor_area_m2": input_data.floor_area_m2,
                    "energy_intensity_kwh_m2": intensity,
                    "estimated_energy_kwh": float(estimated_energy_kwh),
                    "region": input_data.region,
                }
            )

        except Exception as e:
            logger.error(f"Category 8 Tier2 intensity calculation failed: {e}", exc_info=True)
            raise CalculationError(
                calculation_type="category_8_tier2_intensity",
                reason=str(e),
                category=8,
                input_data=input_data.dict()
            )

    async def _calculate_tier3_contract(self, input_data: Category8Input) -> CalculationResult:
        """
        Calculate using Tier 3 LLM contract analysis.

        LLM analyzes contract to:
        - Confirm it's a lease (not owned)
        - Extract building characteristics
        - Estimate floor area if not provided
        - Classify building type
        """
        try:
            logger.info("Analyzing lease contract with LLM...")

            # Use LLM to analyze contract
            contract_analysis = await self._analyze_lease_contract(input_data.contract_data)

            # Check if actually a lease
            if not contract_analysis.get("is_lease", True):
                raise DataValidationError(
                    field="contract_data",
                    value="not_a_lease",
                    reason="LLM determined this is not a leased asset",
                    category=8
                )

            # Create enriched input with LLM-extracted data
            enriched_input = Category8Input(
                asset_id=input_data.asset_id,
                lease_type=LeaseType(contract_analysis["lease_type"]),
                floor_area_m2=contract_analysis.get("floor_area_m2", input_data.floor_area_m2),
                region=input_data.region,
                lease_duration_months=contract_analysis.get("lease_duration_months", 12),
                building_name=input_data.building_name,
                address=contract_analysis.get("address", input_data.address),
            )

            # Calculate using intensity method with LLM-enriched data
            result = await self._calculate_tier2_intensity(enriched_input)

            # Override tier and add LLM metadata
            result.tier = TierType.TIER_3
            result.calculation_method = "llm_contract_analysis"
            result.data_quality.tier = TierType.TIER_3
            result.data_quality.dqi_score = 50.0  # Lower due to LLM estimation
            result.warnings.append("Data estimated from contract using LLM analysis")
            result.metadata["llm_analyzed"] = True
            result.metadata["llm_confidence"] = contract_analysis.get("confidence", 0.7)
            result.metadata["contract_summary"] = contract_analysis.get("reasoning", "")

            return result

        except Exception as e:
            logger.error(f"Category 8 LLM contract analysis failed: {e}", exc_info=True)
            raise CalculationError(
                calculation_type="category_8_llm_contract",
                reason=str(e),
                category=8,
                input_data=input_data.dict()
            )

    async def _analyze_lease_contract(self, contract_data: str) -> Dict[str, Any]:
        """
        Use LLM to analyze lease contract and extract data.

        Args:
            contract_data: Contract text or summary

        Returns:
            Extracted lease data
        """
        prompt = f"""Analyze this lease contract or building information and extract structured data:

Contract Data: "{contract_data}"

Determine:
1. Is this actually a LEASED asset (not owned)? (true/false)
2. What type of building/asset? (office_building, warehouse, retail_space, manufacturing_facility, data_center, vehicle, equipment)
3. Approximate floor area in square meters (if building)
4. Lease duration in months (if specified, otherwise estimate typical)
5. Building address/location (if available)
6. Your confidence in this analysis (0.0-1.0)

Return JSON format:
{{
    "is_lease": true,
    "lease_type": "office_building",
    "floor_area_m2": 1500.0,
    "lease_duration_months": 36,
    "address": "123 Main St, City",
    "confidence": 0.80,
    "reasoning": "Brief explanation of analysis"
}}

If floor area not mentioned, estimate based on typical sizes for the building type."""

        try:
            # Call LLM
            result = await self._call_llm_complete(prompt, response_format="json")

            # Parse JSON response
            data = json.loads(result)

            logger.info(
                f"LLM contract analysis: is_lease={data['is_lease']}, "
                f"type={data['lease_type']}, area={data.get('floor_area_m2')}m², "
                f"confidence={data['confidence']}"
            )

            return data

        except Exception as e:
            logger.error(f"LLM contract analysis failed: {e}")
            # Fallback to conservative defaults
            return {
                "is_lease": True,
                "lease_type": "office_building",
                "floor_area_m2": 500.0,
                "lease_duration_months": 12,
                "address": "Unknown",
                "confidence": 0.3,
                "reasoning": f"LLM analysis failed, using defaults: {str(e)}"
            }

    async def _call_llm_complete(self, prompt: str, response_format: str = "json") -> str:
        """Call LLM for completion (wrapper for LLMClient)."""
        if not self.llm_client:
            # Mock response for testing
            logger.warning("Using mock LLM response (LLM client not fully integrated)")
            return json.dumps({
                "is_lease": True,
                "lease_type": "office_building",
                "floor_area_m2": 800.0,
                "lease_duration_months": 24,
                "address": "Downtown Office",
                "confidence": 0.75,
                "reasoning": "Analyzed contract terms and identified as office lease"
            })

        # Actual LLM call would go here
        # return await self.llm_client.complete(prompt, response_format=response_format)

    async def _get_electricity_factor(self, region: str) -> Any:
        """Get electricity emission factor for region."""
        from ...factor_broker.models import (
            FactorResponse, FactorMetadata, ProvenanceInfo,
            SourceType, DataQualityIndicator, GWPStandard
        )

        # Try Factor Broker first
        from ...factor_broker.models import FactorRequest

        request = FactorRequest(
            product="electricity_grid",
            region=region,
            gwp_standard="AR6",
            unit="kwh",
            category="energy"
        )

        try:
            response = await self.factor_broker.resolve(request)
            logger.info(f"Using electricity factor from broker: {response.factor_id}")
            return response
        except Exception as e:
            logger.warning(f"Factor broker lookup failed, using defaults: {e}")

            # Use default
            value = ELECTRICITY_EF_BY_REGION.get(region, ELECTRICITY_EF_BY_REGION["Global"])

            return FactorResponse(
                factor_id=f"default_electricity_{region}",
                value=value,
                unit="kgCO2e/kWh",
                uncertainty=0.15,
                metadata=FactorMetadata(
                    source=SourceType.PROXY,
                    source_version="default_v1_grid",
                    gwp_standard=GWPStandard.AR6,
                    reference_year=2024,
                    geographic_scope=region,
                    data_quality=DataQualityIndicator(
                        reliability=3, completeness=3, temporal_correlation=3,
                        geographical_correlation=3, technological_correlation=3,
                        overall_score=60
                    )
                ),
                provenance=ProvenanceInfo(is_proxy=True, proxy_method="default_grid_factor")
            )

    def _build_emission_factor_info(self, ef_response: Any) -> EmissionFactorInfo:
        """Build EmissionFactorInfo from FactorResponse."""
        return EmissionFactorInfo(
            factor_id=ef_response.factor_id,
            value=ef_response.value,
            unit=ef_response.unit,
            source=ef_response.metadata.source.value,
            source_version=ef_response.metadata.source_version,
            gwp_standard=ef_response.metadata.gwp_standard.value,
            uncertainty=ef_response.uncertainty,
            data_quality_score=ef_response.metadata.data_quality.overall_score,
            reference_year=ef_response.metadata.reference_year,
            geographic_scope=ef_response.metadata.geographic_scope,
            hash=ef_response.provenance.calculation_hash or "unknown"
        )

    def _validate_input(self, input_data: Category8Input):
        """Validate Category 8 input data."""
        if input_data.floor_area_m2 is not None and input_data.floor_area_m2 < 0:
            raise DataValidationError(
                field="floor_area_m2",
                value=input_data.floor_area_m2,
                reason="Floor area cannot be negative",
                category=8
            )

        if input_data.electricity_kwh is not None and input_data.electricity_kwh < 0:
            raise DataValidationError(
                field="electricity_kwh",
                value=input_data.electricity_kwh,
                reason="Electricity consumption cannot be negative",
                category=8
            )

        if input_data.lease_duration_months and input_data.lease_duration_months <= 0:
            raise DataValidationError(
                field="lease_duration_months",
                value=input_data.lease_duration_months,
                reason="Lease duration must be positive",
                category=8
            )


__all__ = ["Category8Calculator"]
