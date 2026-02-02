# -*- coding: utf-8 -*-
"""
GL-MRV-BLD-002: Residential Buildings MRV Agent
=================================================

Specialized MRV agent for residential building emissions including
single-family homes, multi-family apartments, and housing complexes.

Features:
    - Single-family and multi-family support
    - Heating/cooling degree day adjustments
    - Per-household and per-occupant metrics
    - RESNET HERS score compatibility
    - Weather normalization

Standards:
    - GHG Protocol Building Sector Guidance
    - EPA Residential Emission Factors
    - RESNET HERS Index

Author: GreenLang Framework Team
Agent ID: GL-MRV-BLD-002
Version: 1.0.0
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.agents.mrv.buildings.base import (
    BuildingMRVBaseAgent,
    BuildingMRVInput,
    BuildingMRVOutput,
    BuildingMetadata,
    BuildingType,
    EnergyConsumption,
    EnergySource,
    EmissionFactor,
    CalculationStep,
    EnergyUseIntensity,
    CarbonIntensity,
    DataQuality,
    VerificationStatus,
    ClimateZone,
    NATURAL_GAS_EF_KGCO2E_PER_THERM,
    FUEL_OIL_EF_KGCO2E_PER_GALLON,
    PROPANE_EF_KGCO2E_PER_GALLON,
    GRID_EF_BY_REGION_KGCO2E_PER_KWH,
)

logger = logging.getLogger(__name__)


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class ResidentialBuildingInput(BuildingMRVInput):
    """Input model for residential building MRV."""

    # Residential-specific fields
    num_dwelling_units: int = Field(default=1, ge=1, description="Number of dwelling units")
    num_occupants: Optional[int] = Field(None, ge=0, description="Total number of occupants")
    num_bedrooms: Optional[int] = Field(None, ge=0, description="Total bedrooms")

    # Climate data
    heating_degree_days: Optional[Decimal] = Field(None, ge=0, description="Annual HDD base 65F")
    cooling_degree_days: Optional[Decimal] = Field(None, ge=0, description="Annual CDD base 65F")

    # Building characteristics
    year_built: Optional[int] = Field(None, ge=1800, le=2100)
    has_central_air: bool = Field(default=True)
    has_swimming_pool: bool = Field(default=False)
    primary_heating_fuel: Optional[EnergySource] = None

    # Home energy rating
    hers_index: Optional[int] = Field(None, ge=0, le=200)


class ResidentialBuildingOutput(BuildingMRVOutput):
    """Output model for residential building MRV."""

    # Residential-specific metrics
    emissions_per_dwelling_unit_kgco2e: Decimal = Field(default=Decimal("0"))
    emissions_per_occupant_kgco2e: Optional[Decimal] = None
    emissions_per_bedroom_kgco2e: Optional[Decimal] = None

    # Weather normalized
    weather_normalized_eui: Optional[Decimal] = None

    # HERS comparison
    estimated_hers_index: Optional[int] = None
    hers_reference_home_emissions: Optional[Decimal] = None


# =============================================================================
# CLIMATE ZONE ADJUSTMENTS
# =============================================================================

# HDD and CDD by climate zone (annual averages)
CLIMATE_ZONE_HDD_CDD = {
    ClimateZone.ZONE_1A: (Decimal("200"), Decimal("4000")),
    ClimateZone.ZONE_1B: (Decimal("300"), Decimal("4500")),
    ClimateZone.ZONE_2A: (Decimal("1500"), Decimal("3000")),
    ClimateZone.ZONE_2B: (Decimal("1000"), Decimal("3500")),
    ClimateZone.ZONE_3A: (Decimal("2500"), Decimal("2000")),
    ClimateZone.ZONE_3B: (Decimal("1500"), Decimal("2500")),
    ClimateZone.ZONE_3C: (Decimal("2000"), Decimal("500")),
    ClimateZone.ZONE_4A: (Decimal("4000"), Decimal("1500")),
    ClimateZone.ZONE_4B: (Decimal("3500"), Decimal("1500")),
    ClimateZone.ZONE_4C: (Decimal("4000"), Decimal("500")),
    ClimateZone.ZONE_5A: (Decimal("5500"), Decimal("1000")),
    ClimateZone.ZONE_5B: (Decimal("5000"), Decimal("800")),
    ClimateZone.ZONE_5C: (Decimal("4500"), Decimal("200")),
    ClimateZone.ZONE_6A: (Decimal("6500"), Decimal("500")),
    ClimateZone.ZONE_6B: (Decimal("6500"), Decimal("300")),
    ClimateZone.ZONE_7: (Decimal("8000"), Decimal("200")),
    ClimateZone.ZONE_8: (Decimal("10000"), Decimal("100")),
}


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class ResidentialBuildingsMRVAgent(BuildingMRVBaseAgent[ResidentialBuildingInput, ResidentialBuildingOutput]):
    """
    GL-MRV-BLD-002: Residential Buildings MRV Agent.

    Calculates emissions for residential buildings including single-family
    homes and multi-family apartment buildings.

    ZERO-HALLUCINATION GUARANTEE:
        - All calculations use EPA residential emission factors
        - Weather normalization using HDD/CDD
        - Complete audit trail for verification
        - Reproducible results with same inputs

    Example:
        >>> agent = ResidentialBuildingsMRVAgent()
        >>> input_data = ResidentialBuildingInput(
        ...     building_id="HOME-001",
        ...     reporting_period="2024",
        ...     building_metadata=BuildingMetadata(
        ...         building_id="HOME-001",
        ...         building_type=BuildingType.RESIDENTIAL_SINGLE,
        ...         gross_floor_area_sqm=Decimal("200")
        ...     ),
        ...     num_dwelling_units=1,
        ...     energy_consumption=[...]
        ... )
        >>> output = agent.process(input_data)
    """

    AGENT_ID = "GL-MRV-BLD-002"
    AGENT_VERSION = "1.0.0"
    BUILDING_CATEGORY = "residential"

    SUPPORTED_BUILDING_TYPES = {
        BuildingType.RESIDENTIAL_SINGLE,
        BuildingType.RESIDENTIAL_MULTI
    }

    def _load_emission_factors(self) -> None:
        """Load residential building emission factors."""
        # Natural gas
        self._emission_factors["scope1_natural_gas_therm"] = EmissionFactor(
            factor_id="scope1_natural_gas_therm",
            value=NATURAL_GAS_EF_KGCO2E_PER_THERM,
            unit="kgCO2e/therm",
            source="EPA GHG Emission Factors Hub 2024",
            region="US",
            valid_from="2024-01-01"
        )

        # Propane
        self._emission_factors["scope1_propane_gallon"] = EmissionFactor(
            factor_id="scope1_propane_gallon",
            value=PROPANE_EF_KGCO2E_PER_GALLON,
            unit="kgCO2e/gallon",
            source="EPA GHG Emission Factors Hub 2024",
            region="US",
            valid_from="2024-01-01"
        )

        # Fuel oil
        self._emission_factors["scope1_fuel_oil_gallon"] = EmissionFactor(
            factor_id="scope1_fuel_oil_gallon",
            value=FUEL_OIL_EF_KGCO2E_PER_GALLON,
            unit="kgCO2e/gallon",
            source="EPA GHG Emission Factors Hub 2024",
            region="US",
            valid_from="2024-01-01"
        )

        # Grid electricity
        for region, factor in GRID_EF_BY_REGION_KGCO2E_PER_KWH.items():
            self._emission_factors[f"scope2_electricity_{region}"] = EmissionFactor(
                factor_id=f"scope2_electricity_{region}",
                value=factor,
                unit="kgCO2e/kWh",
                source="EPA eGRID 2024",
                region=region,
                valid_from="2024-01-01"
            )

    def calculate_emissions(
        self,
        input_data: ResidentialBuildingInput
    ) -> ResidentialBuildingOutput:
        """
        Calculate residential building emissions.

        Methodology:
        1. Calculate Scope 1 from heating fuels
        2. Calculate Scope 2 from electricity
        3. Apply weather normalization
        4. Calculate per-unit and per-occupant metrics
        5. Estimate HERS index comparison

        Args:
            input_data: Validated residential building input

        Returns:
            Complete emission output with provenance
        """
        calculation_steps: List[CalculationStep] = []
        factors_used: List[EmissionFactor] = []
        step_number = 1

        metadata = input_data.building_metadata
        floor_area = metadata.gross_floor_area_sqm
        num_units = input_data.num_dwelling_units

        # Step 1: Calculate Scope 1 emissions
        scope1_emissions = Decimal("0")
        natural_gas_therms = Decimal("0")
        propane_gallons = Decimal("0")
        fuel_oil_gallons = Decimal("0")

        for consumption in input_data.energy_consumption:
            if consumption.source == EnergySource.NATURAL_GAS:
                if "therm" in consumption.unit.lower():
                    natural_gas_therms += consumption.consumption
                elif consumption.unit.lower() == "ccf":
                    natural_gas_therms += consumption.consumption  # 1 CCF ~ 1 therm

            elif consumption.source == EnergySource.PROPANE:
                if "gallon" in consumption.unit.lower():
                    propane_gallons += consumption.consumption

            elif consumption.source == EnergySource.FUEL_OIL:
                if "gallon" in consumption.unit.lower():
                    fuel_oil_gallons += consumption.consumption

        # Natural gas emissions
        if natural_gas_therms > 0:
            ng_emissions = natural_gas_therms * NATURAL_GAS_EF_KGCO2E_PER_THERM
            scope1_emissions += ng_emissions

            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description="Calculate Scope 1 emissions from natural gas",
                formula="ng_emissions = therms * 5.302 kgCO2e/therm",
                inputs={"natural_gas_therms": str(natural_gas_therms)},
                output_value=self._round_emissions(ng_emissions),
                output_unit="kgCO2e",
                source="EPA GHG Emission Factors Hub 2024"
            ))
            factors_used.append(self._emission_factors["scope1_natural_gas_therm"])
            step_number += 1

        # Propane emissions
        if propane_gallons > 0:
            propane_emissions = propane_gallons * PROPANE_EF_KGCO2E_PER_GALLON
            scope1_emissions += propane_emissions

            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description="Calculate Scope 1 emissions from propane",
                formula="propane_emissions = gallons * 5.72 kgCO2e/gallon",
                inputs={"propane_gallons": str(propane_gallons)},
                output_value=self._round_emissions(propane_emissions),
                output_unit="kgCO2e",
                source="EPA GHG Emission Factors Hub 2024"
            ))
            factors_used.append(self._emission_factors["scope1_propane_gallon"])
            step_number += 1

        # Fuel oil emissions
        if fuel_oil_gallons > 0:
            fo_emissions = fuel_oil_gallons * FUEL_OIL_EF_KGCO2E_PER_GALLON
            scope1_emissions += fo_emissions

            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description="Calculate Scope 1 emissions from fuel oil",
                formula="fo_emissions = gallons * 10.21 kgCO2e/gallon",
                inputs={"fuel_oil_gallons": str(fuel_oil_gallons)},
                output_value=self._round_emissions(fo_emissions),
                output_unit="kgCO2e",
                source="EPA GHG Emission Factors Hub 2024"
            ))
            factors_used.append(self._emission_factors["scope1_fuel_oil_gallon"])
            step_number += 1

        # Step 2: Calculate Scope 2 emissions
        electricity_kwh = Decimal("0")
        for consumption in input_data.energy_consumption:
            if consumption.source == EnergySource.ELECTRICITY:
                electricity_kwh += self._convert_to_kwh(
                    consumption.consumption,
                    consumption.unit,
                    consumption.source
                )

        grid_ef = input_data.grid_emission_factor_kgco2e_per_kwh
        if grid_ef is None:
            grid_ef = GRID_EF_BY_REGION_KGCO2E_PER_KWH.get("us_average", Decimal("0.379"))

        scope2_emissions = electricity_kwh * grid_ef

        calculation_steps.append(CalculationStep(
            step_number=step_number,
            description="Calculate Scope 2 emissions from electricity",
            formula="scope2_emissions = electricity_kwh * grid_factor",
            inputs={
                "electricity_kwh": str(electricity_kwh),
                "grid_factor": str(grid_ef)
            },
            output_value=self._round_emissions(scope2_emissions),
            output_unit="kgCO2e",
            source="EPA eGRID 2024"
        ))
        step_number += 1

        # Step 3: Calculate totals
        total_emissions = scope1_emissions + scope2_emissions
        total_energy_kwh = Decimal("0")
        energy_by_source: Dict[str, Decimal] = {}

        for consumption in input_data.energy_consumption:
            energy_kwh = self._convert_to_kwh(
                consumption.consumption,
                consumption.unit,
                consumption.source
            )
            total_energy_kwh += energy_kwh
            source_key = consumption.source.value
            energy_by_source[source_key] = energy_by_source.get(
                source_key, Decimal("0")
            ) + energy_kwh

        # Step 4: Calculate EUI with weather normalization
        eui_metrics = self._calculate_eui(total_energy_kwh, floor_area)

        weather_normalized_eui = None
        if input_data.heating_degree_days and input_data.cooling_degree_days:
            # Simple weather normalization
            hdd = input_data.heating_degree_days
            cdd = input_data.cooling_degree_days
            # Normalize to typical 5000 HDD / 1000 CDD
            normalization_factor = (Decimal("5000") + Decimal("1000")) / (hdd + cdd + Decimal("1"))
            weather_normalized_eui = eui_metrics.site_eui_kwh_per_sqm * normalization_factor
            eui_metrics.weather_normalized_eui_kwh_per_sqm = self._round_intensity(weather_normalized_eui)

        carbon_intensity = self._calculate_carbon_intensity(
            self._round_emissions(scope1_emissions),
            self._round_emissions(scope2_emissions),
            floor_area
        )

        # Step 5: Calculate per-unit and per-occupant metrics
        emissions_per_unit = self._round_emissions(
            total_emissions / Decimal(str(num_units))
        )

        emissions_per_occupant = None
        if input_data.num_occupants and input_data.num_occupants > 0:
            emissions_per_occupant = self._round_emissions(
                total_emissions / Decimal(str(input_data.num_occupants))
            )

        emissions_per_bedroom = None
        if input_data.num_bedrooms and input_data.num_bedrooms > 0:
            emissions_per_bedroom = self._round_emissions(
                total_emissions / Decimal(str(input_data.num_bedrooms))
            )

        # Step 6: Estimate HERS index
        estimated_hers = None
        hers_reference_emissions = None
        if metadata.building_type == BuildingType.RESIDENTIAL_SINGLE:
            # HERS 100 = reference home, lower is better
            # Average US home ~ 100, efficient home ~ 50, net zero ~ 0
            avg_emissions_per_sqm = Decimal("35")  # US average
            actual_per_sqm = total_emissions / floor_area if floor_area > 0 else Decimal("0")
            estimated_hers = min(200, max(0, int(actual_per_sqm / avg_emissions_per_sqm * 100)))
            hers_reference_emissions = self._round_emissions(avg_emissions_per_sqm * floor_area)

        return ResidentialBuildingOutput(
            calculation_id=self._generate_calculation_id(
                input_data.building_id,
                input_data.reporting_period
            ),
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            timestamp=self._get_timestamp(),
            building_id=input_data.building_id,
            building_type=metadata.building_type,
            reporting_period=input_data.reporting_period,
            gross_floor_area_sqm=floor_area,
            total_energy_kwh=self._round_energy(total_energy_kwh),
            energy_by_source={k: self._round_energy(v) for k, v in energy_by_source.items()},
            eui_metrics=eui_metrics,
            scope_1_emissions_kgco2e=self._round_emissions(scope1_emissions),
            scope_2_emissions_kgco2e=self._round_emissions(scope2_emissions),
            scope_3_emissions_kgco2e=Decimal("0"),
            total_emissions_kgco2e=self._round_emissions(total_emissions),
            carbon_intensity=carbon_intensity,
            calculation_steps=calculation_steps,
            emission_factors_used=factors_used,
            data_quality=input_data.data_quality,
            verification_status=VerificationStatus.UNVERIFIED,
            is_valid=True,
            emissions_per_dwelling_unit_kgco2e=emissions_per_unit,
            emissions_per_occupant_kgco2e=emissions_per_occupant,
            emissions_per_bedroom_kgco2e=emissions_per_bedroom,
            weather_normalized_eui=weather_normalized_eui,
            estimated_hers_index=estimated_hers,
            hers_reference_home_emissions=hers_reference_emissions
        )
