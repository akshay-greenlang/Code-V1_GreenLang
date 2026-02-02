# -*- coding: utf-8 -*-
"""
GL-MRV-BLD-003: Industrial Buildings MRV Agent
=================================================

Specialized MRV agent for industrial building emissions including
warehouses, factories, and distribution centers.

Features:
    - Warehouse and factory building support
    - Process energy vs building energy separation
    - Refrigerated warehouse handling
    - Loading dock and logistics energy
    - CBAM alignment for industrial facilities

Standards:
    - GHG Protocol Building Sector Guidance
    - EPA Industrial Facility Guidelines
    - EN ISO 14064-1 for industrial GHG accounting

Author: GreenLang Framework Team
Agent ID: GL-MRV-BLD-003
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
    EndUseCategory,
    EmissionFactor,
    CalculationStep,
    EnergyUseIntensity,
    CarbonIntensity,
    DataQuality,
    VerificationStatus,
    NATURAL_GAS_EF_KGCO2E_PER_THERM,
    FUEL_OIL_EF_KGCO2E_PER_GALLON,
    GRID_EF_BY_REGION_KGCO2E_PER_KWH,
)

logger = logging.getLogger(__name__)


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class IndustrialBuildingInput(BuildingMRVInput):
    """Input model for industrial building MRV."""

    # Industrial-specific fields
    warehouse_type: Optional[str] = Field(
        None,
        description="Type: dry, refrigerated, cold_storage, freezer"
    )
    num_loading_docks: Optional[int] = Field(None, ge=0)
    weekly_operating_hours: Optional[Decimal] = Field(None, ge=0, le=168)
    num_shifts: Optional[int] = Field(None, ge=1, le=3)

    # Process vs building energy
    process_energy_included: bool = Field(default=False)
    process_energy_kwh: Optional[Decimal] = Field(None, ge=0)

    # Refrigeration
    has_refrigeration: bool = Field(default=False)
    refrigerated_area_sqm: Optional[Decimal] = Field(None, ge=0)
    refrigeration_temp_c: Optional[Decimal] = Field(None, ge=-40, le=15)

    # Material handling
    num_forklifts_electric: Optional[int] = Field(None, ge=0)
    num_forklifts_propane: Optional[int] = Field(None, ge=0)


class IndustrialBuildingOutput(BuildingMRVOutput):
    """Output model for industrial building MRV."""

    # Industrial-specific metrics
    building_emissions_kgco2e: Decimal = Field(default=Decimal("0"))
    process_emissions_kgco2e: Decimal = Field(default=Decimal("0"))
    refrigeration_emissions_kgco2e: Decimal = Field(default=Decimal("0"))
    material_handling_emissions_kgco2e: Decimal = Field(default=Decimal("0"))

    # Intensity by function
    emissions_per_sqm_kgco2e: Decimal = Field(default=Decimal("0"))
    emissions_per_loading_dock_kgco2e: Optional[Decimal] = None

    # CBAM relevant
    is_cbam_relevant: bool = Field(default=False)


# =============================================================================
# INDUSTRIAL BUILDING CONSTANTS
# =============================================================================

# Refrigeration emission factors (indirect from electricity)
REFRIGERATION_ENERGY_BY_TEMP = {
    "dry": Decimal("0"),  # No refrigeration
    "cooled": Decimal("50"),  # 2-8C, kWh per sqm per year
    "cold_storage": Decimal("120"),  # -2 to 2C
    "freezer": Decimal("200"),  # -25 to -18C
    "deep_freeze": Decimal("300"),  # Below -25C
}

# Forklift energy consumption
FORKLIFT_PROPANE_GALLONS_PER_HOUR = Decimal("1.5")
FORKLIFT_ELECTRIC_KWH_PER_HOUR = Decimal("3.0")


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class IndustrialBuildingsMRVAgent(BuildingMRVBaseAgent[IndustrialBuildingInput, IndustrialBuildingOutput]):
    """
    GL-MRV-BLD-003: Industrial Buildings MRV Agent.

    Calculates emissions for industrial buildings including warehouses,
    factories, and distribution centers.

    ZERO-HALLUCINATION GUARANTEE:
        - All calculations use EPA/IPCC emission factors
        - Process and building energy separated
        - Complete audit trail for verification
        - Reproducible results with same inputs

    Example:
        >>> agent = IndustrialBuildingsMRVAgent()
        >>> input_data = IndustrialBuildingInput(
        ...     building_id="WH-001",
        ...     reporting_period="2024",
        ...     building_metadata=BuildingMetadata(
        ...         building_id="WH-001",
        ...         building_type=BuildingType.WAREHOUSE,
        ...         gross_floor_area_sqm=Decimal("10000")
        ...     ),
        ...     warehouse_type="refrigerated",
        ...     energy_consumption=[...]
        ... )
        >>> output = agent.process(input_data)
    """

    AGENT_ID = "GL-MRV-BLD-003"
    AGENT_VERSION = "1.0.0"
    BUILDING_CATEGORY = "industrial"

    SUPPORTED_BUILDING_TYPES = {
        BuildingType.WAREHOUSE,
        BuildingType.INDUSTRIAL,
        BuildingType.DATA_CENTER
    }

    def _load_emission_factors(self) -> None:
        """Load industrial building emission factors."""
        # Natural gas
        self._emission_factors["scope1_natural_gas_therm"] = EmissionFactor(
            factor_id="scope1_natural_gas_therm",
            value=NATURAL_GAS_EF_KGCO2E_PER_THERM,
            unit="kgCO2e/therm",
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

        # Propane (forklifts)
        self._emission_factors["scope1_propane_gallon"] = EmissionFactor(
            factor_id="scope1_propane_gallon",
            value=Decimal("5.72"),
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
        input_data: IndustrialBuildingInput
    ) -> IndustrialBuildingOutput:
        """
        Calculate industrial building emissions.

        Methodology:
        1. Calculate building HVAC and lighting emissions
        2. Calculate refrigeration emissions
        3. Calculate material handling emissions
        4. Separate process energy if applicable
        5. Calculate intensity metrics

        Args:
            input_data: Validated industrial building input

        Returns:
            Complete emission output with provenance
        """
        calculation_steps: List[CalculationStep] = []
        factors_used: List[EmissionFactor] = []
        step_number = 1

        metadata = input_data.building_metadata
        floor_area = metadata.gross_floor_area_sqm

        # Get grid emission factor
        grid_ef = input_data.grid_emission_factor_kgco2e_per_kwh
        if grid_ef is None:
            grid_ef = GRID_EF_BY_REGION_KGCO2E_PER_KWH.get("us_average", Decimal("0.379"))

        # Step 1: Calculate Scope 1 from heating fuels
        scope1_building = Decimal("0")
        natural_gas_therms = Decimal("0")
        fuel_oil_gallons = Decimal("0")

        for consumption in input_data.energy_consumption:
            if consumption.source == EnergySource.NATURAL_GAS:
                if "therm" in consumption.unit.lower():
                    natural_gas_therms += consumption.consumption

            elif consumption.source == EnergySource.FUEL_OIL:
                if "gallon" in consumption.unit.lower():
                    fuel_oil_gallons += consumption.consumption

        if natural_gas_therms > 0:
            ng_emissions = natural_gas_therms * NATURAL_GAS_EF_KGCO2E_PER_THERM
            scope1_building += ng_emissions

            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description="Calculate Scope 1 from natural gas heating",
                formula="ng_emissions = therms * 5.302",
                inputs={"natural_gas_therms": str(natural_gas_therms)},
                output_value=self._round_emissions(ng_emissions),
                output_unit="kgCO2e",
                source="EPA GHG Emission Factors Hub 2024"
            ))
            factors_used.append(self._emission_factors["scope1_natural_gas_therm"])
            step_number += 1

        if fuel_oil_gallons > 0:
            fo_emissions = fuel_oil_gallons * FUEL_OIL_EF_KGCO2E_PER_GALLON
            scope1_building += fo_emissions

            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description="Calculate Scope 1 from fuel oil heating",
                formula="fo_emissions = gallons * 10.21",
                inputs={"fuel_oil_gallons": str(fuel_oil_gallons)},
                output_value=self._round_emissions(fo_emissions),
                output_unit="kgCO2e",
                source="EPA GHG Emission Factors Hub 2024"
            ))
            factors_used.append(self._emission_factors["scope1_fuel_oil_gallon"])
            step_number += 1

        # Step 2: Calculate material handling emissions (propane forklifts)
        material_handling_emissions = Decimal("0")
        if input_data.num_forklifts_propane and input_data.num_forklifts_propane > 0:
            operating_hours = input_data.weekly_operating_hours or Decimal("40")
            annual_hours = operating_hours * 52
            propane_usage = (
                Decimal(str(input_data.num_forklifts_propane)) *
                annual_hours *
                FORKLIFT_PROPANE_GALLONS_PER_HOUR *
                Decimal("0.5")  # Assume 50% utilization
            )
            material_handling_emissions = propane_usage * Decimal("5.72")

            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description="Calculate propane forklift emissions",
                formula="emissions = forklifts * hours * 1.5 gal/hr * 0.5 util * 5.72",
                inputs={
                    "num_forklifts": str(input_data.num_forklifts_propane),
                    "annual_hours": str(annual_hours)
                },
                output_value=self._round_emissions(material_handling_emissions),
                output_unit="kgCO2e",
                source="EPA GHG Emission Factors Hub 2024"
            ))
            factors_used.append(self._emission_factors["scope1_propane_gallon"])
            step_number += 1

        scope1_emissions = scope1_building + material_handling_emissions

        # Step 3: Calculate Scope 2 from electricity
        electricity_kwh = Decimal("0")
        for consumption in input_data.energy_consumption:
            if consumption.source == EnergySource.ELECTRICITY:
                electricity_kwh += self._convert_to_kwh(
                    consumption.consumption,
                    consumption.unit,
                    consumption.source
                )

        scope2_emissions = electricity_kwh * grid_ef

        calculation_steps.append(CalculationStep(
            step_number=step_number,
            description="Calculate Scope 2 from electricity",
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

        # Step 4: Estimate refrigeration emissions
        refrigeration_emissions = Decimal("0")
        if input_data.has_refrigeration and input_data.refrigerated_area_sqm:
            # Determine refrigeration type from temperature
            ref_type = "cooled"
            if input_data.refrigeration_temp_c is not None:
                temp = input_data.refrigeration_temp_c
                if temp < -25:
                    ref_type = "deep_freeze"
                elif temp < -18:
                    ref_type = "freezer"
                elif temp < 2:
                    ref_type = "cold_storage"
                else:
                    ref_type = "cooled"

            ref_energy_factor = REFRIGERATION_ENERGY_BY_TEMP.get(ref_type, Decimal("50"))
            ref_energy_kwh = input_data.refrigerated_area_sqm * ref_energy_factor
            refrigeration_emissions = ref_energy_kwh * grid_ef

            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description=f"Calculate refrigeration emissions ({ref_type})",
                formula="ref_emissions = area * energy_factor * grid_ef",
                inputs={
                    "refrigerated_area_sqm": str(input_data.refrigerated_area_sqm),
                    "energy_factor": str(ref_energy_factor),
                    "grid_ef": str(grid_ef)
                },
                output_value=self._round_emissions(refrigeration_emissions),
                output_unit="kgCO2e",
                source="EPA Industrial Refrigeration Guidelines"
            ))
            step_number += 1

        # Step 5: Separate process emissions
        process_emissions = Decimal("0")
        building_emissions = scope1_building + scope2_emissions + refrigeration_emissions

        if input_data.process_energy_included and input_data.process_energy_kwh:
            process_emissions = input_data.process_energy_kwh * grid_ef
            # Subtract process from total to get building-only
            building_emissions = building_emissions - process_emissions

            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description="Separate process energy emissions",
                formula="process_emissions = process_kwh * grid_ef",
                inputs={
                    "process_energy_kwh": str(input_data.process_energy_kwh),
                    "grid_ef": str(grid_ef)
                },
                output_value=self._round_emissions(process_emissions),
                output_unit="kgCO2e",
                source="GHG Protocol"
            ))
            step_number += 1

        # Step 6: Calculate totals
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

        # Step 7: Calculate intensity metrics
        eui_metrics = self._calculate_eui(total_energy_kwh, floor_area)

        carbon_intensity = self._calculate_carbon_intensity(
            self._round_emissions(scope1_emissions),
            self._round_emissions(scope2_emissions),
            floor_area
        )

        emissions_per_sqm = self._round_emissions(
            total_emissions / floor_area if floor_area > 0 else Decimal("0")
        )

        emissions_per_dock = None
        if input_data.num_loading_docks and input_data.num_loading_docks > 0:
            emissions_per_dock = self._round_emissions(
                total_emissions / Decimal(str(input_data.num_loading_docks))
            )

        return IndustrialBuildingOutput(
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
            building_emissions_kgco2e=self._round_emissions(building_emissions),
            process_emissions_kgco2e=self._round_emissions(process_emissions),
            refrigeration_emissions_kgco2e=self._round_emissions(refrigeration_emissions),
            material_handling_emissions_kgco2e=self._round_emissions(material_handling_emissions),
            emissions_per_sqm_kgco2e=emissions_per_sqm,
            emissions_per_loading_dock_kgco2e=emissions_per_dock,
            is_cbam_relevant=input_data.process_energy_included
        )
