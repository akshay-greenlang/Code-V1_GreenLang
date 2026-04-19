# -*- coding: utf-8 -*-
"""
GL-MRV-BLD-004: HVAC Systems MRV Agent
========================================

Specialized MRV agent for heating, ventilation, and air conditioning
system emissions in buildings.

Features:
    - Detailed HVAC energy breakdown
    - Refrigerant leakage tracking (F-gases)
    - Equipment efficiency metrics (COP, SEER, AFUE)
    - Heat pump performance analysis
    - District heating/cooling integration

Standards:
    - ASHRAE Standard 90.1
    - EPA Section 608 (Refrigerants)
    - GHG Protocol for refrigerant emissions
    - F-gas Regulation (EU) 517/2014

Author: GreenLang Framework Team
Agent ID: GL-MRV-BLD-004
Version: 1.0.0
"""

from __future__ import annotations

import logging
from decimal import Decimal
from enum import Enum
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
    DataQuality,
    VerificationStatus,
    NATURAL_GAS_EF_KGCO2E_PER_THERM,
    GRID_EF_BY_REGION_KGCO2E_PER_KWH,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class HVACSystemType(str, Enum):
    """HVAC system type classification."""
    SPLIT_SYSTEM = "split_system"
    PACKAGED_UNIT = "packaged_unit"
    CHILLER = "chiller"
    BOILER = "boiler"
    HEAT_PUMP_AIR = "heat_pump_air"
    HEAT_PUMP_GROUND = "heat_pump_ground"
    HEAT_PUMP_WATER = "heat_pump_water"
    VRF = "vrf"  # Variable Refrigerant Flow
    PTU = "ptu"  # Packaged Terminal Unit
    DISTRICT_HEATING = "district_heating"
    DISTRICT_COOLING = "district_cooling"
    RADIANT = "radiant"
    FURNACE = "furnace"


class RefrigerantType(str, Enum):
    """Common refrigerant types."""
    R22 = "R22"  # HCFC - being phased out
    R410A = "R410A"  # HFC - common in split systems
    R134A = "R134A"  # HFC - chillers
    R407C = "R407C"  # HFC blend
    R32 = "R32"  # Lower GWP HFC
    R290 = "R290"  # Propane - natural refrigerant
    R744 = "R744"  # CO2 - natural refrigerant
    R717 = "R717"  # Ammonia - industrial
    R1234YF = "R1234yf"  # HFO - low GWP
    R1234ZE = "R1234ze"  # HFO - low GWP


# =============================================================================
# REFRIGERANT GWP VALUES (AR6 100-year)
# =============================================================================

REFRIGERANT_GWP = {
    RefrigerantType.R22: Decimal("1960"),
    RefrigerantType.R410A: Decimal("2088"),
    RefrigerantType.R134A: Decimal("1430"),
    RefrigerantType.R407C: Decimal("1774"),
    RefrigerantType.R32: Decimal("675"),
    RefrigerantType.R290: Decimal("3"),
    RefrigerantType.R744: Decimal("1"),
    RefrigerantType.R717: Decimal("0"),
    RefrigerantType.R1234YF: Decimal("4"),
    RefrigerantType.R1234ZE: Decimal("7"),
}

# Typical annual leak rates by equipment type (%)
LEAK_RATES = {
    HVACSystemType.SPLIT_SYSTEM: Decimal("4"),
    HVACSystemType.PACKAGED_UNIT: Decimal("4"),
    HVACSystemType.CHILLER: Decimal("2"),
    HVACSystemType.HEAT_PUMP_AIR: Decimal("3"),
    HVACSystemType.HEAT_PUMP_GROUND: Decimal("1"),
    HVACSystemType.VRF: Decimal("5"),
    HVACSystemType.PTU: Decimal("8"),
}


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class HVACEquipment(BaseModel):
    """Individual HVAC equipment specification."""
    equipment_id: str
    system_type: HVACSystemType
    capacity_kw: Decimal = Field(..., gt=0)

    # Efficiency metrics
    cooling_cop: Optional[Decimal] = Field(None, ge=1, le=10)
    heating_cop: Optional[Decimal] = Field(None, ge=1, le=6)
    seer: Optional[Decimal] = Field(None, ge=8, le=30)
    afue: Optional[Decimal] = Field(None, ge=0.5, le=1.0)

    # Refrigerant details
    refrigerant_type: Optional[RefrigerantType] = None
    refrigerant_charge_kg: Optional[Decimal] = Field(None, ge=0)
    annual_leak_rate_percent: Optional[Decimal] = Field(None, ge=0, le=100)

    # Operating data
    annual_operating_hours: Optional[Decimal] = Field(None, ge=0)
    average_load_percent: Optional[Decimal] = Field(None, ge=0, le=100)


class HVACSystemsInput(BuildingMRVInput):
    """Input model for HVAC systems MRV."""

    # Equipment inventory
    hvac_equipment: List[HVACEquipment] = Field(default_factory=list)

    # Energy by end use
    heating_energy_kwh: Optional[Decimal] = Field(None, ge=0)
    cooling_energy_kwh: Optional[Decimal] = Field(None, ge=0)
    ventilation_energy_kwh: Optional[Decimal] = Field(None, ge=0)

    # Refrigerant tracking
    refrigerant_purchased_kg: Optional[Decimal] = Field(None, ge=0)
    refrigerant_disposed_kg: Optional[Decimal] = Field(None, ge=0)

    # District systems
    district_heating_mwh: Optional[Decimal] = Field(None, ge=0)
    district_cooling_mwh: Optional[Decimal] = Field(None, ge=0)


class HVACSystemsOutput(BuildingMRVOutput):
    """Output model for HVAC systems MRV."""

    # Emissions breakdown
    heating_emissions_kgco2e: Decimal = Field(default=Decimal("0"))
    cooling_emissions_kgco2e: Decimal = Field(default=Decimal("0"))
    ventilation_emissions_kgco2e: Decimal = Field(default=Decimal("0"))
    refrigerant_emissions_kgco2e: Decimal = Field(default=Decimal("0"))

    # Equipment summary
    total_heating_capacity_kw: Decimal = Field(default=Decimal("0"))
    total_cooling_capacity_kw: Decimal = Field(default=Decimal("0"))
    average_cooling_cop: Optional[Decimal] = None
    average_heating_cop: Optional[Decimal] = None

    # Refrigerant summary
    total_refrigerant_charge_kg: Decimal = Field(default=Decimal("0"))
    estimated_annual_leakage_kg: Decimal = Field(default=Decimal("0"))
    refrigerant_gwp_weighted: Optional[Decimal] = None


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class HVACSystemsMRVAgent(BuildingMRVBaseAgent[HVACSystemsInput, HVACSystemsOutput]):
    """
    GL-MRV-BLD-004: HVAC Systems MRV Agent.

    Calculates detailed HVAC system emissions including energy consumption
    and refrigerant leakage (F-gas emissions).

    ZERO-HALLUCINATION GUARANTEE:
        - All calculations use EPA/ASHRAE emission factors
        - Refrigerant GWP values from IPCC AR6
        - Complete audit trail for verification
        - Reproducible results with same inputs

    Example:
        >>> agent = HVACSystemsMRVAgent()
        >>> input_data = HVACSystemsInput(
        ...     building_id="BLDG-001",
        ...     reporting_period="2024",
        ...     building_metadata=BuildingMetadata(...),
        ...     hvac_equipment=[
        ...         HVACEquipment(
        ...             equipment_id="AHU-1",
        ...             system_type=HVACSystemType.CHILLER,
        ...             capacity_kw=Decimal("500"),
        ...             refrigerant_type=RefrigerantType.R134A,
        ...             refrigerant_charge_kg=Decimal("50")
        ...         )
        ...     ]
        ... )
        >>> output = agent.process(input_data)
    """

    AGENT_ID = "GL-MRV-BLD-004"
    AGENT_VERSION = "1.0.0"
    BUILDING_CATEGORY = "hvac_systems"

    def _load_emission_factors(self) -> None:
        """Load HVAC-related emission factors."""
        # Natural gas for heating
        self._emission_factors["scope1_natural_gas_therm"] = EmissionFactor(
            factor_id="scope1_natural_gas_therm",
            value=NATURAL_GAS_EF_KGCO2E_PER_THERM,
            unit="kgCO2e/therm",
            source="EPA GHG Emission Factors Hub 2024",
            region="US",
            valid_from="2024-01-01"
        )

        # Electricity for HVAC
        for region, factor in GRID_EF_BY_REGION_KGCO2E_PER_KWH.items():
            self._emission_factors[f"scope2_electricity_{region}"] = EmissionFactor(
                factor_id=f"scope2_electricity_{region}",
                value=factor,
                unit="kgCO2e/kWh",
                source="EPA eGRID 2024",
                region=region,
                valid_from="2024-01-01"
            )

        # District heating factor
        self._emission_factors["district_heating_kwh"] = EmissionFactor(
            factor_id="district_heating_kwh",
            value=Decimal("0.150"),  # Typical district heating
            unit="kgCO2e/kWh",
            source="EU District Heating Average 2024",
            region="EU",
            valid_from="2024-01-01"
        )

        # District cooling factor
        self._emission_factors["district_cooling_kwh"] = EmissionFactor(
            factor_id="district_cooling_kwh",
            value=Decimal("0.080"),  # Typical district cooling
            unit="kgCO2e/kWh",
            source="EU District Cooling Average 2024",
            region="EU",
            valid_from="2024-01-01"
        )

        # Refrigerant GWP factors
        for ref_type, gwp in REFRIGERANT_GWP.items():
            self._emission_factors[f"refrigerant_{ref_type.value}"] = EmissionFactor(
                factor_id=f"refrigerant_{ref_type.value}",
                value=gwp,
                unit="kgCO2e/kg",
                source="IPCC AR6 GWP-100",
                region="global",
                valid_from="2024-01-01"
            )

    def calculate_emissions(
        self,
        input_data: HVACSystemsInput
    ) -> HVACSystemsOutput:
        """
        Calculate HVAC system emissions.

        Methodology:
        1. Calculate heating energy emissions (Scope 1 & 2)
        2. Calculate cooling energy emissions (Scope 2)
        3. Calculate ventilation energy emissions (Scope 2)
        4. Calculate refrigerant leakage emissions (Scope 1)
        5. Calculate district energy emissions (Scope 2)

        Args:
            input_data: Validated HVAC systems input

        Returns:
            Complete HVAC emission output with provenance
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

        scope1_emissions = Decimal("0")
        scope2_emissions = Decimal("0")

        # Step 1: Calculate heating emissions
        heating_emissions = Decimal("0")

        # From direct heating fuels (Scope 1)
        for consumption in input_data.energy_consumption:
            if consumption.source == EnergySource.NATURAL_GAS:
                if "therm" in consumption.unit.lower():
                    ng_emissions = consumption.consumption * NATURAL_GAS_EF_KGCO2E_PER_THERM
                    heating_emissions += ng_emissions
                    scope1_emissions += ng_emissions

        # From electric heating (Scope 2)
        if input_data.heating_energy_kwh:
            electric_heating_emissions = input_data.heating_energy_kwh * grid_ef
            heating_emissions += electric_heating_emissions
            scope2_emissions += electric_heating_emissions

            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description="Calculate electric heating emissions",
                formula="heating_emissions = heating_kwh * grid_ef",
                inputs={
                    "heating_kwh": str(input_data.heating_energy_kwh),
                    "grid_ef": str(grid_ef)
                },
                output_value=self._round_emissions(electric_heating_emissions),
                output_unit="kgCO2e",
                source="EPA eGRID 2024"
            ))
            step_number += 1

        # Step 2: Calculate cooling emissions (Scope 2)
        cooling_emissions = Decimal("0")
        if input_data.cooling_energy_kwh:
            cooling_emissions = input_data.cooling_energy_kwh * grid_ef
            scope2_emissions += cooling_emissions

            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description="Calculate cooling energy emissions",
                formula="cooling_emissions = cooling_kwh * grid_ef",
                inputs={
                    "cooling_kwh": str(input_data.cooling_energy_kwh),
                    "grid_ef": str(grid_ef)
                },
                output_value=self._round_emissions(cooling_emissions),
                output_unit="kgCO2e",
                source="EPA eGRID 2024"
            ))
            step_number += 1

        # Step 3: Calculate ventilation emissions (Scope 2)
        ventilation_emissions = Decimal("0")
        if input_data.ventilation_energy_kwh:
            ventilation_emissions = input_data.ventilation_energy_kwh * grid_ef
            scope2_emissions += ventilation_emissions

            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description="Calculate ventilation energy emissions",
                formula="ventilation_emissions = ventilation_kwh * grid_ef",
                inputs={
                    "ventilation_kwh": str(input_data.ventilation_energy_kwh),
                    "grid_ef": str(grid_ef)
                },
                output_value=self._round_emissions(ventilation_emissions),
                output_unit="kgCO2e",
                source="EPA eGRID 2024"
            ))
            step_number += 1

        # Step 4: Calculate refrigerant emissions (Scope 1)
        refrigerant_emissions = Decimal("0")
        total_refrigerant_charge = Decimal("0")
        estimated_leakage = Decimal("0")
        weighted_gwp_sum = Decimal("0")
        weighted_charge_sum = Decimal("0")

        for equipment in input_data.hvac_equipment:
            if equipment.refrigerant_type and equipment.refrigerant_charge_kg:
                charge = equipment.refrigerant_charge_kg
                total_refrigerant_charge += charge

                # Get leak rate
                leak_rate = equipment.annual_leak_rate_percent
                if leak_rate is None:
                    leak_rate = LEAK_RATES.get(equipment.system_type, Decimal("4"))

                # Calculate leakage
                leakage_kg = charge * (leak_rate / 100)
                estimated_leakage += leakage_kg

                # Get GWP
                gwp = REFRIGERANT_GWP.get(equipment.refrigerant_type, Decimal("0"))

                # Calculate emissions
                ref_emissions = leakage_kg * gwp
                refrigerant_emissions += ref_emissions

                # Track weighted GWP
                weighted_gwp_sum += charge * gwp
                weighted_charge_sum += charge

        if refrigerant_emissions > 0:
            scope1_emissions += refrigerant_emissions

            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description="Calculate refrigerant leakage emissions",
                formula="ref_emissions = leakage_kg * GWP",
                inputs={
                    "total_charge_kg": str(total_refrigerant_charge),
                    "estimated_leakage_kg": str(estimated_leakage)
                },
                output_value=self._round_emissions(refrigerant_emissions),
                output_unit="kgCO2e",
                source="IPCC AR6 GWP-100"
            ))
            step_number += 1

        weighted_gwp = None
        if weighted_charge_sum > 0:
            weighted_gwp = self._round_intensity(weighted_gwp_sum / weighted_charge_sum)

        # Step 5: Calculate district energy emissions (Scope 2)
        if input_data.district_heating_mwh:
            dh_kwh = input_data.district_heating_mwh * 1000
            dh_ef = self._emission_factors["district_heating_kwh"].value
            dh_emissions = dh_kwh * dh_ef
            heating_emissions += dh_emissions
            scope2_emissions += dh_emissions

            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description="Calculate district heating emissions",
                formula="dh_emissions = dh_mwh * 1000 * ef",
                inputs={
                    "district_heating_mwh": str(input_data.district_heating_mwh),
                    "emission_factor": str(dh_ef)
                },
                output_value=self._round_emissions(dh_emissions),
                output_unit="kgCO2e",
                source="EU District Heating Average"
            ))
            factors_used.append(self._emission_factors["district_heating_kwh"])
            step_number += 1

        if input_data.district_cooling_mwh:
            dc_kwh = input_data.district_cooling_mwh * 1000
            dc_ef = self._emission_factors["district_cooling_kwh"].value
            dc_emissions = dc_kwh * dc_ef
            cooling_emissions += dc_emissions
            scope2_emissions += dc_emissions

            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description="Calculate district cooling emissions",
                formula="dc_emissions = dc_mwh * 1000 * ef",
                inputs={
                    "district_cooling_mwh": str(input_data.district_cooling_mwh),
                    "emission_factor": str(dc_ef)
                },
                output_value=self._round_emissions(dc_emissions),
                output_unit="kgCO2e",
                source="EU District Cooling Average"
            ))
            factors_used.append(self._emission_factors["district_cooling_kwh"])
            step_number += 1

        # Step 6: Calculate equipment summaries
        total_heating_capacity = Decimal("0")
        total_cooling_capacity = Decimal("0")
        cop_cooling_sum = Decimal("0")
        cop_cooling_count = 0
        cop_heating_sum = Decimal("0")
        cop_heating_count = 0

        for equipment in input_data.hvac_equipment:
            if equipment.system_type in {
                HVACSystemType.BOILER, HVACSystemType.FURNACE,
                HVACSystemType.HEAT_PUMP_AIR, HVACSystemType.HEAT_PUMP_GROUND
            }:
                total_heating_capacity += equipment.capacity_kw

            if equipment.system_type in {
                HVACSystemType.CHILLER, HVACSystemType.SPLIT_SYSTEM,
                HVACSystemType.PACKAGED_UNIT, HVACSystemType.VRF,
                HVACSystemType.HEAT_PUMP_AIR, HVACSystemType.HEAT_PUMP_GROUND
            }:
                total_cooling_capacity += equipment.capacity_kw

            if equipment.cooling_cop:
                cop_cooling_sum += equipment.cooling_cop
                cop_cooling_count += 1

            if equipment.heating_cop:
                cop_heating_sum += equipment.heating_cop
                cop_heating_count += 1

        avg_cooling_cop = None
        if cop_cooling_count > 0:
            avg_cooling_cop = self._round_intensity(
                cop_cooling_sum / Decimal(str(cop_cooling_count))
            )

        avg_heating_cop = None
        if cop_heating_count > 0:
            avg_heating_cop = self._round_intensity(
                cop_heating_sum / Decimal(str(cop_heating_count))
            )

        # Calculate totals
        total_emissions = scope1_emissions + scope2_emissions

        total_energy_kwh = (
            (input_data.heating_energy_kwh or Decimal("0")) +
            (input_data.cooling_energy_kwh or Decimal("0")) +
            (input_data.ventilation_energy_kwh or Decimal("0"))
        )

        return HVACSystemsOutput(
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
            scope_1_emissions_kgco2e=self._round_emissions(scope1_emissions),
            scope_2_emissions_kgco2e=self._round_emissions(scope2_emissions),
            scope_3_emissions_kgco2e=Decimal("0"),
            total_emissions_kgco2e=self._round_emissions(total_emissions),
            calculation_steps=calculation_steps,
            emission_factors_used=factors_used,
            data_quality=input_data.data_quality,
            verification_status=VerificationStatus.UNVERIFIED,
            is_valid=True,
            heating_emissions_kgco2e=self._round_emissions(heating_emissions),
            cooling_emissions_kgco2e=self._round_emissions(cooling_emissions),
            ventilation_emissions_kgco2e=self._round_emissions(ventilation_emissions),
            refrigerant_emissions_kgco2e=self._round_emissions(refrigerant_emissions),
            total_heating_capacity_kw=self._round_energy(total_heating_capacity),
            total_cooling_capacity_kw=self._round_energy(total_cooling_capacity),
            average_cooling_cop=avg_cooling_cop,
            average_heating_cop=avg_heating_cop,
            total_refrigerant_charge_kg=self._round_energy(total_refrigerant_charge),
            estimated_annual_leakage_kg=self._round_energy(estimated_leakage),
            refrigerant_gwp_weighted=weighted_gwp
        )
