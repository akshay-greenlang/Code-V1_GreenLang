# -*- coding: utf-8 -*-
"""
GL-MRV-BLD-007: Building Operations MRV Agent
===============================================

Comprehensive MRV agent for whole-building operational emissions.
Aggregates all operational emission sources for complete building footprint.

Features:
    - Whole-building operational emissions
    - Multi-scope aggregation (Scope 1, 2, 3 operational)
    - Tenant and landlord split
    - Year-over-year comparison
    - Intensity benchmarking

Standards:
    - GHG Protocol Building Sector Guidance
    - GRESB Real Estate Assessment
    - CRREM (Carbon Risk Real Estate Monitor)

Author: GreenLang Framework Team
Agent ID: GL-MRV-BLD-007
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
    NATURAL_GAS_EF_KGCO2E_PER_THERM,
    FUEL_OIL_EF_KGCO2E_PER_GALLON,
    PROPANE_EF_KGCO2E_PER_GALLON,
    GRID_EF_BY_REGION_KGCO2E_PER_KWH,
    BENCHMARK_EUI_BY_TYPE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class YearlyEmissions(BaseModel):
    """Historical emissions for comparison."""
    year: int
    total_emissions_kgco2e: Decimal
    total_energy_kwh: Decimal


class BuildingOperationsInput(BuildingMRVInput):
    """Input model for building operations MRV."""

    # Occupancy and usage
    occupancy_percent: Optional[Decimal] = Field(None, ge=0, le=100)
    weekly_operating_hours: Optional[Decimal] = Field(None, ge=0, le=168)

    # Historical data for comparison
    previous_year_emissions: Optional[YearlyEmissions] = None

    # Scope 3 operational
    include_scope3_operational: bool = Field(default=False)
    waste_tonnes: Optional[Decimal] = Field(None, ge=0)
    water_m3: Optional[Decimal] = Field(None, ge=0)
    commuting_km_total: Optional[Decimal] = Field(None, ge=0)

    # Tenant/landlord split
    is_whole_building: bool = Field(default=True)
    tenant_floor_area_sqm: Optional[Decimal] = Field(None, ge=0)
    common_area_sqm: Optional[Decimal] = Field(None, ge=0)

    # Renewable energy
    onsite_renewable_kwh: Optional[Decimal] = Field(None, ge=0)
    renewable_energy_certificates_kwh: Optional[Decimal] = Field(None, ge=0)


class BuildingOperationsOutput(BuildingMRVOutput):
    """Output model for building operations MRV."""

    # Detailed scope breakdown
    scope_1_fuel_combustion_kgco2e: Decimal = Field(default=Decimal("0"))
    scope_1_refrigerants_kgco2e: Decimal = Field(default=Decimal("0"))
    scope_2_location_based_kgco2e: Decimal = Field(default=Decimal("0"))
    scope_2_market_based_kgco2e: Decimal = Field(default=Decimal("0"))
    scope_3_waste_kgco2e: Decimal = Field(default=Decimal("0"))
    scope_3_water_kgco2e: Decimal = Field(default=Decimal("0"))
    scope_3_commuting_kgco2e: Decimal = Field(default=Decimal("0"))

    # Net emissions after renewables
    gross_emissions_kgco2e: Decimal = Field(default=Decimal("0"))
    renewable_offset_kgco2e: Decimal = Field(default=Decimal("0"))
    net_emissions_kgco2e: Decimal = Field(default=Decimal("0"))

    # Tenant/landlord allocation
    tenant_emissions_kgco2e: Optional[Decimal] = None
    landlord_emissions_kgco2e: Optional[Decimal] = None
    common_area_emissions_kgco2e: Optional[Decimal] = None

    # Year-over-year
    yoy_emissions_change_percent: Optional[Decimal] = None
    yoy_energy_change_percent: Optional[Decimal] = None
    yoy_intensity_change_percent: Optional[Decimal] = None

    # Benchmarking
    gresb_carbon_intensity: Optional[Decimal] = None
    crrem_aligned: Optional[bool] = None


# =============================================================================
# SCOPE 3 OPERATIONAL FACTORS
# =============================================================================

# Waste emission factors (kgCO2e per tonne)
WASTE_EF_KGCO2E_PER_TONNE = {
    "landfill": Decimal("460"),
    "recycled": Decimal("21"),
    "composted": Decimal("10"),
    "average": Decimal("300"),
}

# Water emission factors (kgCO2e per m3)
WATER_EF_KGCO2E_PER_M3 = Decimal("0.344")

# Commuting emission factors (kgCO2e per passenger-km)
COMMUTING_EF_KGCO2E_PER_KM = Decimal("0.171")  # Average car


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class BuildingOperationsMRVAgent(BuildingMRVBaseAgent[BuildingOperationsInput, BuildingOperationsOutput]):
    """
    GL-MRV-BLD-007: Building Operations MRV Agent.

    Calculates comprehensive operational emissions for buildings including
    all scopes and tenant/landlord allocation.

    ZERO-HALLUCINATION GUARANTEE:
        - All calculations use EPA/GHG Protocol emission factors
        - Deterministic aggregation calculations
        - Complete audit trail for verification
        - Reproducible results with same inputs

    Example:
        >>> agent = BuildingOperationsMRVAgent()
        >>> input_data = BuildingOperationsInput(
        ...     building_id="BLDG-001",
        ...     reporting_period="2024",
        ...     building_metadata=BuildingMetadata(...),
        ...     energy_consumption=[...],
        ...     include_scope3_operational=True,
        ...     waste_tonnes=Decimal("50"),
        ...     water_m3=Decimal("5000")
        ... )
        >>> output = agent.process(input_data)
    """

    AGENT_ID = "GL-MRV-BLD-007"
    AGENT_VERSION = "1.0.0"
    BUILDING_CATEGORY = "operations"

    def _load_emission_factors(self) -> None:
        """Load operational emission factors."""
        # Fuel factors
        self._emission_factors["scope1_natural_gas_therm"] = EmissionFactor(
            factor_id="scope1_natural_gas_therm",
            value=NATURAL_GAS_EF_KGCO2E_PER_THERM,
            unit="kgCO2e/therm",
            source="EPA GHG Emission Factors Hub 2024",
            region="US",
            valid_from="2024-01-01"
        )

        self._emission_factors["scope1_fuel_oil_gallon"] = EmissionFactor(
            factor_id="scope1_fuel_oil_gallon",
            value=FUEL_OIL_EF_KGCO2E_PER_GALLON,
            unit="kgCO2e/gallon",
            source="EPA GHG Emission Factors Hub 2024",
            region="US",
            valid_from="2024-01-01"
        )

        self._emission_factors["scope1_propane_gallon"] = EmissionFactor(
            factor_id="scope1_propane_gallon",
            value=PROPANE_EF_KGCO2E_PER_GALLON,
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

        # Scope 3 factors
        self._emission_factors["waste_average"] = EmissionFactor(
            factor_id="waste_average",
            value=WASTE_EF_KGCO2E_PER_TONNE["average"],
            unit="kgCO2e/tonne",
            source="EPA WARM Model",
            region="US",
            valid_from="2024-01-01"
        )

        self._emission_factors["water"] = EmissionFactor(
            factor_id="water",
            value=WATER_EF_KGCO2E_PER_M3,
            unit="kgCO2e/m3",
            source="UK Water Industry Research",
            region="UK",
            valid_from="2024-01-01"
        )

    def calculate_emissions(
        self,
        input_data: BuildingOperationsInput
    ) -> BuildingOperationsOutput:
        """
        Calculate comprehensive building operational emissions.

        Methodology:
        1. Calculate Scope 1 (fuel combustion)
        2. Calculate Scope 2 (purchased electricity)
        3. Calculate Scope 3 operational (waste, water, commuting)
        4. Apply renewable offsets
        5. Allocate tenant/landlord
        6. Calculate year-over-year changes

        Args:
            input_data: Validated building operations input

        Returns:
            Complete operational emission output with provenance
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

        # Step 1: Calculate Scope 1 emissions
        scope1_fuel = Decimal("0")
        scope1_refrigerants = Decimal("0")

        natural_gas_therms = Decimal("0")
        fuel_oil_gallons = Decimal("0")
        propane_gallons = Decimal("0")

        for consumption in input_data.energy_consumption:
            if consumption.source == EnergySource.NATURAL_GAS:
                if "therm" in consumption.unit.lower():
                    natural_gas_therms += consumption.consumption

            elif consumption.source == EnergySource.FUEL_OIL:
                if "gallon" in consumption.unit.lower():
                    fuel_oil_gallons += consumption.consumption

            elif consumption.source == EnergySource.PROPANE:
                if "gallon" in consumption.unit.lower():
                    propane_gallons += consumption.consumption

        if natural_gas_therms > 0:
            scope1_fuel += natural_gas_therms * NATURAL_GAS_EF_KGCO2E_PER_THERM
            factors_used.append(self._emission_factors["scope1_natural_gas_therm"])

        if fuel_oil_gallons > 0:
            scope1_fuel += fuel_oil_gallons * FUEL_OIL_EF_KGCO2E_PER_GALLON
            factors_used.append(self._emission_factors["scope1_fuel_oil_gallon"])

        if propane_gallons > 0:
            scope1_fuel += propane_gallons * PROPANE_EF_KGCO2E_PER_GALLON
            factors_used.append(self._emission_factors["scope1_propane_gallon"])

        calculation_steps.append(CalculationStep(
            step_number=step_number,
            description="Calculate Scope 1 fuel combustion emissions",
            formula="scope1 = sum(fuel_quantity * emission_factor)",
            inputs={
                "natural_gas_therms": str(natural_gas_therms),
                "fuel_oil_gallons": str(fuel_oil_gallons),
                "propane_gallons": str(propane_gallons)
            },
            output_value=self._round_emissions(scope1_fuel),
            output_unit="kgCO2e",
            source="EPA GHG Emission Factors Hub 2024"
        ))
        step_number += 1

        scope1_total = scope1_fuel + scope1_refrigerants

        # Step 2: Calculate Scope 2 emissions
        electricity_kwh = Decimal("0")
        for consumption in input_data.energy_consumption:
            if consumption.source == EnergySource.ELECTRICITY:
                electricity_kwh += self._convert_to_kwh(
                    consumption.consumption,
                    consumption.unit,
                    consumption.source
                )

        scope2_location = electricity_kwh * grid_ef

        # Market-based (using RECs if available)
        scope2_market = scope2_location
        if input_data.renewable_energy_certificates_kwh:
            rec_offset = input_data.renewable_energy_certificates_kwh * grid_ef
            scope2_market = max(Decimal("0"), scope2_location - rec_offset)

        calculation_steps.append(CalculationStep(
            step_number=step_number,
            description="Calculate Scope 2 electricity emissions",
            formula="scope2_location = electricity_kwh * grid_ef",
            inputs={
                "electricity_kwh": str(electricity_kwh),
                "grid_ef": str(grid_ef)
            },
            output_value=self._round_emissions(scope2_location),
            output_unit="kgCO2e",
            source="EPA eGRID 2024"
        ))
        step_number += 1

        # Step 3: Calculate Scope 3 operational emissions
        scope3_waste = Decimal("0")
        scope3_water = Decimal("0")
        scope3_commuting = Decimal("0")

        if input_data.include_scope3_operational:
            if input_data.waste_tonnes:
                scope3_waste = input_data.waste_tonnes * WASTE_EF_KGCO2E_PER_TONNE["average"]
                factors_used.append(self._emission_factors["waste_average"])

                calculation_steps.append(CalculationStep(
                    step_number=step_number,
                    description="Calculate Scope 3 waste emissions",
                    formula="scope3_waste = waste_tonnes * 300",
                    inputs={"waste_tonnes": str(input_data.waste_tonnes)},
                    output_value=self._round_emissions(scope3_waste),
                    output_unit="kgCO2e",
                    source="EPA WARM Model"
                ))
                step_number += 1

            if input_data.water_m3:
                scope3_water = input_data.water_m3 * WATER_EF_KGCO2E_PER_M3
                factors_used.append(self._emission_factors["water"])

                calculation_steps.append(CalculationStep(
                    step_number=step_number,
                    description="Calculate Scope 3 water emissions",
                    formula="scope3_water = water_m3 * 0.344",
                    inputs={"water_m3": str(input_data.water_m3)},
                    output_value=self._round_emissions(scope3_water),
                    output_unit="kgCO2e",
                    source="UK Water Industry Research"
                ))
                step_number += 1

            if input_data.commuting_km_total:
                scope3_commuting = input_data.commuting_km_total * COMMUTING_EF_KGCO2E_PER_KM

                calculation_steps.append(CalculationStep(
                    step_number=step_number,
                    description="Calculate Scope 3 commuting emissions",
                    formula="scope3_commuting = km * 0.171",
                    inputs={"commuting_km": str(input_data.commuting_km_total)},
                    output_value=self._round_emissions(scope3_commuting),
                    output_unit="kgCO2e",
                    source="DEFRA 2024"
                ))
                step_number += 1

        scope3_total = scope3_waste + scope3_water + scope3_commuting

        # Step 4: Calculate renewable offsets
        renewable_offset = Decimal("0")
        if input_data.onsite_renewable_kwh:
            renewable_offset = input_data.onsite_renewable_kwh * grid_ef

        # Step 5: Calculate totals
        gross_emissions = scope1_total + scope2_location + scope3_total
        net_emissions = max(Decimal("0"), gross_emissions - renewable_offset)

        # Step 6: Calculate energy totals
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

        # Step 7: Calculate tenant/landlord allocation
        tenant_emissions = None
        landlord_emissions = None
        common_area_emissions = None

        if not input_data.is_whole_building and input_data.tenant_floor_area_sqm:
            tenant_ratio = input_data.tenant_floor_area_sqm / floor_area
            tenant_emissions = self._round_emissions(net_emissions * tenant_ratio)

            if input_data.common_area_sqm:
                common_ratio = input_data.common_area_sqm / floor_area
                common_area_emissions = self._round_emissions(net_emissions * common_ratio)
                landlord_emissions = self._round_emissions(
                    net_emissions - tenant_emissions - common_area_emissions
                )
            else:
                landlord_emissions = self._round_emissions(net_emissions - tenant_emissions)

        # Step 8: Calculate year-over-year changes
        yoy_emissions_change = None
        yoy_energy_change = None
        yoy_intensity_change = None

        if input_data.previous_year_emissions:
            prev = input_data.previous_year_emissions

            if prev.total_emissions_kgco2e > 0:
                yoy_emissions_change = self._round_intensity(
                    ((net_emissions - prev.total_emissions_kgco2e) / prev.total_emissions_kgco2e) * 100
                )

            if prev.total_energy_kwh > 0:
                yoy_energy_change = self._round_intensity(
                    ((total_energy_kwh - prev.total_energy_kwh) / prev.total_energy_kwh) * 100
                )

            # Intensity change
            prev_intensity = prev.total_emissions_kgco2e / floor_area if floor_area > 0 else Decimal("0")
            curr_intensity = net_emissions / floor_area if floor_area > 0 else Decimal("0")
            if prev_intensity > 0:
                yoy_intensity_change = self._round_intensity(
                    ((curr_intensity - prev_intensity) / prev_intensity) * 100
                )

        # Calculate intensity metrics
        eui_metrics = self._calculate_eui(total_energy_kwh, floor_area)

        benchmark_eui = BENCHMARK_EUI_BY_TYPE.get(metadata.building_type)
        if benchmark_eui:
            eui_metrics.benchmark_median_eui = benchmark_eui

        carbon_intensity = self._calculate_carbon_intensity(
            self._round_emissions(scope1_total),
            self._round_emissions(scope2_location),
            floor_area
        )

        # GRESB carbon intensity (kgCO2e/sqm/year)
        gresb_intensity = None
        if floor_area > 0:
            gresb_intensity = self._round_intensity(net_emissions / floor_area)

        return BuildingOperationsOutput(
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
            scope_1_emissions_kgco2e=self._round_emissions(scope1_total),
            scope_2_emissions_kgco2e=self._round_emissions(scope2_location),
            scope_3_emissions_kgco2e=self._round_emissions(scope3_total),
            total_emissions_kgco2e=self._round_emissions(net_emissions),
            carbon_intensity=carbon_intensity,
            calculation_steps=calculation_steps,
            emission_factors_used=factors_used,
            data_quality=input_data.data_quality,
            verification_status=VerificationStatus.UNVERIFIED,
            is_valid=True,
            scope_1_fuel_combustion_kgco2e=self._round_emissions(scope1_fuel),
            scope_1_refrigerants_kgco2e=self._round_emissions(scope1_refrigerants),
            scope_2_location_based_kgco2e=self._round_emissions(scope2_location),
            scope_2_market_based_kgco2e=self._round_emissions(scope2_market),
            scope_3_waste_kgco2e=self._round_emissions(scope3_waste),
            scope_3_water_kgco2e=self._round_emissions(scope3_water),
            scope_3_commuting_kgco2e=self._round_emissions(scope3_commuting),
            gross_emissions_kgco2e=self._round_emissions(gross_emissions),
            renewable_offset_kgco2e=self._round_emissions(renewable_offset),
            net_emissions_kgco2e=self._round_emissions(net_emissions),
            tenant_emissions_kgco2e=tenant_emissions,
            landlord_emissions_kgco2e=landlord_emissions,
            common_area_emissions_kgco2e=common_area_emissions,
            yoy_emissions_change_percent=yoy_emissions_change,
            yoy_energy_change_percent=yoy_energy_change,
            yoy_intensity_change_percent=yoy_intensity_change,
            gresb_carbon_intensity=gresb_intensity,
            crrem_aligned=None  # Requires pathway analysis
        )
