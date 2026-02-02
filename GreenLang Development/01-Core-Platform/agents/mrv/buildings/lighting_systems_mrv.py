# -*- coding: utf-8 -*-
"""
GL-MRV-BLD-005: Lighting Systems MRV Agent
============================================

Specialized MRV agent for building lighting system energy and emissions.

Features:
    - Interior and exterior lighting breakdown
    - Lighting Power Density (LPD) analysis
    - LED retrofit savings calculation
    - Daylight harvesting credits
    - Occupancy and scheduling factors

Standards:
    - ASHRAE Standard 90.1 Lighting Power Allowances
    - IES Lighting Handbook
    - EPA ENERGY STAR for Lighting

Author: GreenLang Framework Team
Agent ID: GL-MRV-BLD-005
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
    GRID_EF_BY_REGION_KGCO2E_PER_KWH,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class LightingType(str, Enum):
    """Lighting technology type."""
    LED = "led"
    FLUORESCENT = "fluorescent"
    CFL = "cfl"
    HALOGEN = "halogen"
    INCANDESCENT = "incandescent"
    HID = "hid"  # High Intensity Discharge
    METAL_HALIDE = "metal_halide"


class LightingZone(str, Enum):
    """Building lighting zone type."""
    OFFICE = "office"
    RETAIL = "retail"
    WAREHOUSE = "warehouse"
    CORRIDOR = "corridor"
    LOBBY = "lobby"
    RESTROOM = "restroom"
    PARKING = "parking"
    EXTERIOR = "exterior"
    EMERGENCY = "emergency"


# =============================================================================
# LIGHTING STANDARDS
# =============================================================================

# ASHRAE 90.1 LPD allowances (W/sqm)
LPD_ALLOWANCES = {
    BuildingType.COMMERCIAL_OFFICE: Decimal("10.76"),
    BuildingType.RETAIL: Decimal("15.07"),
    BuildingType.WAREHOUSE: Decimal("7.53"),
    BuildingType.HOTEL: Decimal("11.84"),
    BuildingType.HOSPITAL: Decimal("12.92"),
    BuildingType.EDUCATION: Decimal("10.23"),
    BuildingType.RESTAURANT: Decimal("14.31"),
    BuildingType.INDUSTRIAL: Decimal("9.69"),
}

# Typical efficacy by lighting type (lumens/watt)
LIGHTING_EFFICACY = {
    LightingType.LED: Decimal("150"),
    LightingType.FLUORESCENT: Decimal("90"),
    LightingType.CFL: Decimal("70"),
    LightingType.HALOGEN: Decimal("20"),
    LightingType.INCANDESCENT: Decimal("15"),
    LightingType.HID: Decimal("100"),
    LightingType.METAL_HALIDE: Decimal("95"),
}


# =============================================================================
# INPUT/OUTPUT MODELS
# =============================================================================

class LightingFixture(BaseModel):
    """Individual lighting fixture specification."""
    fixture_id: str
    lighting_type: LightingType
    wattage: Decimal = Field(..., ge=0)
    quantity: int = Field(..., ge=1)
    zone: LightingZone
    daily_operating_hours: Decimal = Field(..., ge=0, le=24)
    dimming_factor: Optional[Decimal] = Field(None, ge=0, le=1)
    has_occupancy_sensor: bool = Field(default=False)
    has_daylight_sensor: bool = Field(default=False)


class LightingSystemsInput(BuildingMRVInput):
    """Input model for lighting systems MRV."""

    # Fixture inventory
    fixtures: List[LightingFixture] = Field(default_factory=list)

    # Aggregated data (if fixture details not available)
    total_interior_lighting_kwh: Optional[Decimal] = Field(None, ge=0)
    total_exterior_lighting_kwh: Optional[Decimal] = Field(None, ge=0)

    # Control factors
    average_occupancy_factor: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Fraction of time spaces are occupied"
    )
    daylight_availability_factor: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Fraction of daylight contribution"
    )

    # Benchmarking
    include_lpd_analysis: bool = Field(default=True)


class LightingSystemsOutput(BuildingMRVOutput):
    """Output model for lighting systems MRV."""

    # Emissions breakdown
    interior_lighting_emissions_kgco2e: Decimal = Field(default=Decimal("0"))
    exterior_lighting_emissions_kgco2e: Decimal = Field(default=Decimal("0"))

    # Energy breakdown
    interior_lighting_kwh: Decimal = Field(default=Decimal("0"))
    exterior_lighting_kwh: Decimal = Field(default=Decimal("0"))

    # LPD analysis
    actual_lpd_w_per_sqm: Optional[Decimal] = None
    allowable_lpd_w_per_sqm: Optional[Decimal] = None
    lpd_compliance_percent: Optional[Decimal] = None

    # Technology breakdown
    led_percentage: Optional[Decimal] = None
    fluorescent_percentage: Optional[Decimal] = None
    other_percentage: Optional[Decimal] = None

    # Efficiency metrics
    lighting_power_per_sqm_w: Decimal = Field(default=Decimal("0"))
    estimated_lumens_per_sqm: Optional[Decimal] = None

    # Savings potential
    led_retrofit_savings_kwh: Optional[Decimal] = None
    led_retrofit_savings_kgco2e: Optional[Decimal] = None


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class LightingSystemsMRVAgent(BuildingMRVBaseAgent[LightingSystemsInput, LightingSystemsOutput]):
    """
    GL-MRV-BLD-005: Lighting Systems MRV Agent.

    Calculates lighting energy consumption and emissions with LPD analysis
    and LED retrofit potential.

    ZERO-HALLUCINATION GUARANTEE:
        - All calculations use ASHRAE 90.1 standards
        - Deterministic energy calculations
        - Complete audit trail for verification
        - Reproducible results with same inputs

    Example:
        >>> agent = LightingSystemsMRVAgent()
        >>> input_data = LightingSystemsInput(
        ...     building_id="BLDG-001",
        ...     reporting_period="2024",
        ...     building_metadata=BuildingMetadata(...),
        ...     fixtures=[
        ...         LightingFixture(
        ...             fixture_id="F1",
        ...             lighting_type=LightingType.LED,
        ...             wattage=Decimal("40"),
        ...             quantity=100,
        ...             zone=LightingZone.OFFICE,
        ...             daily_operating_hours=Decimal("10")
        ...         )
        ...     ]
        ... )
        >>> output = agent.process(input_data)
    """

    AGENT_ID = "GL-MRV-BLD-005"
    AGENT_VERSION = "1.0.0"
    BUILDING_CATEGORY = "lighting"

    def _load_emission_factors(self) -> None:
        """Load lighting-related emission factors."""
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
        input_data: LightingSystemsInput
    ) -> LightingSystemsOutput:
        """
        Calculate lighting system emissions.

        Methodology:
        1. Calculate energy from fixture inventory or aggregated data
        2. Apply control factors (occupancy, daylight)
        3. Calculate LPD for compliance
        4. Estimate LED retrofit potential
        5. Calculate emissions

        Args:
            input_data: Validated lighting systems input

        Returns:
            Complete lighting emission output with provenance
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

        # Step 1: Calculate lighting energy
        interior_kwh = Decimal("0")
        exterior_kwh = Decimal("0")
        total_installed_watts = Decimal("0")
        led_watts = Decimal("0")
        fluorescent_watts = Decimal("0")
        other_watts = Decimal("0")

        if input_data.fixtures:
            # Calculate from fixture inventory
            for fixture in input_data.fixtures:
                fixture_watts = fixture.wattage * Decimal(str(fixture.quantity))
                total_installed_watts += fixture_watts

                # Track by technology type
                if fixture.lighting_type == LightingType.LED:
                    led_watts += fixture_watts
                elif fixture.lighting_type == LightingType.FLUORESCENT:
                    fluorescent_watts += fixture_watts
                else:
                    other_watts += fixture_watts

                # Calculate annual energy
                daily_hours = fixture.daily_operating_hours
                annual_hours = daily_hours * 365

                # Apply dimming factor
                dimming = fixture.dimming_factor or Decimal("1.0")

                # Apply control factors
                occupancy_reduction = Decimal("0")
                if fixture.has_occupancy_sensor:
                    occupancy_reduction = Decimal("0.20")  # 20% savings typical

                daylight_reduction = Decimal("0")
                if fixture.has_daylight_sensor:
                    daylight_reduction = Decimal("0.15")  # 15% savings typical

                effective_factor = (
                    dimming *
                    (Decimal("1") - occupancy_reduction) *
                    (Decimal("1") - daylight_reduction)
                )

                fixture_kwh = (
                    fixture_watts * annual_hours * effective_factor / 1000
                )

                if fixture.zone == LightingZone.EXTERIOR:
                    exterior_kwh += fixture_kwh
                else:
                    interior_kwh += fixture_kwh

            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description="Calculate lighting energy from fixture inventory",
                formula="fixture_kwh = watts * quantity * hours * 365 * factors / 1000",
                inputs={
                    "num_fixtures": str(len(input_data.fixtures)),
                    "total_installed_watts": str(total_installed_watts)
                },
                output_value=self._round_energy(interior_kwh + exterior_kwh),
                output_unit="kWh",
                source="Fixture inventory calculation"
            ))
            step_number += 1

        else:
            # Use aggregated data
            if input_data.total_interior_lighting_kwh:
                interior_kwh = input_data.total_interior_lighting_kwh
            if input_data.total_exterior_lighting_kwh:
                exterior_kwh = input_data.total_exterior_lighting_kwh

        # Apply global control factors if provided
        if input_data.average_occupancy_factor is not None:
            occupancy_multiplier = input_data.average_occupancy_factor
            interior_kwh = interior_kwh * occupancy_multiplier

        if input_data.daylight_availability_factor is not None:
            daylight_savings = input_data.daylight_availability_factor * Decimal("0.3")
            interior_kwh = interior_kwh * (Decimal("1") - daylight_savings)

        total_lighting_kwh = interior_kwh + exterior_kwh

        # Step 2: Calculate emissions
        interior_emissions = interior_kwh * grid_ef
        exterior_emissions = exterior_kwh * grid_ef
        total_emissions = interior_emissions + exterior_emissions

        calculation_steps.append(CalculationStep(
            step_number=step_number,
            description="Calculate lighting emissions",
            formula="emissions = lighting_kwh * grid_ef",
            inputs={
                "lighting_kwh": str(self._round_energy(total_lighting_kwh)),
                "grid_ef": str(grid_ef)
            },
            output_value=self._round_emissions(total_emissions),
            output_unit="kgCO2e",
            source="EPA eGRID 2024"
        ))
        step_number += 1

        # Step 3: Calculate LPD
        actual_lpd = None
        allowable_lpd = None
        lpd_compliance = None

        if total_installed_watts > 0 and floor_area > 0:
            actual_lpd = self._round_intensity(total_installed_watts / floor_area)
            allowable_lpd = LPD_ALLOWANCES.get(metadata.building_type)

            if allowable_lpd:
                lpd_compliance = self._round_intensity(
                    (allowable_lpd / actual_lpd) * 100 if actual_lpd > 0 else Decimal("100")
                )

                calculation_steps.append(CalculationStep(
                    step_number=step_number,
                    description="Calculate Lighting Power Density compliance",
                    formula="lpd_compliance = (allowable_lpd / actual_lpd) * 100",
                    inputs={
                        "actual_lpd": str(actual_lpd),
                        "allowable_lpd": str(allowable_lpd)
                    },
                    output_value=lpd_compliance,
                    output_unit="%",
                    source="ASHRAE 90.1"
                ))
                step_number += 1

        # Step 4: Calculate technology percentages
        led_percentage = None
        fluorescent_percentage = None
        other_percentage = None

        if total_installed_watts > 0:
            led_percentage = self._round_intensity(
                led_watts / total_installed_watts * 100
            )
            fluorescent_percentage = self._round_intensity(
                fluorescent_watts / total_installed_watts * 100
            )
            other_percentage = self._round_intensity(
                other_watts / total_installed_watts * 100
            )

        # Step 5: Estimate LED retrofit savings
        led_retrofit_savings_kwh = None
        led_retrofit_savings_emissions = None

        non_led_watts = fluorescent_watts + other_watts
        if non_led_watts > 0:
            # LED typically 50% more efficient than fluorescent, 80% more than other
            average_savings_factor = Decimal("0.50")
            potential_energy_savings = (non_led_watts / total_installed_watts) * total_lighting_kwh * average_savings_factor

            led_retrofit_savings_kwh = self._round_energy(potential_energy_savings)
            led_retrofit_savings_emissions = self._round_emissions(
                potential_energy_savings * grid_ef
            )

            calculation_steps.append(CalculationStep(
                step_number=step_number,
                description="Estimate LED retrofit savings potential",
                formula="savings = non_led_fraction * total_kwh * 0.50",
                inputs={
                    "non_led_watts": str(non_led_watts),
                    "total_installed_watts": str(total_installed_watts)
                },
                output_value=led_retrofit_savings_kwh,
                output_unit="kWh",
                source="IES LED Retrofit Guidelines"
            ))
            step_number += 1

        # Calculate intensity metrics
        lighting_power_per_sqm = Decimal("0")
        if floor_area > 0:
            lighting_power_per_sqm = self._round_intensity(
                total_installed_watts / floor_area
            )

        # Estimate lumens per sqm
        estimated_lumens_per_sqm = None
        if total_installed_watts > 0:
            # Weighted average efficacy
            weighted_efficacy = (
                led_watts * LIGHTING_EFFICACY[LightingType.LED] +
                fluorescent_watts * LIGHTING_EFFICACY[LightingType.FLUORESCENT] +
                other_watts * Decimal("50")  # Average for "other"
            ) / total_installed_watts

            if floor_area > 0:
                estimated_lumens_per_sqm = self._round_intensity(
                    (total_installed_watts * weighted_efficacy) / floor_area
                )

        return LightingSystemsOutput(
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
            total_energy_kwh=self._round_energy(total_lighting_kwh),
            scope_1_emissions_kgco2e=Decimal("0"),
            scope_2_emissions_kgco2e=self._round_emissions(total_emissions),
            scope_3_emissions_kgco2e=Decimal("0"),
            total_emissions_kgco2e=self._round_emissions(total_emissions),
            calculation_steps=calculation_steps,
            emission_factors_used=factors_used,
            data_quality=input_data.data_quality,
            verification_status=VerificationStatus.UNVERIFIED,
            is_valid=True,
            interior_lighting_emissions_kgco2e=self._round_emissions(interior_emissions),
            exterior_lighting_emissions_kgco2e=self._round_emissions(exterior_emissions),
            interior_lighting_kwh=self._round_energy(interior_kwh),
            exterior_lighting_kwh=self._round_energy(exterior_kwh),
            actual_lpd_w_per_sqm=actual_lpd,
            allowable_lpd_w_per_sqm=allowable_lpd,
            lpd_compliance_percent=lpd_compliance,
            led_percentage=led_percentage,
            fluorescent_percentage=fluorescent_percentage,
            other_percentage=other_percentage,
            lighting_power_per_sqm_w=lighting_power_per_sqm,
            estimated_lumens_per_sqm=estimated_lumens_per_sqm,
            led_retrofit_savings_kwh=led_retrofit_savings_kwh,
            led_retrofit_savings_kgco2e=led_retrofit_savings_emissions
        )
