# -*- coding: utf-8 -*-
"""
Category 7: Employee Commuting Calculator

Calculates emissions from employees commuting between their homes
and worksites. This includes:

1. Regular commuting (daily travel to/from work)
2. Remote work emissions (home office energy use)
3. Various transportation modes

Supported Methods:
1. Distance-based method (commute distance x employees)
2. Average data method (employees x average commute factors)
3. Spend-based method (commute benefits/reimbursements)

Reference: GHG Protocol Scope 3 Standard, Chapter 6

Example:
    >>> calculator = Category07CommutingCalculator()
    >>> input_data = CommutingInput(
    ...     reporting_year=2024,
    ...     organization_id="ORG001",
    ...     commute_profiles=[
    ...         CommuteProfile(mode="car_gasoline", employees=500, distance_km=25),
    ...         CommuteProfile(mode="public_transit", employees=300, distance_km=15),
    ...         CommuteProfile(mode="remote", employees=200, days_per_week=5),
    ...     ]
    ... )
    >>> result = calculator.calculate(input_data)
"""

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator

from .base import (
    Scope3CategoryCalculator,
    Scope3CalculationInput,
    Scope3CalculationResult,
    CalculationMethod,
    CalculationStep,
    EmissionFactorRecord,
    EmissionFactorSource,
    DataQualityTier,
)

logger = logging.getLogger(__name__)


class CommuteProfile(BaseModel):
    """Employee commute profile by transportation mode."""

    mode: str = Field(..., description="Commute mode")
    employees: int = Field(..., ge=0, description="Number of employees using this mode")
    distance_km: Optional[Decimal] = Field(
        None, ge=0, description="One-way commute distance (km)"
    )
    days_per_week: Decimal = Field(
        Decimal("5"), ge=0, le=7, description="Days commuting per week"
    )
    weeks_per_year: Decimal = Field(
        Decimal("48"), ge=0, le=52, description="Working weeks per year"
    )
    vehicle_type: Optional[str] = Field(None, description="Vehicle type for cars")
    fuel_type: Optional[str] = Field(None, description="Fuel type for cars")
    carpooling_factor: Optional[Decimal] = Field(
        None, ge=1, description="Average passengers per vehicle"
    )

    @validator("mode")
    def normalize_mode(cls, v: str) -> str:
        """Normalize commute mode."""
        mode_map = {
            "car": "car_average",
            "car_gasoline": "car_gasoline",
            "car_petrol": "car_gasoline",
            "car_diesel": "car_diesel",
            "car_hybrid": "car_hybrid",
            "car_electric": "car_electric",
            "ev": "car_electric",
            "public_transit": "public_transit",
            "bus": "bus",
            "train": "rail",
            "rail": "rail",
            "metro": "metro",
            "subway": "metro",
            "light_rail": "light_rail",
            "motorcycle": "motorcycle",
            "scooter": "motorcycle",
            "bicycle": "bicycle",
            "bike": "bicycle",
            "e_bike": "e_bicycle",
            "walking": "walking",
            "walk": "walking",
            "remote": "remote",
            "wfh": "remote",
            "telecommute": "remote",
        }
        normalized = v.lower().strip().replace(" ", "_")
        return mode_map.get(normalized, normalized)


class CommutingInput(Scope3CalculationInput):
    """Input model for Category 7: Employee Commuting."""

    # Commute profile data
    commute_profiles: List[CommuteProfile] = Field(
        default_factory=list, description="Commute profiles by mode"
    )

    # Aggregated inputs (alternative)
    total_employees: Optional[int] = Field(None, ge=0, description="Total employees")
    average_commute_km: Optional[Decimal] = Field(
        None, ge=0, description="Average one-way commute distance"
    )

    # Working patterns
    working_days_per_year: int = Field(
        240, ge=0, le=365, description="Working days per year"
    )
    remote_work_percentage: Optional[Decimal] = Field(
        None, ge=0, le=100, description="Percentage working remotely"
    )

    # Configuration
    include_remote_work_emissions: bool = Field(
        True, description="Include home office energy emissions"
    )


# Commute emission factors (kg CO2e per passenger-km)
# Source: DEFRA 2024, EPA
COMMUTE_FACTORS: Dict[str, Dict[str, Decimal]] = {
    # Private vehicles
    "car_average": {
        "factor": Decimal("0.171"),
        "unit": "passenger-km",
    },
    "car_gasoline": {
        "factor": Decimal("0.192"),
        "unit": "passenger-km",
    },
    "car_diesel": {
        "factor": Decimal("0.168"),
        "unit": "passenger-km",
    },
    "car_hybrid": {
        "factor": Decimal("0.117"),
        "unit": "passenger-km",
    },
    "car_electric": {
        "factor": Decimal("0.047"),  # Depends on grid mix
        "unit": "passenger-km",
    },
    "motorcycle": {
        "factor": Decimal("0.113"),
        "unit": "passenger-km",
    },

    # Public transit
    "public_transit": {
        "factor": Decimal("0.089"),  # Average public transit
        "unit": "passenger-km",
    },
    "bus": {
        "factor": Decimal("0.089"),
        "unit": "passenger-km",
    },
    "rail": {
        "factor": Decimal("0.035"),
        "unit": "passenger-km",
    },
    "metro": {
        "factor": Decimal("0.029"),
        "unit": "passenger-km",
    },
    "light_rail": {
        "factor": Decimal("0.029"),
        "unit": "passenger-km",
    },

    # Zero/low emission
    "bicycle": {
        "factor": Decimal("0"),
        "unit": "passenger-km",
    },
    "e_bicycle": {
        "factor": Decimal("0.005"),
        "unit": "passenger-km",
    },
    "walking": {
        "factor": Decimal("0"),
        "unit": "passenger-km",
    },
}

# Remote work emission factors (kg CO2e per employee per day)
# Based on home office energy consumption
REMOTE_WORK_FACTORS: Dict[str, Decimal] = {
    "US": Decimal("3.2"),  # Higher due to larger homes, more cooling
    "UK": Decimal("1.8"),
    "EU": Decimal("1.5"),
    "default": Decimal("2.0"),
}

# Average commute distances by country (km one-way)
AVERAGE_COMMUTE_DISTANCES: Dict[str, Decimal] = {
    "US": Decimal("21.0"),
    "UK": Decimal("15.0"),
    "EU": Decimal("17.0"),
    "default": Decimal("18.0"),
}


class Category07CommutingCalculator(Scope3CategoryCalculator):
    """
    Calculator for Scope 3 Category 7: Employee Commuting.

    Calculates emissions from employee travel between home and work.

    Attributes:
        CATEGORY_NUMBER: 7
        CATEGORY_NAME: "Employee Commuting"

    Example:
        >>> calculator = Category07CommutingCalculator()
        >>> result = calculator.calculate(input_data)
    """

    CATEGORY_NUMBER = 7
    CATEGORY_NAME = "Employee Commuting"
    SUPPORTED_METHODS = [
        CalculationMethod.DISTANCE_BASED,
        CalculationMethod.AVERAGE_DATA,
    ]

    def __init__(self):
        """Initialize the Category 7 calculator."""
        super().__init__()
        self._commute_factors = COMMUTE_FACTORS
        self._remote_factors = REMOTE_WORK_FACTORS
        self._avg_distances = AVERAGE_COMMUTE_DISTANCES

    def calculate(self, input_data: CommutingInput) -> Scope3CalculationResult:
        """
        Calculate Category 7 emissions.

        Args:
            input_data: Commuting input data

        Returns:
            Complete calculation result with audit trail
        """
        start_time = datetime.utcnow()
        self._validate_method(input_data.calculation_method)

        if input_data.calculation_method == CalculationMethod.AVERAGE_DATA:
            return self._calculate_average_data(input_data, start_time)
        else:
            return self._calculate_distance_based(input_data, start_time)

    def _calculate_distance_based(
        self,
        input_data: CommutingInput,
        start_time: datetime,
    ) -> Scope3CalculationResult:
        """
        Calculate emissions using distance-based method.

        Formula: Emissions = Employees x Distance x 2 (round trip) x Days x Factor

        Args:
            input_data: Input data
            start_time: Calculation start time

        Returns:
            Calculation result
        """
        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")
        total_employees = 0
        commute_emissions = Decimal("0")
        remote_emissions = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize employee commuting calculation",
            inputs={
                "num_profiles": len(input_data.commute_profiles),
                "working_days_per_year": input_data.working_days_per_year,
                "include_remote": input_data.include_remote_work_emissions,
            },
        ))

        for profile in input_data.commute_profiles:
            total_employees += profile.employees

            if profile.mode == "remote":
                # Calculate remote work emissions
                if input_data.include_remote_work_emissions:
                    remote_factor = self._remote_factors.get(
                        input_data.region, self._remote_factors["default"]
                    )
                    working_days = int(
                        profile.days_per_week * profile.weeks_per_year
                    )
                    profile_emissions = (
                        Decimal(str(profile.employees))
                        * Decimal(str(working_days))
                        * remote_factor
                    ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                    remote_emissions += profile_emissions

                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Calculate remote work emissions",
                        formula="emissions = employees x days x remote_factor",
                        inputs={
                            "employees": profile.employees,
                            "working_days": working_days,
                            "remote_factor": str(remote_factor),
                        },
                        output=str(profile_emissions),
                    ))
            else:
                # Calculate commute emissions
                distance = profile.distance_km
                if distance is None:
                    distance = self._avg_distances.get(
                        input_data.region, self._avg_distances["default"]
                    )
                    warnings.append(
                        f"Using default commute distance ({distance} km) for {profile.mode}"
                    )

                factor = self._get_commute_factor(
                    profile.mode, profile.vehicle_type, profile.fuel_type
                )

                # Calculate annual commute km
                # Round trip = distance x 2
                # Annual = round_trip x days_per_week x weeks_per_year
                working_days = int(profile.days_per_week * profile.weeks_per_year)
                annual_pax_km = distance * 2 * Decimal(str(working_days))

                # Apply carpooling factor if specified
                if profile.carpooling_factor and profile.carpooling_factor > 1:
                    annual_pax_km = annual_pax_km / profile.carpooling_factor

                profile_emissions = (
                    Decimal(str(profile.employees)) * annual_pax_km * factor
                ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                commute_emissions += profile_emissions

                steps.append(CalculationStep(
                    step_number=len(steps) + 1,
                    description=f"Calculate commute emissions for {profile.mode}",
                    formula="emissions = employees x annual_pax_km x factor",
                    inputs={
                        "mode": profile.mode,
                        "employees": profile.employees,
                        "distance_km": str(distance),
                        "working_days": working_days,
                        "factor": str(factor),
                        "carpooling": str(profile.carpooling_factor) if profile.carpooling_factor else "N/A",
                    },
                    output=str(profile_emissions),
                ))

        total_emissions_kg = commute_emissions + remote_emissions

        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Sum commute and remote work emissions",
            formula="total = commute_emissions + remote_emissions",
            inputs={
                "commute_emissions_kg": str(commute_emissions),
                "remote_emissions_kg": str(remote_emissions),
            },
            output=str(total_emissions_kg),
        ))

        emission_factor = EmissionFactorRecord(
            factor_id="commuting_composite",
            factor_value=Decimal("0.171"),  # Average car factor
            factor_unit="kg CO2e/passenger-km",
            source=EmissionFactorSource.DEFRA,
            source_uri="https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024",
            version="2024",
            last_updated="2024-01-01",
            data_quality_tier=DataQualityTier.TIER_2,
        )

        activity_data = {
            "total_employees": total_employees,
            "num_profiles": len(input_data.commute_profiles),
            "commute_emissions_kg": str(commute_emissions),
            "remote_emissions_kg": str(remote_emissions),
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_emissions_kg,
            method=CalculationMethod.DISTANCE_BASED,
            emission_factor=emission_factor,
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
            warnings=warnings,
        )

    def _calculate_average_data(
        self,
        input_data: CommutingInput,
        start_time: datetime,
    ) -> Scope3CalculationResult:
        """
        Calculate emissions using average data method.

        Uses average commute patterns when detailed data unavailable.

        Args:
            input_data: Input data
            start_time: Calculation start time

        Returns:
            Calculation result
        """
        steps: List[CalculationStep] = []
        warnings: List[str] = []

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize average data calculation",
        ))

        # Get employee count
        if input_data.commute_profiles:
            total_employees = sum(p.employees for p in input_data.commute_profiles)
        else:
            total_employees = input_data.total_employees or 0

        # Get average commute distance
        avg_distance = input_data.average_commute_km
        if avg_distance is None:
            avg_distance = self._avg_distances.get(
                input_data.region, self._avg_distances["default"]
            )
            warnings.append(f"Using default average commute distance: {avg_distance} km")

        # Use average commute factor (weighted by typical mode split)
        # Typical US mode split: 76% drive alone, 9% carpool, 5% transit, 3% walk, 3% WFH
        avg_factor = Decimal("0.145")  # Weighted average

        # Calculate annual commute km per employee
        # Round trip x working days
        annual_km_per_employee = avg_distance * 2 * Decimal(str(input_data.working_days_per_year))

        # Apply remote work adjustment
        if input_data.remote_work_percentage:
            in_office_pct = (100 - input_data.remote_work_percentage) / 100
            annual_km_per_employee = annual_km_per_employee * Decimal(str(in_office_pct))

        commute_emissions = (
            Decimal(str(total_employees)) * annual_km_per_employee * avg_factor
        ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        steps.append(CalculationStep(
            step_number=2,
            description="Calculate commute emissions using average factors",
            formula="emissions = employees x annual_km x avg_factor",
            inputs={
                "employees": total_employees,
                "avg_distance_km": str(avg_distance),
                "working_days": input_data.working_days_per_year,
                "annual_km_per_employee": str(annual_km_per_employee),
                "avg_factor": str(avg_factor),
            },
            output=str(commute_emissions),
        ))

        # Add remote work emissions
        remote_emissions = Decimal("0")
        if input_data.include_remote_work_emissions and input_data.remote_work_percentage:
            remote_employees = int(
                total_employees * float(input_data.remote_work_percentage) / 100
            )
            remote_factor = self._remote_factors.get(
                input_data.region, self._remote_factors["default"]
            )
            remote_emissions = (
                Decimal(str(remote_employees))
                * Decimal(str(input_data.working_days_per_year))
                * remote_factor
            ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

            steps.append(CalculationStep(
                step_number=3,
                description="Calculate remote work emissions",
                inputs={
                    "remote_employees": remote_employees,
                    "working_days": input_data.working_days_per_year,
                    "remote_factor": str(remote_factor),
                },
                output=str(remote_emissions),
            ))

        total_emissions_kg = commute_emissions + remote_emissions

        emission_factor = EmissionFactorRecord(
            factor_id="commuting_average",
            factor_value=avg_factor,
            factor_unit="kg CO2e/passenger-km",
            source=EmissionFactorSource.GHG_PROTOCOL,
            source_uri="https://ghgprotocol.org/",
            version="2024",
            last_updated="2024-01-01",
            data_quality_tier=DataQualityTier.TIER_3,
        )

        activity_data = {
            "total_employees": total_employees,
            "avg_commute_km": str(avg_distance),
            "working_days_per_year": input_data.working_days_per_year,
            "remote_work_percentage": str(input_data.remote_work_percentage or 0),
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_emissions_kg,
            method=CalculationMethod.AVERAGE_DATA,
            emission_factor=emission_factor,
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
            warnings=warnings,
        )

    def _get_commute_factor(
        self,
        mode: str,
        vehicle_type: Optional[str] = None,
        fuel_type: Optional[str] = None,
    ) -> Decimal:
        """
        Get commute emission factor.

        Args:
            mode: Commute mode
            vehicle_type: Vehicle type for cars
            fuel_type: Fuel type for cars

        Returns:
            Emission factor in kg CO2e/passenger-km
        """
        # Try fuel-specific car factor first
        if mode.startswith("car") and fuel_type:
            fuel_key = f"car_{fuel_type.lower()}"
            if fuel_key in self._commute_factors:
                return self._commute_factors[fuel_key]["factor"]

        # Try mode
        if mode in self._commute_factors:
            return self._commute_factors[mode]["factor"]

        # Default to average car
        self.logger.warning(f"No factor for mode '{mode}', using car average")
        return self._commute_factors["car_average"]["factor"]
