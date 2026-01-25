# -*- coding: utf-8 -*-
"""
Category 4: Upstream Transportation and Distribution Calculator

Calculates emissions from transportation and distribution of products
purchased by the reporting organization, including:

1. Inbound logistics (raw materials, components)
2. Third-party transportation services
3. Distribution of products between own facilities

Supported Methods:
1. Distance-based method (tonne-km)
2. Spend-based method (transportation spend)
3. Fuel-based method (actual fuel consumption)

Reference: GHG Protocol Scope 3 Standard, Chapter 6

Example:
    >>> calculator = Category04UpstreamTransportCalculator()
    >>> input_data = UpstreamTransportInput(
    ...     reporting_year=2024,
    ...     organization_id="ORG001",
    ...     shipments=[
    ...         Shipment(mode="truck", distance_km=500, weight_tonnes=10),
    ...         Shipment(mode="ocean", distance_km=15000, weight_tonnes=100),
    ...     ]
    ... )
    >>> result = calculator.calculate(input_data)
"""

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator

from greenlang.calculators.sb253.scope3.base import (
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


class Shipment(BaseModel):
    """Individual shipment record for transportation."""

    # Basic shipment info
    description: Optional[str] = Field(None, description="Shipment description")
    mode: str = Field(..., description="Transport mode (truck, rail, ocean, air)")

    # Distance-based inputs
    distance_km: Optional[Decimal] = Field(None, ge=0, description="Distance in km")
    weight_tonnes: Optional[Decimal] = Field(None, ge=0, description="Weight in metric tonnes")

    # Spend-based inputs
    transport_spend_usd: Optional[Decimal] = Field(None, ge=0, description="Transport spend in USD")

    # Fuel-based inputs
    fuel_liters: Optional[Decimal] = Field(None, ge=0, description="Fuel consumed in liters")
    fuel_type: Optional[str] = Field(None, description="Fuel type")

    # Additional details
    vehicle_type: Optional[str] = Field(None, description="Specific vehicle type")
    load_factor: Optional[Decimal] = Field(
        None, ge=0, le=1, description="Load factor (0-1)"
    )
    empty_running_pct: Optional[Decimal] = Field(
        None, ge=0, le=100, description="Percentage of empty running"
    )
    supplier_name: Optional[str] = Field(None, description="Logistics provider")
    origin: Optional[str] = Field(None, description="Origin location")
    destination: Optional[str] = Field(None, description="Destination location")

    @validator("mode")
    def normalize_mode(cls, v: str) -> str:
        """Normalize transport mode."""
        mode_map = {
            "truck": "truck",
            "road": "truck",
            "lorry": "truck",
            "hgv": "truck",
            "rail": "rail",
            "train": "rail",
            "ocean": "ocean",
            "sea": "ocean",
            "ship": "ocean",
            "container": "ocean",
            "air": "air",
            "flight": "air",
            "plane": "air",
            "barge": "barge",
            "inland_waterway": "barge",
        }
        normalized = v.lower().strip()
        return mode_map.get(normalized, normalized)


class UpstreamTransportInput(Scope3CalculationInput):
    """Input model for Category 4: Upstream Transportation."""

    # Shipment data
    shipments: List[Shipment] = Field(
        default_factory=list, description="List of shipments"
    )

    # Aggregated inputs (alternative to shipments)
    total_tonne_km: Optional[Decimal] = Field(
        None, ge=0, description="Total tonne-km (all modes)"
    )
    total_transport_spend_usd: Optional[Decimal] = Field(
        None, ge=0, description="Total transport spend in USD"
    )
    default_mode: str = Field("truck", description="Default transport mode")

    # Configuration
    include_empty_running: bool = Field(
        True, description="Include empty return trips"
    )
    default_empty_running_pct: Decimal = Field(
        Decimal("30"), ge=0, le=100, description="Default empty running percentage"
    )


# Transport emission factors (kg CO2e per tonne-km)
# Source: DEFRA 2024, EPA SmartWay
TRANSPORT_FACTORS: Dict[str, Dict[str, Decimal]] = {
    # Road transport
    "truck": {
        "factor": Decimal("0.107"),  # Average articulated truck
        "unit": "tonne-km",
        "description": "Average HGV articulated",
    },
    "truck_small": {
        "factor": Decimal("0.295"),
        "unit": "tonne-km",
        "description": "Small rigid truck (<7.5t)",
    },
    "truck_medium": {
        "factor": Decimal("0.178"),
        "unit": "tonne-km",
        "description": "Medium rigid truck (7.5-17t)",
    },
    "truck_large": {
        "factor": Decimal("0.107"),
        "unit": "tonne-km",
        "description": "Large articulated (>33t)",
    },

    # Rail transport
    "rail": {
        "factor": Decimal("0.028"),
        "unit": "tonne-km",
        "description": "Freight rail",
    },
    "rail_electric": {
        "factor": Decimal("0.018"),
        "unit": "tonne-km",
        "description": "Electric rail",
    },
    "rail_diesel": {
        "factor": Decimal("0.032"),
        "unit": "tonne-km",
        "description": "Diesel rail",
    },

    # Maritime transport
    "ocean": {
        "factor": Decimal("0.016"),
        "unit": "tonne-km",
        "description": "Container ship average",
    },
    "ocean_container": {
        "factor": Decimal("0.016"),
        "unit": "tonne-km",
        "description": "Container ship",
    },
    "ocean_bulk": {
        "factor": Decimal("0.005"),
        "unit": "tonne-km",
        "description": "Bulk carrier",
    },
    "ocean_tanker": {
        "factor": Decimal("0.008"),
        "unit": "tonne-km",
        "description": "Tanker",
    },

    # Air transport
    "air": {
        "factor": Decimal("0.602"),
        "unit": "tonne-km",
        "description": "Air freight average",
    },
    "air_short_haul": {
        "factor": Decimal("1.128"),
        "unit": "tonne-km",
        "description": "Air freight <1000km",
    },
    "air_long_haul": {
        "factor": Decimal("0.459"),
        "unit": "tonne-km",
        "description": "Air freight >3700km",
    },

    # Inland waterway
    "barge": {
        "factor": Decimal("0.032"),
        "unit": "tonne-km",
        "description": "Inland barge",
    },
}

# Spend-based transport factors (kg CO2e per USD)
TRANSPORT_SPEND_FACTORS: Dict[str, Decimal] = {
    "truck": Decimal("0.45"),
    "rail": Decimal("0.12"),
    "ocean": Decimal("0.08"),
    "air": Decimal("0.85"),
    "barge": Decimal("0.15"),
    "default": Decimal("0.35"),
}


class Category04UpstreamTransportCalculator(Scope3CategoryCalculator):
    """
    Calculator for Scope 3 Category 4: Upstream Transportation.

    Calculates emissions from inbound logistics and transportation
    of purchased goods using multiple methodologies.

    Attributes:
        CATEGORY_NUMBER: 4
        CATEGORY_NAME: "Upstream Transportation and Distribution"

    Example:
        >>> calculator = Category04UpstreamTransportCalculator()
        >>> result = calculator.calculate(input_data)
    """

    CATEGORY_NUMBER = 4
    CATEGORY_NAME = "Upstream Transportation and Distribution"
    SUPPORTED_METHODS = [
        CalculationMethod.DISTANCE_BASED,
        CalculationMethod.SPEND_BASED,
        CalculationMethod.ACTIVITY_BASED,
    ]

    def __init__(self):
        """Initialize the Category 4 calculator."""
        super().__init__()
        self._transport_factors = TRANSPORT_FACTORS
        self._spend_factors = TRANSPORT_SPEND_FACTORS

    def calculate(
        self, input_data: UpstreamTransportInput
    ) -> Scope3CalculationResult:
        """
        Calculate Category 4 emissions.

        Args:
            input_data: Transportation input data

        Returns:
            Complete calculation result with audit trail
        """
        start_time = datetime.utcnow()
        self._validate_method(input_data.calculation_method)

        if input_data.calculation_method == CalculationMethod.SPEND_BASED:
            return self._calculate_spend_based(input_data, start_time)
        else:
            return self._calculate_distance_based(input_data, start_time)

    def _calculate_distance_based(
        self,
        input_data: UpstreamTransportInput,
        start_time: datetime,
    ) -> Scope3CalculationResult:
        """
        Calculate emissions using distance-based method.

        Formula: Emissions = SUM(Weight x Distance x EF_mode)

        Args:
            input_data: Input data
            start_time: Calculation start time

        Returns:
            Calculation result
        """
        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")
        total_tonne_km = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize distance-based transport calculation",
            inputs={
                "num_shipments": len(input_data.shipments),
                "include_empty_running": input_data.include_empty_running,
            },
        ))

        if input_data.shipments:
            for idx, shipment in enumerate(input_data.shipments):
                if shipment.distance_km and shipment.weight_tonnes:
                    # Calculate tonne-km
                    tonne_km = shipment.distance_km * shipment.weight_tonnes

                    # Adjust for empty running
                    if input_data.include_empty_running:
                        empty_pct = (
                            shipment.empty_running_pct
                            or input_data.default_empty_running_pct
                        ) / 100
                        empty_adjustment = Decimal("1") + empty_pct
                    else:
                        empty_adjustment = Decimal("1")

                    adjusted_tonne_km = tonne_km * empty_adjustment
                    total_tonne_km += adjusted_tonne_km

                    # Get emission factor for mode
                    factor = self._get_transport_factor(
                        shipment.mode, shipment.vehicle_type
                    )

                    shipment_emissions = (adjusted_tonne_km * factor).quantize(
                        Decimal("0.001"), rounding=ROUND_HALF_UP
                    )
                    total_emissions_kg += shipment_emissions

                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Calculate emissions for shipment: {shipment.description or shipment.mode}",
                        formula="emissions = tonne_km x empty_adjustment x factor",
                        inputs={
                            "mode": shipment.mode,
                            "distance_km": str(shipment.distance_km),
                            "weight_tonnes": str(shipment.weight_tonnes),
                            "tonne_km": str(tonne_km),
                            "empty_adjustment": str(empty_adjustment),
                            "factor": str(factor),
                        },
                        output=str(shipment_emissions),
                    ))
                else:
                    warnings.append(
                        f"Shipment missing distance or weight: {shipment.description}"
                    )
        elif input_data.total_tonne_km:
            # Use aggregated tonne-km
            factor = self._get_transport_factor(input_data.default_mode)
            total_emissions_kg = (input_data.total_tonne_km * factor).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            total_tonne_km = input_data.total_tonne_km

            steps.append(CalculationStep(
                step_number=2,
                description="Calculate from aggregated tonne-km",
                formula="emissions = total_tonne_km x factor",
                inputs={
                    "total_tonne_km": str(input_data.total_tonne_km),
                    "mode": input_data.default_mode,
                    "factor": str(factor),
                },
                output=str(total_emissions_kg),
            ))

        emission_factor = EmissionFactorRecord(
            factor_id="transport_distance_composite",
            factor_value=Decimal("0.107"),  # Average truck factor
            factor_unit="kg CO2e/tonne-km",
            source=EmissionFactorSource.DEFRA,
            source_uri="https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024",
            version="2024",
            last_updated="2024-01-01",
            data_quality_tier=DataQualityTier.TIER_2,
        )

        activity_data = {
            "total_tonne_km": str(total_tonne_km),
            "num_shipments": len(input_data.shipments),
            "modes_used": list(set(s.mode for s in input_data.shipments)),
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

    def _calculate_spend_based(
        self,
        input_data: UpstreamTransportInput,
        start_time: datetime,
    ) -> Scope3CalculationResult:
        """
        Calculate emissions using spend-based method.

        Formula: Emissions = Transport_Spend x EF_spend

        Args:
            input_data: Input data
            start_time: Calculation start time

        Returns:
            Calculation result
        """
        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")
        total_spend = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize spend-based transport calculation",
        ))

        if input_data.shipments:
            for idx, shipment in enumerate(input_data.shipments):
                if shipment.transport_spend_usd:
                    spend_factor = self._spend_factors.get(
                        shipment.mode, self._spend_factors["default"]
                    )
                    shipment_emissions = (
                        shipment.transport_spend_usd * spend_factor
                    ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
                    total_emissions_kg += shipment_emissions
                    total_spend += shipment.transport_spend_usd

                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Calculate from spend: {shipment.mode}",
                        formula="emissions = spend x factor",
                        inputs={
                            "spend_usd": str(shipment.transport_spend_usd),
                            "mode": shipment.mode,
                            "factor": str(spend_factor),
                        },
                        output=str(shipment_emissions),
                    ))
        elif input_data.total_transport_spend_usd:
            spend_factor = self._spend_factors.get(
                input_data.default_mode, self._spend_factors["default"]
            )
            total_emissions_kg = (
                input_data.total_transport_spend_usd * spend_factor
            ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            total_spend = input_data.total_transport_spend_usd

            steps.append(CalculationStep(
                step_number=2,
                description="Calculate from total transport spend",
                formula="emissions = total_spend x factor",
                inputs={
                    "total_spend_usd": str(total_spend),
                    "factor": str(spend_factor),
                },
                output=str(total_emissions_kg),
            ))

        emission_factor = EmissionFactorRecord(
            factor_id="transport_spend_average",
            factor_value=Decimal("0.35"),
            factor_unit="kg CO2e/USD",
            source=EmissionFactorSource.EPA_EEIO,
            source_uri="https://cfpub.epa.gov/si/si_public_record_Report.cfm?dirEntryId=349324",
            version="2023",
            last_updated="2023-01-01",
            data_quality_tier=DataQualityTier.TIER_3,
        )

        activity_data = {
            "total_transport_spend_usd": str(total_spend),
            "num_shipments": len(input_data.shipments),
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_emissions_kg,
            method=CalculationMethod.SPEND_BASED,
            emission_factor=emission_factor,
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
            warnings=warnings,
        )

    def _get_transport_factor(
        self,
        mode: str,
        vehicle_type: Optional[str] = None,
    ) -> Decimal:
        """
        Get emission factor for transport mode.

        Args:
            mode: Transport mode
            vehicle_type: Specific vehicle type

        Returns:
            Emission factor in kg CO2e/tonne-km
        """
        # Try vehicle-specific factor first
        if vehicle_type:
            key = f"{mode}_{vehicle_type}".lower()
            if key in self._transport_factors:
                return self._transport_factors[key]["factor"]

        # Fall back to mode average
        if mode in self._transport_factors:
            return self._transport_factors[mode]["factor"]

        # Default to truck
        self.logger.warning(f"No factor found for mode '{mode}', using truck average")
        return self._transport_factors["truck"]["factor"]
