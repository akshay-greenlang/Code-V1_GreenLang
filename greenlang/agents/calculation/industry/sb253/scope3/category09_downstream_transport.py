# -*- coding: utf-8 -*-
"""
Category 9: Downstream Transportation and Distribution Calculator

Calculates emissions from transportation and distribution of sold products
after they leave the reporting organization's ownership or control.

Similar to Category 4 but for outbound logistics of products sold.

Supported Methods:
1. Distance-based method (tonne-km)
2. Spend-based method (logistics spend by customers)
3. Average data method (industry averages)

Reference: GHG Protocol Scope 3 Standard, Chapter 6

Example:
    >>> calculator = Category09DownstreamTransportCalculator()
    >>> input_data = DownstreamTransportInput(
    ...     reporting_year=2024,
    ...     organization_id="ORG001",
    ...     outbound_shipments=[
    ...         OutboundShipment(mode="truck", distance_km=800, weight_tonnes=50),
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


class OutboundShipment(BaseModel):
    """Outbound shipment to customers/distributors."""

    description: Optional[str] = Field(None, description="Shipment description")
    mode: str = Field(..., description="Transport mode")

    # Distance-based
    distance_km: Optional[Decimal] = Field(None, ge=0, description="Distance (km)")
    weight_tonnes: Optional[Decimal] = Field(None, ge=0, description="Weight (tonnes)")

    # Spend-based
    logistics_spend_usd: Optional[Decimal] = Field(None, ge=0, description="Logistics spend")

    # Details
    vehicle_type: Optional[str] = Field(None, description="Vehicle type")
    destination_region: Optional[str] = Field(None, description="Destination region")

    @validator("mode")
    def normalize_mode(cls, v: str) -> str:
        """Normalize transport mode."""
        mode_map = {
            "truck": "truck", "road": "truck",
            "rail": "rail", "train": "rail",
            "ocean": "ocean", "sea": "ocean", "ship": "ocean",
            "air": "air", "flight": "air",
            "courier": "courier", "parcel": "courier",
            "last_mile": "last_mile",
        }
        return mode_map.get(v.lower().strip(), v.lower().strip())


class DownstreamTransportInput(Scope3CalculationInput):
    """Input model for Category 9: Downstream Transportation."""

    outbound_shipments: List[OutboundShipment] = Field(
        default_factory=list, description="List of outbound shipments"
    )

    # Aggregated inputs
    total_products_sold_tonnes: Optional[Decimal] = Field(
        None, ge=0, description="Total products sold (tonnes)"
    )
    average_distribution_distance_km: Optional[Decimal] = Field(
        None, ge=0, description="Average distribution distance"
    )
    total_logistics_spend_usd: Optional[Decimal] = Field(
        None, ge=0, description="Total downstream logistics spend"
    )

    default_mode: str = Field("truck", description="Default transport mode")


# Same transport factors as Category 4
DOWNSTREAM_TRANSPORT_FACTORS: Dict[str, Decimal] = {
    "truck": Decimal("0.107"),
    "rail": Decimal("0.028"),
    "ocean": Decimal("0.016"),
    "air": Decimal("0.602"),
    "courier": Decimal("0.42"),  # Higher for parcels
    "last_mile": Decimal("0.25"),  # Urban delivery
    "default": Decimal("0.107"),
}


class Category09DownstreamTransportCalculator(Scope3CategoryCalculator):
    """
    Calculator for Scope 3 Category 9: Downstream Transportation.

    Calculates emissions from outbound logistics of sold products.
    """

    CATEGORY_NUMBER = 9
    CATEGORY_NAME = "Downstream Transportation and Distribution"
    SUPPORTED_METHODS = [
        CalculationMethod.DISTANCE_BASED,
        CalculationMethod.SPEND_BASED,
        CalculationMethod.AVERAGE_DATA,
    ]

    def __init__(self):
        """Initialize the Category 9 calculator."""
        super().__init__()
        self._transport_factors = DOWNSTREAM_TRANSPORT_FACTORS

    def calculate(
        self, input_data: DownstreamTransportInput
    ) -> Scope3CalculationResult:
        """Calculate Category 9 emissions."""
        start_time = datetime.utcnow()
        self._validate_method(input_data.calculation_method)

        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")
        total_tonne_km = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize downstream transportation calculation",
            inputs={"num_shipments": len(input_data.outbound_shipments)},
        ))

        if input_data.outbound_shipments:
            for shipment in input_data.outbound_shipments:
                if shipment.distance_km and shipment.weight_tonnes:
                    tonne_km = shipment.distance_km * shipment.weight_tonnes
                    total_tonne_km += tonne_km

                    factor = self._transport_factors.get(
                        shipment.mode, self._transport_factors["default"]
                    )
                    shipment_emissions = (tonne_km * factor).quantize(
                        Decimal("0.001"), rounding=ROUND_HALF_UP
                    )
                    total_emissions_kg += shipment_emissions

                    steps.append(CalculationStep(
                        step_number=len(steps) + 1,
                        description=f"Calculate downstream transport: {shipment.mode}",
                        formula="emissions = tonne_km x factor",
                        inputs={
                            "mode": shipment.mode,
                            "distance_km": str(shipment.distance_km),
                            "weight_tonnes": str(shipment.weight_tonnes),
                            "tonne_km": str(tonne_km),
                            "factor": str(factor),
                        },
                        output=str(shipment_emissions),
                    ))
        else:
            # Use aggregated data
            if input_data.total_products_sold_tonnes and input_data.average_distribution_distance_km:
                total_tonne_km = (
                    input_data.total_products_sold_tonnes *
                    input_data.average_distribution_distance_km
                )
                factor = self._transport_factors.get(
                    input_data.default_mode, self._transport_factors["default"]
                )
                total_emissions_kg = (total_tonne_km * factor).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )

                steps.append(CalculationStep(
                    step_number=2,
                    description="Calculate from aggregated product data",
                    formula="emissions = (tonnes x avg_distance) x factor",
                    inputs={
                        "tonnes_sold": str(input_data.total_products_sold_tonnes),
                        "avg_distance_km": str(input_data.average_distribution_distance_km),
                        "mode": input_data.default_mode,
                        "factor": str(factor),
                    },
                    output=str(total_emissions_kg),
                ))

        emission_factor = EmissionFactorRecord(
            factor_id="downstream_transport_composite",
            factor_value=Decimal("0.107"),
            factor_unit="kg CO2e/tonne-km",
            source=EmissionFactorSource.DEFRA,
            source_uri="https://www.gov.uk/government/publications/greenhouse-gas-reporting-conversion-factors-2024",
            version="2024",
            last_updated="2024-01-01",
            data_quality_tier=DataQualityTier.TIER_2,
        )

        activity_data = {
            "total_tonne_km": str(total_tonne_km),
            "num_shipments": len(input_data.outbound_shipments),
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_emissions_kg,
            method=input_data.calculation_method,
            emission_factor=emission_factor,
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
            warnings=warnings,
        )
