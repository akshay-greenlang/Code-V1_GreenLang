# -*- coding: utf-8 -*-
"""
Category 5: Waste Generated in Operations Calculator

Calculates emissions from third-party disposal and treatment of waste
generated in the reporting organization's operations.

Includes:
1. Solid waste (landfill, incineration, recycling, composting)
2. Wastewater treatment
3. Hazardous waste treatment

Supported Methods:
1. Waste-type-specific method (weight x factor)
2. Average data method (total waste x average factor)
3. Spend-based method (waste management spend)

Reference: GHG Protocol Scope 3 Standard, Chapter 6

Example:
    >>> calculator = Category05WasteCalculator()
    >>> input_data = WasteInput(
    ...     reporting_year=2024,
    ...     organization_id="ORG001",
    ...     waste_streams=[
    ...         WasteStream(waste_type="mixed_municipal", weight_tonnes=100, treatment="landfill"),
    ...         WasteStream(waste_type="paper", weight_tonnes=50, treatment="recycling"),
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


class WasteStream(BaseModel):
    """Individual waste stream for disposal."""

    waste_type: str = Field(..., description="Type of waste")
    weight_tonnes: Optional[Decimal] = Field(None, ge=0, description="Weight in metric tonnes")
    weight_kg: Optional[Decimal] = Field(None, ge=0, description="Weight in kg")
    treatment: str = Field("landfill", description="Treatment method")
    description: Optional[str] = Field(None, description="Description")
    disposal_spend_usd: Optional[Decimal] = Field(None, ge=0, description="Disposal spend")

    @validator("waste_type")
    def normalize_waste_type(cls, v: str) -> str:
        """Normalize waste type name."""
        return v.lower().strip().replace(" ", "_")

    @validator("treatment")
    def normalize_treatment(cls, v: str) -> str:
        """Normalize treatment method."""
        treatment_map = {
            "landfill": "landfill",
            "landfilled": "landfill",
            "incineration": "incineration",
            "incinerated": "incineration",
            "combustion": "incineration",
            "recycling": "recycling",
            "recycled": "recycling",
            "composting": "composting",
            "composted": "composting",
            "anaerobic_digestion": "anaerobic_digestion",
            "ad": "anaerobic_digestion",
            "wastewater": "wastewater_treatment",
        }
        normalized = v.lower().strip()
        return treatment_map.get(normalized, normalized)

    def get_weight_tonnes(self) -> Decimal:
        """Get weight in tonnes."""
        if self.weight_tonnes:
            return self.weight_tonnes
        if self.weight_kg:
            return self.weight_kg / Decimal("1000")
        return Decimal("0")


class WastewaterData(BaseModel):
    """Wastewater treatment data."""

    volume_m3: Decimal = Field(..., ge=0, description="Volume in cubic meters")
    cod_kg: Optional[Decimal] = Field(None, ge=0, description="Chemical oxygen demand (kg)")
    bod_kg: Optional[Decimal] = Field(None, ge=0, description="Biological oxygen demand (kg)")
    treatment_type: str = Field("aerobic", description="Treatment type")
    industry_type: Optional[str] = Field(None, description="Industry type for default COD")


class WasteInput(Scope3CalculationInput):
    """Input model for Category 5: Waste Generated in Operations."""

    # Solid waste data
    waste_streams: List[WasteStream] = Field(
        default_factory=list, description="List of waste streams"
    )

    # Aggregated waste (alternative)
    total_waste_tonnes: Optional[Decimal] = Field(
        None, ge=0, description="Total waste in tonnes"
    )
    default_treatment: str = Field("landfill", description="Default treatment method")
    default_waste_type: str = Field("mixed_municipal", description="Default waste type")

    # Wastewater data
    wastewater: Optional[WastewaterData] = Field(None, description="Wastewater data")

    # Configuration
    include_recycling_credit: bool = Field(
        False, description="Include emission credits for recycling"
    )


# Waste emission factors (kg CO2e per tonne)
# Source: EPA WARM Model, DEFRA 2024
WASTE_FACTORS: Dict[str, Dict[str, Decimal]] = {
    # Municipal solid waste by treatment
    "mixed_municipal": {
        "landfill": Decimal("587"),
        "incineration": Decimal("21.3"),
        "recycling": Decimal("-182"),  # Credit for avoided virgin production
        "composting": Decimal("36"),
    },
    # Paper and cardboard
    "paper": {
        "landfill": Decimal("1095"),  # High methane potential
        "incineration": Decimal("21.3"),
        "recycling": Decimal("-680"),
        "composting": Decimal("36"),
    },
    "cardboard": {
        "landfill": Decimal("729"),
        "incineration": Decimal("21.3"),
        "recycling": Decimal("-520"),
        "composting": Decimal("36"),
    },
    # Plastics
    "plastic": {
        "landfill": Decimal("21.3"),  # Low decomposition
        "incineration": Decimal("2760"),  # High CO2 from combustion
        "recycling": Decimal("-1440"),
    },
    "pet": {
        "landfill": Decimal("21.3"),
        "incineration": Decimal("2130"),
        "recycling": Decimal("-1530"),
    },
    "hdpe": {
        "landfill": Decimal("21.3"),
        "incineration": Decimal("2760"),
        "recycling": Decimal("-1440"),
    },
    # Metals
    "metal_aluminum": {
        "landfill": Decimal("21.3"),
        "recycling": Decimal("-9120"),  # High savings from recycling
    },
    "metal_steel": {
        "landfill": Decimal("21.3"),
        "recycling": Decimal("-1820"),
    },
    # Glass
    "glass": {
        "landfill": Decimal("21.3"),
        "recycling": Decimal("-315"),
    },
    # Organic waste
    "food_waste": {
        "landfill": Decimal("1824"),  # Very high methane
        "composting": Decimal("36"),
        "anaerobic_digestion": Decimal("-72"),  # Biogas capture credit
    },
    "yard_waste": {
        "landfill": Decimal("421"),
        "composting": Decimal("36"),
    },
    # Construction & demolition
    "concrete": {
        "landfill": Decimal("21.3"),
        "recycling": Decimal("-8"),
    },
    "wood": {
        "landfill": Decimal("729"),
        "recycling": Decimal("-516"),
        "composting": Decimal("36"),
    },
    # Electronic waste
    "e_waste": {
        "landfill": Decimal("21.3"),
        "recycling": Decimal("-2500"),  # Recoverable metals
    },
    # Textile waste
    "textiles": {
        "landfill": Decimal("21.3"),
        "incineration": Decimal("2760"),
        "recycling": Decimal("-2130"),
    },
    # Default
    "default": {
        "landfill": Decimal("450"),
        "incineration": Decimal("21.3"),
        "recycling": Decimal("-200"),
        "composting": Decimal("36"),
    },
}

# Wastewater emission factors
WASTEWATER_FACTORS: Dict[str, Decimal] = {
    "aerobic": Decimal("0.22"),  # kg CO2e per kg COD
    "anaerobic": Decimal("2.0"),  # Higher methane from anaerobic
    "lagoon": Decimal("1.5"),
    "septic": Decimal("0.5"),
}


class Category05WasteCalculator(Scope3CategoryCalculator):
    """
    Calculator for Scope 3 Category 5: Waste Generated in Operations.

    Calculates emissions from third-party waste disposal and treatment.

    Attributes:
        CATEGORY_NUMBER: 5
        CATEGORY_NAME: "Waste Generated in Operations"

    Example:
        >>> calculator = Category05WasteCalculator()
        >>> result = calculator.calculate(input_data)
    """

    CATEGORY_NUMBER = 5
    CATEGORY_NAME = "Waste Generated in Operations"
    SUPPORTED_METHODS = [
        CalculationMethod.ACTIVITY_BASED,
        CalculationMethod.AVERAGE_DATA,
        CalculationMethod.SPEND_BASED,
    ]

    def __init__(self):
        """Initialize the Category 5 calculator."""
        super().__init__()
        self._waste_factors = WASTE_FACTORS
        self._wastewater_factors = WASTEWATER_FACTORS

    def calculate(self, input_data: WasteInput) -> Scope3CalculationResult:
        """
        Calculate Category 5 emissions.

        Args:
            input_data: Waste input data

        Returns:
            Complete calculation result with audit trail
        """
        start_time = datetime.utcnow()

        steps: List[CalculationStep] = []
        warnings: List[str] = []
        total_emissions_kg = Decimal("0")

        steps.append(CalculationStep(
            step_number=1,
            description="Initialize waste emissions calculation",
            inputs={
                "num_waste_streams": len(input_data.waste_streams),
                "has_wastewater": input_data.wastewater is not None,
                "include_recycling_credit": input_data.include_recycling_credit,
            },
        ))

        # Calculate solid waste emissions
        solid_waste_emissions = self._calculate_solid_waste(
            input_data, steps, warnings
        )
        total_emissions_kg += solid_waste_emissions

        # Calculate wastewater emissions
        if input_data.wastewater:
            wastewater_emissions = self._calculate_wastewater(
                input_data.wastewater, steps, warnings
            )
            total_emissions_kg += wastewater_emissions

        # Final summary
        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Sum all waste emissions",
            output=str(total_emissions_kg),
        ))

        emission_factor = EmissionFactorRecord(
            factor_id="waste_composite",
            factor_value=Decimal("450"),  # Average landfill factor
            factor_unit="kg CO2e/tonne",
            source=EmissionFactorSource.EPA_GHG,
            source_uri="https://www.epa.gov/warm",
            version="2024",
            last_updated="2024-01-01",
            data_quality_tier=DataQualityTier.TIER_2,
        )

        total_waste_tonnes = sum(
            ws.get_weight_tonnes() for ws in input_data.waste_streams
        ) or input_data.total_waste_tonnes or Decimal("0")

        activity_data = {
            "total_waste_tonnes": str(total_waste_tonnes),
            "num_waste_streams": len(input_data.waste_streams),
            "has_wastewater": input_data.wastewater is not None,
            "reporting_year": input_data.reporting_year,
        }

        return self._create_result(
            emissions_kg=total_emissions_kg,
            method=CalculationMethod.ACTIVITY_BASED,
            emission_factor=emission_factor,
            activity_data=activity_data,
            steps=steps,
            start_time=start_time,
            warnings=warnings,
        )

    def _calculate_solid_waste(
        self,
        input_data: WasteInput,
        steps: List[CalculationStep],
        warnings: List[str],
    ) -> Decimal:
        """
        Calculate emissions from solid waste disposal.

        Args:
            input_data: Input data
            steps: Calculation steps
            warnings: Warnings list

        Returns:
            Solid waste emissions in kg CO2e
        """
        total_emissions = Decimal("0")

        if input_data.waste_streams:
            for waste in input_data.waste_streams:
                weight_tonnes = waste.get_weight_tonnes()
                if weight_tonnes == 0:
                    continue

                factor = self._get_waste_factor(
                    waste.waste_type,
                    waste.treatment,
                    input_data.include_recycling_credit,
                )

                # Convert tonnes to kg for factor application
                # Factors are per tonne, so multiply tonnes * factor = kg CO2e
                waste_emissions = (weight_tonnes * factor * Decimal("1000")).quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP
                )
                total_emissions += waste_emissions

                steps.append(CalculationStep(
                    step_number=len(steps) + 1,
                    description=f"Calculate emissions for {waste.waste_type} ({waste.treatment})",
                    formula="emissions_kg = tonnes x factor_per_tonne x 1000",
                    inputs={
                        "waste_type": waste.waste_type,
                        "weight_tonnes": str(weight_tonnes),
                        "treatment": waste.treatment,
                        "factor_kg_per_tonne": str(factor),
                    },
                    output=str(waste_emissions),
                ))
        elif input_data.total_waste_tonnes:
            factor = self._get_waste_factor(
                input_data.default_waste_type,
                input_data.default_treatment,
                input_data.include_recycling_credit,
            )
            total_emissions = (
                input_data.total_waste_tonnes * factor * Decimal("1000")
            ).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

            steps.append(CalculationStep(
                step_number=len(steps) + 1,
                description="Calculate from total waste",
                formula="emissions_kg = total_tonnes x factor x 1000",
                inputs={
                    "total_tonnes": str(input_data.total_waste_tonnes),
                    "treatment": input_data.default_treatment,
                    "factor": str(factor),
                },
                output=str(total_emissions),
            ))

        return total_emissions

    def _calculate_wastewater(
        self,
        wastewater: WastewaterData,
        steps: List[CalculationStep],
        warnings: List[str],
    ) -> Decimal:
        """
        Calculate emissions from wastewater treatment.

        Formula: Emissions = COD (kg) x EF_treatment

        Args:
            wastewater: Wastewater data
            steps: Calculation steps
            warnings: Warnings list

        Returns:
            Wastewater emissions in kg CO2e
        """
        # Get COD value (use provided or estimate)
        cod_kg = wastewater.cod_kg
        if not cod_kg:
            # Estimate COD from volume (default 0.5 kg COD/m3 for municipal)
            default_cod_concentration = Decimal("0.5")  # kg/m3
            cod_kg = wastewater.volume_m3 * default_cod_concentration
            warnings.append(
                "COD not provided, estimated from volume using default concentration"
            )

        # Get treatment factor
        treatment_factor = self._wastewater_factors.get(
            wastewater.treatment_type.lower(),
            self._wastewater_factors["aerobic"],
        )

        wastewater_emissions = (cod_kg * treatment_factor * Decimal("1000")).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        steps.append(CalculationStep(
            step_number=len(steps) + 1,
            description="Calculate wastewater treatment emissions",
            formula="emissions = cod_kg x treatment_factor x 1000",
            inputs={
                "volume_m3": str(wastewater.volume_m3),
                "cod_kg": str(cod_kg),
                "treatment_type": wastewater.treatment_type,
                "treatment_factor": str(treatment_factor),
            },
            output=str(wastewater_emissions),
        ))

        return wastewater_emissions

    def _get_waste_factor(
        self,
        waste_type: str,
        treatment: str,
        include_credit: bool = False,
    ) -> Decimal:
        """
        Get waste emission factor.

        Args:
            waste_type: Type of waste
            treatment: Treatment method
            include_credit: Include recycling credits

        Returns:
            Emission factor in kg CO2e per tonne
        """
        # Normalize inputs
        waste_key = waste_type.lower().strip().replace(" ", "_")
        treatment_key = treatment.lower().strip()

        # Get waste type factors
        type_factors = self._waste_factors.get(
            waste_key, self._waste_factors["default"]
        )

        # Get treatment factor
        factor = type_factors.get(treatment_key)

        if factor is None:
            # Try default treatment for this waste type
            factor = type_factors.get("landfill", Decimal("450"))
            self.logger.warning(
                f"No factor for {waste_type}/{treatment}, using landfill default"
            )

        # Handle recycling credits
        if factor < 0 and not include_credit:
            # Return zero if credits not included and factor is negative
            factor = Decimal("0")

        return factor
