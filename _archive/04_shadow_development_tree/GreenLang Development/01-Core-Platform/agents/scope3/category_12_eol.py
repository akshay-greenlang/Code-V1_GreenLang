"""
Category 12: End-of-Life Treatment of Sold Products Agent

Calculates emissions from disposal and treatment of sold products at end of life

GHG Protocol Reference:
- Chapter 5.12 of Scope 3 Standard
- Category 12 Calculation Guidance
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field

from .base import Scope3BaseAgent, Scope3InputData, Scope3Result


class EndOfLifeTreatmentInput(Scope3InputData):
    """Input data for Category 12: End-of-Life Treatment of Sold Products."""

    calculation_method: str = Field(
        ...,
        description="Calculation method: waste-type-specific, average-data"
    )

    # Add category-specific fields
    products_eol: Dict[str, Any] = Field(
        ...,
        description="End-of-life data for sold products"
    )

    material_composition: Dict[str, float] = Field(
        ...,
        description="Material composition of products"
    )

    treatment_scenario: Dict[str, float] = Field(
        ...,
        description="Treatment method distribution"
    )


class EndOfLifeTreatmentAgent(Scope3BaseAgent):
    """
    Agent for calculating Category 12: End-of-Life Treatment of Sold Products.

    Includes:
    - Landfilling
    - Incineration
    - Recycling
    - Material recovery
    """

    def __init__(self):
        """Initialize End-of-Life Treatment of Sold Products agent."""
        super().__init__()
        self.category_number = 12
        self.category_name = "End-of-Life Treatment of Sold Products"
        self.emission_factors = self._load_eol_factors()

    def _load_eol_factors(self) -> Dict[str, Decimal]:
        """Load emission factors for End-of-Life Treatment of Sold Products."""
        return {
            # EOL treatment factors (kg CO2e/tonne)
            "landfill_plastic": Decimal("2.8"),
            "landfill_paper": Decimal("1184"),
            "landfill_metal": Decimal("2.8"),

            "incineration_plastic": Decimal("2766"),
            "incineration_paper": Decimal("15.8"),

            "recycling_plastic": Decimal("-1666"),  # Avoided emissions
            "recycling_paper": Decimal("-3530"),
            "recycling_metal": Decimal("-1811")
        }

    def _parse_input(self, input_data: Dict[str, Any]) -> EndOfLifeTreatmentInput:
        """Parse and validate input data."""
        return EndOfLifeTreatmentInput(**input_data)

    async def calculate_emissions(self, input_data: EndOfLifeTreatmentInput) -> Scope3Result:
        """
        Calculate Category 12 emissions.

        Formula:
        Emissions = Sum(Product_Weight × Material_% × Treatment_% × Treatment_EF)
        """
        self.calculation_steps = []
        self.factors_used = {}
        total_emissions = Decimal("0")

        # Implement calculation logic
        total_weight = Decimal(str(input_data.products_eol.get("total_weight_tonnes", 0)))

        for material, composition in input_data.material_composition.items():
            material_weight = total_weight * Decimal(str(composition))

            for treatment, percentage in input_data.treatment_scenario.items():
                treatment_weight = material_weight * Decimal(str(percentage))

                ef_key = f"{treatment}_{material}"
                ef = self.emission_factors.get(ef_key, Decimal("0"))

                emissions = treatment_weight * ef
                total_emissions += emissions

        # Generate result
        result = Scope3Result(
            category="Category 12: End-of-Life Treatment of Sold Products",
            category_number=12,
            total_emissions_kg_co2e=total_emissions,
            total_emissions_t_co2e=total_emissions / Decimal("1000"),
            calculation_methodology=input_data.calculation_method,
            data_quality_score=self._calculate_data_quality_score(2.5, 2.5, 2.5, 2.5, 2.5),
            calculation_steps=self.calculation_steps,
            emission_factors_used=self.factors_used,
            provenance_hash=self._calculate_provenance_hash(
                input_data.dict(),
                self.calculation_steps,
                total_emissions
            ),
            calculation_timestamp=str(self._get_timestamp()),
            ghg_protocol_compliance=True,
            uncertainty_range=self._estimate_uncertainty(2.5, input_data.calculation_method)
        )

        return result

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()
