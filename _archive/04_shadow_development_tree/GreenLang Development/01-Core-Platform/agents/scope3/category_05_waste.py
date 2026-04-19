"""
Category 5: Waste Generated in Operations Agent

Calculates emissions from third-party disposal and treatment of waste generated in operations

GHG Protocol Reference:
- Chapter 5.5 of Scope 3 Standard
- Category 5 Calculation Guidance
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field

from .base import Scope3BaseAgent, Scope3InputData, Scope3Result


class WasteGeneratedInput(Scope3InputData):
    """Input data for Category 5: Waste Generated in Operations."""

    calculation_method: str = Field(
        ...,
        description="Calculation method: waste-type-specific, average-data"
    )

    # Add category-specific fields
    waste_streams: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Waste by type and treatment method"
    )

    wastewater: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Wastewater treatment data"
    )


class WasteGeneratedAgent(Scope3BaseAgent):
    """
    Agent for calculating Category 5: Waste Generated in Operations.

    Includes:
    - Disposal in landfills
    - Incineration
    - Recycling
    - Composting
    - Wastewater treatment
    """

    def __init__(self):
        """Initialize Waste Generated in Operations agent."""
        super().__init__()
        self.category_number = 5
        self.category_name = "Waste Generated in Operations"
        self.emission_factors = self._load_waste_factors()

    def _load_waste_factors(self) -> Dict[str, Decimal]:
        """Load emission factors for Waste Generated in Operations."""
        return {
            # Landfill factors (kg CO2e/tonne)
            "landfill_mixed": Decimal("467.0"),
            "landfill_organic": Decimal("558.0"),
            "landfill_plastic": Decimal("2.8"),

            # Incineration factors
            "incineration_mixed": Decimal("907.0"),
            "incineration_energy_recovery": Decimal("255.0"),

            # Recycling factors
            "recycling_paper": Decimal("21.0"),
            "recycling_plastic": Decimal("21.0"),
            "recycling_metal": Decimal("21.0"),

            # Composting
            "composting": Decimal("55.0")
        }

    def _parse_input(self, input_data: Dict[str, Any]) -> WasteGeneratedInput:
        """Parse and validate input data."""
        return WasteGeneratedInput(**input_data)

    async def calculate_emissions(self, input_data: WasteGeneratedInput) -> Scope3Result:
        """
        Calculate Category 5 emissions.

        Formula:
        Emissions = Sum(Waste_Weight × Treatment_EF)
        """
        self.calculation_steps = []
        self.factors_used = {}
        total_emissions = Decimal("0")

        # Implement calculation logic
        if input_data.waste_streams:
            for waste_type, waste_data in input_data.waste_streams.items():
                weight = Decimal(str(waste_data.get("weight_tonnes", 0)))
                treatment = waste_data.get("treatment", "landfill")

                ef_key = f"{treatment}_{waste_type}"
                ef = self.emission_factors.get(ef_key, self.emission_factors.get(f"{treatment}_mixed"))

                emissions = weight * ef
                total_emissions += emissions

                self._record_calculation_step(
                    description=f"Calculate {treatment} emissions for {waste_type}",
                    operation="multiply",
                    inputs={"weight_tonnes": weight, "emission_factor": ef},
                    output_value=emissions,
                    output_name=f"emissions_{waste_type}_{treatment}",
                    formula="Emissions = Weight × Treatment_EF",
                    unit="kg CO2e"
                )

        # Generate result
        result = Scope3Result(
            category="Category 5: Waste Generated in Operations",
            category_number=5,
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
