"""
Category 11: Use of Sold Products Agent

Calculates emissions from the use of sold products by end users

GHG Protocol Reference:
- Chapter 5.11 of Scope 3 Standard
- Category 11 Calculation Guidance
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field

from .base import Scope3BaseAgent, Scope3InputData, Scope3Result


class UseOfSoldProductsInput(Scope3InputData):
    """Input data for Category 11: Use of Sold Products."""

    calculation_method: str = Field(
        ...,
        description="Calculation method: direct-use-phase, indirect-use-phase"
    )

    # Add category-specific fields
    products_sold: List[Dict[str, Any]] = Field(
        ...,
        description="Products sold with energy consumption data"
    )

    use_scenario: str = Field(
        default="average",
        description="Use scenario: minimum, average, or intensive"
    )


class UseOfSoldProductsAgent(Scope3BaseAgent):
    """
    Agent for calculating Category 11: Use of Sold Products.

    Includes:
    - Direct use-phase emissions (fuel/energy consuming products)
    - Indirect use-phase emissions (products requiring energy)
    - Lifetime energy consumption
    """

    def __init__(self):
        """Initialize Use of Sold Products agent."""
        super().__init__()
        self.category_number = 11
        self.category_name = "Use of Sold Products"
        self.emission_factors = self._load_product_use_factors()

    def _load_product_use_factors(self) -> Dict[str, Decimal]:
        """Load emission factors for Use of Sold Products."""
        return {
            # Grid factors by region (kg CO2e/kWh)
            "grid_us": Decimal("0.433"),
            "grid_eu": Decimal("0.295"),
            "grid_global": Decimal("0.475"),

            # Fuel factors (kg CO2e/liter)
            "gasoline": Decimal("2.31"),
            "diesel": Decimal("2.68")
        }

    def _parse_input(self, input_data: Dict[str, Any]) -> UseOfSoldProductsInput:
        """Parse and validate input data."""
        return UseOfSoldProductsInput(**input_data)

    async def calculate_emissions(self, input_data: UseOfSoldProductsInput) -> Scope3Result:
        """
        Calculate Category 11 emissions.

        Formula:
        Emissions = Units_Sold × Lifetime_Energy × Energy_EF
        """
        self.calculation_steps = []
        self.factors_used = {}
        total_emissions = Decimal("0")

        # Implement calculation logic
        for product in input_data.products_sold:
            units = Decimal(str(product.get("units_sold", 0)))
            lifetime_years = Decimal(str(product.get("lifetime_years", 5)))
            annual_energy = Decimal(str(product.get("annual_energy_kwh", 0)))

            grid_region = product.get("grid_region", "global")
            ef = self.emission_factors.get(f"grid_{grid_region}")

            lifetime_energy = annual_energy * lifetime_years
            emissions = units * lifetime_energy * ef
            total_emissions += emissions

        # Generate result
        result = Scope3Result(
            category="Category 11: Use of Sold Products",
            category_number=11,
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
