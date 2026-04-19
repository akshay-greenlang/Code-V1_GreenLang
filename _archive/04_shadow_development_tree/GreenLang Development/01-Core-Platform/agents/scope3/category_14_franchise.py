"""
Category 14: Franchises Agent

Calculates emissions from franchise operations

GHG Protocol Reference:
- Chapter 5.14 of Scope 3 Standard
- Category 14 Calculation Guidance
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field

from .base import Scope3BaseAgent, Scope3InputData, Scope3Result


class FranchisesInput(Scope3InputData):
    """Input data for Category 14: Franchises."""

    calculation_method: str = Field(
        ...,
        description="Calculation method: franchise-specific, average-data"
    )

    # Add category-specific fields
    franchises: List[Dict[str, Any]] = Field(
        ...,
        description="Franchise locations with energy data"
    )

    franchise_count: Optional[int] = Field(
        default=None,
        description="Number of franchises (for average-data method)"
    )


class FranchisesAgent(Scope3BaseAgent):
    """
    Agent for calculating Category 14: Franchises.

    Includes:
    - Scope 1 and 2 emissions of franchises
    - Energy use at franchise locations
    """

    def __init__(self):
        """Initialize Franchises agent."""
        super().__init__()
        self.category_number = 14
        self.category_name = "Franchises"
        self.emission_factors = self._load_franchise_factors()

    def _load_franchise_factors(self) -> Dict[str, Decimal]:
        """Load emission factors for Franchises."""
        return {
            # Average franchise emissions (kg CO2e/year)
            "restaurant": Decimal("150000"),
            "retail": Decimal("75000"),
            "service": Decimal("25000")
        }

    def _parse_input(self, input_data: Dict[str, Any]) -> FranchisesInput:
        """Parse and validate input data."""
        return FranchisesInput(**input_data)

    async def calculate_emissions(self, input_data: FranchisesInput) -> Scope3Result:
        """
        Calculate Category 14 emissions.

        Formula:
        Emissions = Sum(Franchise_Energy × EF) + Sum(Franchise_Fuel × Fuel_EF)
        """
        self.calculation_steps = []
        self.factors_used = {}
        total_emissions = Decimal("0")

        # Implement calculation logic
        if input_data.calculation_method == "franchise-specific":
            for franchise in input_data.franchises:
                energy = Decimal(str(franchise.get("energy_kwh", 0)))
                fuel = Decimal(str(franchise.get("fuel_liters", 0)))

                energy_emissions = energy * self.emission_factors.get("electricity_grid")
                fuel_emissions = fuel * Decimal("2.68")  # Diesel factor

                total_emissions += energy_emissions + fuel_emissions
        else:
            # Average-data method
            franchise_type = input_data.franchises[0].get("type", "service")
            count = Decimal(str(input_data.franchise_count or len(input_data.franchises)))

            avg_emissions = self.emission_factors.get(franchise_type)
            total_emissions = count * avg_emissions

        # Generate result
        result = Scope3Result(
            category="Category 14: Franchises",
            category_number=14,
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
