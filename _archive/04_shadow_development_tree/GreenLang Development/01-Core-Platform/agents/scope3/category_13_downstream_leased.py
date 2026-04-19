"""
Category 13: Downstream Leased Assets Agent

Calculates emissions from assets leased to others

GHG Protocol Reference:
- Chapter 5.13 of Scope 3 Standard
- Category 13 Calculation Guidance
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field

from .base import Scope3BaseAgent, Scope3InputData, Scope3Result


class DownstreamLeasedAssetsInput(Scope3InputData):
    """Input data for Category 13: Downstream Leased Assets."""

    calculation_method: str = Field(
        ...,
        description="Calculation method: asset-specific, average-data"
    )

    # Add category-specific fields
    leased_out_assets: List[Dict[str, Any]] = Field(
        ...,
        description="Assets leased to others"
    )


class DownstreamLeasedAssetsAgent(Scope3BaseAgent):
    """
    Agent for calculating Category 13: Downstream Leased Assets.

    Includes:
    - Buildings leased to tenants
    - Vehicles leased to customers
    - Equipment leased to others
    """

    def __init__(self):
        """Initialize Downstream Leased Assets agent."""
        super().__init__()
        self.category_number = 13
        self.category_name = "Downstream Leased Assets"
        self.emission_factors = self._load_downstream_leased_factors()

    def _load_downstream_leased_factors(self) -> Dict[str, Decimal]:
        """Load emission factors for Downstream Leased Assets."""
        return {
            # Similar to upstream leased
            "electricity_grid": Decimal("0.433"),
            "natural_gas": Decimal("0.185")
        }

    def _parse_input(self, input_data: Dict[str, Any]) -> DownstreamLeasedAssetsInput:
        """Parse and validate input data."""
        return DownstreamLeasedAssetsInput(**input_data)

    async def calculate_emissions(self, input_data: DownstreamLeasedAssetsInput) -> Scope3Result:
        """
        Calculate Category 13 emissions.

        Formula:
        Emissions = Sum(Leased_Asset_Energy × EF)
        """
        self.calculation_steps = []
        self.factors_used = {}
        total_emissions = Decimal("0")

        # Implement calculation logic
        for asset in input_data.leased_out_assets:
            energy_kwh = Decimal(str(asset.get("energy_kwh", 0)))
            ef = self.emission_factors.get("electricity_grid")

            emissions = energy_kwh * ef
            total_emissions += emissions

        # Generate result
        result = Scope3Result(
            category="Category 13: Downstream Leased Assets",
            category_number=13,
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
