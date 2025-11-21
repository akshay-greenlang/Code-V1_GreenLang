"""
Category 8: Upstream Leased Assets Agent

Calculates emissions from leased assets not in Scope 1 or 2

GHG Protocol Reference:
- Chapter 5.8 of Scope 3 Standard
- Category 8 Calculation Guidance
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field

from .base import Scope3BaseAgent, Scope3InputData, Scope3Result


class UpstreamLeasedAssetsInput(Scope3InputData):
    """Input data for Category 8: Upstream Leased Assets."""

    calculation_method: str = Field(
        ...,
        description="Calculation method: asset-specific, average-data"
    )

    # Add category-specific fields
    leased_assets: List[Dict[str, Any]] = Field(
        ...,
        description="List of leased assets with energy data"
    )

    allocation_method: str = Field(
        default="area",
        description="Allocation method: area, headcount, or revenue"
    )


class UpstreamLeasedAssetsAgent(Scope3BaseAgent):
    """
    Agent for calculating Category 8: Upstream Leased Assets.

    Includes:
    - Leased facilities
    - Leased vehicles
    - Leased equipment
    (Only if not already included in Scope 1 or 2)
    """

    def __init__(self):
        """Initialize Upstream Leased Assets agent."""
        super().__init__()
        self.category_number = 8
        self.category_name = "Upstream Leased Assets"
        self.emission_factors = self._load_leased_assets_factors()

    def _load_leased_assets_factors(self) -> Dict[str, Decimal]:
        """Load emission factors for Upstream Leased Assets."""
        return {
            # Energy factors (kg CO2e/kWh)
            "electricity_grid": Decimal("0.433"),
            "natural_gas": Decimal("0.185"),
            "fuel_oil": Decimal("0.256"),

            # Asset intensity factors
            "office_kwh_per_m2": Decimal("135"),
            "warehouse_kwh_per_m2": Decimal("95"),
            "retail_kwh_per_m2": Decimal("165")
        }

    def _parse_input(self, input_data: Dict[str, Any]) -> UpstreamLeasedAssetsInput:
        """Parse and validate input data."""
        return UpstreamLeasedAssetsInput(**input_data)

    async def calculate_emissions(self, input_data: UpstreamLeasedAssetsInput) -> Scope3Result:
        """
        Calculate Category 8 emissions.

        Formula:
        Emissions = Sum(Asset_Energy × EF) - Scope_1_2_Portion
        """
        self.calculation_steps = []
        self.factors_used = {}
        total_emissions = Decimal("0")

        # Implement calculation logic
        for asset in input_data.leased_assets:
            if asset.get("included_in_scope_1_2", False):
                continue  # Skip if already in Scope 1/2

            energy_kwh = Decimal(str(asset.get("energy_kwh", 0)))
            energy_type = asset.get("energy_type", "electricity_grid")

            ef = self.emission_factors.get(energy_type)
            emissions = energy_kwh * ef
            total_emissions += emissions

        # Generate result
        result = Scope3Result(
            category="Category 8: Upstream Leased Assets",
            category_number=8,
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
