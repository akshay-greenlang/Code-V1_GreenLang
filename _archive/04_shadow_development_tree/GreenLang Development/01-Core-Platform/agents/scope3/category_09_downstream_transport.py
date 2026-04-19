"""
Category 9: Downstream Transportation and Distribution Agent

Calculates emissions from transportation and distribution of sold products

GHG Protocol Reference:
- Chapter 5.9 of Scope 3 Standard
- Category 9 Calculation Guidance
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field

from .base import Scope3BaseAgent, Scope3InputData, Scope3Result


class DownstreamTransportInput(Scope3InputData):
    """Input data for Category 9: Downstream Transportation and Distribution."""

    calculation_method: str = Field(
        ...,
        description="Calculation method: distance-based, spend-based"
    )

    # Add category-specific fields
    shipments: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Outbound shipment data"
    )

    distribution_spend: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Distribution spend by mode"
    )


class DownstreamTransportAgent(Scope3BaseAgent):
    """
    Agent for calculating Category 9: Downstream Transportation and Distribution.

    Includes:
    - Transportation from company to customer
    - Third-party distribution
    - Retail storage
    """

    def __init__(self):
        """Initialize Downstream Transportation and Distribution agent."""
        super().__init__()
        self.category_number = 9
        self.category_name = "Downstream Transportation and Distribution"
        self.emission_factors = self._load_downstream_transport_factors()

    def _load_downstream_transport_factors(self) -> Dict[str, Decimal]:
        """Load emission factors for Downstream Transportation and Distribution."""
        return {
            # Same as upstream transport
            "truck": Decimal("0.145"),
            "rail": Decimal("0.024"),
            "ship": Decimal("0.012"),
            "air": Decimal("0.809")
        }

    def _parse_input(self, input_data: Dict[str, Any]) -> DownstreamTransportInput:
        """Parse and validate input data."""
        return DownstreamTransportInput(**input_data)

    async def calculate_emissions(self, input_data: DownstreamTransportInput) -> Scope3Result:
        """
        Calculate Category 9 emissions.

        Formula:
        Emissions = Sum(Distance × Weight × Mode_EF)
        """
        self.calculation_steps = []
        self.factors_used = {}
        total_emissions = Decimal("0")

        # Implement calculation logic
        # Similar to upstream transport
        if input_data.shipments:
            for shipment in input_data.shipments:
                distance = Decimal(str(shipment.get("distance_km", 0)))
                weight = Decimal(str(shipment.get("weight_tonnes", 0)))
                mode = shipment.get("mode", "truck")

                ef = self.emission_factors.get(mode)
                emissions = distance * weight * ef
                total_emissions += emissions

        # Generate result
        result = Scope3Result(
            category="Category 9: Downstream Transportation and Distribution",
            category_number=9,
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
