# -*- coding: utf-8 -*-
"""
Scope 3 Category 4: Upstream Transportation and Distribution

Calculates emissions from:
- Transportation and distribution of purchased products in vehicles not owned by reporting company
- Third-party transportation and distribution services purchased by reporting company
- Inbound logistics

GHG Protocol Definition:
Transportation and distribution of purchased goods/services in the reporting year
between tier 1 suppliers and own operations in vehicles not owned or operated
by the reporting company.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator

from greenlang.agents.scope3.base import (
    Scope3BaseAgent,
    Scope3InputData,
    Scope3Result,
    AgentConfig
)


class UpstreamTransportInput(Scope3InputData):
    """Input data model for Upstream Transportation calculation."""

    shipments: List[Dict[str, Any]] = Field(
        ...,
        description="List of upstream transportation activities"
    )

    class ShipmentData(BaseModel):
        """Individual shipment data."""
        mode: str = Field(..., description="Transport mode: road, rail, air, sea, inland_waterway")
        distance_km: float = Field(..., gt=0, description="Distance in kilometers")
        weight_tonnes: Optional[float] = Field(None, gt=0, description="Weight in tonnes")
        volume_m3: Optional[float] = Field(None, gt=0, description="Volume in cubic meters")
        vehicle_type: Optional[str] = Field(None, description="Specific vehicle type")
        fuel_type: Optional[str] = Field(None, description="Fuel type used")
        utilization_rate: Optional[float] = Field(
            None,
            ge=0,
            le=100,
            description="Vehicle utilization rate %"
        )
        refrigerated: bool = Field(False, description="Whether refrigerated transport")


class UpstreamTransportAgent(Scope3BaseAgent):
    """
    Agent for calculating Scope 3 Category 4: Upstream Transportation and Distribution.

    Calculation Methods:
    1. Distance-based: tonne-km × emission_factor
    2. Fuel-based: fuel_consumed × emission_factor
    3. Spend-based: transport_spend × EEIO_factor
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize Upstream Transport Agent."""
        if config is None:
            config = AgentConfig(
                name="UpstreamTransportAgent",
                description="Calculates Scope 3 Category 4: Upstream Transportation",
                version="1.0.0"
            )
        super().__init__(config)

        self.transport_factors = self._load_transport_factors()

    def _load_transport_factors(self) -> Dict[str, Any]:
        """Load transportation emission factors."""
        # kg CO2e per tonne-km by mode and type
        return {
            "road": {
                "small_truck": Decimal("0.249"),  # <7.5t
                "medium_truck": Decimal("0.178"),  # 7.5-17t
                "large_truck": Decimal("0.089"),  # >17t
                "refrigerated": Decimal("0.123"),  # Refrigerated truck
                "default": Decimal("0.105")  # Average truck
            },
            "rail": {
                "freight": Decimal("0.028"),
                "electric": Decimal("0.019"),
                "diesel": Decimal("0.034"),
                "default": Decimal("0.028")
            },
            "air": {
                "domestic": Decimal("1.205"),
                "international": Decimal("0.895"),
                "express": Decimal("1.516"),
                "default": Decimal("1.122")
            },
            "sea": {
                "container": Decimal("0.016"),
                "bulk": Decimal("0.006"),
                "tanker": Decimal("0.009"),
                "refrigerated": Decimal("0.022"),
                "default": Decimal("0.012")
            },
            "inland_waterway": {
                "barge": Decimal("0.031"),
                "default": Decimal("0.031")
            }
        }

    def _parse_input(self, input_data: Dict[str, Any]) -> UpstreamTransportInput:
        """Parse and validate input data."""
        return UpstreamTransportInput(**input_data)

    async def calculate_emissions(
        self,
        input_data: UpstreamTransportInput
    ) -> Scope3Result:
        """
        Calculate Upstream Transportation emissions.

        Formula: Σ(distance_km × weight_tonnes × emission_factor)
        """
        total_emissions = Decimal("0")

        for idx, shipment in enumerate(input_data.shipments):
            mode = shipment['mode'].lower()
            distance = Decimal(str(shipment['distance_km']))
            weight = Decimal(str(shipment.get('weight_tonnes', 1)))

            # Get emission factor
            mode_factors = self.transport_factors.get(mode, {})
            vehicle_type = shipment.get('vehicle_type', 'default')
            refrigerated = shipment.get('refrigerated', False)

            if refrigerated and 'refrigerated' in mode_factors:
                factor = mode_factors['refrigerated']
            elif vehicle_type in mode_factors:
                factor = mode_factors[vehicle_type]
            else:
                factor = mode_factors.get('default', Decimal("0.1"))

            # Calculate tonne-km
            tonne_km = distance * weight

            # Apply utilization adjustment if provided
            if shipment.get('utilization_rate'):
                utilization = Decimal(str(shipment['utilization_rate'])) / Decimal("100")
                factor = factor / utilization

            # Calculate emissions
            emissions = tonne_km * factor

            self._record_calculation_step(
                description=f"Calculate emissions for {mode} transport",
                operation="multiply",
                inputs={
                    "distance_km": distance,
                    "weight_tonnes": weight,
                    "tonne_km": tonne_km,
                    "emission_factor": factor,
                    "mode": mode
                },
                output_value=emissions,
                output_name=f"shipment_{idx}_emissions",
                formula="tonne_km × emission_factor",
                unit="kg CO2e"
            )

            total_emissions += emissions
            self.factors_used[f"shipment_{idx}"] = {
                "mode": mode,
                "factor": float(factor),
                "tonne_km": float(tonne_km)
            }

        # Convert to tonnes
        total_emissions_tonnes = total_emissions / Decimal("1000")

        self._record_calculation_step(
            description="Convert total emissions to tonnes CO2e",
            operation="divide",
            inputs={
                "total_kg_co2e": total_emissions,
                "conversion_factor": Decimal("1000")
            },
            output_value=total_emissions_tonnes,
            output_name="total_emissions_t_co2e",
            formula="total_kg_co2e / 1000",
            unit="t CO2e"
        )

        # Calculate data quality and uncertainty
        data_quality = 2.5
        uncertainty = self._estimate_uncertainty(data_quality, "distance-based")

        # Generate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            input_data.dict(),
            self.calculation_steps,
            total_emissions
        )

        return Scope3Result(
            category="Category 4: Upstream Transportation and Distribution",
            category_number=4,
            total_emissions_kg_co2e=self._apply_precision(total_emissions),
            total_emissions_t_co2e=self._apply_precision(total_emissions_tonnes),
            calculation_methodology="distance-based",
            data_quality_score=data_quality,
            calculation_steps=self.calculation_steps,
            emission_factors_used=self.factors_used,
            provenance_hash=provenance_hash,
            calculation_timestamp=self.clock.now().isoformat(),
            ghg_protocol_compliance=True,
            uncertainty_range=uncertainty
        )

    def get_example_calculation(self) -> Dict[str, Any]:
        """Provide example calculation."""
        return {
            "example_input": {
                "reporting_year": 2024,
                "reporting_entity": "Example Corp",
                "shipments": [
                    {
                        "mode": "road",
                        "distance_km": 500,
                        "weight_tonnes": 20,
                        "vehicle_type": "large_truck"
                    },
                    {
                        "mode": "sea",
                        "distance_km": 8000,
                        "weight_tonnes": 100,
                        "vehicle_type": "container"
                    }
                ]
            },
            "calculation": """
            1. Road transport:
               500 km × 20 t × 0.089 kg CO2e/t-km = 890 kg CO2e

            2. Sea transport:
               8,000 km × 100 t × 0.016 kg CO2e/t-km = 12,800 kg CO2e

            Total: 13,690 kg CO2e = 13.69 t CO2e
            """,
            "result": "13.69 t CO2e"
        }