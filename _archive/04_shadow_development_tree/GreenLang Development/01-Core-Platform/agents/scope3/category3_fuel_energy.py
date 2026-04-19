# -*- coding: utf-8 -*-
"""
Scope 3 Category 3: Fuel- and Energy-Related Activities (not included in Scope 1 or 2)

Calculates emissions from:
- Upstream emissions of purchased fuels
- Upstream emissions of purchased electricity
- Transmission and distribution (T&D) losses
- Generation of purchased electricity sold to end users

GHG Protocol Definition:
Emissions related to the production of fuels and energy purchased and consumed
by the reporting company that are not included in Scope 1 or Scope 2.
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


class FuelEnergyInput(Scope3InputData):
    """Input data model for Fuel and Energy Related Activities calculation."""

    # Scope 1 fuel consumption (for upstream emissions calculation)
    fuel_consumption: List[Dict[str, Any]] = Field(
        default=[],
        description="Fuels consumed in Scope 1 activities"
    )

    # Scope 2 electricity consumption (for upstream and T&D losses)
    electricity_consumption: Dict[str, Any] = Field(
        default={},
        description="Electricity consumed in Scope 2 activities"
    )

    # Energy sold to end users (if applicable)
    energy_sold: Optional[Dict[str, Any]] = Field(
        None,
        description="Energy sold to end users"
    )

    class FuelConsumption(BaseModel):
        """Fuel consumption data."""
        fuel_type: str = Field(..., description="Type of fuel")
        quantity: float = Field(..., gt=0, description="Quantity consumed")
        unit: str = Field(..., description="Unit (liters, m3, kg, etc.)")
        heating_value: Optional[float] = Field(None, description="Heating value if known")

    class ElectricityConsumption(BaseModel):
        """Electricity consumption data."""
        grid_region: str = Field(..., description="Grid region/country")
        consumption_kwh: float = Field(..., gt=0, description="Electricity consumed in kWh")
        renewable_percentage: Optional[float] = Field(
            0,
            ge=0,
            le=100,
            description="Percentage from renewable sources"
        )


class FuelEnergyAgent(Scope3BaseAgent):
    """
    Agent for calculating Scope 3 Category 3: Fuel and Energy Related Activities.

    Covers four distinct activities:
    1. Upstream emissions of purchased fuels (extraction, production, transportation)
    2. Upstream emissions of purchased electricity (fuel extraction, production)
    3. T&D losses from purchased electricity
    4. Generation of purchased electricity sold to end users
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize Fuel and Energy Agent."""
        if config is None:
            config = AgentConfig(
                name="FuelEnergyAgent",
                description="Calculates Scope 3 Category 3: Fuel and Energy Related Activities",
                version="1.0.0"
            )
        super().__init__(config)

        # Load fuel and energy specific factors
        self.upstream_factors = self._load_upstream_factors()
        self.td_loss_factors = self._load_td_loss_factors()

    def _load_upstream_factors(self) -> Dict[str, Any]:
        """Load upstream emission factors for fuels and electricity."""
        # Well-to-tank (WTT) factors - kg CO2e per unit
        # In production, these would be loaded from YAML/database
        return {
            "fuels": {
                # Upstream factors (kg CO2e per liter/m3/kg)
                "natural_gas": {
                    "factor": Decimal("0.38"),  # kg CO2e per m3
                    "unit": "m3",
                    "description": "Extraction, processing, and transport"
                },
                "diesel": {
                    "factor": Decimal("0.61"),  # kg CO2e per liter
                    "unit": "liter",
                    "description": "Crude extraction, refining, and transport"
                },
                "gasoline": {
                    "factor": Decimal("0.57"),  # kg CO2e per liter
                    "unit": "liter",
                    "description": "Crude extraction, refining, and transport"
                },
                "coal": {
                    "factor": Decimal("0.12"),  # kg CO2e per kg
                    "unit": "kg",
                    "description": "Mining and transport"
                },
                "propane": {
                    "factor": Decimal("0.23"),  # kg CO2e per liter
                    "unit": "liter",
                    "description": "Extraction and processing"
                },
                "fuel_oil": {
                    "factor": Decimal("0.65"),  # kg CO2e per liter
                    "unit": "liter",
                    "description": "Crude extraction, refining, and transport"
                }
            },
            "electricity": {
                # Upstream factors for electricity generation (kg CO2e per kWh)
                "us_average": Decimal("0.082"),
                "eu_average": Decimal("0.065"),
                "china": Decimal("0.125"),
                "india": Decimal("0.115"),
                "global_average": Decimal("0.095")
            }
        }

    def _load_td_loss_factors(self) -> Dict[str, Decimal]:
        """Load transmission and distribution loss factors."""
        # T&D loss percentages by region
        return {
            "us": Decimal("0.045"),  # 4.5% loss
            "eu": Decimal("0.040"),  # 4.0% loss
            "china": Decimal("0.055"),  # 5.5% loss
            "india": Decimal("0.075"),  # 7.5% loss
            "global": Decimal("0.050")  # 5.0% loss
        }

    def _parse_input(self, input_data: Dict[str, Any]) -> FuelEnergyInput:
        """Parse and validate input data."""
        return FuelEnergyInput(**input_data)

    async def calculate_emissions(
        self,
        input_data: FuelEnergyInput
    ) -> Scope3Result:
        """
        Calculate Fuel and Energy Related Activities emissions.

        Formulas:
        1. Upstream fuel: Σ(fuel_consumed × upstream_emission_factor)
        2. Upstream electricity: electricity_consumed × upstream_factor
        3. T&D losses: electricity_consumed × T&D_loss_% × grid_emission_factor
        4. Sold energy: energy_sold × generation_emission_factor
        """
        total_emissions = Decimal("0")
        components = {}

        # 1. Calculate upstream emissions from purchased fuels
        if input_data.fuel_consumption:
            fuel_upstream_emissions = Decimal("0")

            for fuel in input_data.fuel_consumption:
                fuel_type = fuel['fuel_type'].lower()
                quantity = Decimal(str(fuel['quantity']))

                if fuel_type in self.upstream_factors['fuels']:
                    factor_data = self.upstream_factors['fuels'][fuel_type]
                    factor = factor_data['factor']
                    emissions = quantity * factor

                    self._record_calculation_step(
                        description=f"Calculate upstream emissions for {fuel_type}",
                        operation="multiply",
                        inputs={
                            "quantity": quantity,
                            "unit": fuel['unit'],
                            "upstream_factor": factor
                        },
                        output_value=emissions,
                        output_name=f"{fuel_type}_upstream_emissions",
                        formula=f"quantity × upstream_emission_factor",
                        unit="kg CO2e"
                    )

                    fuel_upstream_emissions += emissions
                    self.factors_used[f"{fuel_type}_upstream"] = {
                        "factor": float(factor),
                        "unit": factor_data['unit'],
                        "description": factor_data['description']
                    }

            components['fuel_upstream'] = float(fuel_upstream_emissions)
            total_emissions += fuel_upstream_emissions

        # 2. Calculate upstream emissions from purchased electricity
        if input_data.electricity_consumption:
            elec_data = input_data.electricity_consumption
            consumption_kwh = Decimal(str(elec_data.get('consumption_kwh', 0)))
            grid_region = elec_data.get('grid_region', 'global').lower()

            # Get upstream factor for electricity
            upstream_factor = self.upstream_factors['electricity'].get(
                f"{grid_region}_average",
                self.upstream_factors['electricity']['global_average']
            )

            elec_upstream_emissions = consumption_kwh * upstream_factor

            self._record_calculation_step(
                description="Calculate upstream emissions for purchased electricity",
                operation="multiply",
                inputs={
                    "consumption_kwh": consumption_kwh,
                    "upstream_factor": upstream_factor,
                    "grid_region": grid_region
                },
                output_value=elec_upstream_emissions,
                output_name="electricity_upstream_emissions",
                formula="electricity_kwh × upstream_emission_factor",
                unit="kg CO2e"
            )

            components['electricity_upstream'] = float(elec_upstream_emissions)
            total_emissions += elec_upstream_emissions

            # 3. Calculate T&D losses
            td_loss_rate = self.td_loss_factors.get(
                grid_region.split('_')[0],
                self.td_loss_factors['global']
            )

            # Assume grid emission factor (would be loaded from factors)
            grid_emission_factor = Decimal("0.45")  # kg CO2e per kWh

            td_loss_emissions = consumption_kwh * td_loss_rate * grid_emission_factor

            self._record_calculation_step(
                description="Calculate T&D loss emissions",
                operation="multiply",
                inputs={
                    "consumption_kwh": consumption_kwh,
                    "td_loss_rate": td_loss_rate,
                    "grid_factor": grid_emission_factor
                },
                output_value=td_loss_emissions,
                output_name="td_loss_emissions",
                formula="electricity_kwh × T&D_loss_% × grid_emission_factor",
                unit="kg CO2e"
            )

            components['td_losses'] = float(td_loss_emissions)
            total_emissions += td_loss_emissions

        # 4. Calculate emissions from energy sold to end users (if applicable)
        if input_data.energy_sold:
            energy_sold_kwh = Decimal(str(input_data.energy_sold.get('kwh', 0)))
            generation_factor = Decimal("0.45")  # Would be specific to generation type

            sold_energy_emissions = energy_sold_kwh * generation_factor

            self._record_calculation_step(
                description="Calculate emissions from energy sold to end users",
                operation="multiply",
                inputs={
                    "energy_sold_kwh": energy_sold_kwh,
                    "generation_factor": generation_factor
                },
                output_value=sold_energy_emissions,
                output_name="sold_energy_emissions",
                formula="energy_sold × generation_emission_factor",
                unit="kg CO2e"
            )

            components['energy_sold'] = float(sold_energy_emissions)
            total_emissions += sold_energy_emissions

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

        # Determine data quality
        data_quality = self._calculate_data_quality_score(
            temporal_correlation=2.0,  # Recent data
            geographical_correlation=2.5,  # Regional factors
            technological_correlation=2.0,  # Technology-specific
            completeness=2.0,  # Complete data
            reliability=2.5  # Published sources
        )

        # Calculate uncertainty
        uncertainty = self._estimate_uncertainty(data_quality, "average-data")

        # Generate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            input_data.dict(),
            self.calculation_steps,
            total_emissions
        )

        return Scope3Result(
            category="Category 3: Fuel- and Energy-Related Activities",
            category_number=3,
            total_emissions_kg_co2e=self._apply_precision(total_emissions),
            total_emissions_t_co2e=self._apply_precision(total_emissions_tonnes),
            calculation_methodology="average-data",
            data_quality_score=data_quality,
            calculation_steps=self.calculation_steps,
            emission_factors_used=self.factors_used,
            provenance_hash=provenance_hash,
            calculation_timestamp=self.clock.now().isoformat(),
            ghg_protocol_compliance=True,
            uncertainty_range=uncertainty
        )

    def get_example_calculation(self) -> Dict[str, Any]:
        """
        Provide example calculation for documentation.
        """
        example_input = {
            "reporting_year": 2024,
            "reporting_entity": "Example Corp",
            "fuel_consumption": [
                {
                    "fuel_type": "natural_gas",
                    "quantity": 10000,
                    "unit": "m3"
                },
                {
                    "fuel_type": "diesel",
                    "quantity": 5000,
                    "unit": "liters"
                }
            ],
            "electricity_consumption": {
                "grid_region": "us",
                "consumption_kwh": 1000000,
                "renewable_percentage": 20
            }
        }

        example_calculation = """
        Example Calculation:

        1. Upstream emissions from natural gas:
           10,000 m³ × 0.38 kg CO2e/m³ = 3,800 kg CO2e

        2. Upstream emissions from diesel:
           5,000 L × 0.61 kg CO2e/L = 3,050 kg CO2e

        3. Upstream emissions from electricity:
           1,000,000 kWh × 0.082 kg CO2e/kWh = 82,000 kg CO2e

        4. T&D losses:
           1,000,000 kWh × 0.045 × 0.45 kg CO2e/kWh = 20,250 kg CO2e

        Total: 109,100 kg CO2e = 109.10 t CO2e
        """

        return {
            "example_input": example_input,
            "calculation": example_calculation,
            "result": "109.10 t CO2e"
        }