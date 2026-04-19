"""
Category 3: Fuel and Energy Related Activities Agent

Calculates emissions from extraction, production, and transportation of fuels
and energy purchased and consumed by the reporting company that are not
included in Scope 1 or Scope 2.

GHG Protocol Reference:
- Chapter 5.3 of Scope 3 Standard
- Category 3 Calculation Guidance
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field, validator

from .base import Scope3BaseAgent, Scope3InputData, Scope3Result, Scope3CalculationStep


class FuelEnergyInput(Scope3InputData):
    """Input data for Category 3: Fuel and Energy Related Activities."""

    # Upstream emissions of purchased fuels (Well-to-Tank)
    purchased_fuels: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Purchased fuels with quantities (e.g., diesel, natural gas, coal)"
    )

    # Upstream emissions of purchased electricity
    purchased_electricity: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Electricity consumption by source/grid (kWh)"
    )

    # Transmission and distribution losses
    td_losses: Optional[Dict[str, Any]] = Field(
        default=None,
        description="T&D loss data for electricity and steam"
    )

    # Generation of purchased electricity sold to end users
    electricity_sold: Optional[Decimal] = Field(
        default=None,
        description="Electricity generated and sold (kWh)"
    )

    grid_region: str = Field(default="US", description="Grid region for electricity factors")
    include_wtt: bool = Field(default=True, description="Include well-to-tank emissions")
    include_td_losses: bool = Field(default=True, description="Include T&D losses")


class FuelEnergyRelatedAgent(Scope3BaseAgent):
    """
    Agent for calculating Category 3: Fuel and Energy Related Activities.

    Includes four distinct activities:
    1. Upstream emissions of purchased fuels (extraction, production, transport)
    2. Upstream emissions of purchased electricity (extraction, production of fuels)
    3. T&D losses for purchased electricity and steam
    4. Generation of purchased electricity sold to end users (utility companies only)

    These are NOT included in Scope 1 or 2 but are related to fuel/energy use.
    """

    def __init__(self):
        """Initialize Fuel and Energy Related Activities agent."""
        super().__init__()
        self.category_number = 3
        self.category_name = "Fuel and Energy Related Activities"

        # Load WTT and T&D emission factors
        self.wtt_factors = self._load_wtt_factors()
        self.td_loss_factors = self._load_td_factors()
        self.grid_factors = self._load_grid_factors()

    def _load_wtt_factors(self) -> Dict[str, Decimal]:
        """
        Load Well-to-Tank (WTT) emission factors.

        These represent upstream emissions from extraction, refining,
        and transportation of fuels.

        Sources: DEFRA, EPA, IEA
        """
        return {
            # WTT factors (kg CO2e per unit)
            "diesel_liter": Decimal("0.611"),  # kg CO2e/liter WTT
            "gasoline_liter": Decimal("0.584"),
            "natural_gas_m3": Decimal("0.184"),  # kg CO2e/m3 WTT
            "coal_tonne": Decimal("35.2"),  # kg CO2e/tonne WTT
            "lpg_liter": Decimal("0.201"),
            "fuel_oil_liter": Decimal("0.558"),
            "aviation_fuel_liter": Decimal("0.521"),
            "marine_fuel_tonne": Decimal("298.5"),

            # Electricity WTT factors by grid (kg CO2e/kWh)
            "electricity_wtt_us": Decimal("0.075"),
            "electricity_wtt_eu": Decimal("0.062"),
            "electricity_wtt_cn": Decimal("0.118"),
            "electricity_wtt_global": Decimal("0.085")
        }

    def _load_td_factors(self) -> Dict[str, Decimal]:
        """
        Load Transmission & Distribution loss factors.

        T&D losses typically 5-10% of electricity consumed.
        """
        return {
            # T&D loss rates (percentage)
            "td_loss_rate_us": Decimal("0.048"),  # 4.8% loss
            "td_loss_rate_eu": Decimal("0.065"),  # 6.5% loss
            "td_loss_rate_cn": Decimal("0.057"),  # 5.7% loss
            "td_loss_rate_global": Decimal("0.080"),  # 8% loss

            # Grid emission factors for T&D losses (kg CO2e/kWh)
            "grid_factor_us": Decimal("0.433"),
            "grid_factor_eu": Decimal("0.295"),
            "grid_factor_cn": Decimal("0.581"),
            "grid_factor_global": Decimal("0.475")
        }

    def _load_grid_factors(self) -> Dict[str, Decimal]:
        """Load grid emission factors by region."""
        return {
            "US": Decimal("0.433"),
            "EU": Decimal("0.295"),
            "UK": Decimal("0.233"),
            "CN": Decimal("0.581"),
            "IN": Decimal("0.708"),
            "JP": Decimal("0.465"),
            "global": Decimal("0.475")
        }

    def _parse_input(self, input_data: Dict[str, Any]) -> FuelEnergyInput:
        """Parse and validate input data."""
        return FuelEnergyInput(**input_data)

    async def calculate_emissions(self, input_data: FuelEnergyInput) -> Scope3Result:
        """
        Calculate Category 3 emissions.

        Formulas:
        1. WTT Fuels: Σ(Fuel_Quantity × WTT_Factor)
        2. WTT Electricity: Electricity_Consumed × WTT_Factor_Grid
        3. T&D Losses: Electricity_Consumed × Loss_Rate × Grid_Factor
        4. Sold Electricity: Electricity_Sold × Grid_Factor
        """
        self.calculation_steps = []
        self.factors_used = {}
        total_emissions = Decimal("0")

        # Activity 1: Upstream emissions of purchased fuels
        if input_data.purchased_fuels and input_data.include_wtt:
            wtt_fuel_emissions = self._calculate_wtt_fuels(input_data)
            total_emissions += wtt_fuel_emissions

        # Activity 2: Upstream emissions of purchased electricity
        if input_data.purchased_electricity and input_data.include_wtt:
            wtt_elec_emissions = self._calculate_wtt_electricity(input_data)
            total_emissions += wtt_elec_emissions

        # Activity 3: T&D losses
        if input_data.purchased_electricity and input_data.include_td_losses:
            td_emissions = self._calculate_td_losses(input_data)
            total_emissions += td_emissions

        # Activity 4: Generation of sold electricity (utility companies)
        if input_data.electricity_sold:
            sold_emissions = self._calculate_sold_electricity(input_data)
            total_emissions += sold_emissions

        # Calculate data quality score
        data_quality = self._calculate_data_quality_score(
            temporal_correlation=2.0 if input_data.reporting_year == 2024 else 3.0,
            geographical_correlation=2.0 if input_data.grid_region != "global" else 3.5,
            technological_correlation=2.0,
            completeness=2.5,
            reliability=2.5
        )

        # Generate result
        result = Scope3Result(
            category="Category 3: Fuel and Energy Related Activities",
            category_number=3,
            total_emissions_kg_co2e=total_emissions,
            total_emissions_t_co2e=total_emissions / Decimal("1000"),
            calculation_methodology="activity-based",
            data_quality_score=data_quality,
            calculation_steps=self.calculation_steps,
            emission_factors_used=self.factors_used,
            provenance_hash=self._calculate_provenance_hash(
                input_data.dict(),
                self.calculation_steps,
                total_emissions
            ),
            calculation_timestamp=str(self._get_timestamp()),
            ghg_protocol_compliance=True,
            uncertainty_range=self._estimate_uncertainty(
                data_quality,
                "activity-based"
            )
        )

        return result

    def _calculate_wtt_fuels(self, input_data: FuelEnergyInput) -> Decimal:
        """
        Calculate Well-to-Tank emissions for purchased fuels.

        These are upstream emissions from extraction, production,
        and transportation of fuels before combustion.
        """
        total = Decimal("0")

        for fuel_type, fuel_data in input_data.purchased_fuels.items():
            quantity = Decimal(str(fuel_data.get("quantity", 0)))
            unit = fuel_data.get("unit", "liter")

            # Get WTT factor
            factor_key = f"{fuel_type}_{unit}"
            wtt_factor = self.wtt_factors.get(factor_key, Decimal("0.5"))

            # Calculate WTT emissions
            emissions = quantity * wtt_factor

            self._record_calculation_step(
                description=f"Calculate WTT emissions for {fuel_type}",
                operation="multiply",
                inputs={
                    "quantity": quantity,
                    "unit": unit,
                    "wtt_factor": wtt_factor
                },
                output_value=emissions,
                output_name=f"wtt_emissions_{fuel_type}",
                formula="WTT_Emissions = Fuel_Quantity × WTT_Factor",
                unit="kg CO2e"
            )

            self.factors_used[f"wtt_{fuel_type}"] = {
                "value": float(wtt_factor),
                "unit": f"kg CO2e/{unit}",
                "source": "DEFRA 2024",
                "type": "Well-to-Tank"
            }

            total += emissions

        return total

    def _calculate_wtt_electricity(self, input_data: FuelEnergyInput) -> Decimal:
        """
        Calculate Well-to-Tank emissions for purchased electricity.

        These are upstream emissions from extraction and production
        of fuels used to generate electricity.
        """
        total = Decimal("0")

        for source, kwh in input_data.purchased_electricity.items():
            kwh_decimal = Decimal(str(kwh))

            # Get WTT factor for grid region
            factor_key = f"electricity_wtt_{input_data.grid_region.lower()}"
            wtt_factor = self.wtt_factors.get(
                factor_key,
                self.wtt_factors["electricity_wtt_global"]
            )

            # Calculate WTT emissions
            emissions = kwh_decimal * wtt_factor

            self._record_calculation_step(
                description=f"Calculate WTT emissions for electricity from {source}",
                operation="multiply",
                inputs={
                    "electricity_kwh": kwh,
                    "wtt_factor": wtt_factor,
                    "grid_region": input_data.grid_region
                },
                output_value=emissions,
                output_name=f"wtt_electricity_{source}",
                formula="WTT_Elec = Electricity_kWh × WTT_Factor_Grid",
                unit="kg CO2e"
            )

            self.factors_used[f"wtt_electricity_{source}"] = {
                "value": float(wtt_factor),
                "unit": "kg CO2e/kWh",
                "source": "IEA 2024",
                "type": "Electricity WTT",
                "region": input_data.grid_region
            }

            total += emissions

        return total

    def _calculate_td_losses(self, input_data: FuelEnergyInput) -> Decimal:
        """
        Calculate emissions from transmission and distribution losses.

        Energy is lost as heat during T&D, requiring additional generation.
        """
        total = Decimal("0")

        # Get T&D loss rate for region
        loss_rate_key = f"td_loss_rate_{input_data.grid_region.lower()}"
        loss_rate = self.td_loss_factors.get(
            loss_rate_key,
            self.td_loss_factors["td_loss_rate_global"]
        )

        # Get grid emission factor
        grid_factor_key = f"grid_factor_{input_data.grid_region.lower()}"
        grid_factor = self.td_loss_factors.get(
            grid_factor_key,
            self.td_loss_factors["grid_factor_global"]
        )

        for source, kwh in input_data.purchased_electricity.items():
            kwh_decimal = Decimal(str(kwh))

            # Calculate T&D loss emissions
            # Emissions = Electricity × Loss_Rate × Grid_Factor
            lost_electricity = kwh_decimal * loss_rate
            emissions = lost_electricity * grid_factor

            self._record_calculation_step(
                description=f"Calculate T&D loss emissions for {source}",
                operation="multiply",
                inputs={
                    "electricity_kwh": kwh,
                    "loss_rate": loss_rate,
                    "grid_factor": grid_factor
                },
                output_value=emissions,
                output_name=f"td_losses_{source}",
                formula="T&D_Emissions = Electricity × Loss_Rate × Grid_Factor",
                unit="kg CO2e"
            )

            self.factors_used[f"td_losses_{source}"] = {
                "loss_rate": float(loss_rate),
                "grid_factor": float(grid_factor),
                "unit": "kg CO2e/kWh",
                "source": "EPA eGRID 2024",
                "type": "T&D Losses",
                "region": input_data.grid_region
            }

            total += emissions

        return total

    def _calculate_sold_electricity(self, input_data: FuelEnergyInput) -> Decimal:
        """
        Calculate emissions from generation of electricity sold to end users.

        Only applicable to utility companies that generate and sell electricity.
        """
        kwh_sold = Decimal(str(input_data.electricity_sold))

        # Get grid emission factor
        grid_factor = self.grid_factors.get(
            input_data.grid_region,
            self.grid_factors["global"]
        )

        # Calculate emissions
        emissions = kwh_sold * grid_factor

        self._record_calculation_step(
            description="Calculate emissions from electricity sold to end users",
            operation="multiply",
            inputs={
                "electricity_sold_kwh": input_data.electricity_sold,
                "grid_factor": grid_factor
            },
            output_value=emissions,
            output_name="sold_electricity_emissions",
            formula="Sold_Emissions = Electricity_Sold × Grid_Factor",
            unit="kg CO2e"
        )

        self.factors_used["sold_electricity"] = {
            "value": float(grid_factor),
            "unit": "kg CO2e/kWh",
            "source": "IEA Grid Factors 2024",
            "type": "Grid Emissions",
            "region": input_data.grid_region
        }

        return emissions

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()