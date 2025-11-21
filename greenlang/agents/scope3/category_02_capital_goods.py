"""
Category 2: Capital Goods Agent

Calculates emissions from extraction, production, and transportation of capital goods
purchased or acquired by the reporting company in the reporting year.

GHG Protocol Reference:
- Chapter 5.2 of Scope 3 Standard
- Category 2 Calculation Guidance
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field, validator

from .base import Scope3BaseAgent, Scope3InputData, Scope3Result, Scope3CalculationStep


class CapitalGoodsInput(Scope3InputData):
    """Input data for Category 2: Capital Goods calculations."""

    calculation_method: str = Field(
        ...,
        description="Calculation method: spend-based, average-data, supplier-specific, hybrid"
    )

    # Spend-based method fields
    capital_spend: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Capital expenditure by category (USD)"
    )

    # Average-data method fields
    equipment_purchases: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of equipment/machinery purchased"
    )

    # Building construction
    building_construction: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Building construction data (area, type, materials)"
    )

    # Supplier-specific data
    supplier_emissions: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Supplier-provided emission data"
    )

    region: str = Field(default="global", description="Geographic region")
    currency: str = Field(default="USD", description="Currency for spend data")

    @validator('calculation_method')
    def validate_method(cls, v):
        valid_methods = ['spend-based', 'average-data', 'supplier-specific', 'hybrid']
        if v not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        return v


class CapitalGoodsAgent(Scope3BaseAgent):
    """
    Agent for calculating Category 2: Capital Goods emissions.

    Includes emissions from:
    - Production of capital goods (equipment, machinery, buildings, facilities, vehicles)
    - Transportation of capital goods to reporting company

    Excludes:
    - Use of capital goods (Scope 1 & 2)
    - End-of-life treatment (Category 12)
    """

    def __init__(self):
        """Initialize Capital Goods agent."""
        super().__init__()
        self.category_number = 2
        self.category_name = "Capital Goods"

        # Load capital goods emission factors
        self.emission_factors = self._load_capital_goods_factors()

    def _load_capital_goods_factors(self) -> Dict[str, Decimal]:
        """
        Load emission factors for capital goods.

        Sources:
        - EPA EEIO (Environmentally-Extended Input-Output) factors
        - DEFRA conversion factors
        - Ecoinvent database
        """
        return {
            # Spend-based factors (kg CO2e per USD)
            "machinery_equipment": Decimal("0.385"),
            "computer_electronic": Decimal("0.242"),
            "electrical_equipment": Decimal("0.298"),
            "motor_vehicles": Decimal("0.451"),
            "furniture_fixtures": Decimal("0.275"),
            "construction": Decimal("0.416"),
            "manufacturing_equipment": Decimal("0.392"),

            # Material-based factors (kg CO2e per kg)
            "steel": Decimal("2.32"),
            "aluminum": Decimal("8.24"),
            "concrete": Decimal("0.158"),
            "glass": Decimal("1.09"),
            "plastic": Decimal("3.71"),

            # Building factors (kg CO2e per m2)
            "office_building": Decimal("525"),
            "warehouse": Decimal("385"),
            "manufacturing_facility": Decimal("650"),
            "retail_space": Decimal("425")
        }

    def _parse_input(self, input_data: Dict[str, Any]) -> CapitalGoodsInput:
        """Parse and validate input data."""
        return CapitalGoodsInput(**input_data)

    async def calculate_emissions(self, input_data: CapitalGoodsInput) -> Scope3Result:
        """
        Calculate Category 2 emissions using appropriate method.

        Formula (spend-based):
        Emissions = Σ(Spend_category × EF_category)

        Formula (average-data):
        Emissions = Σ(Quantity × EF_per_unit)

        Formula (supplier-specific):
        Emissions = Σ(Supplier_reported_emissions)
        """
        self.calculation_steps = []
        self.factors_used = {}
        total_emissions = Decimal("0")

        if input_data.calculation_method == "spend-based":
            total_emissions = self._calculate_spend_based(input_data)
        elif input_data.calculation_method == "average-data":
            total_emissions = self._calculate_average_data(input_data)
        elif input_data.calculation_method == "supplier-specific":
            total_emissions = self._calculate_supplier_specific(input_data)
        elif input_data.calculation_method == "hybrid":
            total_emissions = self._calculate_hybrid(input_data)

        # Calculate data quality score
        data_quality = self._calculate_data_quality_score(
            temporal_correlation=2.0 if input_data.reporting_year == 2024 else 3.0,
            geographical_correlation=2.0 if input_data.region != "global" else 3.5,
            technological_correlation=2.5,
            completeness=2.0,
            reliability=2.0 if input_data.calculation_method == "supplier-specific" else 3.0
        )

        # Generate result
        result = Scope3Result(
            category="Category 2: Capital Goods",
            category_number=2,
            total_emissions_kg_co2e=total_emissions,
            total_emissions_t_co2e=total_emissions / Decimal("1000"),
            calculation_methodology=input_data.calculation_method,
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
                input_data.calculation_method
            )
        )

        return result

    def _calculate_spend_based(self, input_data: CapitalGoodsInput) -> Decimal:
        """
        Calculate emissions using spend-based method.

        Most common method for capital goods.
        Uses economic input-output emission factors.
        """
        total = Decimal("0")

        if not input_data.capital_spend:
            raise ValueError("capital_spend required for spend-based method")

        for category, spend in input_data.capital_spend.items():
            # Get emission factor
            ef = self.emission_factors.get(category, Decimal("0.35"))  # Default factor

            # Calculate emissions
            emissions = Decimal(str(spend)) * ef

            # Record step
            self._record_calculation_step(
                description=f"Calculate emissions for {category}",
                operation="multiply",
                inputs={"spend_usd": spend, "emission_factor": ef},
                output_value=emissions,
                output_name=f"emissions_{category}",
                formula="Emissions = Spend × Emission_Factor",
                unit="kg CO2e"
            )

            # Record factor used
            self.factors_used[category] = {
                "value": float(ef),
                "unit": "kg CO2e/USD",
                "source": "EPA EEIO",
                "year": 2024
            }

            total += emissions

        return total

    def _calculate_average_data(self, input_data: CapitalGoodsInput) -> Decimal:
        """
        Calculate emissions using average-data method.

        Uses physical quantities and average emission factors.
        """
        total = Decimal("0")

        # Equipment purchases
        if input_data.equipment_purchases:
            for equipment in input_data.equipment_purchases:
                quantity = Decimal(str(equipment.get("quantity", 1)))
                weight_kg = Decimal(str(equipment.get("weight_kg", 0)))
                material = equipment.get("primary_material", "steel")

                # Get material emission factor
                ef = self.emission_factors.get(material, Decimal("2.0"))

                # Calculate emissions
                emissions = weight_kg * ef * quantity

                self._record_calculation_step(
                    description=f"Calculate emissions for {equipment.get('name', 'equipment')}",
                    operation="multiply",
                    inputs={
                        "weight_kg": weight_kg,
                        "quantity": quantity,
                        "emission_factor": ef
                    },
                    output_value=emissions,
                    output_name=f"emissions_{equipment.get('name', 'equipment')}",
                    formula="Emissions = Weight × Quantity × Material_EF",
                    unit="kg CO2e"
                )

                total += emissions

        # Building construction
        if input_data.building_construction:
            area_m2 = Decimal(str(input_data.building_construction.get("area_m2", 0)))
            building_type = input_data.building_construction.get("type", "office_building")

            # Get building emission factor
            ef = self.emission_factors.get(building_type, Decimal("500"))

            # Calculate emissions
            emissions = area_m2 * ef

            self._record_calculation_step(
                description=f"Calculate emissions for {building_type}",
                operation="multiply",
                inputs={"area_m2": area_m2, "emission_factor": ef},
                output_value=emissions,
                output_name="emissions_building",
                formula="Emissions = Area × Building_EF",
                unit="kg CO2e"
            )

            total += emissions

        return total

    def _calculate_supplier_specific(self, input_data: CapitalGoodsInput) -> Decimal:
        """
        Calculate emissions using supplier-specific data.

        Most accurate method when suppliers provide cradle-to-gate emissions.
        """
        total = Decimal("0")

        if not input_data.supplier_emissions:
            raise ValueError("supplier_emissions required for supplier-specific method")

        for supplier, emissions in input_data.supplier_emissions.items():
            emissions_decimal = Decimal(str(emissions))

            self._record_calculation_step(
                description=f"Add supplier emissions from {supplier}",
                operation="add",
                inputs={"supplier_emissions": emissions},
                output_value=emissions_decimal,
                output_name=f"emissions_{supplier}",
                formula="Direct supplier data",
                unit="kg CO2e"
            )

            total += emissions_decimal

        return total

    def _calculate_hybrid(self, input_data: CapitalGoodsInput) -> Decimal:
        """
        Calculate emissions using hybrid method.

        Combines supplier-specific data where available with
        spend-based or average-data for remaining items.
        """
        total = Decimal("0")

        # Start with supplier-specific if available
        if input_data.supplier_emissions:
            total += self._calculate_supplier_specific(input_data)

        # Add spend-based for remaining categories
        if input_data.capital_spend:
            # Filter out categories covered by supplier data
            remaining_spend = {
                k: v for k, v in input_data.capital_spend.items()
                if k not in (input_data.supplier_emissions or {})
            }

            if remaining_spend:
                temp_input = CapitalGoodsInput(
                    **{**input_data.dict(), "capital_spend": remaining_spend}
                )
                total += self._calculate_spend_based(temp_input)

        return total

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.utcnow().isoformat()