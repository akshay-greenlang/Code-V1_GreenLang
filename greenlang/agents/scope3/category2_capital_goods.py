# -*- coding: utf-8 -*-
"""
Scope 3 Category 2: Capital Goods Agent

Calculates emissions from extraction, production, and transportation of capital goods
purchased or acquired by the reporting company in the reporting year.

GHG Protocol Definition:
Capital goods are final products that have an extended life and are used by the company
to manufacture a product or provide a service.
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


class CapitalGoodsInput(Scope3InputData):
    """Input data model for Capital Goods emissions calculation."""

    purchases: List[Dict[str, Any]] = Field(
        ...,
        description="List of capital goods purchases"
    )

    class PurchaseItem(BaseModel):
        """Individual capital goods purchase."""
        asset_type: str = Field(..., description="Type of capital asset")
        quantity: float = Field(..., gt=0, description="Quantity purchased")
        unit: str = Field(..., description="Unit of quantity (e.g., units, kg, USD)")
        purchase_value_usd: Optional[float] = Field(None, gt=0, description="Purchase value in USD")
        material_type: Optional[str] = Field(None, description="Primary material (steel, aluminum, etc.)")
        weight_kg: Optional[float] = Field(None, gt=0, description="Weight in kg if known")
        supplier_emission_factor: Optional[float] = Field(
            None,
            description="Supplier-specific emission factor if available"
        )

    @validator('purchases')
    def validate_purchases(cls, v):
        """Validate purchase items."""
        if not v:
            raise ValueError("At least one capital goods purchase required")
        return v


class CapitalGoodsAgent(Scope3BaseAgent):
    """
    Agent for calculating Scope 3 Category 2: Capital Goods emissions.

    Calculation Methods (in order of preference per GHG Protocol):
    1. Supplier-specific method: Using supplier-provided data
    2. Average-data method: Using industry-average emission factors
    3. Spend-based method: Using EEIO factors based on purchase value
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize Capital Goods Agent."""
        if config is None:
            config = AgentConfig(
                name="CapitalGoodsAgent",
                description="Calculates Scope 3 Category 2: Capital Goods emissions",
                version="1.0.0"
            )
        super().__init__(config)

        # Load capital goods specific emission factors
        self.capital_goods_factors = self._load_capital_goods_factors()

    def _load_capital_goods_factors(self) -> Dict[str, Any]:
        """Load capital goods emission factors."""
        # Default factors (kg CO2e per unit)
        # In production, these would be loaded from YAML/database
        return {
            # Material-based factors (kg CO2e per kg)
            "materials": {
                "steel": Decimal("2.32"),  # kg CO2e per kg steel
                "aluminum": Decimal("11.89"),  # kg CO2e per kg aluminum
                "copper": Decimal("3.81"),  # kg CO2e per kg copper
                "plastic": Decimal("3.31"),  # kg CO2e per kg plastic
                "concrete": Decimal("0.13"),  # kg CO2e per kg concrete
                "glass": Decimal("0.85"),  # kg CO2e per kg glass
                "wood": Decimal("0.72"),  # kg CO2e per kg wood
            },
            # Asset type factors (kg CO2e per unit or USD)
            "asset_types": {
                "it_equipment": Decimal("500"),  # kg CO2e per unit
                "machinery": Decimal("0.45"),  # kg CO2e per USD
                "vehicles": Decimal("0.38"),  # kg CO2e per USD
                "buildings": Decimal("0.31"),  # kg CO2e per USD
                "furniture": Decimal("0.28"),  # kg CO2e per USD
                "hvac_equipment": Decimal("0.42"),  # kg CO2e per USD
                "solar_panels": Decimal("2500"),  # kg CO2e per kW
            },
            # EEIO factors (kg CO2e per USD spent)
            "eeio": {
                "computers_electronics": Decimal("0.31"),
                "machinery_equipment": Decimal("0.45"),
                "motor_vehicles": Decimal("0.38"),
                "construction": Decimal("0.31"),
                "furniture_fixtures": Decimal("0.28"),
                "default": Decimal("0.35"),  # Default EEIO factor
            }
        }

    def _parse_input(self, input_data: Dict[str, Any]) -> CapitalGoodsInput:
        """Parse and validate input data."""
        return CapitalGoodsInput(**input_data)

    async def calculate_emissions(
        self,
        input_data: CapitalGoodsInput
    ) -> Scope3Result:
        """
        Calculate Capital Goods emissions using best available method.

        Formula varies by method:
        1. Supplier-specific: Σ(quantity × supplier_emission_factor)
        2. Average-data: Σ(weight_kg × material_emission_factor)
        3. Spend-based: Σ(spend_USD × EEIO_factor)
        """
        total_emissions = Decimal("0")
        methodology_used = []

        for idx, purchase in enumerate(input_data.purchases):
            purchase_emissions = Decimal("0")
            method = ""

            # Method 1: Supplier-specific (highest quality)
            if purchase.get('supplier_emission_factor'):
                factor = Decimal(str(purchase['supplier_emission_factor']))
                quantity = Decimal(str(purchase['quantity']))
                purchase_emissions = quantity * factor
                method = "supplier-specific"

                self._record_calculation_step(
                    description=f"Calculate emissions for {purchase['asset_type']} using supplier data",
                    operation="multiply",
                    inputs={
                        "quantity": quantity,
                        "supplier_factor": factor
                    },
                    output_value=purchase_emissions,
                    output_name=f"purchase_{idx}_emissions",
                    formula="quantity × supplier_emission_factor",
                    unit="kg CO2e"
                )

            # Method 2: Average-data using weight and material
            elif purchase.get('weight_kg') and purchase.get('material_type'):
                weight = Decimal(str(purchase['weight_kg']))
                material = purchase['material_type'].lower()

                if material in self.capital_goods_factors['materials']:
                    factor = self.capital_goods_factors['materials'][material]
                    purchase_emissions = weight * factor
                    method = "average-data"

                    self._record_calculation_step(
                        description=f"Calculate emissions for {purchase['asset_type']} using material factor",
                        operation="multiply",
                        inputs={
                            "weight_kg": weight,
                            "material_factor": factor,
                            "material": material
                        },
                        output_value=purchase_emissions,
                        output_name=f"purchase_{idx}_emissions",
                        formula="weight_kg × material_emission_factor",
                        unit="kg CO2e"
                    )

            # Method 3: Spend-based (lowest quality but always available)
            elif purchase.get('purchase_value_usd'):
                spend = Decimal(str(purchase['purchase_value_usd']))
                asset_type = purchase['asset_type'].lower()

                # Try specific asset type factor
                if asset_type in self.capital_goods_factors['asset_types']:
                    factor = self.capital_goods_factors['asset_types'][asset_type]
                else:
                    # Fall back to EEIO factors
                    factor = self.capital_goods_factors['eeio'].get(
                        asset_type,
                        self.capital_goods_factors['eeio']['default']
                    )

                purchase_emissions = spend * factor
                method = "spend-based"

                self._record_calculation_step(
                    description=f"Calculate emissions for {purchase['asset_type']} using spend-based method",
                    operation="multiply",
                    inputs={
                        "purchase_value_usd": spend,
                        "eeio_factor": factor,
                        "asset_type": asset_type
                    },
                    output_value=purchase_emissions,
                    output_name=f"purchase_{idx}_emissions",
                    formula="spend_USD × EEIO_factor",
                    unit="kg CO2e"
                )

            # Add to total
            if purchase_emissions > 0:
                total_emissions += purchase_emissions
                methodology_used.append(method)

                self.factors_used[f"purchase_{idx}"] = {
                    "asset_type": purchase['asset_type'],
                    "method": method,
                    "emissions": float(purchase_emissions)
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

        # Determine overall methodology
        if "supplier-specific" in methodology_used:
            primary_methodology = "supplier-specific"
            data_quality = 2.0
        elif "average-data" in methodology_used:
            primary_methodology = "average-data"
            data_quality = 3.0
        else:
            primary_methodology = "spend-based"
            data_quality = 4.0

        # Calculate uncertainty
        uncertainty = self._estimate_uncertainty(data_quality, primary_methodology)

        # Generate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            input_data.dict(),
            self.calculation_steps,
            total_emissions
        )

        return Scope3Result(
            category="Category 2: Capital Goods",
            category_number=2,
            total_emissions_kg_co2e=self._apply_precision(total_emissions),
            total_emissions_t_co2e=self._apply_precision(total_emissions_tonnes),
            calculation_methodology=primary_methodology,
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

        Example: Company purchases IT equipment and machinery
        """
        example_input = {
            "reporting_year": 2024,
            "reporting_entity": "Example Corp",
            "purchases": [
                {
                    "asset_type": "IT_Equipment",
                    "quantity": 100,
                    "unit": "units",
                    "purchase_value_usd": 150000,
                    "supplier_emission_factor": 450  # kg CO2e per unit
                },
                {
                    "asset_type": "Machinery",
                    "quantity": 5,
                    "unit": "units",
                    "weight_kg": 2000,
                    "material_type": "steel",
                    "purchase_value_usd": 500000
                }
            ]
        }

        example_calculation = """
        Example Calculation:

        1. IT Equipment (Supplier-specific method):
           100 units × 450 kg CO2e/unit = 45,000 kg CO2e

        2. Machinery (Average-data method):
           2,000 kg × 2.32 kg CO2e/kg steel = 4,640 kg CO2e

        Total: 49,640 kg CO2e = 49.64 t CO2e
        """

        return {
            "example_input": example_input,
            "calculation": example_calculation,
            "result": "49.64 t CO2e"
        }