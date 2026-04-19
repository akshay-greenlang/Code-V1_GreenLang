"""
Script to generate remaining Scope 3 category implementations
"""

import os
from pathlib import Path

# Template for each category
CATEGORY_TEMPLATE = '''"""
Category {number}: {name} Agent

{description}

GHG Protocol Reference:
- Chapter 5.{number} of Scope 3 Standard
- Category {number} Calculation Guidance
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field

from .base import Scope3BaseAgent, Scope3InputData, Scope3Result


class {class_name}Input(Scope3InputData):
    """Input data for Category {number}: {name}."""

    calculation_method: str = Field(
        ...,
        description="Calculation method: {methods}"
    )

    # Add category-specific fields
    {input_fields}


class {class_name}Agent(Scope3BaseAgent):
    """
    Agent for calculating Category {number}: {name}.

    {detailed_description}
    """

    def __init__(self):
        """Initialize {name} agent."""
        super().__init__()
        self.category_number = {number}
        self.category_name = "{name}"
        self.emission_factors = self._load_{factor_name}_factors()

    def _load_{factor_name}_factors(self) -> Dict[str, Decimal]:
        """Load emission factors for {name}."""
        return {{
            {emission_factors}
        }}

    def _parse_input(self, input_data: Dict[str, Any]) -> {class_name}Input:
        """Parse and validate input data."""
        return {class_name}Input(**input_data)

    async def calculate_emissions(self, input_data: {class_name}Input) -> Scope3Result:
        """
        Calculate Category {number} emissions.

        Formula:
        {formula}
        """
        self.calculation_steps = []
        self.factors_used = {{}}
        total_emissions = Decimal("0")

        # Implement calculation logic
        {calculation_logic}

        # Generate result
        result = Scope3Result(
            category="Category {number}: {name}",
            category_number={number},
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
'''

# Category definitions
CATEGORIES = [
    {
        "number": 5,
        "name": "Waste Generated in Operations",
        "class_name": "WasteGenerated",
        "factor_name": "waste",
        "methods": "waste-type-specific, average-data",
        "description": "Calculates emissions from third-party disposal and treatment of waste generated in operations",
        "detailed_description": """Includes:
    - Disposal in landfills
    - Incineration
    - Recycling
    - Composting
    - Wastewater treatment""",
        "input_fields": """waste_streams: Optional[Dict[str, Dict[str, Any]]] = Field(
        default=None,
        description="Waste by type and treatment method"
    )

    wastewater: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Wastewater treatment data"
    )""",
        "emission_factors": """# Landfill factors (kg CO2e/tonne)
            "landfill_mixed": Decimal("467.0"),
            "landfill_organic": Decimal("558.0"),
            "landfill_plastic": Decimal("2.8"),

            # Incineration factors
            "incineration_mixed": Decimal("907.0"),
            "incineration_energy_recovery": Decimal("255.0"),

            # Recycling factors
            "recycling_paper": Decimal("21.0"),
            "recycling_plastic": Decimal("21.0"),
            "recycling_metal": Decimal("21.0"),

            # Composting
            "composting": Decimal("55.0")""",
        "formula": "Emissions = Σ(Waste_Weight × Treatment_EF)",
        "calculation_logic": """if input_data.waste_streams:
            for waste_type, waste_data in input_data.waste_streams.items():
                weight = Decimal(str(waste_data.get("weight_tonnes", 0)))
                treatment = waste_data.get("treatment", "landfill")

                ef_key = f"{treatment}_{waste_type}"
                ef = self.emission_factors.get(ef_key, self.emission_factors.get(f"{treatment}_mixed"))

                emissions = weight * ef
                total_emissions += emissions

                self._record_calculation_step(
                    description=f"Calculate {treatment} emissions for {waste_type}",
                    operation="multiply",
                    inputs={"weight_tonnes": weight, "emission_factor": ef},
                    output_value=emissions,
                    output_name=f"emissions_{waste_type}_{treatment}",
                    formula="Emissions = Weight × Treatment_EF",
                    unit="kg CO2e"
                )"""
    },
    {
        "number": 6,
        "name": "Business Travel",
        "class_name": "BusinessTravel",
        "factor_name": "travel",
        "methods": "distance-based, spend-based",
        "description": "Calculates emissions from employee business travel",
        "detailed_description": """Includes:
    - Air travel
    - Rail travel
    - Road travel (rental cars, taxis)
    - Hotel stays""",
        "input_fields": """air_travel: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Air travel by distance and class"
    )

    rail_travel: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Rail travel distances"
    )

    road_travel: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Car rental and taxi data"
    )

    hotel_stays: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Hotel nights by country"
    )

    travel_spend: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Travel spend by category (for spend-based)"
    )""",
        "emission_factors": """# Air travel factors (kg CO2e/km)
            "air_domestic": Decimal("0.244"),
            "air_short_haul": Decimal("0.156"),
            "air_long_haul": Decimal("0.193"),

            # Class multipliers
            "economy_class": Decimal("1.0"),
            "premium_economy": Decimal("1.6"),
            "business_class": Decimal("2.9"),
            "first_class": Decimal("4.0"),

            # Rail factors
            "rail_national": Decimal("0.041"),
            "rail_international": Decimal("0.014"),

            # Road factors (kg CO2e/km)
            "car_small": Decimal("0.147"),
            "car_medium": Decimal("0.186"),
            "car_large": Decimal("0.279"),
            "taxi": Decimal("0.208"),

            # Hotel factors (kg CO2e/night)
            "hotel_us": Decimal("31.1"),
            "hotel_eu": Decimal("21.0"),
            "hotel_global": Decimal("25.0")""",
        "formula": "Emissions = Σ(Distance × Mode_EF × Class_Multiplier) + Hotel_Nights × Hotel_EF",
        "calculation_logic": """if input_data.calculation_method == "distance-based":
            # Air travel
            if input_data.air_travel:
                for flight_type, data in input_data.air_travel.items():
                    distance = Decimal(str(data.get("distance_km", 0)))
                    class_type = data.get("class", "economy_class")

                    ef = self.emission_factors.get(f"air_{flight_type}")
                    multiplier = self.emission_factors.get(class_type, Decimal("1.0"))

                    emissions = distance * ef * multiplier
                    total_emissions += emissions

            # Hotel stays
            if input_data.hotel_stays:
                nights = Decimal(str(input_data.hotel_stays.get("nights", 0)))
                region = input_data.hotel_stays.get("region", "global")

                ef = self.emission_factors.get(f"hotel_{region}", self.emission_factors["hotel_global"])
                emissions = nights * ef
                total_emissions += emissions"""
    },
    {
        "number": 7,
        "name": "Employee Commuting",
        "class_name": "EmployeeCommuting",
        "factor_name": "commuting",
        "methods": "distance-based, average-data",
        "description": "Calculates emissions from employee commuting",
        "detailed_description": """Includes:
    - Personal vehicles
    - Public transportation
    - Company shuttles
    - Remote work avoided emissions""",
        "input_fields": """total_employees: int = Field(..., description="Total number of employees")

    working_days: int = Field(default=220, description="Average working days per year")

    mode_split: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Commuting mode split and distances"
    )

    remote_work: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Remote work data for avoided emissions"
    )""",
        "emission_factors": """# Commuting factors (kg CO2e/km)
            "car_solo": Decimal("0.171"),
            "car_pool": Decimal("0.086"),
            "bus": Decimal("0.089"),
            "train": Decimal("0.041"),
            "subway": Decimal("0.030"),
            "bike": Decimal("0.0"),
            "walk": Decimal("0.0"),
            "motorcycle": Decimal("0.113"),
            "e_bike": Decimal("0.003"),
            "e_scooter": Decimal("0.025")""",
        "formula": "Emissions = Σ(Employees × Days × Distance × Mode_EF × Mode_Share)",
        "calculation_logic": """employees = Decimal(str(input_data.total_employees))
        working_days = Decimal(str(input_data.working_days))

        for mode, mode_data in input_data.mode_split.items():
            share = Decimal(str(mode_data.get("percentage", 0)))
            avg_distance = Decimal(str(mode_data.get("avg_distance_km", 0)))

            ef = self.emission_factors.get(mode, Decimal("0.15"))

            # Round trip distance
            daily_distance = avg_distance * Decimal("2")
            mode_employees = employees * share

            emissions = mode_employees * working_days * daily_distance * ef
            total_emissions += emissions"""
    },
    {
        "number": 8,
        "name": "Upstream Leased Assets",
        "class_name": "UpstreamLeasedAssets",
        "factor_name": "leased_assets",
        "methods": "asset-specific, average-data",
        "description": "Calculates emissions from leased assets not in Scope 1 or 2",
        "detailed_description": """Includes:
    - Leased facilities
    - Leased vehicles
    - Leased equipment
    (Only if not already included in Scope 1 or 2)""",
        "input_fields": """leased_assets: List[Dict[str, Any]] = Field(
        ...,
        description="List of leased assets with energy data"
    )

    allocation_method: str = Field(
        default="area",
        description="Allocation method: area, headcount, or revenue"
    )""",
        "emission_factors": """# Energy factors (kg CO2e/kWh)
            "electricity_grid": Decimal("0.433"),
            "natural_gas": Decimal("0.185"),
            "fuel_oil": Decimal("0.256"),

            # Asset intensity factors
            "office_kwh_per_m2": Decimal("135"),
            "warehouse_kwh_per_m2": Decimal("95"),
            "retail_kwh_per_m2": Decimal("165")""",
        "formula": "Emissions = Σ(Asset_Energy × EF) - Scope_1_2_Portion",
        "calculation_logic": """for asset in input_data.leased_assets:
            if asset.get("included_in_scope_1_2", False):
                continue  # Skip if already in Scope 1/2

            energy_kwh = Decimal(str(asset.get("energy_kwh", 0)))
            energy_type = asset.get("energy_type", "electricity_grid")

            ef = self.emission_factors.get(energy_type)
            emissions = energy_kwh * ef
            total_emissions += emissions"""
    },
    {
        "number": 9,
        "name": "Downstream Transportation and Distribution",
        "class_name": "DownstreamTransport",
        "factor_name": "downstream_transport",
        "methods": "distance-based, spend-based",
        "description": "Calculates emissions from transportation and distribution of sold products",
        "detailed_description": """Includes:
    - Transportation from company to customer
    - Third-party distribution
    - Retail storage""",
        "input_fields": """shipments: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Outbound shipment data"
    )

    distribution_spend: Optional[Dict[str, Decimal]] = Field(
        default=None,
        description="Distribution spend by mode"
    )""",
        "emission_factors": """# Same as upstream transport
            "truck": Decimal("0.145"),
            "rail": Decimal("0.024"),
            "ship": Decimal("0.012"),
            "air": Decimal("0.809")""",
        "formula": "Emissions = Σ(Distance × Weight × Mode_EF)",
        "calculation_logic": """# Similar to upstream transport
        if input_data.shipments:
            for shipment in input_data.shipments:
                distance = Decimal(str(shipment.get("distance_km", 0)))
                weight = Decimal(str(shipment.get("weight_tonnes", 0)))
                mode = shipment.get("mode", "truck")

                ef = self.emission_factors.get(mode)
                emissions = distance * weight * ef
                total_emissions += emissions"""
    },
    {
        "number": 11,
        "name": "Use of Sold Products",
        "class_name": "UseOfSoldProducts",
        "factor_name": "product_use",
        "methods": "direct-use-phase, indirect-use-phase",
        "description": "Calculates emissions from the use of sold products by end users",
        "detailed_description": """Includes:
    - Direct use-phase emissions (fuel/energy consuming products)
    - Indirect use-phase emissions (products requiring energy)
    - Lifetime energy consumption""",
        "input_fields": """products_sold: List[Dict[str, Any]] = Field(
        ...,
        description="Products sold with energy consumption data"
    )

    use_scenario: str = Field(
        default="average",
        description="Use scenario: minimum, average, or intensive"
    )""",
        "emission_factors": """# Grid factors by region (kg CO2e/kWh)
            "grid_us": Decimal("0.433"),
            "grid_eu": Decimal("0.295"),
            "grid_global": Decimal("0.475"),

            # Fuel factors (kg CO2e/liter)
            "gasoline": Decimal("2.31"),
            "diesel": Decimal("2.68")""",
        "formula": "Emissions = Units_Sold × Lifetime_Energy × Energy_EF",
        "calculation_logic": """for product in input_data.products_sold:
            units = Decimal(str(product.get("units_sold", 0)))
            lifetime_years = Decimal(str(product.get("lifetime_years", 5)))
            annual_energy = Decimal(str(product.get("annual_energy_kwh", 0)))

            grid_region = product.get("grid_region", "global")
            ef = self.emission_factors.get(f"grid_{grid_region}")

            lifetime_energy = annual_energy * lifetime_years
            emissions = units * lifetime_energy * ef
            total_emissions += emissions"""
    },
    {
        "number": 12,
        "name": "End-of-Life Treatment of Sold Products",
        "class_name": "EndOfLifeTreatment",
        "factor_name": "eol",
        "methods": "waste-type-specific, average-data",
        "description": "Calculates emissions from disposal and treatment of sold products at end of life",
        "detailed_description": """Includes:
    - Landfilling
    - Incineration
    - Recycling
    - Material recovery""",
        "input_fields": """products_eol: Dict[str, Any] = Field(
        ...,
        description="End-of-life data for sold products"
    )

    material_composition: Dict[str, float] = Field(
        ...,
        description="Material composition of products"
    )

    treatment_scenario: Dict[str, float] = Field(
        ...,
        description="Treatment method distribution"
    )""",
        "emission_factors": """# EOL treatment factors (kg CO2e/tonne)
            "landfill_plastic": Decimal("2.8"),
            "landfill_paper": Decimal("1184"),
            "landfill_metal": Decimal("2.8"),

            "incineration_plastic": Decimal("2766"),
            "incineration_paper": Decimal("15.8"),

            "recycling_plastic": Decimal("-1666"),  # Avoided emissions
            "recycling_paper": Decimal("-3530"),
            "recycling_metal": Decimal("-1811")""",
        "formula": "Emissions = Σ(Product_Weight × Material_% × Treatment_% × Treatment_EF)",
        "calculation_logic": """total_weight = Decimal(str(input_data.products_eol.get("total_weight_tonnes", 0)))

        for material, composition in input_data.material_composition.items():
            material_weight = total_weight * Decimal(str(composition))

            for treatment, percentage in input_data.treatment_scenario.items():
                treatment_weight = material_weight * Decimal(str(percentage))

                ef_key = f"{treatment}_{material}"
                ef = self.emission_factors.get(ef_key, Decimal("0"))

                emissions = treatment_weight * ef
                total_emissions += emissions"""
    },
    {
        "number": 13,
        "name": "Downstream Leased Assets",
        "class_name": "DownstreamLeasedAssets",
        "factor_name": "downstream_leased",
        "methods": "asset-specific, average-data",
        "description": "Calculates emissions from assets leased to others",
        "detailed_description": """Includes:
    - Buildings leased to tenants
    - Vehicles leased to customers
    - Equipment leased to others""",
        "input_fields": """leased_out_assets: List[Dict[str, Any]] = Field(
        ...,
        description="Assets leased to others"
    )""",
        "emission_factors": """# Similar to upstream leased
            "electricity_grid": Decimal("0.433"),
            "natural_gas": Decimal("0.185")""",
        "formula": "Emissions = Σ(Leased_Asset_Energy × EF)",
        "calculation_logic": """for asset in input_data.leased_out_assets:
            energy_kwh = Decimal(str(asset.get("energy_kwh", 0)))
            ef = self.emission_factors.get("electricity_grid")

            emissions = energy_kwh * ef
            total_emissions += emissions"""
    },
    {
        "number": 14,
        "name": "Franchises",
        "class_name": "Franchises",
        "factor_name": "franchise",
        "methods": "franchise-specific, average-data",
        "description": "Calculates emissions from franchise operations",
        "detailed_description": """Includes:
    - Scope 1 and 2 emissions of franchises
    - Energy use at franchise locations""",
        "input_fields": """franchises: List[Dict[str, Any]] = Field(
        ...,
        description="Franchise locations with energy data"
    )

    franchise_count: Optional[int] = Field(
        default=None,
        description="Number of franchises (for average-data method)"
    )""",
        "emission_factors": """# Average franchise emissions (kg CO2e/year)
            "restaurant": Decimal("150000"),
            "retail": Decimal("75000"),
            "service": Decimal("25000")""",
        "formula": "Emissions = Σ(Franchise_Energy × EF) + Σ(Franchise_Fuel × Fuel_EF)",
        "calculation_logic": """if input_data.calculation_method == "franchise-specific":
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
            total_emissions = count * avg_emissions"""
    }
]

def generate_category_files():
    """Generate all category implementation files."""
    base_dir = Path(__file__).parent

    for cat in CATEGORIES:
        filename = f"category_{cat['number']:02d}_{cat['factor_name']}.py"
        filepath = base_dir / filename

        content = CATEGORY_TEMPLATE.format(**cat)

        # Replace special characters for encoding
        content = content.replace('Σ', 'Sum')

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Created: {filename}")

if __name__ == "__main__":
    generate_category_files()
    print("\nAll category files generated successfully!")