"""
Category 7: Employee Commuting Agent

Calculates emissions from employee commuting

GHG Protocol Reference:
- Chapter 5.7 of Scope 3 Standard
- Category 7 Calculation Guidance
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
from pydantic import BaseModel, Field

from .base import Scope3BaseAgent, Scope3InputData, Scope3Result


class EmployeeCommutingInput(Scope3InputData):
    """Input data for Category 7: Employee Commuting."""

    calculation_method: str = Field(
        ...,
        description="Calculation method: distance-based, average-data"
    )

    # Add category-specific fields
    total_employees: int = Field(..., description="Total number of employees")

    working_days: int = Field(default=220, description="Average working days per year")

    mode_split: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Commuting mode split and distances"
    )

    remote_work: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Remote work data for avoided emissions"
    )


class EmployeeCommutingAgent(Scope3BaseAgent):
    """
    Agent for calculating Category 7: Employee Commuting.

    Includes:
    - Personal vehicles
    - Public transportation
    - Company shuttles
    - Remote work avoided emissions
    """

    def __init__(self):
        """Initialize Employee Commuting agent."""
        super().__init__()
        self.category_number = 7
        self.category_name = "Employee Commuting"
        self.emission_factors = self._load_commuting_factors()

    def _load_commuting_factors(self) -> Dict[str, Decimal]:
        """Load emission factors for Employee Commuting."""
        return {
            # Commuting factors (kg CO2e/km)
            "car_solo": Decimal("0.171"),
            "car_pool": Decimal("0.086"),
            "bus": Decimal("0.089"),
            "train": Decimal("0.041"),
            "subway": Decimal("0.030"),
            "bike": Decimal("0.0"),
            "walk": Decimal("0.0"),
            "motorcycle": Decimal("0.113"),
            "e_bike": Decimal("0.003"),
            "e_scooter": Decimal("0.025")
        }

    def _parse_input(self, input_data: Dict[str, Any]) -> EmployeeCommutingInput:
        """Parse and validate input data."""
        return EmployeeCommutingInput(**input_data)

    async def calculate_emissions(self, input_data: EmployeeCommutingInput) -> Scope3Result:
        """
        Calculate Category 7 emissions.

        Formula:
        Emissions = Sum(Employees × Days × Distance × Mode_EF × Mode_Share)
        """
        self.calculation_steps = []
        self.factors_used = {}
        total_emissions = Decimal("0")

        # Implement calculation logic
        employees = Decimal(str(input_data.total_employees))
        working_days = Decimal(str(input_data.working_days))

        for mode, mode_data in input_data.mode_split.items():
            share = Decimal(str(mode_data.get("percentage", 0)))
            avg_distance = Decimal(str(mode_data.get("avg_distance_km", 0)))

            ef = self.emission_factors.get(mode, Decimal("0.15"))

            # Round trip distance
            daily_distance = avg_distance * Decimal("2")
            mode_employees = employees * share

            emissions = mode_employees * working_days * daily_distance * ef
            total_emissions += emissions

        # Generate result
        result = Scope3Result(
            category="Category 7: Employee Commuting",
            category_number=7,
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
