# -*- coding: utf-8 -*-
"""
Scope 3 Calculator - Value Chain Emissions

Handles emissions from value chain across 15 categories:

Upstream (8 categories):
1. Purchased Goods & Services
2. Capital Goods
3. Fuel & Energy Related Activities (not in Scope 1/2)
4. Upstream Transportation & Distribution
5. Waste Generated in Operations
6. Business Travel
7. Employee Commuting
8. Upstream Leased Assets

Downstream (7 categories):
9. Downstream Transportation & Distribution
10. Processing of Sold Products
11. Use of Sold Products
12. End-of-Life Treatment of Sold Products
13. Downstream Leased Assets
14. Franchises
15. Investments

Reference: GHG Protocol Scope 3 Standard
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Dict, Any
from greenlang.utilities.determinism import FinancialDecimal
from .core_calculator import (
    EmissionCalculator,
    CalculationRequest,
    CalculationResult,
)
from .gas_decomposition import MultiGasCalculator, GasBreakdown


@dataclass
class Scope3Result:
    """
    Scope 3 calculation result with category breakdown.

    Attributes:
        calculation_result: Core calculation result
        category: Scope 3 category (1-15)
        category_name: Human-readable category name
        gas_breakdown: Individual gas contributions
        metadata: Additional context
    """
    calculation_result: CalculationResult
    category: int
    category_name: str
    gas_breakdown: Optional[GasBreakdown] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'scope': 'Scope 3',
            'category': self.category,
            'category_name': self.category_name,
            'calculation_result': self.calculation_result.to_dict(),
            'gas_breakdown': self.gas_breakdown.to_dict() if self.gas_breakdown else None,
            'metadata': self.metadata,
        }


# Scope 3 Category Definitions
SCOPE3_CATEGORIES = {
    1: "Purchased Goods and Services",
    2: "Capital Goods",
    3: "Fuel and Energy Related Activities",
    4: "Upstream Transportation and Distribution",
    5: "Waste Generated in Operations",
    6: "Business Travel",
    7: "Employee Commuting",
    8: "Upstream Leased Assets",
    9: "Downstream Transportation and Distribution",
    10: "Processing of Sold Products",
    11: "Use of Sold Products",
    12: "End-of-Life Treatment of Sold Products",
    13: "Downstream Leased Assets",
    14: "Franchises",
    15: "Investments",
}


class Scope3Calculator:
    """
    Scope 3 Value Chain Emissions Calculator

    Most comprehensive and challenging scope - often 70-90% of total footprint.

    Key Challenges:
    - Data availability (external to organization)
    - Allocation methodologies
    - Double counting prevention
    - 15 diverse categories requiring different methods
    """

    def __init__(self, emission_calculator: Optional[EmissionCalculator] = None):
        """
        Initialize Scope 3 calculator.

        Args:
            emission_calculator: Core calculator (auto-creates if None)
        """
        self.calculator = emission_calculator or EmissionCalculator()
        self.gas_calculator = MultiGasCalculator()

    def calculate_category_1_purchased_goods(
        self,
        material_type: str,
        quantity_kg: float,
        region: Optional[str] = None,
    ) -> Scope3Result:
        """
        Category 1: Purchased Goods and Services

        Cradle-to-gate emissions of purchased goods and services.
        Often largest Scope 3 category.

        Args:
            material_type: Material/product type (e.g., 'steel_blast_furnace', 'aluminum_primary')
            quantity_kg: Quantity purchased (kg)
            region: Geographic region

        Returns:
            Scope3Result for Category 1

        Example:
            >>> calc = Scope3Calculator()
            >>> result = calc.calculate_category_1_purchased_goods(
            ...     material_type='steel_blast_furnace',
            ...     quantity_kg=1000,
            ... )
        """
        request = CalculationRequest(
            factor_id=material_type,
            activity_amount=quantity_kg,
            activity_unit='kg',
            region=region,
        )

        calc_result = self.calculator.calculate(request)

        gas_breakdown = self.gas_calculator.decompose(
            total_co2e_kg=FinancialDecimal.from_string(calc_result.emissions_kg_co2e),
            fuel_type='electricity',  # Mixed composition
        )

        metadata = {
            'material_type': material_type,
            'quantity_kg': quantity_kg,
            'scope_3_category': 1,
        }

        return Scope3Result(
            calculation_result=calc_result,
            category=1,
            category_name=SCOPE3_CATEGORIES[1],
            gas_breakdown=gas_breakdown,
            metadata=metadata,
        )

    def calculate_category_3_fuel_energy(
        self,
        fuel_type: str,
        amount: float,
        unit: str,
        include_wtt: bool = True,
        region: Optional[str] = None,
    ) -> Scope3Result:
        """
        Category 3: Fuel and Energy Related Activities (not in Scope 1 or 2)

        Includes:
        - Well-to-tank (WTT) emissions for fuels
        - Transmission and distribution (T&D) losses for electricity
        - Generation of purchased electricity sold to end users

        Args:
            fuel_type: Fuel type (e.g., 'diesel', 'natural_gas')
            amount: Fuel quantity
            unit: Fuel unit
            include_wtt: Include well-to-tank upstream emissions
            region: Geographic region

        Returns:
            Scope3Result for Category 3

        Example:
            >>> calc = Scope3Calculator()
            >>> result = calc.calculate_category_3_fuel_energy(
            ...     fuel_type='diesel',
            ...     amount=100,
            ...     unit='gallons',
            ...     include_wtt=True
            ... )
        """
        # Calculate base emissions
        request = CalculationRequest(
            factor_id=fuel_type,
            activity_amount=amount,
            activity_unit=unit,
            region=region,
        )

        calc_result = self.calculator.calculate(request)

        # Add WTT emissions (typically 15-25% of combustion emissions)
        if include_wtt:
            wtt_factor = Decimal('0.20')  # 20% upstream emissions
            wtt_emissions = calc_result.emissions_kg_co2e * wtt_factor
            total_emissions = calc_result.emissions_kg_co2e + wtt_emissions
            calc_result.emissions_kg_co2e = total_emissions.quantize(Decimal('0.001'))

        gas_breakdown = self.gas_calculator.decompose(
            total_co2e_kg=FinancialDecimal.from_string(calc_result.emissions_kg_co2e),
            fuel_type=fuel_type,
        )

        metadata = {
            'fuel_type': fuel_type,
            'amount': amount,
            'unit': unit,
            'include_wtt': include_wtt,
            'scope_3_category': 3,
        }

        return Scope3Result(
            calculation_result=calc_result,
            category=3,
            category_name=SCOPE3_CATEGORIES[3],
            gas_breakdown=gas_breakdown,
            metadata=metadata,
        )

    def calculate_category_4_upstream_transport(
        self,
        mode: str,
        distance_km: float,
        weight_tonnes: float,
        region: Optional[str] = None,
    ) -> Scope3Result:
        """
        Category 4: Upstream Transportation and Distribution

        Emissions from transportation of purchased goods and services.

        Args:
            mode: Transport mode ('freight_truck_diesel', 'ocean_freight_container', 'air_freight')
            distance_km: Distance traveled (km)
            weight_tonnes: Weight transported (metric tonnes)
            region: Geographic region

        Returns:
            Scope3Result for Category 4

        Example:
            >>> calc = Scope3Calculator()
            >>> result = calc.calculate_category_4_upstream_transport(
            ...     mode='freight_truck_diesel',
            ...     distance_km=500,
            ...     weight_tonnes=10
            ... )
        """
        # Calculate tonne-km
        tonne_km = distance_km * weight_tonnes

        request = CalculationRequest(
            factor_id=mode,
            activity_amount=tonne_km,
            activity_unit='ton_km',
            region=region,
        )

        calc_result = self.calculator.calculate(request)

        gas_breakdown = self.gas_calculator.decompose(
            total_co2e_kg=FinancialDecimal.from_string(calc_result.emissions_kg_co2e),
            fuel_type='diesel' if 'truck' in mode else 'jet_fuel' if 'air' in mode else 'fuel_oil_no6',
        )

        metadata = {
            'transport_mode': mode,
            'distance_km': distance_km,
            'weight_tonnes': weight_tonnes,
            'tonne_km': tonne_km,
            'scope_3_category': 4,
        }

        return Scope3Result(
            calculation_result=calc_result,
            category=4,
            category_name=SCOPE3_CATEGORIES[4],
            gas_breakdown=gas_breakdown,
            metadata=metadata,
        )

    def calculate_category_6_business_travel(
        self,
        mode: str,
        distance_km: float,
        passengers: int = 1,
        cabin_class: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Scope3Result:
        """
        Category 6: Business Travel

        Emissions from employee business travel in vehicles not owned/controlled
        by the reporting organization.

        Args:
            mode: Travel mode ('air_short_haul', 'air_long_haul', 'rail_passenger', 'hotel_night')
            distance_km: Distance traveled (km) or nights (for hotels)
            passengers: Number of passengers
            cabin_class: For air travel ('economy', 'business', 'first')
            region: Geographic region

        Returns:
            Scope3Result for Category 6

        Example:
            >>> calc = Scope3Calculator()
            >>> # Air travel
            >>> result = calc.calculate_category_6_business_travel(
            ...     mode='air_long_haul',
            ...     distance_km=5000,
            ...     passengers=2,
            ...     cabin_class='economy'
            ... )
            >>> # Hotel nights
            >>> result = calc.calculate_category_6_business_travel(
            ...     mode='hotel_night',
            ...     distance_km=5,  # 5 nights
            ...     passengers=2
            ... )
        """
        # Calculate passenger-km (or passenger-nights for hotels)
        if mode == 'hotel_night':
            activity_amount = distance_km * passengers  # nights × passengers
            activity_unit = 'night'
        else:
            activity_amount = distance_km * passengers  # km × passengers
            activity_unit = 'pax_km'

        request = CalculationRequest(
            factor_id=mode,
            activity_amount=activity_amount,
            activity_unit=activity_unit,
            region=region,
        )

        calc_result = self.calculator.calculate(request)

        # Adjust for cabin class (business class ~2x, first class ~4x economy)
        if cabin_class and 'air' in mode:
            if cabin_class.lower() == 'business':
                calc_result.emissions_kg_co2e *= Decimal('2.0')
            elif cabin_class.lower() == 'first':
                calc_result.emissions_kg_co2e *= Decimal('4.0')

        gas_breakdown = self.gas_calculator.decompose(
            total_co2e_kg=FinancialDecimal.from_string(calc_result.emissions_kg_co2e),
            fuel_type='jet_fuel' if 'air' in mode else 'diesel' if 'rail' in mode else 'electricity',
        )

        metadata = {
            'travel_mode': mode,
            'distance_km': distance_km,
            'passengers': passengers,
            'cabin_class': cabin_class,
            'scope_3_category': 6,
        }

        return Scope3Result(
            calculation_result=calc_result,
            category=6,
            category_name=SCOPE3_CATEGORIES[6],
            gas_breakdown=gas_breakdown,
            metadata=metadata,
        )

    def calculate_category_5_waste(
        self,
        waste_type: str,
        waste_kg: float,
        treatment_method: str = 'landfill',
        region: Optional[str] = None,
    ) -> Scope3Result:
        """
        Category 5: Waste Generated in Operations

        Emissions from third-party disposal and treatment of waste.

        Args:
            waste_type: Type of waste ('organic', 'mixed', 'plastic', 'paper')
            waste_kg: Weight of waste (kg)
            treatment_method: 'landfill', 'incineration', 'composting', 'recycling'
            region: Geographic region

        Returns:
            Scope3Result for Category 5

        Example:
            >>> calc = Scope3Calculator()
            >>> result = calc.calculate_category_5_waste(
            ...     waste_type='organic',
            ...     waste_kg=1000,
            ...     treatment_method='landfill'
            ... )
        """
        factor_id = f"{treatment_method}_{waste_type}_waste"

        request = CalculationRequest(
            factor_id=factor_id,
            activity_amount=waste_kg,
            activity_unit='kg',
            region=region,
        )

        calc_result = self.calculator.calculate(request)

        # Landfill waste generates significant methane
        gas_vector = None
        if treatment_method == 'landfill':
            gas_vector = {'CO2': 0.4, 'CH4_biogenic': 0.6}  # High methane from anaerobic decomposition

        gas_breakdown = self.gas_calculator.decompose(
            total_co2e_kg=FinancialDecimal.from_string(calc_result.emissions_kg_co2e),
            gas_vector=gas_vector,
        )

        metadata = {
            'waste_type': waste_type,
            'waste_kg': waste_kg,
            'treatment_method': treatment_method,
            'scope_3_category': 5,
        }

        return Scope3Result(
            calculation_result=calc_result,
            category=5,
            category_name=SCOPE3_CATEGORIES[5],
            gas_breakdown=gas_breakdown,
            metadata=metadata,
        )

    def calculate_category_7_employee_commuting(
        self,
        mode: str,
        distance_km_per_day: float,
        employees: int,
        working_days_per_year: int = 220,
        region: Optional[str] = None,
    ) -> Scope3Result:
        """
        Category 7: Employee Commuting

        Emissions from employee commuting between home and work.

        Args:
            mode: Commute mode ('car_gasoline', 'car_diesel', 'bus', 'rail', 'bicycle')
            distance_km_per_day: Average one-way commute distance (km)
            employees: Number of employees
            working_days_per_year: Working days per year (default 220)
            region: Geographic region

        Returns:
            Scope3Result for Category 7

        Example:
            >>> calc = Scope3Calculator()
            >>> result = calc.calculate_category_7_employee_commuting(
            ...     mode='car_gasoline',
            ...     distance_km_per_day=20,  # 20 km one-way
            ...     employees=100,
            ...     working_days_per_year=220
            ... )
        """
        # Calculate total commute distance
        # Formula: distance × 2 (round trip) × employees × working days
        total_km = distance_km_per_day * 2 * employees * working_days_per_year

        # Map mode to factor_id
        mode_map = {
            'car_gasoline': 'gasoline',
            'car_diesel': 'diesel',
            'bus': 'diesel',  # Buses typically diesel
            'rail': 'electricity',  # Trains typically electric
            'bicycle': None,  # Zero emissions
            'walking': None,  # Zero emissions
        }

        if mode.lower() in ['bicycle', 'walking']:
            # Zero emissions for active transport
            from .core_calculator import (
                FactorResolution, FallbackLevel, CalculationStatus
            )

            request = CalculationRequest(
                factor_id=mode,
                activity_amount=total_km,
                activity_unit='km',
            )

            calc_result = CalculationResult(
                request=request,
                emissions_kg_co2e=Decimal('0'),
                status=CalculationStatus.SUCCESS,
                calculation_steps=[
                    {
                        'step': 1,
                        'description': f'{mode} has zero emissions',
                        'total_km': total_km,
                        'emissions_kg_co2e': '0.000',
                    }
                ],
            )
        else:
            # For motorized transport, need fuel consumption
            # Simplification: Use average fuel economy
            avg_fuel_economy_l_per_100km = {
                'car_gasoline': 8.0,
                'car_diesel': 6.5,
                'bus': 30.0,  # Per vehicle, divided by passengers
            }

            fuel_type = mode_map.get(mode, 'gasoline')
            fuel_economy = avg_fuel_economy_l_per_100km.get(mode, 8.0)
            fuel_liters = (total_km / 100) * fuel_economy

            request = CalculationRequest(
                factor_id=fuel_type,
                activity_amount=fuel_liters,
                activity_unit='liters',
                region=region,
            )

            calc_result = self.calculator.calculate(request)

        gas_breakdown = None
        if calc_result.emissions_kg_co2e > 0:
            gas_breakdown = self.gas_calculator.decompose(
                total_co2e_kg=FinancialDecimal.from_string(calc_result.emissions_kg_co2e),
                fuel_type='gasoline' if 'gasoline' in mode else 'diesel',
            )

        metadata = {
            'commute_mode': mode,
            'distance_km_per_day': distance_km_per_day,
            'employees': employees,
            'working_days_per_year': working_days_per_year,
            'total_km': total_km,
            'scope_3_category': 7,
        }

        return Scope3Result(
            calculation_result=calc_result,
            category=7,
            category_name=SCOPE3_CATEGORIES[7],
            gas_breakdown=gas_breakdown,
            metadata=metadata,
        )
