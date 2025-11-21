# -*- coding: utf-8 -*-
"""
Scope 1 Calculator - Direct GHG Emissions

Handles emissions from sources owned or controlled by the organization:
1. Stationary Combustion (boilers, furnaces, generators)
2. Mobile Combustion (company vehicles, fleet)
3. Process Emissions (chemical reactions, cement, steel)
4. Fugitive Emissions (refrigerant leaks, gas leaks)

Reference: GHG Protocol Corporate Standard
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Dict, Any
from greenlang.calculation.core_calculator import (
from greenlang.determinism import FinancialDecimal
    EmissionCalculator,
    CalculationRequest,
    CalculationResult,
)
from greenlang.calculation.gas_decomposition import MultiGasCalculator, GasBreakdown


@dataclass
class Scope1Result:
    """
    Scope 1 calculation result with category breakdown.

    Attributes:
        calculation_result: Core calculation result
        category: Scope 1 category (combustion, process, fugitive)
        gas_breakdown: Individual gas contributions
        metadata: Additional context
    """
    calculation_result: CalculationResult
    category: str
    gas_breakdown: Optional[GasBreakdown] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'scope': 'Scope 1',
            'category': self.category,
            'calculation_result': self.calculation_result.to_dict(),
            'gas_breakdown': self.gas_breakdown.to_dict() if self.gas_breakdown else None,
            'metadata': self.metadata,
        }


class Scope1Calculator:
    """
    Scope 1 Direct Emissions Calculator

    Categories:
    - Stationary Combustion: Fuel burned in stationary equipment
    - Mobile Combustion: Fuel burned in vehicles
    - Process Emissions: Industrial processes (cement, steel)
    - Fugitive Emissions: Refrigerant leaks, natural gas leaks
    """

    def __init__(self, emission_calculator: Optional[EmissionCalculator] = None):
        """
        Initialize Scope 1 calculator.

        Args:
            emission_calculator: Core calculator (auto-creates if None)
        """
        self.calculator = emission_calculator or EmissionCalculator()
        self.gas_calculator = MultiGasCalculator()

    def calculate_fuel_combustion(
        self,
        fuel_type: str,
        amount: float,
        unit: str,
        combustion_type: str = "stationary",
        region: Optional[str] = None,
        include_gas_breakdown: bool = True,
    ) -> Scope1Result:
        """
        Calculate emissions from fuel combustion.

        Args:
            fuel_type: Fuel type (e.g., 'natural_gas', 'diesel', 'gasoline')
            amount: Fuel quantity
            unit: Fuel unit (e.g., 'gallons', 'liters', 'therms')
            combustion_type: 'stationary' or 'mobile'
            region: Geographic region for factors
            include_gas_breakdown: Include CO2/CH4/N2O breakdown

        Returns:
            Scope1Result with emissions and breakdown

        Example:
            >>> calc = Scope1Calculator()
            >>> result = calc.calculate_fuel_combustion(
            ...     fuel_type='diesel',
            ...     amount=100,
            ...     unit='gallons',
            ...     combustion_type='stationary'
            ... )
            >>> print(result.calculation_result.emissions_kg_co2e)
        """
        # Create calculation request
        request = CalculationRequest(
            factor_id=fuel_type,
            activity_amount=amount,
            activity_unit=unit,
            region=region,
        )

        # Execute calculation
        calc_result = self.calculator.calculate(request)

        # Gas breakdown
        gas_breakdown = None
        if include_gas_breakdown and calc_result.emissions_kg_co2e > 0:
            gas_breakdown = self.gas_calculator.decompose(
                total_co2e_kg=FinancialDecimal.from_string(calc_result.emissions_kg_co2e),
                fuel_type=fuel_type,
            )

        # Metadata
        metadata = {
            'fuel_type': fuel_type,
            'combustion_type': combustion_type,
            'scope_1_category': f'{combustion_type}_combustion',
        }

        return Scope1Result(
            calculation_result=calc_result,
            category=f'{combustion_type}_combustion',
            gas_breakdown=gas_breakdown,
            metadata=metadata,
        )

    def calculate_process_emissions(
        self,
        process_type: str,
        production_amount: float,
        production_unit: str = 'kg',
        region: Optional[str] = None,
    ) -> Scope1Result:
        """
        Calculate process emissions from industrial processes.

        Process emissions occur from chemical/physical transformations:
        - Cement: Calcination of limestone (CaCO3 → CaO + CO2)
        - Steel: Blast furnace reduction
        - Aluminum: Electrolysis
        - Chemical production

        Args:
            process_type: Process type (e.g., 'cement_production', 'steel_blast_furnace')
            production_amount: Production quantity
            production_unit: Unit (typically 'kg', 'tonnes')
            region: Geographic region

        Returns:
            Scope1Result with process emissions

        Example:
            >>> calc = Scope1Calculator()
            >>> result = calc.calculate_process_emissions(
            ...     process_type='cement_production',
            ...     production_amount=1000,
            ...     production_unit='kg'
            ... )
        """
        request = CalculationRequest(
            factor_id=process_type,
            activity_amount=production_amount,
            activity_unit=production_unit,
            region=region,
        )

        calc_result = self.calculator.calculate(request)

        # Process emissions typically 100% CO2
        gas_breakdown = self.gas_calculator.decompose(
            total_co2e_kg=FinancialDecimal.from_string(calc_result.emissions_kg_co2e),
            gas_vector={'CO2': 1.0},
        )

        metadata = {
            'process_type': process_type,
            'production_amount': production_amount,
            'production_unit': production_unit,
            'scope_1_category': 'process_emissions',
        }

        return Scope1Result(
            calculation_result=calc_result,
            category='process_emissions',
            gas_breakdown=gas_breakdown,
            metadata=metadata,
        )

    def calculate_fugitive_emissions(
        self,
        refrigerant_type: str,
        charge_kg: float,
        annual_leakage_rate: Optional[float] = None,
        leaked_amount_kg: Optional[float] = None,
    ) -> Scope1Result:
        """
        Calculate fugitive emissions from refrigerant/gas leaks.

        Fugitive emissions are intentional or unintentional releases of GHGs:
        - Refrigerant leaks from A/C and refrigeration systems
        - Natural gas leaks from pipelines and equipment
        - Industrial gas leaks (SF6 from electrical equipment)

        Two calculation modes:
        1. Leaked amount: Direct measurement of leaked gas
        2. Leakage rate: Percentage of total charge leaked annually

        Args:
            refrigerant_type: Refrigerant type (e.g., 'HFC-134a', 'R-410A')
            charge_kg: Total refrigerant charge in system (kg)
            annual_leakage_rate: Annual leakage rate (0-1, e.g., 0.15 = 15%)
            leaked_amount_kg: Direct measurement of leaked amount (kg)

        Returns:
            Scope1Result with fugitive emissions

        Example:
            >>> calc = Scope1Calculator()
            >>> # Method 1: Using leakage rate
            >>> result = calc.calculate_fugitive_emissions(
            ...     refrigerant_type='HFC-134a',
            ...     charge_kg=10,
            ...     annual_leakage_rate=0.15  # 15% annual leakage
            ... )
            >>> # Method 2: Using leaked amount
            >>> result = calc.calculate_fugitive_emissions(
            ...     refrigerant_type='HFC-134a',
            ...     charge_kg=10,
            ...     leaked_amount_kg=1.5  # 1.5 kg leaked
            ... )
        """
        # Determine leaked amount
        if leaked_amount_kg is not None:
            leaked_kg = leaked_amount_kg
        elif annual_leakage_rate is not None:
            if not 0 <= annual_leakage_rate <= 1:
                raise ValueError(f"Leakage rate must be between 0 and 1, got {annual_leakage_rate}")
            leaked_kg = charge_kg * annual_leakage_rate
        else:
            raise ValueError("Must provide either 'leaked_amount_kg' or 'annual_leakage_rate'")

        # Look up refrigerant in processes category
        request = CalculationRequest(
            factor_id=f'refrigeration_{refrigerant_type.lower()}_leakage',
            activity_amount=leaked_kg,
            activity_unit='kg',
        )

        calc_result = self.calculator.calculate(request)

        # Fugitive emissions are 100% the refrigerant gas
        gas_breakdown = self.gas_calculator.decompose(
            total_co2e_kg=FinancialDecimal.from_string(calc_result.emissions_kg_co2e),
            gas_vector={refrigerant_type: 1.0},
        )

        metadata = {
            'refrigerant_type': refrigerant_type,
            'charge_kg': charge_kg,
            'leaked_kg': leaked_kg,
            'annual_leakage_rate': annual_leakage_rate,
            'scope_1_category': 'fugitive_emissions',
        }

        return Scope1Result(
            calculation_result=calc_result,
            category='fugitive_emissions',
            gas_breakdown=gas_breakdown,
            metadata=metadata,
        )

    def calculate_mobile_combustion(
        self,
        fuel_type: str,
        amount: float,
        unit: str,
        vehicle_type: Optional[str] = None,
        distance_km: Optional[float] = None,
        region: Optional[str] = None,
    ) -> Scope1Result:
        """
        Calculate emissions from mobile combustion (company vehicles).

        Args:
            fuel_type: Fuel type (e.g., 'gasoline', 'diesel')
            amount: Fuel quantity
            unit: Fuel unit (e.g., 'gallons', 'liters')
            vehicle_type: Vehicle type (e.g., 'sedan', 'truck')
            distance_km: Distance traveled (for fuel economy checks)
            region: Geographic region

        Returns:
            Scope1Result with mobile combustion emissions

        Example:
            >>> calc = Scope1Calculator()
            >>> result = calc.calculate_mobile_combustion(
            ...     fuel_type='gasoline',
            ...     amount=50,
            ...     unit='gallons',
            ...     vehicle_type='sedan',
            ...     distance_km=1000
            ... )
        """
        return self.calculate_fuel_combustion(
            fuel_type=fuel_type,
            amount=amount,
            unit=unit,
            combustion_type='mobile',
            region=region,
            include_gas_breakdown=True,
        )
