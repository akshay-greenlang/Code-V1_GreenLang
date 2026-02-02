# -*- coding: utf-8 -*-
"""
Scope 2 Calculator - Indirect Energy Emissions

Handles emissions from purchased electricity, steam, heating, and cooling.

Two methods:
1. Location-Based: Uses average emission factor for grid region
2. Market-Based: Uses supplier-specific factors and renewable energy certificates

Reference: GHG Protocol Scope 2 Guidance (2015)
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
class Scope2Result:
    """
    Scope 2 calculation result.

    Attributes:
        calculation_result: Core calculation result
        method: 'location_based' or 'market_based'
        energy_type: Type of energy (electricity, steam, etc.)
        gas_breakdown: Individual gas contributions
        metadata: Additional context (grid region, supplier, RECs)
    """
    calculation_result: CalculationResult
    method: str
    energy_type: str
    gas_breakdown: Optional[GasBreakdown] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'scope': 'Scope 2',
            'method': self.method,
            'energy_type': self.energy_type,
            'calculation_result': self.calculation_result.to_dict(),
            'gas_breakdown': self.gas_breakdown.to_dict() if self.gas_breakdown else None,
            'metadata': self.metadata,
        }


class Scope2Calculator:
    """
    Scope 2 Indirect Energy Emissions Calculator

    Supports:
    - Electricity (location-based and market-based)
    - Steam (district heating/cooling)
    - Other purchased energy

    Key Concepts:
    - Location-Based: Uses regional grid average emission factor
    - Market-Based: Uses supplier-specific factors, adjusted for RECs
    - Dual Reporting: Many standards require BOTH methods
    """

    def __init__(self, emission_calculator: Optional[EmissionCalculator] = None):
        """
        Initialize Scope 2 calculator.

        Args:
            emission_calculator: Core calculator (auto-creates if None)
        """
        self.calculator = emission_calculator or EmissionCalculator()
        self.gas_calculator = MultiGasCalculator()

    def calculate_location_based(
        self,
        electricity_kwh: float,
        grid_region: str,
        year: Optional[int] = None,
    ) -> Scope2Result:
        """
        Calculate location-based Scope 2 emissions.

        Uses average emission intensity of the grid where energy is consumed.

        Args:
            electricity_kwh: Electricity consumption (kWh)
            grid_region: Grid region code (e.g., 'US_NATIONAL', 'US_WECC_CA', 'UK', 'DE')
            year: Year for emission factors (defaults to latest)

        Returns:
            Scope2Result with location-based emissions

        Example:
            >>> calc = Scope2Calculator()
            >>> result = calc.calculate_location_based(
            ...     electricity_kwh=10000,
            ...     grid_region='US_WECC_CA',  # California grid
            ...     year=2023
            ... )
            >>> print(result.calculation_result.emissions_kg_co2e)
        """
        # Grid region is the factor_id
        request = CalculationRequest(
            factor_id=grid_region,
            activity_amount=electricity_kwh,
            activity_unit='kwh',
        )

        calc_result = self.calculator.calculate(request)

        # Grid electricity gas composition
        gas_breakdown = None
        if calc_result.emissions_kg_co2e > 0:
            gas_breakdown = self.gas_calculator.decompose(
                total_co2e_kg=FinancialDecimal.from_string(calc_result.emissions_kg_co2e),
                fuel_type='electricity',  # Uses electricity default vector
            )

        metadata = {
            'grid_region': grid_region,
            'electricity_kwh': electricity_kwh,
            'year': year,
            'scope_2_method': 'location_based',
        }

        return Scope2Result(
            calculation_result=calc_result,
            method='location_based',
            energy_type='electricity',
            gas_breakdown=gas_breakdown,
            metadata=metadata,
        )

    def calculate_market_based(
        self,
        electricity_kwh: float,
        supplier_factor_kg_co2e_per_kwh: Optional[float] = None,
        rec_certificates_kwh: float = 0,
        grid_region: Optional[str] = None,
        residual_mix_factor: Optional[float] = None,
    ) -> Scope2Result:
        """
        Calculate market-based Scope 2 emissions.

        Uses contractual instruments (supplier-specific factors, RECs, PPAs).

        Hierarchy for market-based method:
        1. Supplier-specific factor (from contract/disclosure)
        2. Residual mix factor (grid average minus renewable claims)
        3. Location-based factor (fallback if no contractual data)

        Args:
            electricity_kwh: Total electricity consumption (kWh)
            supplier_factor_kg_co2e_per_kwh: Supplier-specific emission factor
            rec_certificates_kwh: Renewable energy certificates (kWh)
            grid_region: Grid region (for fallback)
            residual_mix_factor: Residual mix emission factor (kg CO2e/kWh)

        Returns:
            Scope2Result with market-based emissions

        Example:
            >>> calc = Scope2Calculator()
            >>> # Example 1: Using supplier-specific factor
            >>> result = calc.calculate_market_based(
            ...     electricity_kwh=10000,
            ...     supplier_factor_kg_co2e_per_kwh=0.250,  # Supplier-specific
            ... )
            >>> # Example 2: With RECs
            >>> result = calc.calculate_market_based(
            ...     electricity_kwh=10000,
            ...     supplier_factor_kg_co2e_per_kwh=0.385,
            ...     rec_certificates_kwh=5000,  # 50% renewable via RECs
            ... )
        """
        # Validate inputs
        if rec_certificates_kwh > electricity_kwh:
            raise ValueError(
                f"REC certificates ({rec_certificates_kwh} kWh) cannot exceed "
                f"total consumption ({electricity_kwh} kWh)"
            )

        # Determine emission factor to use
        if supplier_factor_kg_co2e_per_kwh is not None:
            # Use supplier-specific factor
            emission_factor = Decimal(str(supplier_factor_kg_co2e_per_kwh))
            factor_source = "supplier_specific"
        elif residual_mix_factor is not None:
            # Use residual mix
            emission_factor = Decimal(str(residual_mix_factor))
            factor_source = "residual_mix"
        elif grid_region is not None:
            # Fallback to location-based
            location_result = self.calculate_location_based(
                electricity_kwh=electricity_kwh,
                grid_region=grid_region,
            )
            # Adjust for RECs
            if rec_certificates_kwh > 0:
                # RECs have zero emission factor
                net_emissions = (
                    location_result.calculation_result.emissions_kg_co2e *
                    (electricity_kwh - rec_certificates_kwh) / electricity_kwh
                )
                location_result.calculation_result.emissions_kg_co2e = net_emissions.quantize(
                    Decimal('0.001')
                )

            location_result.method = 'market_based'
            location_result.metadata['factor_source'] = 'location_based_fallback'
            location_result.metadata['rec_certificates_kwh'] = rec_certificates_kwh
            return location_result
        else:
            raise ValueError(
                "Must provide one of: supplier_factor_kg_co2e_per_kwh, "
                "residual_mix_factor, or grid_region"
            )

        # Calculate emissions
        # Formula: (total_kwh - rec_kwh) × emission_factor
        net_kwh = electricity_kwh - rec_certificates_kwh
        emissions_kg_co2e = Decimal(str(net_kwh)) * emission_factor
        emissions_kg_co2e = emissions_kg_co2e.quantize(Decimal('0.001'))

        # Create a pseudo calculation result
        # (since we're not using the standard calculator for market-based)
        from .core_calculator import (
            FactorResolution,
            FallbackLevel,
            CalculationStatus,
        )

        factor_resolution = FactorResolution(
            factor_id='market_based_electricity',
            factor_value=emission_factor,
            factor_unit='kg CO2e per kWh',
            source=factor_source,
            uri='',
            last_updated='',
            fallback_level=FallbackLevel.EXACT,
        )

        # Create request for audit trail
        request = CalculationRequest(
            factor_id='market_based_electricity',
            activity_amount=net_kwh,
            activity_unit='kwh',
        )

        calc_result = CalculationResult(
            request=request,
            emissions_kg_co2e=emissions_kg_co2e,
            factor_resolution=factor_resolution,
            calculation_steps=[
                {
                    'step': 1,
                    'description': 'Calculate market-based emissions',
                    'total_kwh': electricity_kwh,
                    'rec_certificates_kwh': rec_certificates_kwh,
                    'net_kwh': float(net_kwh),
                    'emission_factor': str(emission_factor),
                    'emissions_kg_co2e': str(emissions_kg_co2e),
                }
            ],
            status=CalculationStatus.SUCCESS,
        )

        # Gas breakdown
        gas_breakdown = None
        if emissions_kg_co2e > 0:
            gas_breakdown = self.gas_calculator.decompose(
                total_co2e_kg=FinancialDecimal.from_string(emissions_kg_co2e),
                fuel_type='electricity',
            )

        metadata = {
            'electricity_kwh': electricity_kwh,
            'rec_certificates_kwh': rec_certificates_kwh,
            'net_kwh': float(net_kwh),
            'emission_factor': str(emission_factor),
            'factor_source': factor_source,
            'scope_2_method': 'market_based',
        }

        return Scope2Result(
            calculation_result=calc_result,
            method='market_based',
            energy_type='electricity',
            gas_breakdown=gas_breakdown,
            metadata=metadata,
        )

    def calculate_steam(
        self,
        steam_kwh: Optional[float] = None,
        steam_mj: Optional[float] = None,
        region: str = 'US',
        energy_source: str = 'district_heating',
    ) -> Scope2Result:
        """
        Calculate emissions from purchased steam/district heating.

        Args:
            steam_kwh: Steam energy content (kWh)
            steam_mj: Steam energy content (MJ)
            region: Region ('US', 'EU', etc.)
            energy_source: 'district_heating' or 'district_cooling'

        Returns:
            Scope2Result with steam emissions

        Example:
            >>> calc = Scope2Calculator()
            >>> result = calc.calculate_steam(
            ...     steam_kwh=5000,
            ...     region='US',
            ...     energy_source='district_heating'
            ... )
        """
        # Convert MJ to kWh if needed
        if steam_kwh is None and steam_mj is None:
            raise ValueError("Must provide either steam_kwh or steam_mj")

        if steam_kwh is None:
            # 1 MJ = 0.277778 kWh
            steam_kwh = steam_mj * 0.277778

        # Factor ID is based on region
        factor_id = f"{energy_source}_{region.lower()}"

        request = CalculationRequest(
            factor_id=factor_id,
            activity_amount=steam_kwh,
            activity_unit='kwh',
        )

        calc_result = self.calculator.calculate(request)

        # Gas breakdown (similar to electricity)
        gas_breakdown = None
        if calc_result.emissions_kg_co2e > 0:
            gas_breakdown = self.gas_calculator.decompose(
                total_co2e_kg=FinancialDecimal.from_string(calc_result.emissions_kg_co2e),
                fuel_type='electricity',  # Use similar composition
            )

        metadata = {
            'steam_kwh': steam_kwh,
            'region': region,
            'energy_source': energy_source,
            'scope_2_method': 'location_based',
        }

        return Scope2Result(
            calculation_result=calc_result,
            method='location_based',
            energy_type='steam',
            gas_breakdown=gas_breakdown,
            metadata=metadata,
        )
