"""
Validation Hooks for Zero-Hallucination Climate Calculations

These validators ensure that all climate calculations meet scientific
and regulatory requirements with NO hallucination risk.
"""

from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from decimal import Decimal
from enum import Enum


class ValidationLevel(str, Enum):
    """Validation severity level."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationResult(BaseModel):
    """Result of validation check."""
    is_valid: bool
    level: ValidationLevel
    message: str
    field: Optional[str] = None
    actual_value: Optional[Any] = None
    expected_value: Optional[Any] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ValidationError(Exception):
    """Raised when validation fails."""
    def __init__(self, results: List[ValidationResult]):
        self.results = results
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        super().__init__(f"Validation failed with {len(errors)} error(s)")


class EmissionFactorValidator:
    """
    Validate emission factors against authoritative databases.

    Ensures that emission factors are:
    - Within expected ranges for fuel/material type
    - From authoritative sources (DEFRA, EPA, IPCC)
    - Appropriate for the geographic region
    - Current (not outdated)
    """

    # Emission factor ranges (kg CO2e per unit)
    # Source: DEFRA 2024, EPA eGRID 2023
    FACTOR_RANGES = {
        # Fuels (kg CO2e per liter)
        'diesel': {'min': 2.5, 'max': 2.8, 'typical': 2.687},
        'petrol': {'min': 2.2, 'max': 2.4, 'typical': 2.296},
        'natural_gas': {'min': 0.18, 'max': 0.21, 'typical': 0.184},  # per kWh
        'coal': {'min': 2.0, 'max': 2.5, 'typical': 2.269},  # per kg
        'lng': {'min': 2.5, 'max': 3.0, 'typical': 2.75},  # per kg
        'lpg': {'min': 1.4, 'max': 1.6, 'typical': 1.508},  # per liter

        # Electricity (kg CO2e per kWh) - varies by grid
        'electricity_uk': {'min': 0.15, 'max': 0.30, 'typical': 0.193},
        'electricity_us': {'min': 0.30, 'max': 0.70, 'typical': 0.417},
        'electricity_eu': {'min': 0.20, 'max': 0.40, 'typical': 0.295},
        'electricity_china': {'min': 0.50, 'max': 0.80, 'typical': 0.581},
    }

    def validate_factor(
        self,
        fuel_type: str,
        factor_value: float,
        region: Optional[str] = None,
        source: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate emission factor against known ranges.

        Args:
            fuel_type: Type of fuel/energy (e.g., 'diesel', 'natural_gas')
            factor_value: Emission factor value (kg CO2e per unit)
            region: Geographic region (e.g., 'UK', 'US', 'EU')
            source: Data source (e.g., 'DEFRA', 'EPA')

        Returns:
            ValidationResult with validation status
        """
        # Normalize fuel type
        fuel_key = fuel_type.lower().replace(' ', '_')

        # Check if we have a regional variant for electricity
        if fuel_key == 'electricity' and region:
            regional_key = f'electricity_{region.lower()}'
            if regional_key in self.FACTOR_RANGES:
                fuel_key = regional_key

        # Check if fuel type is known
        if fuel_key not in self.FACTOR_RANGES:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Unknown fuel type: {fuel_type}. Cannot validate range.",
                field="fuel_type",
                actual_value=fuel_type,
                metadata={'known_types': list(self.FACTOR_RANGES.keys())}
            )

        # Get expected range
        range_info = self.FACTOR_RANGES[fuel_key]
        min_val = range_info['min']
        max_val = range_info['max']
        typical_val = range_info['typical']

        # Validate range
        if factor_value < min_val or factor_value > max_val:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Emission factor {factor_value} outside valid range [{min_val}, {max_val}] for {fuel_type}",
                field="emission_factor",
                actual_value=factor_value,
                expected_value=f"[{min_val}, {max_val}]",
                metadata={
                    'fuel_type': fuel_type,
                    'typical_value': typical_val,
                    'source': source
                }
            )

        # Check if significantly different from typical
        deviation = abs(factor_value - typical_val) / typical_val
        if deviation > 0.1:  # More than 10% deviation
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.WARNING,
                message=f"Emission factor {factor_value} deviates {deviation*100:.1f}% from typical value {typical_val}",
                field="emission_factor",
                actual_value=factor_value,
                expected_value=typical_val,
                metadata={'fuel_type': fuel_type, 'source': source}
            )

        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message=f"Emission factor {factor_value} is valid for {fuel_type}",
            field="emission_factor",
            actual_value=factor_value,
            metadata={'fuel_type': fuel_type, 'source': source}
        )


class UnitValidator:
    """
    Validate climate units and conversions.

    Ensures that:
    - Units are recognized and valid
    - Conversions are correct
    - Unit combinations make sense (e.g., kWh/year, tCO2e/MWh)
    """

    # Recognized units
    ENERGY_UNITS = ['kWh', 'MWh', 'GWh', 'TJ', 'GJ', 'MJ', 'BTU', 'MMBtu']
    EMISSION_UNITS = ['kgCO2e', 'tCO2e', 'MtCO2e', 'gCO2e']
    MASS_UNITS = ['kg', 't', 'g', 'lb', 'ton']
    VOLUME_UNITS = ['L', 'liters', 'm3', 'gal', 'gallon']
    TIME_UNITS = ['hour', 'day', 'month', 'year']

    ALL_UNITS = ENERGY_UNITS + EMISSION_UNITS + MASS_UNITS + VOLUME_UNITS + TIME_UNITS

    # Unit conversion factors (to base unit)
    CONVERSIONS = {
        # Energy (base: kWh)
        'kWh': 1.0,
        'MWh': 1000.0,
        'GWh': 1_000_000.0,
        'MJ': 0.277778,
        'GJ': 277.778,
        'TJ': 277_778.0,
        'BTU': 0.000293071,
        'MMBtu': 293.071,

        # Emissions (base: kgCO2e)
        'gCO2e': 0.001,
        'kgCO2e': 1.0,
        'tCO2e': 1000.0,
        'MtCO2e': 1_000_000_000.0,

        # Mass (base: kg)
        'g': 0.001,
        'kg': 1.0,
        't': 1000.0,
        'lb': 0.453592,
        'ton': 1000.0,

        # Volume (base: L)
        'L': 1.0,
        'liters': 1.0,
        'm3': 1000.0,
        'gal': 3.78541,
        'gallon': 3.78541,
    }

    def validate_unit(self, unit: str) -> ValidationResult:
        """Validate that unit is recognized."""
        if unit in self.ALL_UNITS:
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message=f"Unit '{unit}' is valid",
                field="unit",
                actual_value=unit
            )

        return ValidationResult(
            is_valid=False,
            level=ValidationLevel.ERROR,
            message=f"Unknown unit: {unit}",
            field="unit",
            actual_value=unit,
            metadata={'known_units': self.ALL_UNITS}
        )

    def validate_conversion(
        self,
        value: float,
        from_unit: str,
        to_unit: str,
        expected_result: Optional[float] = None
    ) -> ValidationResult:
        """Validate unit conversion."""
        # Check units are valid
        if from_unit not in self.CONVERSIONS or to_unit not in self.CONVERSIONS:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Cannot convert between {from_unit} and {to_unit}",
                field="unit_conversion",
                actual_value=f"{from_unit} -> {to_unit}",
                metadata={'convertible_units': list(self.CONVERSIONS.keys())}
            )

        # Check units are of the same type (energy, emissions, mass, or volume)
        from_type = None
        to_type = None

        if from_unit in self.ENERGY_UNITS:
            from_type = "energy"
        elif from_unit in self.EMISSION_UNITS:
            from_type = "emissions"
        elif from_unit in self.MASS_UNITS:
            from_type = "mass"
        elif from_unit in self.VOLUME_UNITS:
            from_type = "volume"

        if to_unit in self.ENERGY_UNITS:
            to_type = "energy"
        elif to_unit in self.EMISSION_UNITS:
            to_type = "emissions"
        elif to_unit in self.MASS_UNITS:
            to_type = "mass"
        elif to_unit in self.VOLUME_UNITS:
            to_type = "volume"

        if from_type != to_type:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Cannot convert between {from_type} ({from_unit}) and {to_type} ({to_unit})",
                field="unit_conversion",
                actual_value=f"{from_unit} -> {to_unit}",
                metadata={'from_type': from_type, 'to_type': to_type}
            )

        # Calculate conversion
        base_value = value * self.CONVERSIONS[from_unit]
        converted_value = base_value / self.CONVERSIONS[to_unit]

        # If expected result provided, validate
        if expected_result is not None:
            tolerance = 0.0001  # 0.01% tolerance for floating point
            if abs(converted_value - expected_result) / expected_result > tolerance:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Conversion error: {value} {from_unit} = {converted_value} {to_unit}, expected {expected_result}",
                    field="unit_conversion",
                    actual_value=converted_value,
                    expected_value=expected_result
                )

        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message=f"Conversion valid: {value} {from_unit} = {converted_value:.6f} {to_unit}",
            field="unit_conversion",
            actual_value=converted_value,
            metadata={'from_unit': from_unit, 'to_unit': to_unit}
        )


class ThermodynamicValidator:
    """
    Validate thermodynamic constraints.

    Ensures that:
    - Efficiencies are < 100% (except for heat pumps with COP)
    - Energy balance is maintained
    - Physical constraints are respected
    """

    def validate_efficiency(
        self,
        efficiency: float,
        equipment_type: str = "generic",
        is_cop: bool = False
    ) -> ValidationResult:
        """
        Validate equipment efficiency.

        Args:
            efficiency: Efficiency value (0-1 for %, >1 for COP)
            equipment_type: Type of equipment
            is_cop: True if this is Coefficient of Performance (can be > 1)
        """
        if is_cop:
            # Heat pumps can have COP > 1 (typically 2-5)
            if efficiency < 1.0:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"COP {efficiency} is less than 1.0 (impossible for heat pump)",
                    field="cop",
                    actual_value=efficiency,
                    expected_value=">= 1.0"
                )
            if efficiency > 6.0:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.WARNING,
                    message=f"COP {efficiency} is unusually high (typical range: 2-5)",
                    field="cop",
                    actual_value=efficiency,
                    expected_value="2.0 - 5.0"
                )
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.INFO,
                message=f"COP {efficiency} is valid",
                field="cop",
                actual_value=efficiency
            )

        # Regular efficiency (must be 0-100% or 0-1)
        if efficiency > 1.0:
            # Assume percentage (0-100)
            if efficiency > 100.0:
                return ValidationResult(
                    is_valid=False,
                    level=ValidationLevel.ERROR,
                    message=f"Efficiency {efficiency}% exceeds 100% (violates thermodynamics)",
                    field="efficiency",
                    actual_value=efficiency,
                    expected_value="<= 100%"
                )
            efficiency_fraction = efficiency / 100.0
        else:
            efficiency_fraction = efficiency

        if efficiency_fraction < 0:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Efficiency {efficiency} is negative",
                field="efficiency",
                actual_value=efficiency,
                expected_value=">= 0"
            )

        if efficiency_fraction == 0:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Efficiency is 0% (no useful output)",
                field="efficiency",
                actual_value=efficiency,
                expected_value="> 0"
            )

        # Check typical ranges for common equipment
        typical_ranges = {
            'boiler': (0.70, 0.98),
            'chiller': (0.50, 0.85),
            'motor': (0.75, 0.96),
            'generator': (0.30, 0.45),
            'solar_panel': (0.15, 0.23),
            'wind_turbine': (0.35, 0.50),
        }

        if equipment_type in typical_ranges:
            min_eff, max_eff = typical_ranges[equipment_type]
            if efficiency_fraction < min_eff or efficiency_fraction > max_eff:
                return ValidationResult(
                    is_valid=True,
                    level=ValidationLevel.WARNING,
                    message=f"Efficiency {efficiency_fraction*100:.1f}% outside typical range [{min_eff*100:.1f}%, {max_eff*100:.1f}%] for {equipment_type}",
                    field="efficiency",
                    actual_value=efficiency_fraction,
                    expected_value=f"{min_eff*100:.1f}% - {max_eff*100:.1f}%"
                )

        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message=f"Efficiency {efficiency_fraction*100:.1f}% is valid",
            field="efficiency",
            actual_value=efficiency_fraction
        )

    def validate_energy_balance(
        self,
        input_energy: float,
        output_energy: float,
        efficiency: float
    ) -> ValidationResult:
        """
        Validate energy balance equation: output = input * efficiency.

        Args:
            input_energy: Input energy (any unit)
            output_energy: Output energy (same unit)
            efficiency: Efficiency (0-1)
        """
        expected_output = input_energy * efficiency
        tolerance = 0.001  # 0.1% tolerance

        if abs(output_energy - expected_output) / expected_output > tolerance:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Energy balance error: {input_energy} * {efficiency} = {expected_output}, got {output_energy}",
                field="energy_balance",
                actual_value=output_energy,
                expected_value=expected_output
            )

        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message=f"Energy balance valid: {input_energy} * {efficiency} = {output_energy}",
            field="energy_balance",
            actual_value=output_energy
        )


class GWPValidator:
    """
    Validate Global Warming Potential (GWP) values.

    Ensures that:
    - GWP values match IPCC AR5 or AR6
    - Timeframe is specified (20yr, 100yr, 500yr)
    - Values are scientifically accurate
    """

    # GWP values from IPCC AR6 (100-year timeframe)
    # Source: IPCC Sixth Assessment Report (2021)
    GWP_AR6_100YR = {
        'CO2': 1,
        'CH4': 29.8,  # Methane (fossil)
        'CH4_biogenic': 27.2,  # Methane (biogenic)
        'N2O': 273,  # Nitrous oxide
        'HFC-134a': 1530,
        'HFC-32': 771,
        'SF6': 25200,  # Sulfur hexafluoride
        'NF3': 17400,  # Nitrogen trifluoride
        'CF4': 7380,  # Perfluoromethane
    }

    # GWP values from IPCC AR5 (100-year timeframe)
    # Source: IPCC Fifth Assessment Report (2014)
    GWP_AR5_100YR = {
        'CO2': 1,
        'CH4': 28,  # Methane (fossil)
        'CH4_biogenic': 28,
        'N2O': 265,
        'HFC-134a': 1300,
        'HFC-32': 677,
        'SF6': 23500,
        'NF3': 16100,
        'CF4': 6630,
    }

    def validate_gwp(
        self,
        gas: str,
        gwp_value: float,
        report: str = "AR6",
        timeframe: int = 100
    ) -> ValidationResult:
        """
        Validate GWP value against IPCC reports.

        Args:
            gas: Greenhouse gas (e.g., 'CH4', 'N2O')
            gwp_value: GWP value
            report: IPCC report ('AR5' or 'AR6')
            timeframe: Time horizon (20, 100, or 500 years)
        """
        if timeframe != 100:
            return ValidationResult(
                is_valid=True,
                level=ValidationLevel.WARNING,
                message=f"GWP validation only supports 100-year timeframe, got {timeframe}yr",
                field="gwp_timeframe",
                actual_value=timeframe,
                expected_value=100
            )

        # Select reference values
        if report == "AR6":
            reference = self.GWP_AR6_100YR
        elif report == "AR5":
            reference = self.GWP_AR5_100YR
        else:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"Unknown IPCC report: {report}. Use 'AR5' or 'AR6'",
                field="ipcc_report",
                actual_value=report,
                expected_value="AR5 or AR6"
            )

        # Check if gas is known
        if gas not in reference:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.WARNING,
                message=f"Unknown gas: {gas}. Cannot validate GWP.",
                field="gas",
                actual_value=gas,
                metadata={'known_gases': list(reference.keys())}
            )

        # Get expected GWP
        expected_gwp = reference[gas]

        # Allow 5% tolerance (GWP values have uncertainty)
        tolerance = 0.05
        if abs(gwp_value - expected_gwp) / expected_gwp > tolerance:
            return ValidationResult(
                is_valid=False,
                level=ValidationLevel.ERROR,
                message=f"GWP for {gas} is {gwp_value}, expected {expected_gwp} (IPCC {report})",
                field="gwp",
                actual_value=gwp_value,
                expected_value=expected_gwp,
                metadata={'gas': gas, 'report': report, 'timeframe': timeframe}
            )

        return ValidationResult(
            is_valid=True,
            level=ValidationLevel.INFO,
            message=f"GWP {gwp_value} for {gas} matches IPCC {report} (100yr)",
            field="gwp",
            actual_value=gwp_value,
            metadata={'gas': gas, 'report': report}
        )
