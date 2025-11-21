# -*- coding: utf-8 -*-
"""
Climate Domain Validation

Enforces GreenLang's "No Naked Numbers" rule and climate data integrity:
- Emission factors must have metadata (value, unit, source, year, region)
- Units must be from approved list (kg_CO2e/kWh, g_CO2e/MJ, etc.)
- Years must be in valid range (1990-2030)
- Regions must follow ISO-3166 or grid codes
- No numbers without units and sources

Architecture:
    Tool Output → ClimateValidator.validate_*() → [if fails] → GLValidationError

Example:
    validator = ClimateValidator()

    # Valid emission factor
    factor = {
        "value": 0.4,
        "unit": "kg_CO2e/kWh",
        "source": "EPA 2024",
        "year": 2024,
        "region": "US-CA"
    }
    validator.validate_emission_factor(factor)  # OK

    # Invalid (naked number)
    factor = {"value": 0.4}
    validator.validate_emission_factor(factor)  # Raises GLValidationError
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError

from greenlang.intelligence.runtime.json_validator import GLValidationError


class ClimateValidator:
    """
    Domain-specific validator for climate data

    Enforces GreenLang's data integrity rules:
    1. No naked numbers - all values must have units and sources
    2. Emission factors must have complete metadata
    3. Units must be from approved list
    4. Years must be in valid range
    5. Regions must follow standards

    Usage:
        validator = ClimateValidator()

        # Validate emission factor
        factor = tool_output["emission_factor"]
        validator.validate_emission_factor(factor)

        # Validate energy consumption
        energy = tool_output["energy_consumption"]
        validator.validate_energy_value(energy)
    """

    # Approved units for different quantities
    APPROVED_UNITS = {
        "emission_intensity": [
            "kg_CO2e/kWh",
            "g_CO2e/kWh",
            "lb_CO2e/kWh",
            "kg_CO2e/MJ",
            "g_CO2e/MJ",
            "kg_CO2e/MMBtu",
            "tonnes_CO2e/MWh",
        ],
        "emissions": [
            "kg_CO2e",
            "tonnes_CO2e",
            "metric_tonnes_CO2e",
            "lb_CO2e",
            "g_CO2e",
            "MT_CO2e",
        ],
        "energy": [
            "kWh",
            "MWh",
            "GWh",
            "TWh",
            "MJ",
            "GJ",
            "MMBtu",
            "therm",
            "kBtu",
        ],
        "fuel": [
            "gallons",
            "liters",
            "cubic_meters",
            "m3",
            "therms",
            "kg",
            "tonnes",
        ],
    }

    # Valid year range for climate data
    MIN_YEAR = 1990
    MAX_YEAR = 2030

    # Valid region patterns
    REGION_PATTERNS = [
        r"^[A-Z]{2}$",  # ISO-3166 Alpha-2 (US, CA, GB)
        r"^[A-Z]{2}-[A-Z]{2}$",  # US states (US-CA, US-NY)
        r"^[A-Z]{2}-[A-Z]{3}$",  # ISO-3166 Alpha-3
        r"^[A-Z]{4,8}$",  # Grid codes (CAISO, ERCOT, PJM)
    ]

    def __init__(self):
        """Initialize validator"""
        pass

    def validate_emission_factor(self, factor: Dict[str, Any]) -> None:
        """
        Validate emission factor has required metadata

        Required fields:
        - value: numeric emission factor
        - unit: approved emission intensity unit
        - source: data source (e.g., "EPA 2024", "IPCC AR6")
        - year: year of data (1990-2030)
        - region: geographic region code

        Args:
            factor: Emission factor dict

        Raises:
            GLValidationError: If validation fails

        Example:
            >>> validator = ClimateValidator()
            >>> factor = {
            ...     "value": 0.4,
            ...     "unit": "kg_CO2e/kWh",
            ...     "source": "EPA 2024",
            ...     "year": 2024,
            ...     "region": "US-CA"
            ... }
            >>> validator.validate_emission_factor(factor)  # OK
        """
        # Check required fields
        required = ["value", "unit", "source", "year", "region"]
        missing = [f for f in required if f not in factor]
        if missing:
            raise GLValidationError(
                f"Emission factor missing required fields: {missing}. "
                f"All emission factors must have: {required}"
            )

        # Validate value is numeric
        value = factor["value"]
        if not isinstance(value, (int, float)):
            raise GLValidationError(
                f"Emission factor 'value' must be numeric, got {type(value).__name__}"
            )

        if value < 0:
            raise GLValidationError(
                f"Emission factor 'value' cannot be negative: {value}"
            )

        # Validate unit
        unit = factor["unit"]
        if unit not in self.APPROVED_UNITS["emission_intensity"]:
            raise GLValidationError(
                f"Invalid emission intensity unit: '{unit}'. "
                f"Approved units: {self.APPROVED_UNITS['emission_intensity']}"
            )

        # Validate source is non-empty
        source = factor["source"]
        if not source or not isinstance(source, str) or len(source.strip()) == 0:
            raise GLValidationError(
                "Emission factor 'source' must be non-empty string "
                "(e.g., 'EPA 2024', 'IPCC AR6')"
            )

        # Validate year
        year = factor["year"]
        if not isinstance(year, int):
            raise GLValidationError(
                f"Emission factor 'year' must be integer, got {type(year).__name__}"
            )

        if not (self.MIN_YEAR <= year <= self.MAX_YEAR):
            raise GLValidationError(
                f"Emission factor 'year' out of range: {year}. "
                f"Must be between {self.MIN_YEAR} and {self.MAX_YEAR}"
            )

        # Validate region
        self._validate_region(factor["region"])

    def validate_energy_value(self, energy: Dict[str, Any]) -> None:
        """
        Validate energy consumption has value, unit, and context

        Required fields:
        - value: numeric energy amount
        - unit: approved energy unit
        - source: data source or calculation method

        Args:
            energy: Energy consumption dict

        Raises:
            GLValidationError: If validation fails

        Example:
            >>> validator = ClimateValidator()
            >>> energy = {
            ...     "value": 1000,
            ...     "unit": "kWh",
            ...     "source": "Utility bill 2024-01"
            ... }
            >>> validator.validate_energy_value(energy)  # OK
        """
        required = ["value", "unit", "source"]
        missing = [f for f in required if f not in energy]
        if missing:
            raise GLValidationError(f"Energy value missing required fields: {missing}")

        # Validate value
        value = energy["value"]
        if not isinstance(value, (int, float)):
            raise GLValidationError(
                f"Energy 'value' must be numeric, got {type(value).__name__}"
            )

        if value < 0:
            raise GLValidationError(f"Energy 'value' cannot be negative: {value}")

        # Validate unit
        unit = energy["unit"]
        if unit not in self.APPROVED_UNITS["energy"]:
            raise GLValidationError(
                f"Invalid energy unit: '{unit}'. "
                f"Approved units: {self.APPROVED_UNITS['energy']}"
            )

        # Validate source
        source = energy["source"]
        if not source or not isinstance(source, str):
            raise GLValidationError("Energy 'source' must be non-empty string")

    def validate_emissions_value(self, emissions: Dict[str, Any]) -> None:
        """
        Validate emissions calculation has value, unit, and methodology

        Required fields:
        - value: numeric emissions amount
        - unit: approved emissions unit (kg_CO2e, tonnes_CO2e, etc.)
        - source: calculation methodology or data source

        Args:
            emissions: Emissions dict

        Raises:
            GLValidationError: If validation fails

        Example:
            >>> validator = ClimateValidator()
            >>> emissions = {
            ...     "value": 268.5,
            ...     "unit": "kg_CO2e",
            ...     "source": "Calculated from EPA factors"
            ... }
            >>> validator.validate_emissions_value(emissions)  # OK
        """
        required = ["value", "unit", "source"]
        missing = [f for f in required if f not in emissions]
        if missing:
            raise GLValidationError(
                f"Emissions value missing required fields: {missing}"
            )

        # Validate value
        value = emissions["value"]
        if not isinstance(value, (int, float)):
            raise GLValidationError(
                f"Emissions 'value' must be numeric, got {type(value).__name__}"
            )

        if value < 0:
            raise GLValidationError(f"Emissions 'value' cannot be negative: {value}")

        # Validate unit
        unit = emissions["unit"]
        if unit not in self.APPROVED_UNITS["emissions"]:
            raise GLValidationError(
                f"Invalid emissions unit: '{unit}'. "
                f"Approved units: {self.APPROVED_UNITS['emissions']}"
            )

        # Validate source
        source = emissions["source"]
        if not source or not isinstance(source, str):
            raise GLValidationError(
                "Emissions 'source' must be non-empty string "
                "(describe calculation methodology)"
            )

    def _validate_region(self, region: str) -> None:
        """
        Validate region code follows standards

        Accepted formats:
        - ISO-3166 Alpha-2: US, CA, GB
        - US states: US-CA, US-NY
        - Grid codes: CAISO, ERCOT, PJM, WECC

        Args:
            region: Region code string

        Raises:
            GLValidationError: If region format invalid
        """
        if not region or not isinstance(region, str):
            raise GLValidationError("Region must be non-empty string")

        region = region.strip()

        # Check against patterns
        valid = any(re.match(pattern, region) for pattern in self.REGION_PATTERNS)

        if not valid:
            raise GLValidationError(
                f"Invalid region code: '{region}'. "
                f"Expected formats: ISO-3166 (US, CA), US states (US-CA), "
                f"or grid codes (CAISO, ERCOT)"
            )

    def validate_no_naked_numbers(
        self,
        data: Dict[str, Any],
        allowed_naked_keys: Optional[List[str]] = None,
    ) -> None:
        """
        Enforce "No Naked Numbers" rule

        Scans data dict recursively for numeric values without units/sources.
        Raises error if found (unless in allowed_naked_keys).

        Args:
            data: Data dict to validate
            allowed_naked_keys: Keys allowed to have naked numbers (e.g., "count", "index")

        Raises:
            GLValidationError: If naked numbers found

        Example:
            >>> validator = ClimateValidator()
            >>> data = {"emissions": 100}  # Naked number!
            >>> validator.validate_no_naked_numbers(data)  # Raises

            >>> data = {"emissions": {"value": 100, "unit": "kg_CO2e"}}
            >>> validator.validate_no_naked_numbers(data)  # OK
        """
        allowed_naked_keys = allowed_naked_keys or []

        def _scan(obj: Any, path: str = "") -> List[str]:
            """Recursively scan for naked numbers"""
            violations = []

            if isinstance(obj, dict):
                for key, value in obj.items():
                    key_path = f"{path}.{key}" if path else key

                    # If key is allowed to be naked, skip
                    if key in allowed_naked_keys:
                        continue

                    # If value is numeric and NOT in a struct with unit/source
                    if isinstance(value, (int, float)):
                        # Check if parent has unit/source
                        has_unit = "unit" in obj
                        has_source = "source" in obj

                        if not (has_unit and has_source):
                            violations.append(
                                f"{key_path} = {value} (naked number without unit/source)"
                            )

                    # Recurse
                    elif isinstance(value, (dict, list)):
                        violations.extend(_scan(value, key_path))

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    violations.extend(_scan(item, f"{path}[{i}]"))

            return violations

        violations = _scan(data)

        if violations:
            raise GLValidationError(
                f"No Naked Numbers violation! Found {len(violations)} numeric values "
                f"without units/sources:\n" + "\n".join(f"  - {v}" for v in violations)
            )

    def validate_tool_output(
        self,
        output: Dict[str, Any],
        expected_keys: Optional[List[str]] = None,
    ) -> None:
        """
        Validate complete tool output for climate data integrity

        Performs comprehensive validation:
        1. Check for expected keys
        2. Validate emission factors if present
        3. Validate energy values if present
        4. Validate emissions values if present
        5. Check for naked numbers

        Args:
            output: Tool output dict
            expected_keys: Keys expected in output (optional)

        Raises:
            GLValidationError: If validation fails

        Example:
            >>> validator = ClimateValidator()
            >>> output = {
            ...     "emission_factor": {
            ...         "value": 0.4,
            ...         "unit": "kg_CO2e/kWh",
            ...         "source": "EPA 2024",
            ...         "year": 2024,
            ...         "region": "US-CA"
            ...     },
            ...     "emissions": {
            ...         "value": 400,
            ...         "unit": "kg_CO2e",
            ...         "source": "Calculated"
            ...     }
            ... }
            >>> validator.validate_tool_output(output)  # OK
        """
        # Check expected keys
        if expected_keys:
            missing = [k for k in expected_keys if k not in output]
            if missing:
                raise GLValidationError(f"Tool output missing expected keys: {missing}")

        # Validate emission factors
        if "emission_factor" in output:
            self.validate_emission_factor(output["emission_factor"])

        # Validate energy
        if "energy" in output or "energy_consumption" in output:
            energy = output.get("energy") or output.get("energy_consumption")
            if isinstance(energy, dict):
                self.validate_energy_value(energy)

        # Validate emissions
        if "emissions" in output or "co2e" in output or "co2e_kg" in output:
            emissions = (
                output.get("emissions") or output.get("co2e") or output.get("co2e_kg")
            )
            if isinstance(emissions, dict):
                self.validate_emissions_value(emissions)

        # Check for naked numbers (allow common keys like count, id, index)
        self.validate_no_naked_numbers(
            output, allowed_naked_keys=["count", "id", "index", "page", "total"]
        )


# Global validator instance
_global_validator: Optional[ClimateValidator] = None


def get_global_validator() -> ClimateValidator:
    """
    Get global climate validator (singleton)

    Returns:
        Global ClimateValidator instance

    Example:
        validator = get_global_validator()
        validator.validate_emission_factor(factor)
    """
    global _global_validator
    if _global_validator is None:
        _global_validator = ClimateValidator()
    return _global_validator
