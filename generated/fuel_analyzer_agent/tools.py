"""
Tool implementations for Fuel Emissions Analyzer.

This module provides tool wrapper classes for zero-hallucination calculations.
All tools are deterministic and track provenance.

Generated from AgentSpec: emissions/fuel_analyzer_v1
Updated with real emission factor database integration.
"""

from typing import Any, Dict, List, Optional
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "core"))

from greenlang.data.emission_factor_db import get_database, GWPSet

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Base
# =============================================================================

class BaseTool:
    """Base class for all tools."""

    name: str = "base_tool"
    safe: bool = True

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tool - must be overridden."""
        raise NotImplementedError

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate parameters."""
        return True


# =============================================================================
# Tool Implementations
# =============================================================================

class LookupEmissionFactorTool:
    """
    Look up emission factor for a fuel type from the IPCC/EPA emission factor database. Returns the emission factor value, unit, and source citation. This is a DETERMINISTIC lookup - same inputs always return same outputs.


    Implementation: python://greenlang.tools.emission_factors:lookup_emission_factor
    Safe: True
    """

    def __init__(self):
        """Initialize LookupEmissionFactorTool."""
        self.name = "lookup_emission_factor"
        self.safe = True

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tool with given parameters.

        Args:
            params: Tool parameters

        Returns:
            Tool execution result
        """
        # Validate required parameters
        if "fuel_type" not in params:
            raise ValueError("Missing required parameter: fuel_type")
        if "region" not in params:
            raise ValueError("Missing required parameter: region")
        if "year" not in params:
            raise ValueError("Missing required parameter: year")

        # Execute tool logic (ZERO-HALLUCINATION)
        result = await self._execute_internal(params)

        return result

    async def _execute_internal(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal execution logic - connects to emission factor database.

        ZERO-HALLUCINATION: This is a deterministic database lookup.
        Same inputs always return same outputs.
        """
        # Get database instance
        db = get_database()

        # Extract parameters
        fuel_type = params["fuel_type"]
        region = params["region"]
        year = params["year"]
        gwp_set_str = params.get("gwp_set", "AR6GWP100")

        # Convert GWP set string to enum
        try:
            gwp_set = GWPSet(gwp_set_str)
        except ValueError:
            gwp_set = GWPSet.AR6GWP100

        # Lookup emission factor
        ef_record = db.lookup(fuel_type, region, year, gwp_set)

        if ef_record is None:
            raise ValueError(
                f"Emission factor not found for fuel_type={fuel_type}, "
                f"region={region}, year={year}, gwp_set={gwp_set_str}"
            )

        # Return emission factor data
        return {
            "ef_uri": ef_record.ef_uri,
            "ef_value": ef_record.ef_value,
            "ef_unit": ef_record.ef_unit,
            "source": ef_record.source,
            "gwp_set": ef_record.gwp_set.value,
            "uncertainty": ef_record.uncertainty
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        # TODO: Add validation based on schema
        return True

class CalculateEmissionsTool:
    """
    Calculate GHG emissions from fuel combustion using emission factor and activity data. Uses deterministic formula: emissions = activity * emission_factor. All calculations are traceable and reproducible.


    Implementation: python://greenlang.tools.calculators:calculate_emissions
    Safe: True
    """

    def __init__(self):
        """Initialize CalculateEmissionsTool."""
        self.name = "calculate_emissions"
        self.safe = True

        # Unit conversion factors (to base units)
        self.energy_conversions = {
            "MJ": 1.0,
            "GJ": 1000.0,
            "kWh": 3.6,  # 1 kWh = 3.6 MJ
            "MMBTU": 1055.06,  # 1 MMBTU = 1055.06 MJ
        }

        self.emissions_conversions = {
            "kgCO2e": 1.0,
            "tCO2e": 1000.0,
            "MtCO2e": 1000000.0,
        }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tool with given parameters.

        Args:
            params: Tool parameters

        Returns:
            Tool execution result
        """
        # Validate required parameters
        if "activity_value" not in params:
            raise ValueError("Missing required parameter: activity_value")
        if "activity_unit" not in params:
            raise ValueError("Missing required parameter: activity_unit")
        if "ef_value" not in params:
            raise ValueError("Missing required parameter: ef_value")
        if "ef_unit" not in params:
            raise ValueError("Missing required parameter: ef_unit")

        # Execute tool logic (ZERO-HALLUCINATION)
        result = await self._execute_internal(params)

        return result

    async def _execute_internal(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal execution logic - deterministic emissions calculation.

        ZERO-HALLUCINATION: Uses exact formula: emissions = activity × emission_factor
        All calculations are deterministic and reproducible.
        """
        activity_value = float(params["activity_value"])
        activity_unit = params["activity_unit"]
        ef_value = float(params["ef_value"])
        ef_unit = params["ef_unit"]
        output_unit = params.get("output_unit", "tCO2e")

        # Parse emission factor unit (e.g., "kgCO2e/MJ" or "kgCO2e/L")
        if "/" in ef_unit:
            ef_emissions_unit, ef_activity_unit = ef_unit.split("/")
        else:
            ef_emissions_unit = "kgCO2e"
            ef_activity_unit = activity_unit

        # Convert activity to emission factor's activity unit if needed
        if activity_unit != ef_activity_unit:
            # Try energy conversion
            if activity_unit in self.energy_conversions and ef_activity_unit in self.energy_conversions:
                activity_value = activity_value * self.energy_conversions[activity_unit] / self.energy_conversions[ef_activity_unit]
            else:
                # Units must match (L, kg, m3, etc.)
                if activity_unit != ef_activity_unit:
                    raise ValueError(
                        f"Activity unit '{activity_unit}' incompatible with "
                        f"emission factor unit '{ef_unit}'"
                    )

        # Calculate emissions (deterministic formula)
        emissions_value = activity_value * ef_value

        # Convert to output unit
        if ef_emissions_unit in self.emissions_conversions:
            emissions_kg = emissions_value * self.emissions_conversions[ef_emissions_unit]
        else:
            emissions_kg = emissions_value  # Assume kgCO2e

        # Convert to desired output unit
        if output_unit in self.emissions_conversions:
            final_emissions = emissions_kg / self.emissions_conversions[output_unit]
        else:
            final_emissions = emissions_kg / 1000.0  # Default to tCO2e

        # Build calculation formula for provenance
        formula = f"{activity_value:.4f} {activity_unit} × {ef_value:.6f} {ef_unit} = {final_emissions:.6f} {output_unit}"

        # Return result
        return {
            "emissions_value": final_emissions,
            "emissions_unit": output_unit,
            "calculation_formula": formula,
            "conversion_factor": 1.0  # No conversion needed if units matched
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        # TODO: Add validation based on schema
        return True

class ValidateFuelInputTool:
    """
    Validate fuel input for physical plausibility. Checks that values are within reasonable ranges and units are compatible.


    Implementation: python://greenlang.tools.validators:validate_fuel_input
    Safe: True
    """

    def __init__(self):
        """Initialize ValidateFuelInputTool."""
        self.name = "validate_fuel_input"
        self.safe = True

        # Define plausible ranges for each fuel type (min, max, typical_max, unit)
        self.fuel_ranges = {
            "natural_gas": {
                "MJ": (0, 1e9, 1e6, "Industrial/utility scale"),
                "GJ": (0, 1e6, 1e3, "Large industrial"),
                "m3": (0, 1e6, 1e5, "Volume measurement"),
            },
            "diesel": {
                "L": (0, 1e7, 1e5, "Vehicle/generator/industrial"),
                "gal": (0, 3e6, 3e4, "US gallons"),
            },
            "gasoline": {
                "L": (0, 1e7, 5e4, "Vehicle fleet"),
                "gal": (0, 3e6, 1.5e4, "US gallons"),
            },
            "lpg": {
                "kg": (0, 1e6, 1e5, "Industrial/residential"),
                "L": (0, 2e6, 2e5, "Liquid volume"),
            },
            "fuel_oil": {
                "L": (0, 1e7, 1e6, "Industrial/heating"),
            },
        }

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute tool with given parameters.

        Args:
            params: Tool parameters

        Returns:
            Tool execution result
        """
        # Validate required parameters
        if "fuel_type" not in params:
            raise ValueError("Missing required parameter: fuel_type")
        if "quantity" not in params:
            raise ValueError("Missing required parameter: quantity")
        if "unit" not in params:
            raise ValueError("Missing required parameter: unit")

        # Execute tool logic (ZERO-HALLUCINATION)
        result = await self._execute_internal(params)

        return result

    async def _execute_internal(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Internal execution logic - validates input for physical plausibility.

        ZERO-HALLUCINATION: Uses deterministic range checks based on
        real-world industrial and residential consumption patterns.
        """
        fuel_type = params["fuel_type"]
        quantity = float(params["quantity"])
        unit = params["unit"]

        warnings = []
        valid = True
        suggested_value = None
        plausibility_score = 1.0

        # Check 1: Fuel type is recognized
        if fuel_type not in self.fuel_ranges:
            warnings.append(f"Fuel type '{fuel_type}' not in standard list")
            plausibility_score *= 0.5

        # Check 2: Quantity is non-negative
        if quantity < 0:
            valid = False
            warnings.append("Quantity cannot be negative")
            suggested_value = 0.0
            plausibility_score = 0.0

        # Check 3: Quantity is not zero (would be unusual for emission calculation)
        if quantity == 0:
            warnings.append("Zero quantity will result in zero emissions")
            plausibility_score *= 0.8

        # Check 4: Check if unit is compatible with fuel type
        if fuel_type in self.fuel_ranges:
            if unit not in self.fuel_ranges[fuel_type]:
                valid = False
                warnings.append(
                    f"Unit '{unit}' not compatible with fuel type '{fuel_type}'. "
                    f"Expected: {list(self.fuel_ranges[fuel_type].keys())}"
                )
                plausibility_score *= 0.3

            else:
                # Check 5: Quantity within plausible range
                min_val, max_val, typical_max, description = self.fuel_ranges[fuel_type][unit]

                if quantity > max_val:
                    valid = False
                    warnings.append(
                        f"Quantity {quantity} {unit} exceeds maximum plausible value "
                        f"({max_val} {unit}) for {fuel_type}"
                    )
                    suggested_value = typical_max
                    plausibility_score = 0.0

                elif quantity > typical_max:
                    warnings.append(
                        f"Quantity {quantity} {unit} is unusually high for {fuel_type}. "
                        f"Typical maximum: {typical_max} {unit} ({description}). "
                        f"Please verify this is correct."
                    )
                    plausibility_score *= 0.6

                elif quantity < min_val:
                    warnings.append(f"Quantity below minimum: {min_val} {unit}")
                    plausibility_score *= 0.7

        # Check 6: Extremely small quantities (likely data entry error)
        if 0 < quantity < 0.001:
            warnings.append(
                "Quantity is very small (< 0.001). "
                "This may be a data entry error. Please verify."
            )
            plausibility_score *= 0.7

        # Return validation result
        return {
            "valid": valid,
            "warnings": warnings,
            "suggested_value": suggested_value,
            "plausibility_score": plausibility_score
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate input parameters."""
        # TODO: Add validation based on schema
        return True


# =============================================================================
# Tool Registry
# =============================================================================

TOOL_REGISTRY: Dict[str, type] = {
    "lookup_emission_factor": LookupEmissionFactorTool,
    "calculate_emissions": CalculateEmissionsTool,
    "validate_fuel_input": ValidateFuelInputTool,
}


def get_tool(name: str) -> Optional[BaseTool]:
    """Get tool instance by name."""
    tool_class = TOOL_REGISTRY.get(name)
    if tool_class:
        return tool_class()
    return None


def list_tools() -> List[str]:
    """List all available tool names."""
    return list(TOOL_REGISTRY.keys())
