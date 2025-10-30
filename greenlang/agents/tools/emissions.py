"""
GreenLang Emissions Calculation Tools
======================================

Shared tools for emission calculations across all agents.

Tools:
- CalculateEmissionsTool: Calculate CO2e emissions from fuel consumption
- AggregateEmissionsTool: Aggregate emissions from multiple sources
- LookupEmissionFactorTool: Look up emission factors from database
- CalculateBreakdownTool: Calculate percentage breakdown by source

Author: GreenLang Framework Team
Date: October 2025
Status: Production Ready
"""

from typing import Any, Dict, List

from .base import BaseTool, ToolDef, ToolResult, ToolSafety
from greenlang.agents.citations import (
    CalculationCitation,
    create_emission_factor_citation,
)


# ==============================================================================
# Calculate Emissions Tool
# ==============================================================================

class CalculateEmissionsTool(BaseTool):
    """
    Calculate exact CO2e emissions from fuel consumption.

    This tool performs the core emission calculation:
        emissions_kg_co2e = amount × emission_factor

    Deterministic and safe for all agents.
    """

    def __init__(self):
        super().__init__(
            name="calculate_emissions",
            description="Calculate exact CO2e emissions from fuel consumption using authoritative emission factors",
            safety=ToolSafety.DETERMINISTIC
        )

    def execute(
        self,
        fuel_type: str,
        amount: float,
        unit: str,
        emission_factor: float,
        emission_factor_unit: str,
        country: str = "US",
    ) -> ToolResult:
        """
        Execute emission calculation.

        Args:
            fuel_type: Type of fuel
            amount: Amount of fuel consumed
            unit: Unit of consumption
            emission_factor: Emission factor value
            emission_factor_unit: Unit of emission factor
            country: Country code (for context)

        Returns:
            ToolResult with emissions_kg_co2e
        """
        try:
            # Calculate emissions (simple multiplication)
            # NOTE: Unit conversion should be handled by caller
            emissions_kg_co2e = amount * emission_factor

            # Create calculation citation
            calc_citation = CalculationCitation(
                step_name="calculate_emissions",
                formula="emissions_kg_co2e = amount × emission_factor",
                inputs={
                    "fuel_type": fuel_type,
                    "amount": amount,
                    "unit": unit,
                    "emission_factor": emission_factor,
                    "emission_factor_unit": emission_factor_unit,
                },
                output={
                    "emissions_kg_co2e": emissions_kg_co2e,
                    "unit": "kgCO2e"
                }
            )

            return ToolResult(
                success=True,
                data={
                    "emissions_kg_co2e": emissions_kg_co2e,
                    "fuel_type": fuel_type,
                    "amount_consumed": amount,
                    "unit": unit,
                    "emission_factor_used": emission_factor,
                },
                citations=[calc_citation],
                metadata={
                    "calculation": f"{amount} {unit} × {emission_factor} {emission_factor_unit} = {emissions_kg_co2e} kgCO2e",
                    "country": country,
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Calculation failed: {str(e)}"
            )

    def get_tool_def(self) -> ToolDef:
        """Get tool definition for ChatSession."""
        return ToolDef(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "required": ["fuel_type", "amount", "unit", "emission_factor", "emission_factor_unit"],
                "properties": {
                    "fuel_type": {
                        "type": "string",
                        "description": "Type of fuel (natural_gas, coal, diesel, etc.)"
                    },
                    "amount": {
                        "type": "number",
                        "description": "Amount of fuel consumed",
                        "minimum": 0
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit of fuel consumption (therms, m3, gallons, etc.)"
                    },
                    "emission_factor": {
                        "type": "number",
                        "description": "Emission factor value"
                    },
                    "emission_factor_unit": {
                        "type": "string",
                        "description": "Unit of emission factor (kgCO2e/therm, kgCO2e/m3, etc.)"
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code (default: US)",
                        "default": "US"
                    }
                }
            },
            safety=self.safety
        )


# ==============================================================================
# Aggregate Emissions Tool
# ==============================================================================

class AggregateEmissionsTool(BaseTool):
    """
    Aggregate total emissions from multiple sources.

    Sums emissions from different fuel types and provides:
    - Total in kg
    - Total in tons (metric tonnes)
    - Breakdown by source

    Deterministic and safe.
    """

    def __init__(self):
        super().__init__(
            name="aggregate_emissions",
            description="Aggregate total emissions from multiple sources into kg and tons CO2e",
            safety=ToolSafety.DETERMINISTIC
        )

    def execute(self, emissions: List[Dict[str, Any]]) -> ToolResult:
        """
        Execute aggregation.

        Args:
            emissions: List of emission dictionaries with co2e_emissions_kg

        Returns:
            ToolResult with aggregated emissions
        """
        try:
            # Aggregate total emissions in kg
            total_kg = sum(e.get("co2e_emissions_kg", 0.0) for e in emissions)

            # Convert to metric tons
            total_tons = total_kg / 1000.0

            # Breakdown by fuel type
            by_fuel = {}
            for emission in emissions:
                fuel_type = emission.get("fuel_type", "unknown")
                co2e_kg = emission.get("co2e_emissions_kg", 0.0)

                if fuel_type in by_fuel:
                    by_fuel[fuel_type] += co2e_kg
                else:
                    by_fuel[fuel_type] = co2e_kg

            # Create calculation citation
            calc_citation = CalculationCitation(
                step_name="aggregate_emissions",
                formula="sum(emissions[i].co2e_emissions_kg for i in range(len(emissions)))",
                inputs={
                    "num_sources": len(emissions),
                    "sources": [e.get("fuel_type", "Unknown") for e in emissions]
                },
                output={
                    "total_kg": total_kg,
                    "total_tons": total_tons,
                    "unit": "kgCO2e"
                }
            )

            return ToolResult(
                success=True,
                data={
                    "total_co2e_kg": total_kg,
                    "total_co2e_tons": total_tons,
                    "by_fuel": by_fuel,
                    "num_sources": len(emissions),
                },
                citations=[calc_citation],
                metadata={
                    "calculation": f"Aggregated {len(emissions)} sources = {total_kg} kgCO2e"
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Aggregation failed: {str(e)}"
            )

    def get_tool_def(self) -> ToolDef:
        """Get tool definition."""
        return ToolDef(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "required": ["emissions"],
                "properties": {
                    "emissions": {
                        "type": "array",
                        "description": "Array of emission objects with co2e_emissions_kg field",
                        "items": {
                            "type": "object",
                            "properties": {
                                "fuel_type": {"type": "string"},
                                "co2e_emissions_kg": {"type": "number"}
                            }
                        }
                    }
                }
            },
            safety=self.safety
        )


# ==============================================================================
# Calculate Breakdown Tool
# ==============================================================================

class CalculateBreakdownTool(BaseTool):
    """
    Calculate percentage breakdown of emissions by source.

    Deterministic and safe.
    """

    def __init__(self):
        super().__init__(
            name="calculate_breakdown",
            description="Calculate percentage breakdown of emissions by fuel source",
            safety=ToolSafety.DETERMINISTIC
        )

    def execute(
        self,
        emissions: List[Dict[str, Any]],
        total_emissions: float
    ) -> ToolResult:
        """
        Execute breakdown calculation.

        Args:
            emissions: List of emission dictionaries
            total_emissions: Total emissions for percentage calculation

        Returns:
            ToolResult with percentage breakdown
        """
        try:
            breakdown = {}

            for emission in emissions:
                fuel_type = emission.get("fuel_type", "unknown")
                co2e_kg = emission.get("co2e_emissions_kg", 0.0)

                # Calculate percentage
                if total_emissions > 0:
                    percentage = (co2e_kg / total_emissions) * 100.0
                else:
                    percentage = 0.0

                breakdown[fuel_type] = round(percentage, 2)

            # Find largest and smallest sources
            if breakdown:
                largest = max(breakdown.items(), key=lambda x: x[1])
                smallest = min(breakdown.items(), key=lambda x: x[1])
            else:
                largest = ("none", 0.0)
                smallest = ("none", 0.0)

            return ToolResult(
                success=True,
                data={
                    "by_fuel_percent": breakdown,
                    "largest_source": largest[0],
                    "largest_percentage": largest[1],
                    "smallest_source": smallest[0],
                    "smallest_percentage": smallest[1],
                },
                metadata={
                    "num_sources": len(emissions),
                    "total_emissions": total_emissions
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Breakdown calculation failed: {str(e)}"
            )

    def get_tool_def(self) -> ToolDef:
        """Get tool definition."""
        return ToolDef(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "required": ["emissions", "total_emissions"],
                "properties": {
                    "emissions": {
                        "type": "array",
                        "description": "Array of emission objects",
                        "items": {
                            "type": "object",
                            "properties": {
                                "fuel_type": {"type": "string"},
                                "co2e_emissions_kg": {"type": "number"}
                            }
                        }
                    },
                    "total_emissions": {
                        "type": "number",
                        "description": "Total emissions for percentage calculation"
                    }
                }
            },
            safety=self.safety
        )
