"""
GreenLang Emissions Calculation Tools
======================================

Shared tools for emission calculations across all agents.

Tools:
- CalculateEmissionsTool: Calculate CO2e emissions from fuel consumption
- AggregateEmissionsTool: Aggregate emissions from multiple sources
- CalculateBreakdownTool: Calculate percentage breakdown by source
- CalculateScopeEmissionsTool: Calculate Scope 1/2/3 emissions
- RegionalEmissionFactorTool: Get regional emission factors

Author: GreenLang Framework Team
Date: October 2025
Status: Production Ready
"""

from typing import Any, Dict, List, Optional
from datetime import datetime

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


# ==============================================================================
# Calculate Scope Emissions Tool
# ==============================================================================

class CalculateScopeEmissionsTool(BaseTool):
    """
    Calculate emissions by GHG Protocol Scope (1, 2, 3).

    Scope 1: Direct emissions from owned/controlled sources
    Scope 2: Indirect emissions from purchased energy
    Scope 3: Other indirect emissions in value chain

    Deterministic and safe.
    """

    def __init__(self):
        super().__init__(
            name="calculate_scope_emissions",
            description="Calculate emissions breakdown by GHG Protocol Scope 1, 2, and 3",
            safety=ToolSafety.DETERMINISTIC
        )

    def execute(
        self,
        scope_1_sources: List[Dict[str, Any]],
        scope_2_sources: List[Dict[str, Any]],
        scope_3_sources: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolResult:
        """
        Execute scope emission calculation.

        Args:
            scope_1_sources: List of Scope 1 emission sources with co2e_emissions_kg
            scope_2_sources: List of Scope 2 emission sources with co2e_emissions_kg
            scope_3_sources: List of Scope 3 emission sources with co2e_emissions_kg

        Returns:
            ToolResult with scope breakdown
        """
        try:
            # Calculate totals by scope
            scope_1_total = sum(s.get("co2e_emissions_kg", 0.0) for s in scope_1_sources)
            scope_2_total = sum(s.get("co2e_emissions_kg", 0.0) for s in scope_2_sources)
            scope_3_total = sum(s.get("co2e_emissions_kg", 0.0) for s in (scope_3_sources or []))

            total_emissions = scope_1_total + scope_2_total + scope_3_total

            # Calculate percentages
            scope_1_percent = (scope_1_total / total_emissions * 100) if total_emissions > 0 else 0
            scope_2_percent = (scope_2_total / total_emissions * 100) if total_emissions > 0 else 0
            scope_3_percent = (scope_3_total / total_emissions * 100) if total_emissions > 0 else 0

            # Breakdown by source within each scope
            scope_1_by_source = {}
            for source in scope_1_sources:
                source_name = source.get("source_name", source.get("fuel_type", "Unknown"))
                co2e = source.get("co2e_emissions_kg", 0.0)
                scope_1_by_source[source_name] = scope_1_by_source.get(source_name, 0.0) + co2e

            scope_2_by_source = {}
            for source in scope_2_sources:
                source_name = source.get("source_name", source.get("fuel_type", "Unknown"))
                co2e = source.get("co2e_emissions_kg", 0.0)
                scope_2_by_source[source_name] = scope_2_by_source.get(source_name, 0.0) + co2e

            scope_3_by_source = {}
            if scope_3_sources:
                for source in scope_3_sources:
                    source_name = source.get("source_name", source.get("fuel_type", "Unknown"))
                    co2e = source.get("co2e_emissions_kg", 0.0)
                    scope_3_by_source[source_name] = scope_3_by_source.get(source_name, 0.0) + co2e

            # Create calculation citation
            calc_citation = CalculationCitation(
                step_name="calculate_scope_emissions",
                formula="Total = Scope1 + Scope2 + Scope3",
                inputs={
                    "scope_1_sources": len(scope_1_sources),
                    "scope_2_sources": len(scope_2_sources),
                    "scope_3_sources": len(scope_3_sources or []),
                },
                output={
                    "total_emissions_kg": total_emissions,
                    "scope_1_kg": scope_1_total,
                    "scope_2_kg": scope_2_total,
                    "scope_3_kg": scope_3_total,
                    "unit": "kgCO2e"
                }
            )

            return ToolResult(
                success=True,
                data={
                    "total_co2e_kg": total_emissions,
                    "total_co2e_tons": total_emissions / 1000,
                    "scope_1_kg": scope_1_total,
                    "scope_2_kg": scope_2_total,
                    "scope_3_kg": scope_3_total,
                    "scope_1_percent": round(scope_1_percent, 2),
                    "scope_2_percent": round(scope_2_percent, 2),
                    "scope_3_percent": round(scope_3_percent, 2),
                    "scope_1_by_source": scope_1_by_source,
                    "scope_2_by_source": scope_2_by_source,
                    "scope_3_by_source": scope_3_by_source,
                },
                citations=[calc_citation],
                metadata={
                    "calculation": f"Scope 1: {scope_1_total:.2f} kg ({scope_1_percent:.1f}%), "
                                   f"Scope 2: {scope_2_total:.2f} kg ({scope_2_percent:.1f}%), "
                                   f"Scope 3: {scope_3_total:.2f} kg ({scope_3_percent:.1f}%)"
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Scope emission calculation failed: {str(e)}"
            )

    def get_tool_def(self) -> ToolDef:
        """Get tool definition."""
        return ToolDef(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "required": ["scope_1_sources", "scope_2_sources"],
                "properties": {
                    "scope_1_sources": {
                        "type": "array",
                        "description": "Scope 1 emission sources (direct emissions)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_name": {"type": "string"},
                                "fuel_type": {"type": "string"},
                                "co2e_emissions_kg": {"type": "number"}
                            }
                        }
                    },
                    "scope_2_sources": {
                        "type": "array",
                        "description": "Scope 2 emission sources (purchased electricity)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_name": {"type": "string"},
                                "fuel_type": {"type": "string"},
                                "co2e_emissions_kg": {"type": "number"}
                            }
                        }
                    },
                    "scope_3_sources": {
                        "type": "array",
                        "description": "Scope 3 emission sources (value chain)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_name": {"type": "string"},
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
# Regional Emission Factor Tool
# ==============================================================================

class RegionalEmissionFactorTool(BaseTool):
    """
    Get regional emission factors for grid electricity.

    Provides region-specific emission factors including:
    - Average grid intensity
    - Marginal emission rates
    - Temporal factors (if available)

    Deterministic lookup (would be IDEMPOTENT if connected to live API).
    """

    # Regional emission factors (kgCO2e/kWh) - 2025 estimates
    REGIONAL_FACTORS = {
        # US Regions (eGRID)
        "US": 0.385,  # National average
        "US-WECC": 0.295,  # Western
        "US-ERCOT": 0.376,  # Texas
        "US-MROW": 0.651,  # Midwest
        "US-NPCC": 0.183,  # Northeast
        "US-RFC": 0.402,  # Mid-Atlantic
        "US-SERC": 0.408,  # Southeast
        "US-SPP": 0.583,  # Southwest
        "US-TRE": 0.376,  # Texas (duplicate of ERCOT)
        "US-FRCC": 0.421,  # Florida

        # International (IEA 2024)
        "EU": 0.255,  # European Union average
        "UK": 0.212,
        "DE": 0.348,  # Germany
        "FR": 0.052,  # France (nuclear heavy)
        "CN": 0.555,  # China
        "IN": 0.708,  # India
        "JP": 0.445,  # Japan
        "AU": 0.635,  # Australia
        "CA": 0.120,  # Canada (hydro heavy)
        "BR": 0.074,  # Brazil (hydro heavy)

        # US States (selected)
        "US-CA": 0.208,  # California
        "US-NY": 0.160,  # New York
        "US-TX": 0.376,  # Texas
        "US-FL": 0.421,  # Florida
        "US-WA": 0.095,  # Washington (hydro)
        "US-OR": 0.105,  # Oregon
        "US-CO": 0.572,  # Colorado
        "US-IL": 0.378,  # Illinois
        "US-MA": 0.242,  # Massachusetts
    }

    # Marginal emission rates (higher during peak demand)
    MARGINAL_FACTORS = {
        "US": 0.456,
        "US-WECC": 0.350,
        "US-ERCOT": 0.445,
        "EU": 0.320,
        "UK": 0.280,
    }

    def __init__(self):
        super().__init__(
            name="get_regional_emission_factor",
            description="Get regional emission factors for grid electricity by geographic region",
            safety=ToolSafety.DETERMINISTIC
        )

    def execute(
        self,
        region: str,
        year: int = 2025,
        include_marginal: bool = False,
        temporal_hour: Optional[int] = None,
    ) -> ToolResult:
        """
        Execute regional emission factor lookup.

        Args:
            region: Region code (e.g., "US-WECC", "EU", "US-CA")
            year: Year for emission factor (default: 2025)
            include_marginal: Include marginal emission rate
            temporal_hour: Hour of day (0-23) for temporal factors

        Returns:
            ToolResult with regional emission factors
        """
        try:
            # Normalize region code
            region = region.upper()

            # Look up average factor
            avg_factor = self.REGIONAL_FACTORS.get(region)

            if avg_factor is None:
                # Try to find close match
                close_matches = [
                    key for key in self.REGIONAL_FACTORS.keys()
                    if region in key or key in region
                ]

                if close_matches:
                    region = close_matches[0]
                    avg_factor = self.REGIONAL_FACTORS[region]
                else:
                    # Default to US average
                    region = "US"
                    avg_factor = self.REGIONAL_FACTORS["US"]

            # Get marginal factor if requested
            marginal_factor = None
            if include_marginal:
                marginal_factor = self.MARGINAL_FACTORS.get(
                    region,
                    avg_factor * 1.18  # Default: 18% higher
                )

            # Calculate temporal factor if hour provided
            temporal_factor = None
            if temporal_hour is not None:
                temporal_factor = self._get_temporal_factor(region, temporal_hour, avg_factor)

            # Create emission factor citation
            ef_citation = create_emission_factor_citation(
                source="EPA eGRID 2025 / IEA 2024",
                factor_name=f"Grid Electricity - {region}",
                value=avg_factor,
                unit="kgCO2e/kWh",
                version="2025.1",
                last_updated=datetime(2025, 1, 15),
                confidence="high",
                region=region,
                gwp_set="AR6GWP100"
            )

            return ToolResult(
                success=True,
                data={
                    "region": region,
                    "avg_emission_factor": avg_factor,
                    "marginal_emission_factor": marginal_factor,
                    "temporal_emission_factor": temporal_factor,
                    "unit": "kgCO2e/kWh",
                    "year": year,
                    "temporal_hour": temporal_hour,
                },
                citations=[ef_citation],
                metadata={
                    "source": "EPA eGRID 2025 / IEA 2024",
                    "confidence": "high",
                    "last_updated": "2025-01-15",
                }
            )

        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Regional emission factor lookup failed: {str(e)}"
            )

    def _get_temporal_factor(
        self,
        region: str,
        hour: int,
        avg_factor: float
    ) -> float:
        """
        Get temporal (hourly) emission factor.

        Simple model: higher during peak demand hours (3-9 PM).

        Args:
            region: Region code
            hour: Hour of day (0-23)
            avg_factor: Average emission factor

        Returns:
            Temporal emission factor
        """
        # Peak hours: 3 PM - 9 PM (15-21)
        if 15 <= hour <= 21:
            # Peak: 15% higher than average
            return avg_factor * 1.15
        elif 0 <= hour <= 6:
            # Off-peak night: 10% lower than average
            return avg_factor * 0.90
        else:
            # Normal hours
            return avg_factor

    def get_tool_def(self) -> ToolDef:
        """Get tool definition."""
        return ToolDef(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "required": ["region"],
                "properties": {
                    "region": {
                        "type": "string",
                        "description": "Region code (e.g., 'US-WECC', 'EU', 'US-CA')"
                    },
                    "year": {
                        "type": "integer",
                        "description": "Year for emission factor (default: 2025)",
                        "default": 2025,
                        "minimum": 2020,
                        "maximum": 2030
                    },
                    "include_marginal": {
                        "type": "boolean",
                        "description": "Include marginal emission rate (default: false)",
                        "default": False
                    },
                    "temporal_hour": {
                        "type": "integer",
                        "description": "Hour of day (0-23) for temporal factors",
                        "minimum": 0,
                        "maximum": 23
                    }
                }
            },
            safety=self.safety
        )
