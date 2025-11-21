# -*- coding: utf-8 -*-
"""
greenlang/agents/fuel_tools_v2.py

Enhanced tool implementations for FuelAgentAI v2

Provides:
- lookup_emission_factor_v2: Multi-gas factors with provenance
- calculate_emissions_v2: CO2/CH4/N2O breakdown
- generate_recommendations: Unchanged from v1

These tools are called by AI but return deterministic results (zero hallucinated numbers).
"""

from typing import Dict, Any, Optional, List
from datetime import date
import logging

from ..data.emission_factor_database import EmissionFactorDatabase
from ..data.emission_factor_record import EmissionFactorRecord
from greenlang.intelligence.schemas.tools import ToolDef

logger = logging.getLogger(__name__)


class FuelToolsV2:
    """Enhanced tool implementations for FuelAgentAI v2"""

    def __init__(self):
        """Initialize v2 tools with emission factor database."""
        self.db = EmissionFactorDatabase()
        self._tool_call_count = 0

    # ==================== TOOL DEFINITIONS ====================

    def get_tool_definitions(self) -> List[ToolDef]:
        """
        Get all tool definitions for AI orchestration.

        Returns:
            List of ToolDef objects for ChatSession
        """
        return [
            self._get_lookup_tool_def(),
            self._get_calculate_tool_def(),
            self._get_recommendations_tool_def(),
        ]

    def _get_lookup_tool_def(self) -> ToolDef:
        """Define lookup_emission_factor tool (v2 enhanced)."""
        return ToolDef(
            name="lookup_emission_factor",
            description=(
                "Lookup emission factor from database with multi-gas breakdown. "
                "Returns CO2, CH4, N2O vectors separately plus full provenance. "
                "Use this BEFORE calculate_emissions to get the factor."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "fuel_type": {
                        "type": "string",
                        "description": "Fuel type (diesel, natural_gas, electricity, coal, etc.)",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit (gallons, kWh, therms, tons, etc.)",
                    },
                    "country": {
                        "type": "string",
                        "description": "ISO country code (US, UK, EU, IN, etc.)",
                        "default": "US",
                    },
                    # ========== V2 ENHANCEMENTS ==========
                    "scope": {
                        "type": "string",
                        "description": "GHG scope (1=direct, 2=electricity, 3=indirect)",
                        "enum": ["1", "2", "3"],
                        "default": "1",
                    },
                    "boundary": {
                        "type": "string",
                        "description": "Emission boundary (combustion=direct only, WTT=upstream, WTW=full lifecycle)",
                        "enum": ["combustion", "WTT", "WTW"],
                        "default": "combustion",
                    },
                    "gwp_set": {
                        "type": "string",
                        "description": "GWP reference set for CO2e calculation",
                        "enum": ["IPCC_AR6_100", "IPCC_AR6_20", "IPCC_AR5_100"],
                        "default": "IPCC_AR6_100",
                    },
                },
                "required": ["fuel_type", "unit"],
            },
        )

    def _get_calculate_tool_def(self) -> ToolDef:
        """Define calculate_emissions tool (v2 enhanced)."""
        return ToolDef(
            name="calculate_emissions",
            description=(
                "Calculate emissions from fuel consumption with multi-gas breakdown. "
                "Returns CO2, CH4, N2O separately plus aggregated CO2e. "
                "All values are deterministic (no AI-generated numbers)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "fuel_type": {
                        "type": "string",
                        "description": "Fuel type",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Consumption amount (positive for consumption, negative for renewable generation)",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit of amount",
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code",
                        "default": "US",
                    },
                    "renewable_percentage": {
                        "type": "number",
                        "description": "Renewable offset percentage (0-100)",
                        "default": 0.0,
                    },
                    "efficiency": {
                        "type": "number",
                        "description": "Equipment efficiency (0-1)",
                        "default": 1.0,
                    },
                    # ========== V2 ENHANCEMENTS ==========
                    "scope": {
                        "type": "string",
                        "enum": ["1", "2", "3"],
                        "default": "1",
                    },
                    "boundary": {
                        "type": "string",
                        "enum": ["combustion", "WTT", "WTW"],
                        "default": "combustion",
                    },
                    "gwp_set": {
                        "type": "string",
                        "enum": ["IPCC_AR6_100", "IPCC_AR6_20"],
                        "default": "IPCC_AR6_100",
                    },
                },
                "required": ["fuel_type", "amount", "unit"],
            },
        )

    def _get_recommendations_tool_def(self) -> ToolDef:
        """Define generate_recommendations tool (unchanged from v1)."""
        return ToolDef(
            name="generate_recommendations",
            description="Generate fuel switching and efficiency improvement recommendations",
            parameters={
                "type": "object",
                "properties": {
                    "fuel_type": {
                        "type": "string",
                        "description": "Current fuel type",
                    },
                    "emissions_kg": {
                        "type": "number",
                        "description": "Calculated emissions in kg CO2e",
                    },
                    "country": {
                        "type": "string",
                        "description": "Country code",
                        "default": "US",
                    },
                },
                "required": ["fuel_type", "emissions_kg"],
            },
        )

    # ==================== TOOL IMPLEMENTATIONS ====================

    def lookup_emission_factor(
        self,
        fuel_type: str,
        unit: str,
        country: str = "US",
        scope: str = "1",
        boundary: str = "combustion",
        gwp_set: str = "IPCC_AR6_100",
    ) -> Dict[str, Any]:
        """
        Lookup emission factor from database (v2 API).

        Returns multi-gas breakdown with provenance.

        Args:
            fuel_type: Fuel type
            unit: Unit
            country: Country code
            scope: GHG scope (1, 2, or 3)
            boundary: Emission boundary (combustion, WTT, WTW)
            gwp_set: GWP reference set

        Returns:
            Dict with:
            - vectors_kg_per_unit: {CO2, CH4, N2O} kg per unit
            - co2e_kg_per_unit: Aggregated CO2e (kg/unit)
            - provenance: {source_org, publication, year, citation}
            - dqs: {overall_score, rating, dimensions}
            - uncertainty_95ci_pct: ±X%
        """
        self._tool_call_count += 1

        # Lookup factor from database
        factor = self.db.get_factor_record(
            fuel_type=fuel_type,
            unit=unit,
            geography=country,
            scope=scope,
            boundary=boundary,
            gwp_set=gwp_set,
        )

        if not factor:
            raise ValueError(
                f"No emission factor found for {fuel_type} ({unit}) in {country} "
                f"[scope={scope}, boundary={boundary}, gwp_set={gwp_set}]"
            )

        # Extract GWP horizon (100yr or 20yr)
        gwp_horizon = "100yr" if "100" in gwp_set else "20yr"
        co2e_value = factor.get_co2e(gwp_horizon)

        # Build response
        return {
            # Multi-gas vectors
            "vectors_kg_per_unit": {
                "CO2": factor.vectors.CO2,
                "CH4": factor.vectors.CH4,
                "N2O": factor.vectors.N2O,
            },
            # Aggregated CO2e
            "co2e_kg_per_unit": co2e_value,
            "gwp_set": gwp_set,
            # Provenance
            "provenance": {
                "factor_id": factor.factor_id,
                "source_org": factor.provenance.source_org,
                "source_publication": factor.provenance.source_publication,
                "source_year": factor.provenance.source_year,
                "methodology": factor.provenance.methodology.value,
                "citation": factor.provenance.citation,
            },
            # Quality
            "dqs": {
                "overall_score": factor.dqs.overall_score,
                "rating": factor.dqs.rating.value,
                "temporal": factor.dqs.temporal,
                "geographical": factor.dqs.geographical,
                "technological": factor.dqs.technological,
                "representativeness": factor.dqs.representativeness,
                "methodological": factor.dqs.methodological,
            },
            "uncertainty_95ci_pct": factor.uncertainty_95ci * 100,
            # Metadata
            "unit": unit,
            "fuel_type": fuel_type,
            "country": country,
            "scope": scope,
            "boundary": boundary,
        }

    def calculate_emissions(
        self,
        fuel_type: str,
        amount: float,
        unit: str,
        country: str = "US",
        renewable_percentage: float = 0.0,
        efficiency: float = 1.0,
        scope: str = "1",
        boundary: str = "combustion",
        gwp_set: str = "IPCC_AR6_100",
    ) -> Dict[str, Any]:
        """
        Calculate emissions with multi-gas breakdown (v2 API).

        Args:
            fuel_type: Fuel type
            amount: Consumption amount
            unit: Unit
            country: Country code
            renewable_percentage: Renewable offset (0-100)
            efficiency: Equipment efficiency (0-1)
            scope: GHG scope
            boundary: Emission boundary
            gwp_set: GWP reference set

        Returns:
            Dict with:
            - vectors_kg: {CO2, CH4, N2O} total emissions
            - co2e_kg: Total CO2e emissions
            - breakdown: Calculation details
        """
        self._tool_call_count += 1

        # 1. Lookup factor
        factor_data = self.lookup_emission_factor(
            fuel_type=fuel_type,
            unit=unit,
            country=country,
            scope=scope,
            boundary=boundary,
            gwp_set=gwp_set,
        )

        # 2. Calculate emissions for each gas
        vectors = factor_data["vectors_kg_per_unit"]

        # Adjust for efficiency and renewable offset
        effective_amount = abs(amount) * efficiency * (1 - renewable_percentage / 100)

        co2_kg = effective_amount * vectors["CO2"]
        ch4_kg = effective_amount * vectors["CH4"]
        n2o_kg = effective_amount * vectors["N2O"]

        # 3. Calculate CO2e
        co2e_kg = effective_amount * factor_data["co2e_kg_per_unit"]

        # 4. Calculate renewable offset applied
        renewable_offset_kg = 0.0
        if renewable_percentage > 0:
            renewable_offset_kg = abs(amount) * efficiency * (renewable_percentage / 100) * factor_data["co2e_kg_per_unit"]

        # Build response
        return {
            # Multi-gas breakdown
            "vectors_kg": {
                "CO2": co2_kg,
                "CH4": ch4_kg,
                "N2O": n2o_kg,
            },
            # Aggregated
            "co2e_kg": co2e_kg,
            "co2e_100yr_kg": co2e_kg,  # Alias for clarity
            # Details
            "fuel_type": fuel_type,
            "amount": amount,
            "unit": unit,
            "scope": scope,
            "boundary": boundary,
            "renewable_offset_kg": renewable_offset_kg,
            "renewable_offset_applied": renewable_percentage > 0,
            "efficiency_adjusted": efficiency != 1.0,
            # Calculation breakdown
            "breakdown": {
                "effective_amount": effective_amount,
                "emission_factor_co2e": factor_data["co2e_kg_per_unit"],
                "calculation": f"{effective_amount:.2f} {unit} × {factor_data['co2e_kg_per_unit']:.4f} kgCO2e/{unit} = {co2e_kg:.2f} kgCO2e",
            },
            # Propagate provenance and quality
            "provenance": factor_data["provenance"],
            "dqs": factor_data["dqs"],
            "uncertainty_95ci_pct": factor_data["uncertainty_95ci_pct"],
        }

    def generate_recommendations(
        self,
        fuel_type: str,
        emissions_kg: float,
        country: str = "US",
    ) -> Dict[str, Any]:
        """
        Generate fuel switching and efficiency recommendations.

        Args:
            fuel_type: Current fuel type
            emissions_kg: Calculated emissions
            country: Country code

        Returns:
            Dict with recommendations list
        """
        self._tool_call_count += 1

        recommendations = []

        # Rule-based recommendations (deterministic)
        fuel_recommendations = {
            "diesel": [
                {
                    "action": "Switch to biodiesel (B20)",
                    "potential_reduction_pct": 15,
                    "feasibility": "high",
                    "description": "Blend 20% biodiesel to reduce lifecycle emissions by ~15%",
                },
                {
                    "action": "Upgrade to electric vehicle fleet",
                    "potential_reduction_pct": 65,
                    "feasibility": "medium",
                    "description": "Transition to EVs for 50-70% emission reduction (grid-dependent)",
                },
            ],
            "natural_gas": [
                {
                    "action": "Improve boiler efficiency",
                    "potential_reduction_pct": 10,
                    "feasibility": "high",
                    "description": "Upgrade to high-efficiency condensing boiler (90%+ efficiency)",
                },
                {
                    "action": "Switch to renewable natural gas (RNG)",
                    "potential_reduction_pct": 80,
                    "feasibility": "low",
                    "description": "Source RNG from anaerobic digestion or biogas",
                },
            ],
            "electricity": [
                {
                    "action": "Purchase renewable energy certificates (RECs)",
                    "potential_reduction_pct": 100,
                    "feasibility": "high",
                    "description": "Offset Scope 2 emissions via market-based accounting",
                },
                {
                    "action": "Install on-site solar PV",
                    "potential_reduction_pct": 30,
                    "feasibility": "medium",
                    "description": "Generate 20-40% of electricity demand on-site",
                },
            ],
            "coal": [
                {
                    "action": "Switch to natural gas",
                    "potential_reduction_pct": 50,
                    "feasibility": "high",
                    "description": "Replace coal boilers with natural gas (50%+ emission reduction)",
                },
                {
                    "action": "Co-firing with biomass",
                    "potential_reduction_pct": 20,
                    "feasibility": "medium",
                    "description": "Blend 20-30% biomass to reduce net fossil emissions",
                },
            ],
        }

        # Get recommendations for fuel type
        fuel_recs = fuel_recommendations.get(fuel_type, [])
        recommendations.extend(fuel_recs)

        # Add general efficiency recommendation if emissions > 1000 kg
        if emissions_kg > 1000:
            recommendations.append({
                "action": "Conduct energy audit",
                "potential_reduction_pct": 15,
                "feasibility": "high",
                "description": "Professional energy audit to identify 10-20% efficiency gains",
            })

        return {
            "recommendations": recommendations,
            "current_fuel": fuel_type,
            "current_emissions_kg": emissions_kg,
            "country": country,
        }

    # ==================== TOOL ROUTING ====================

    def handle_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route tool call to appropriate implementation.

        Args:
            tool_name: Name of tool to call
            arguments: Tool arguments

        Returns:
            Tool result

        Raises:
            ValueError: If tool not found
        """
        handlers = {
            "lookup_emission_factor": self.lookup_emission_factor,
            "calculate_emissions": self.calculate_emissions,
            "generate_recommendations": self.generate_recommendations,
        }

        if tool_name not in handlers:
            raise ValueError(f"Unknown tool: {tool_name}")

        handler = handlers[tool_name]

        try:
            result = handler(**arguments)
            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            raise

    def get_tool_call_count(self) -> int:
        """Get total number of tool calls made."""
        return self._tool_call_count

    def reset_tool_call_count(self):
        """Reset tool call counter."""
        self._tool_call_count = 0
