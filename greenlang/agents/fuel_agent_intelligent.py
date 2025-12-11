# -*- coding: utf-8 -*-
"""
Intelligent Fuel Agent - AI-Native Fuel Emissions Calculator

This is the RETROFITTED version of FuelAgent with full LLM intelligence.
It demonstrates the pattern for upgrading any existing agent to be AI-native.

BEFORE (fuel_agent.py):
    - Calculates fuel emissions deterministically
    - Returns static recommendations based on fuel type
    - No AI insights or contextual explanations

AFTER (this file):
    - Same deterministic calculations (zero-hallucination)
    - AI-generated explanations of fuel emissions and factors
    - AI-powered recommendations for fuel switching and efficiency
    - Anomaly detection for unusual consumption patterns
    - Regulatory context awareness (GHG Protocol, CSRD, SB253)

Migration Pattern:
    The original FuelAgent calculation logic is UNCHANGED.
    Intelligence is ADDED on top via IntelligentAgentBase.

Usage:
    from greenlang.agents.fuel_agent_intelligent import IntelligentFuelAgent

    agent = IntelligentFuelAgent()
    result = agent.run({
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "country": "US"
    })

    # Result now includes AI-generated content:
    print(result.data["explanation"])  # "Your natural gas consumption of..."
    print(result.data["recommendations"])  # [{...}]

Author: GreenLang Intelligence Framework
Date: December 2025
Status: Production Ready
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from functools import lru_cache
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from greenlang.agents.base import AgentConfig, AgentResult
from greenlang.agents.intelligent_base import (
    IntelligentAgentBase,
    IntelligentAgentConfig,
    IntelligenceLevel,
    Recommendation,
    Anomaly,
)
from greenlang.agents.intelligence_interface import (
    IntelligenceCapabilities,
    require_intelligence,
)
from greenlang.data.emission_factors import EmissionFactors
from greenlang.utils.unit_converter import UnitConverter
from greenlang.utils.performance_tracker import PerformanceTracker
from greenlang.agents.tools import get_registry
from greenlang.exceptions import ValidationError, ExecutionError, MissingData
from greenlang.determinism import DeterministicClock
from templates.agent_monitoring import OperationalMonitoringMixin

logger = logging.getLogger(__name__)


class IntelligentFuelAgentConfig(IntelligentAgentConfig):
    """Configuration for Intelligent Fuel Agent."""

    class Config:
        extra = "allow"


@require_intelligence
class IntelligentFuelAgent(OperationalMonitoringMixin, IntelligentAgentBase):
    """
    AI-Native Fuel Emissions Calculator.

    This agent calculates emissions from fuel consumption with FULL LLM INTELLIGENCE:

    1. DETERMINISTIC CALCULATIONS (Zero-Hallucination):
       - Emission factor lookup and application
       - Energy content calculations
       - Scope determination (1, 2, 3)
       - Renewable offset calculations

    2. AI-POWERED INTELLIGENCE:
       - Natural language explanations of emissions
       - Fuel switching recommendations with ROI
       - Anomaly detection for unusual patterns
       - Regulatory context (GHG Protocol, CSRD, SB253)

    Intelligence Level: STANDARD (explanations + recommendations)
    """

    agent_id: str = "intelligent_fuel"
    name: str = "Intelligent Fuel Emissions Calculator"
    version: str = "1.0.0"

    # Cache configuration
    CACHE_TTL_SECONDS = 3600

    @classmethod
    @lru_cache(maxsize=1)
    def load_fuel_config(cls) -> Dict:
        """Load fuel properties configuration from external file."""
        config_path = Path(__file__).parent.parent / "configs" / "fuel_properties.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        else:
            return {
                "fuel_properties": {
                    "electricity": {"energy_content": {"value": 3412, "unit": "Btu/kWh"}},
                    "natural_gas": {"energy_content": {"value": 100000, "unit": "Btu/therm"}},
                    "diesel": {"energy_content": {"value": 138690, "unit": "Btu/gallon"}},
                }
            }

    def __init__(self, config: Optional[IntelligentFuelAgentConfig] = None):
        """Initialize the intelligent fuel agent."""
        if config is None:
            config = IntelligentFuelAgentConfig(
                name="IntelligentFuelAgent",
                description="AI-native fuel emissions calculator with LLM intelligence",
                intelligence_level=IntelligenceLevel.STANDARD,
                enable_explanations=True,
                enable_recommendations=True,
                enable_anomaly_detection=True,
                regulatory_context="GHG Protocol Scope 1/2, CSRD ESRS E1, SB253",
                domain_context="fuel combustion emissions and energy conversion"
            )

        super().__init__(config)
        self.setup_monitoring(agent_name="intelligent_fuel_agent")

        # Initialize fuel agent components
        self.emission_factors = EmissionFactors()
        self.fuel_config = self.load_fuel_config()
        self.unit_converter = UnitConverter()
        self.performance_tracker = PerformanceTracker(self.agent_id)
        self._cache = {}
        self._cache_timestamps = {}
        self._historical_data = []
        self._execution_times = []
        self._cache_hits = 0
        self._cache_misses = 0

        # Get shared calculation tool
        registry = get_registry()
        self.calc_tool = registry.get("calculate_emissions")

        logger.info(
            f"Initialized IntelligentFuelAgent with intelligence level: "
            f"{config.intelligence_level.value}"
        )

    # =========================================================================
    # MANDATORY: Intelligence Interface Implementation
    # =========================================================================

    def get_intelligence_level(self) -> IntelligenceLevel:
        """Return the agent's intelligence level."""
        return self.intelligent_config.intelligence_level

    def get_intelligence_capabilities(self) -> IntelligenceCapabilities:
        """Return the agent's intelligence capabilities."""
        return IntelligenceCapabilities(
            can_explain=True,
            can_recommend=True,
            can_detect_anomalies=True,
            can_reason=True,
            can_validate=True,
            uses_rag=False,
            uses_tools=False
        )

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data structure."""
        if "fuel_type" not in input_data:
            logger.error("Missing required field: fuel_type")
            return False
        if "amount" not in input_data:
            logger.error("Missing required field: amount")
            return False
        if "unit" not in input_data:
            logger.error("Missing required field: unit")
            return False
        return True

    # =========================================================================
    # CORE EXECUTION
    # =========================================================================

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute fuel emissions calculation with AI intelligence.

        This method:
        1. Performs deterministic emission calculation
        2. Generates AI explanation of the emissions
        3. Generates AI recommendations for reduction
        4. Detects anomalies in consumption patterns

        Args:
            input_data: Dictionary with fuel consumption details

        Returns:
            AgentResult with calculation data and AI intelligence
        """
        with self.track_execution(input_data) as tracker:
            start_time = DeterministicClock.now()

            # =========================================================
            # STEP 1: DETERMINISTIC CALCULATION (Zero-Hallucination)
            # =========================================================
            calculation_result = self._calculate_emissions(input_data)

            if not calculation_result["success"]:
                return AgentResult(
                    success=False,
                    error=calculation_result.get("error", "Calculation failed"),
                    timestamp=DeterministicClock.now()
                )

            calc_data = calculation_result["data"]
            calculation_steps = self._get_calculation_steps(input_data, calc_data)

            # =========================================================
            # STEP 2: AI-POWERED EXPLANATION
            # =========================================================
            explanation = self.generate_explanation(
                input_data=input_data,
                output_data=calc_data,
                calculation_steps=calculation_steps
            )

            # =========================================================
            # STEP 3: AI-POWERED RECOMMENDATIONS
            # =========================================================
            recommendations = []
            if self.intelligent_config.enable_recommendations:
                recommendations = self.generate_recommendations(
                    analysis={
                        **calc_data,
                        "fuel_type": input_data.get("fuel_type"),
                        "amount": input_data.get("amount"),
                        "unit": input_data.get("unit"),
                        "country": input_data.get("country", "US"),
                        "alternative_fuels": self._get_alternative_fuels(input_data.get("fuel_type")),
                        "efficiency_potential": self._estimate_efficiency_potential(input_data)
                    },
                    max_recommendations=5,
                    focus_areas=["fuel_switching", "efficiency", "renewable_energy", "electrification"]
                )

            # =========================================================
            # STEP 4: ANOMALY DETECTION
            # =========================================================
            anomalies = []
            if self.intelligent_config.enable_anomaly_detection:
                anomalies = self.detect_anomalies(
                    data=input_data,
                    expected_ranges=self._get_expected_consumption_ranges(input_data.get("fuel_type"))
                )

            # =========================================================
            # STEP 5: BUILD INTELLIGENT RESULT
            # =========================================================
            duration = (DeterministicClock.now() - start_time).total_seconds()

            result_data = {
                **calc_data,
                "explanation": explanation,
                "recommendations": [r.dict() if hasattr(r, 'dict') else r for r in recommendations],
                "anomalies": [a.dict() if hasattr(a, 'dict') else a for a in anomalies],
                "intelligence_level": self.get_intelligence_level().value,
                "calculation_time_ms": duration * 1000,
            }

            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent": "IntelligentFuelAgent",
                    "calculation": f"{abs(input_data.get('amount', 0))} {input_data.get('unit')} x {calc_data.get('emission_factor')} kgCO2e/{input_data.get('unit')}",
                    "intelligence_metrics": self.get_intelligence_metrics().dict(),
                    "regulatory_context": self.intelligent_config.regulatory_context,
                    "cache_hits": self._cache_hits,
                    "cache_misses": self._cache_misses,
                },
                timestamp=DeterministicClock.now()
            )

    # =========================================================================
    # DETERMINISTIC CALCULATION METHODS (Zero-Hallucination)
    # =========================================================================

    @lru_cache(maxsize=256)
    def _get_cached_emission_factor(
        self, fuel_type: str, unit: str, region: str
    ) -> Optional[float]:
        """Get emission factor with caching for performance."""
        cache_key = f"fuel_{fuel_type}_{unit}_{region}"

        if cache_key in self._cache:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

        return self.emission_factors.get_factor(
            fuel_type=fuel_type, unit=unit, region=region
        )

    def _calculate_emissions(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate fuel emissions - FULLY DETERMINISTIC.

        This method contains ZERO LLM calls.
        """
        try:
            fuel_type = input_data["fuel_type"]
            amount = input_data["amount"]
            unit = input_data["unit"]
            country = input_data.get("country", "US")

            # Map fuel type aliases
            fuel_type_mapping = {
                "lpg": "propane",
                "heating_oil": "fuel_oil",
                "wood": "biomass",
                "electric": "electricity",
            }
            fuel_type = fuel_type_mapping.get(fuel_type, fuel_type)

            # Get emission factor
            emission_factor = self._get_cached_emission_factor(fuel_type, unit, country)

            if emission_factor is None:
                return {
                    "success": False,
                    "error": f"No emission factor found for {fuel_type} in {country}"
                }

            # Calculate emissions
            calc_result = self.calc_tool(
                fuel_type=fuel_type,
                amount=abs(amount),
                unit=unit,
                emission_factor=emission_factor,
                emission_factor_unit=f"kgCO2e/{unit}",
                country=country
            )

            if not calc_result.success:
                return {"success": False, "error": calc_result.error}

            co2e_emissions_kg = calc_result.data["emissions_kg_co2e"]

            # Apply renewable offset if specified
            renewable_percentage = input_data.get("renewable_percentage", 0)
            if renewable_percentage > 0 and fuel_type == "electricity":
                offset = co2e_emissions_kg * (renewable_percentage / 100)
                co2e_emissions_kg -= offset

            # Apply efficiency if specified
            efficiency = input_data.get("efficiency", 1.0)
            if efficiency < 1.0:
                co2e_emissions_kg = co2e_emissions_kg / efficiency

            # Calculate energy content
            energy_mmbtu = self._calculate_energy_content(amount, unit, fuel_type)

            # Determine scope
            scope = self._determine_scope(fuel_type)

            return {
                "success": True,
                "data": {
                    "co2e_emissions_kg": round(co2e_emissions_kg, 2),
                    "co2e_emissions_tons": round(co2e_emissions_kg / 1000, 4),
                    "fuel_type": fuel_type,
                    "consumption_amount": amount,
                    "consumption_unit": unit,
                    "emission_factor": emission_factor,
                    "emission_factor_unit": f"kgCO2e/{unit}",
                    "country": country,
                    "scope": scope,
                    "energy_content_mmbtu": energy_mmbtu,
                    "renewable_offset_applied": renewable_percentage > 0,
                    "efficiency_adjusted": efficiency < 1.0,
                }
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _calculate_energy_content(
        self, amount: float, unit: str, fuel_type: str
    ) -> float:
        """Calculate energy content in MMBtu."""
        try:
            return self.unit_converter.convert_fuel_to_energy(
                abs(amount), unit, fuel_type, "MMBtu"
            )
        except Exception:
            if unit == "kWh":
                return abs(amount) * 0.003412
            elif unit == "therms":
                return abs(amount) * 0.1
            elif unit == "gallons" and fuel_type == "diesel":
                return abs(amount) * 0.138
            return 0.0

    def _determine_scope(self, fuel_type: str) -> str:
        """Determine GHG Protocol scope for fuel type."""
        scope_mapping = {
            "natural_gas": "1",
            "diesel": "1",
            "gasoline": "1",
            "propane": "1",
            "fuel_oil": "1",
            "coal": "1",
            "biomass": "1",
            "electricity": "2",
            "district_heating": "2",
            "district_cooling": "2",
        }
        return scope_mapping.get(fuel_type, "1")

    def _get_calculation_steps(
        self,
        input_data: Dict[str, Any],
        result: Dict[str, Any]
    ) -> List[str]:
        """Document calculation steps for explainability."""
        steps = [
            f"Input: {input_data.get('amount')} {input_data.get('unit')} of {input_data.get('fuel_type')}",
            f"Country/region: {input_data.get('country', 'US')}",
            f"Retrieved emission factor: {result.get('emission_factor')} kgCO2e/{input_data.get('unit')}",
            f"Calculated: {input_data.get('amount')} x {result.get('emission_factor')} = {result.get('co2e_emissions_kg')} kg CO2e",
            f"Determined GHG Protocol Scope: {result.get('scope')}",
        ]

        if input_data.get("renewable_percentage", 0) > 0:
            steps.append(f"Applied {input_data.get('renewable_percentage')}% renewable offset")

        if input_data.get("efficiency", 1.0) < 1.0:
            steps.append(f"Adjusted for {input_data.get('efficiency') * 100}% equipment efficiency")

        steps.append(f"Final result: {result.get('co2e_emissions_tons')} metric tons CO2e")

        return steps

    def _get_alternative_fuels(self, current_fuel: str) -> List[str]:
        """Get alternative fuels for switching recommendations."""
        alternatives = {
            "coal": ["natural_gas", "biomass", "electricity"],
            "fuel_oil": ["natural_gas", "propane", "electricity"],
            "diesel": ["biodiesel", "electricity", "hydrogen"],
            "gasoline": ["electricity", "hydrogen", "ethanol"],
            "natural_gas": ["renewable_natural_gas", "hydrogen", "electricity"],
            "propane": ["natural_gas", "electricity"],
            "electricity": ["solar_pv", "wind", "green_tariff"],
        }
        return alternatives.get(current_fuel, ["electricity"])

    def _estimate_efficiency_potential(self, input_data: Dict[str, Any]) -> str:
        """Estimate efficiency improvement potential."""
        fuel_type = input_data.get("fuel_type", "")
        current_efficiency = input_data.get("efficiency", 0.8)

        if fuel_type in ["natural_gas", "fuel_oil", "propane"]:
            if current_efficiency < 0.85:
                return "High (15-25% potential improvement with condensing equipment)"
            elif current_efficiency < 0.95:
                return "Medium (5-10% potential improvement)"
            else:
                return "Low (already near optimal efficiency)"
        elif fuel_type == "electricity":
            return "Focus on demand reduction and renewable sourcing"
        else:
            return "Moderate (10-15% typical improvement potential)"

    def _get_expected_consumption_ranges(self, fuel_type: str) -> Dict[str, tuple]:
        """Define expected ranges for anomaly detection."""
        ranges = {
            "natural_gas": {"amount": (0, 100000)},  # therms
            "electricity": {"amount": (0, 1000000)},  # kWh
            "diesel": {"amount": (0, 50000)},  # gallons
            "gasoline": {"amount": (0, 50000)},  # gallons
        }
        return ranges.get(fuel_type, {"amount": (0, 1000000)})


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_intelligent_fuel_agent(
    intelligence_level: IntelligenceLevel = IntelligenceLevel.STANDARD,
    regulatory_context: str = "GHG Protocol Scope 1/2, CSRD ESRS E1",
    **kwargs
) -> IntelligentFuelAgent:
    """
    Factory function to create IntelligentFuelAgent.

    Args:
        intelligence_level: Level of LLM intelligence
        regulatory_context: Regulatory context for prompts
        **kwargs: Additional configuration

    Returns:
        Configured IntelligentFuelAgent
    """
    config = IntelligentFuelAgentConfig(
        name="IntelligentFuelAgent",
        description="AI-native fuel emissions calculator",
        intelligence_level=intelligence_level,
        regulatory_context=regulatory_context,
        **kwargs
    )

    return IntelligentFuelAgent(config)


# =============================================================================
# BACKWARD COMPATIBILITY: Retrofit existing FuelAgent
# =============================================================================

def retrofit_fuel_agent():
    """
    Retrofit the existing FuelAgent with intelligence.

    This function demonstrates how to add intelligence to any
    existing agent without modifying its original code.
    """
    from greenlang.agents.fuel_agent import FuelAgent
    from greenlang.agents.intelligence_mixin import retrofit_agent_class

    IntelligentFuelAgentRetrofit = retrofit_agent_class(FuelAgent)
    return IntelligentFuelAgentRetrofit


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """Example usage of IntelligentFuelAgent."""

    agent = create_intelligent_fuel_agent()

    example_input = {
        "fuel_type": "natural_gas",
        "amount": 5000,
        "unit": "therms",
        "country": "US",
        "efficiency": 0.85
    }

    result = agent.run(example_input)

    print("=" * 60)
    print("INTELLIGENT FUEL AGENT RESULTS")
    print("=" * 60)
    print(f"\nSuccess: {result.success}")
    print(f"\n--- DETERMINISTIC DATA ---")
    print(f"CO2e Emissions: {result.data['co2e_emissions_kg']} kg")
    print(f"Scope: {result.data['scope']}")
    print(f"\n--- AI-GENERATED EXPLANATION ---")
    print(result.data.get("explanation", "N/A"))
    print(f"\n--- AI-GENERATED RECOMMENDATIONS ---")
    for rec in result.data.get("recommendations", []):
        title = rec.get("title", "Unknown") if isinstance(rec, dict) else "Unknown"
        print(f"  - {title}")
