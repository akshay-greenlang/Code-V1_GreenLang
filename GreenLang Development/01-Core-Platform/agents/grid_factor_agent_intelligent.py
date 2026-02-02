# -*- coding: utf-8 -*-
"""
Intelligent Grid Factor Agent - AI-Native Emission Factor Provider

This is the RETROFITTED version of GridFactorAgent with full LLM intelligence.
It demonstrates the pattern for upgrading any existing agent to be AI-native.

BEFORE (grid_factor_agent.py):
    - Retrieves emission factors from database
    - Returns static factor values
    - No context or explanation

AFTER (this file):
    - Same deterministic factor retrieval (zero-hallucination)
    - AI-generated explanations of emission factors
    - AI-powered insights on grid mix and decarbonization
    - Anomaly detection for unusual factor values
    - Regulatory context awareness (CSRD, SB253, ISO 14064)

Migration Pattern:
    The original GridFactorAgent lookup logic is UNCHANGED.
    Intelligence is ADDED on top via IntelligentAgentBase.

Usage:
    from greenlang.agents.grid_factor_agent_intelligent import IntelligentGridFactorAgent

    agent = IntelligentGridFactorAgent()
    result = agent.run({
        "country": "US",
        "fuel_type": "electricity",
        "unit": "kWh"
    })

    # Result now includes AI-generated content:
    print(result.data["explanation"])  # "The US grid emission factor of..."
    print(result.data["recommendations"])  # [{...}]

Author: GreenLang Intelligence Framework
Date: December 2025
Status: Production Ready
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import json
from pathlib import Path

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
from greenlang.utilities.determinism import DeterministicClock
from greenlang.templates.agent_monitoring import OperationalMonitoringMixin

logger = logging.getLogger(__name__)


class IntelligentGridFactorAgentConfig(IntelligentAgentConfig):
    """Configuration for Intelligent Grid Factor Agent."""

    class Config:
        extra = "allow"


@require_intelligence
class IntelligentGridFactorAgent(OperationalMonitoringMixin, IntelligentAgentBase):
    """
    AI-Native Grid Emission Factor Provider.

    This agent retrieves emission factors with FULL LLM INTELLIGENCE:

    1. DETERMINISTIC OPERATIONS (Zero-Hallucination):
       - Emission factor database lookup
       - Country/region mapping
       - Grid mix calculation
       - Factor unit conversion

    2. AI-POWERED INTELLIGENCE:
       - Natural language explanations of emission factors
       - Grid decarbonization trend insights
       - Recommendations for location-based vs market-based reporting
       - Anomaly detection for unusual factor values
       - Regulatory context (ISO 14064, GHG Protocol Scope 2)

    Intelligence Level: STANDARD (explanations + recommendations)
    """

    agent_id: str = "intelligent_grid_factor"
    name: str = "Intelligent Grid Emission Factor Provider"
    version: str = "1.0.0"

    def __init__(self, config: Optional[IntelligentGridFactorAgentConfig] = None):
        """Initialize the intelligent grid factor agent."""
        if config is None:
            config = IntelligentGridFactorAgentConfig(
                name="IntelligentGridFactorAgent",
                description="AI-native grid emission factor provider with LLM intelligence",
                intelligence_level=IntelligenceLevel.STANDARD,
                enable_explanations=True,
                enable_recommendations=True,
                enable_anomaly_detection=True,
                regulatory_context="GHG Protocol Scope 2, ISO 14064, CSRD ESRS E1",
                domain_context="electricity grid emission factors and grid decarbonization"
            )

        super().__init__(config)
        self.setup_monitoring(agent_name="intelligent_grid_factor_agent")

        # Load emission factors database
        self.factors_path = (
            Path(__file__).parent.parent / "data" / "global_emission_factors.json"
        )
        self.emission_factors: Dict[str, Dict[str, Dict[str, float]]] = (
            self._load_emission_factors()
        )

        logger.info(
            f"Initialized IntelligentGridFactorAgent with intelligence level: "
            f"{config.intelligence_level.value}"
        )

    def _load_emission_factors(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Load emission factors from JSON file."""
        try:
            with open(self.factors_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load emission factors: {e}")
            return {
                "US": {
                    "electricity": {"emission_factor": 0.385, "unit": "kgCO2e/kWh"},
                    "natural_gas": {"emission_factor": 5.3, "unit": "kgCO2e/therm"},
                }
            }

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
        if not input_data.get("country"):
            logger.error("Missing required field: country")
            return False
        if not input_data.get("fuel_type"):
            logger.error("Missing required field: fuel_type")
            return False
        if not input_data.get("unit"):
            logger.error("Missing required field: unit")
            return False
        return True

    # =========================================================================
    # CORE EXECUTION
    # =========================================================================

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Execute grid factor retrieval with AI intelligence.

        This method:
        1. Retrieves emission factor deterministically
        2. Generates AI explanation of the factor
        3. Generates AI recommendations for reporting
        4. Detects anomalies in factor values

        Args:
            input_data: Dictionary with country, fuel_type, unit

        Returns:
            AgentResult with factor data and AI intelligence
        """
        with self.track_execution(input_data) as tracker:
            # =========================================================
            # STEP 1: DETERMINISTIC RETRIEVAL (Zero-Hallucination)
            # =========================================================
            retrieval_result = self._retrieve_emission_factor(input_data)

            if not retrieval_result["success"]:
                return AgentResult(
                    success=False,
                    error=retrieval_result.get("error", "Factor retrieval failed"),
                    timestamp=DeterministicClock.now()
                )

            factor_data = retrieval_result["data"]
            retrieval_steps = self._get_retrieval_steps(input_data, factor_data)

            # =========================================================
            # STEP 2: AI-POWERED EXPLANATION
            # =========================================================
            explanation = self.generate_explanation(
                input_data=input_data,
                output_data=factor_data,
                calculation_steps=retrieval_steps
            )

            # =========================================================
            # STEP 3: AI-POWERED RECOMMENDATIONS
            # =========================================================
            recommendations = []
            if self.intelligent_config.enable_recommendations:
                recommendations = self.generate_recommendations(
                    analysis={
                        **factor_data,
                        "grid_context": self._get_grid_context(input_data.get("country")),
                        "decarbonization_trend": self._get_decarbonization_trend(input_data.get("country")),
                        "reporting_options": self._get_reporting_options(input_data.get("country"))
                    },
                    max_recommendations=5,
                    focus_areas=["location_vs_market", "renewable_procurement", "grid_decarbonization", "reporting_accuracy"]
                )

            # =========================================================
            # STEP 4: ANOMALY DETECTION
            # =========================================================
            anomalies = []
            if self.intelligent_config.enable_anomaly_detection:
                anomalies = self.detect_anomalies(
                    data=factor_data,
                    expected_ranges=self._get_expected_factor_ranges(input_data.get("fuel_type"))
                )

            # =========================================================
            # STEP 5: BUILD INTELLIGENT RESULT
            # =========================================================
            result_data = {
                **factor_data,
                "explanation": explanation,
                "recommendations": [r.dict() if hasattr(r, 'dict') else r for r in recommendations],
                "anomalies": [a.dict() if hasattr(a, 'dict') else a for a in anomalies],
                "intelligence_level": self.get_intelligence_level().value,
            }

            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent": "IntelligentGridFactorAgent",
                    "source": self.emission_factors.get("metadata", {}).get("source", "IPCC"),
                    "methodology": self.emission_factors.get("metadata", {}).get("methodology", "IPCC Guidelines"),
                    "intelligence_metrics": self.get_intelligence_metrics().dict(),
                    "regulatory_context": self.intelligent_config.regulatory_context,
                },
                timestamp=DeterministicClock.now()
            )

    # =========================================================================
    # DETERMINISTIC RETRIEVAL METHODS (Zero-Hallucination)
    # =========================================================================

    def _retrieve_emission_factor(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve emission factor - FULLY DETERMINISTIC.

        This method contains ZERO LLM calls.
        """
        try:
            country = input_data["country"]
            fuel_type = input_data["fuel_type"]
            unit = input_data["unit"]

            # Map country codes
            country_mapping = {
                "USA": "US", "INDIA": "IN", "CHINA": "CN", "JAPAN": "JP",
                "BRAZIL": "BR", "SOUTH_KOREA": "KR", "KOREA": "KR",
                "GERMANY": "DE", "FRANCE": "FR", "CANADA": "CA",
                "AUSTRALIA": "AU", "UNITED_KINGDOM": "UK",
            }
            mapped_country = country_mapping.get(country.upper(), country)

            # Check if country exists
            if mapped_country not in self.emission_factors:
                logger.warning(f"Country {mapped_country} not found, using US factors")
                mapped_country = "US"

            country_factors = self.emission_factors[mapped_country]

            # Get fuel type factors
            if fuel_type not in country_factors:
                return {
                    "success": False,
                    "error": f"Fuel type '{fuel_type}' not found for country {mapped_country}"
                }

            fuel_data = country_factors[fuel_type]

            # Get emission factor based on unit
            if isinstance(fuel_data, dict) and unit in fuel_data:
                factor = fuel_data[unit]
            elif isinstance(fuel_data, dict) and "emission_factor" in fuel_data:
                factor = fuel_data["emission_factor"]
            else:
                factor = 0.0

            if factor == 0.0:
                return {
                    "success": False,
                    "error": f"No emission factor found for {fuel_type} with unit {unit} in {mapped_country}"
                }

            # Build output
            output = {
                "emission_factor": factor,
                "unit": f"kgCO2e/{unit}",
                "source": fuel_data.get("source", fuel_data.get("description", "GreenLang Global Dataset")) if isinstance(fuel_data, dict) else "GreenLang Global Dataset",
                "version": fuel_data.get("version", "1.0.0") if isinstance(fuel_data, dict) else "1.0.0",
                "last_updated": fuel_data.get("last_updated", "2025-08-14") if isinstance(fuel_data, dict) else "2025-08-14",
                "country": mapped_country,
                "fuel_type": fuel_type,
            }

            # Add grid mix if available
            if "grid_renewable_share" in country_factors:
                output["grid_mix"] = {
                    "renewable": country_factors["grid_renewable_share"],
                    "fossil": 1.0 - country_factors["grid_renewable_share"],
                }

            return {"success": True, "data": output}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _get_retrieval_steps(
        self,
        input_data: Dict[str, Any],
        result: Dict[str, Any]
    ) -> List[str]:
        """Document retrieval steps for explainability."""
        steps = [
            f"Input: {input_data.get('fuel_type')} for {input_data.get('country')}",
            f"Mapped country code: {result.get('country')}",
            f"Retrieved emission factor: {result.get('emission_factor')} {result.get('unit')}",
            f"Data source: {result.get('source')}",
            f"Last updated: {result.get('last_updated')}",
        ]

        if "grid_mix" in result:
            steps.append(f"Grid renewable share: {result['grid_mix']['renewable'] * 100:.1f}%")

        return steps

    def _get_grid_context(self, country: str) -> Dict[str, Any]:
        """Get grid context for a country."""
        grid_contexts = {
            "US": {
                "grid_type": "Highly variable by state/region",
                "key_regions": ["ERCOT (Texas)", "CAISO (California)", "PJM (Mid-Atlantic)"],
                "renewable_trend": "Increasing solar and wind",
                "coal_phase_out": "Ongoing in many states"
            },
            "DE": {
                "grid_type": "Energiewende transition",
                "key_regions": ["National grid"],
                "renewable_trend": "Strong wind and solar growth",
                "coal_phase_out": "2038 target"
            },
            "IN": {
                "grid_type": "Rapidly expanding",
                "key_regions": ["Regional grids"],
                "renewable_trend": "Aggressive solar deployment",
                "coal_phase_out": "Coal still dominant"
            },
        }
        return grid_contexts.get(country, {"grid_type": "National grid"})

    def _get_decarbonization_trend(self, country: str) -> str:
        """Get decarbonization trend for a country."""
        trends = {
            "US": "Declining 2-3% annually due to coal retirement and renewable growth",
            "DE": "Declining 4-5% annually due to Energiewende",
            "FR": "Already low due to nuclear, limited further reduction",
            "IN": "Rising due to demand growth, but improving on per-unit basis",
            "CN": "Peaking, expected decline post-2030",
        }
        return trends.get(country, "Trend data not available")

    def _get_reporting_options(self, country: str) -> List[str]:
        """Get available reporting options for a country."""
        return [
            "Location-based (grid average factor)",
            "Market-based (contractual instruments)",
            "Residual mix factor (where available)",
            "Supplier-specific factor (if disclosed)"
        ]

    def _get_expected_factor_ranges(self, fuel_type: str) -> Dict[str, tuple]:
        """Define expected ranges for anomaly detection."""
        ranges = {
            "electricity": {"emission_factor": (0.05, 1.2)},  # kgCO2e/kWh
            "natural_gas": {"emission_factor": (1.8, 2.5)},  # kgCO2e/m3 or therm
            "diesel": {"emission_factor": (2.5, 3.5)},  # kgCO2e/liter
        }
        return ranges.get(fuel_type, {"emission_factor": (0, 10)})

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_available_countries(self) -> List[str]:
        """Get list of available countries."""
        return [k for k in self.emission_factors.keys() if k != "metadata"]

    def get_available_fuel_types(self, country: str) -> List[str]:
        """Get available fuel types for a country."""
        country = country.upper()
        if country in self.emission_factors:
            return [
                k for k in self.emission_factors[country].keys()
                if not k.startswith("grid_renewable")
            ]
        return []


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_intelligent_grid_factor_agent(
    intelligence_level: IntelligenceLevel = IntelligenceLevel.STANDARD,
    regulatory_context: str = "GHG Protocol Scope 2, ISO 14064",
    **kwargs
) -> IntelligentGridFactorAgent:
    """
    Factory function to create IntelligentGridFactorAgent.

    Args:
        intelligence_level: Level of LLM intelligence
        regulatory_context: Regulatory context for prompts
        **kwargs: Additional configuration

    Returns:
        Configured IntelligentGridFactorAgent
    """
    config = IntelligentGridFactorAgentConfig(
        name="IntelligentGridFactorAgent",
        description="AI-native grid emission factor provider",
        intelligence_level=intelligence_level,
        regulatory_context=regulatory_context,
        **kwargs
    )

    return IntelligentGridFactorAgent(config)


# =============================================================================
# BACKWARD COMPATIBILITY: Retrofit existing GridFactorAgent
# =============================================================================

def retrofit_grid_factor_agent():
    """
    Retrofit the existing GridFactorAgent with intelligence.
    """
    from greenlang.agents.grid_factor_agent import GridFactorAgent
    from greenlang.agents.intelligence_mixin import retrofit_agent_class

    IntelligentGridFactorAgentRetrofit = retrofit_agent_class(GridFactorAgent)
    return IntelligentGridFactorAgentRetrofit


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    """Example usage of IntelligentGridFactorAgent."""

    agent = create_intelligent_grid_factor_agent()

    example_input = {
        "country": "US",
        "fuel_type": "electricity",
        "unit": "kWh"
    }

    result = agent.run(example_input)

    print("=" * 60)
    print("INTELLIGENT GRID FACTOR AGENT RESULTS")
    print("=" * 60)
    print(f"\nSuccess: {result.success}")
    print(f"\n--- DETERMINISTIC DATA ---")
    print(f"Emission Factor: {result.data['emission_factor']} {result.data['unit']}")
    print(f"Country: {result.data['country']}")
    print(f"\n--- AI-GENERATED EXPLANATION ---")
    print(result.data.get("explanation", "N/A"))
    print(f"\n--- AI-GENERATED RECOMMENDATIONS ---")
    for rec in result.data.get("recommendations", []):
        title = rec.get("title", "Unknown") if isinstance(rec, dict) else "Unknown"
        print(f"  - {title}")
