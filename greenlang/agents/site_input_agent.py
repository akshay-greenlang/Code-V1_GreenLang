# -*- coding: utf-8 -*-
"""
SiteInputAgent - AI-Native Site Feasibility Input Loader

This agent loads, validates, and normalizes site feasibility input YAML with
FULL LLM INTELLIGENCE capabilities via IntelligenceMixin.

Intelligence Level: BASIC
Capabilities: Explanations

Regulatory Context: ISO 50001, Building Codes
"""

import yaml
import sys
import os
from typing import Dict, Any
from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from greenlang.agents.intelligence_mixin import IntelligenceMixin
from greenlang.agents.intelligence_interface import IntelligenceCapabilities, IntelligenceLevel


# Dynamic import for climatenza_app
def get_feasibility_input():
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    from climatenza_app.schemas.feasibility import FeasibilityInput

    return FeasibilityInput


class SiteInputAgent(IntelligenceMixin, BaseAgent):
    """
    AI-Native Site Feasibility Input Loader Agent.

    This agent loads and validates site data with FULL LLM INTELLIGENCE:

    1. DETERMINISTIC VALIDATION (Zero-Hallucination):
       - YAML parsing and schema validation
       - Required field checks
       - Data type validation

    2. AI-POWERED INTELLIGENCE:
       - Natural language explanations of validation results
       - Suggestions for missing or incorrect data
    """

    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="SiteInputAgent",
                description="AI-native site feasibility input validator with LLM intelligence",
            )
        super().__init__(config)
        # Intelligence auto-initializes on first use

    def get_intelligence_level(self) -> IntelligenceLevel:
        """Return the agent's intelligence level."""
        return IntelligenceLevel.BASIC

    def get_intelligence_capabilities(self) -> IntelligenceCapabilities:
        """Return the agent's intelligence capabilities."""
        return IntelligenceCapabilities(
            can_explain=True,
            can_recommend=False,
            can_detect_anomalies=False,
            can_reason=False,
            can_validate=True,
            uses_rag=False,
            uses_tools=False
        )

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data."""
        return "site_file" in input_data and input_data.get("site_file")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Load and validate site data with AI-powered insights."""
        try:
            site_file = input_data.get("site_file")
            if not site_file:
                return AgentResult(success=False, error="site_file not provided")

            # =================================================================
            # STEP 1: DETERMINISTIC VALIDATION (Zero-Hallucination)
            # =================================================================
            with open(site_file, "r") as f:
                data = yaml.safe_load(f)

            # Pydantic does the heavy lifting of validation here
            FeasibilityInput = get_feasibility_input()
            validated_data = FeasibilityInput(**data)

            # Convert to dictionary
            result_data = validated_data.dict()

            # =================================================================
            # STEP 2: AI-POWERED EXPLANATION
            # =================================================================
            calculation_steps = [
                f"Loaded YAML from: {site_file}",
                f"Validated against FeasibilityInput schema",
                f"Validated {len(result_data)} fields successfully",
                "All required fields present and valid"
            ]

            explanation = self.generate_explanation(
                input_data={"site_file": site_file},
                output_data={"fields_validated": list(result_data.keys())[:10]},  # First 10 fields
                calculation_steps=calculation_steps
            )
            result_data["explanation"] = explanation

            # =================================================================
            # STEP 3: ADD INTELLIGENCE METADATA
            # =================================================================
            result_data["intelligence_level"] = self.get_intelligence_level().value

            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent": "SiteInputAgent",
                    "fields_validated": len(result_data),
                    "intelligence_metrics": self.get_intelligence_metrics(),
                    "regulatory_context": "ISO 50001, Building Codes"
                },
            )
        except Exception as e:
            return AgentResult(success=False, error=str(e))
