# greenlang/agents/site_input_agent.py

import yaml
import sys
import os
from typing import Dict, Any
from greenlang.agents.base import BaseAgent, AgentResult


# Dynamic import for climatenza_app
def get_feasibility_input():
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    from climatenza_app.schemas.feasibility import FeasibilityInput

    return FeasibilityInput


class SiteInputAgent(BaseAgent):
    """
    Loads, validates, and normalizes the site feasibility input YAML.
    """

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Args:
            input_data: Dictionary containing 'site_file' path.

        Returns:
            AgentResult with validated and parsed site data.
        """
        try:
            site_file = input_data.get("site_file")
            if not site_file:
                return AgentResult(success=False, error="site_file not provided")

            with open(site_file, "r") as f:
                data = yaml.safe_load(f)

            # Pydantic does the heavy lifting of validation here
            FeasibilityInput = get_feasibility_input()
            validated_data = FeasibilityInput(**data)

            # In a real scenario, you would perform unit conversions here
            # For now, we return the parsed model as a dictionary
            return AgentResult(success=True, data=validated_data.dict())
        except Exception as e:
            return AgentResult(success=False, error=str(e))
