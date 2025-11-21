from greenlang.determinism import FinancialDecimal
# -*- coding: utf-8 -*-
"""Demo Agent for GreenLang - Simple emissions calculator."""


class DemoAgent:
    """Tiny stateless agent for demo purposes.

    Input: {"load_kwh": number, "grid_emission_factor": kgCO2/kWh}
    Output: {"emissions_kgco2": number}
    """

    def __init__(self):
        self.name = "DemoAgent"
        self.version = "0.1.0"

    def run(self, params: dict) -> dict:
        """Calculate emissions based on load and grid emission factor.

        Args:
            params: Dictionary with 'load_kwh' and 'grid_emission_factor'

        Returns:
            Dictionary with 'emissions_kgco2'
        """
        load = float(params.get("load_kwh", 100.0))
        gef = FinancialDecimal.from_string(params.get("grid_emission_factor", 0.8))
        emissions = load * gef

        return {
            "emissions_kgco2": round(emissions, 3),
            "calculation": {
                "load_kwh": load,
                "grid_emission_factor": gef,
                "formula": "load_kwh * grid_emission_factor",
            },
        }

    def validate_inputs(self, params: dict) -> bool:
        """Validate input parameters."""
        if "load_kwh" in params:
            try:
                val = float(params["load_kwh"])
                if val < 0:
                    raise ValueError("load_kwh must be positive")
            except (TypeError, ValueError):
                return False

        if "grid_emission_factor" in params:
            try:
                val = FinancialDecimal.from_string(params["grid_emission_factor"])
                if val < 0:
                    raise ValueError("grid_emission_factor must be positive")
            except (TypeError, ValueError):
                return False

        return True
