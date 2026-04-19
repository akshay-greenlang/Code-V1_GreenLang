# -*- coding: utf-8 -*-
"""
LoadProfileAgent - AI-Native Thermal Load Profile Generator

This agent generates hourly thermal load profiles from process demand data with
FULL LLM INTELLIGENCE capabilities via IntelligenceMixin.

Intelligence Level: STANDARD
Capabilities: Explanations, Recommendations

Regulatory Context: ISO 50001, ASHRAE
"""

try:
    import pandas as pd
except ImportError:
    raise ImportError(
        "pandas is required for the LoadProfileAgent. "
        "Install it with: pip install greenlang[analytics]"
    )

from typing import Dict, Any
from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from greenlang.agents.intelligence_mixin import IntelligenceMixin
from greenlang.agents.intelligence_interface import IntelligenceCapabilities, IntelligenceLevel

# Constants
SPECIFIC_HEAT_WATER_KJ_KG_C = 4.186  # Specific heat capacity of water in kJ/kg-C
SECONDS_PER_HOUR = 3600


class LoadProfileAgent(IntelligenceMixin, BaseAgent):
    """
    AI-Native Thermal Load Profile Generator Agent.

    This agent generates hourly thermal load profiles with FULL LLM INTELLIGENCE:

    1. DETERMINISTIC CALCULATIONS (Zero-Hallucination):
       - Heat demand calculation from mass flow and temperature delta
       - Hourly profile generation
       - Annual demand totalization

    2. AI-POWERED INTELLIGENCE:
       - Natural language explanations of load profiles
       - Load optimization recommendations
       - Process efficiency suggestions
    """

    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="LoadProfileAgent",
                description="AI-native thermal load profile generator with LLM intelligence",
            )
        super().__init__(config)
        # Intelligence auto-initializes on first use

    def get_intelligence_level(self) -> IntelligenceLevel:
        """Return the agent's intelligence level."""
        return IntelligenceLevel.STANDARD

    def get_intelligence_capabilities(self) -> IntelligenceCapabilities:
        """Return the agent's intelligence capabilities."""
        return IntelligenceCapabilities(
            can_explain=True,
            can_recommend=True,
            can_detect_anomalies=False,
            can_reason=False,
            can_validate=True,
            uses_rag=False,
            uses_tools=False
        )

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data."""
        process_demand = input_data.get("process_demand")
        if not process_demand:
            return False
        return all(k in process_demand for k in ["flow_profile", "temp_in_C", "temp_out_C"])

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Generate thermal load profile with AI-powered insights."""
        try:
            process_demand = input_data.get("process_demand")
            if not process_demand:
                return AgentResult(success=False, error="process_demand not provided")

            # =================================================================
            # STEP 1: DETERMINISTIC CALCULATION (Zero-Hallucination)
            # =================================================================
            flow_profile_path = process_demand["flow_profile"]
            temp_in = process_demand["temp_in_C"]
            temp_out = process_demand["temp_out_C"]

            # Load the flow profile data
            df = pd.read_csv(flow_profile_path, index_col="timestamp", parse_dates=True)

            # Calculate the temperature difference (delta T)
            delta_t = temp_out - temp_in

            # Calculate the energy demand in kWh for each timestamp
            # Formula: Q (kJ) = m (kg/s) * c (kJ/kg-C) * delta_T (C) * 3600 (s/h)
            # Then convert kJ to kWh by dividing by 3600
            # The 3600 terms cancel out, so Q (kWh) = m * c * delta_T
            df["demand_kWh"] = (
                df["flow_kg_s"]
                * SPECIFIC_HEAT_WATER_KJ_KG_C
                * delta_t
                / SECONDS_PER_HOUR
                * SECONDS_PER_HOUR
            )

            total_annual_demand_gwh = df["demand_kWh"].sum() / 1e6
            peak_demand_kw = df["demand_kWh"].max()
            avg_demand_kw = df["demand_kWh"].mean()

            # Build deterministic result
            result_data = {
                "load_profile_df_json": df[["demand_kWh"]].to_json(orient="split"),
                "total_annual_demand_gwh": round(total_annual_demand_gwh, 4),
                "peak_demand_kw": round(peak_demand_kw, 2),
                "avg_demand_kw": round(avg_demand_kw, 2),
                "load_factor": round(avg_demand_kw / peak_demand_kw, 4) if peak_demand_kw > 0 else 0,
            }

            # =================================================================
            # STEP 2: AI-POWERED EXPLANATION
            # =================================================================
            calculation_steps = [
                f"Loaded flow profile from: {flow_profile_path}",
                f"Temperature rise: {temp_out} - {temp_in} = {delta_t} C",
                f"Using specific heat: {SPECIFIC_HEAT_WATER_KJ_KG_C} kJ/kg-C",
                f"Total annual demand: {total_annual_demand_gwh:.4f} GWh",
                f"Peak demand: {peak_demand_kw:.2f} kW",
                f"Average demand: {avg_demand_kw:.2f} kW",
                f"Load factor: {result_data['load_factor']:.2%}"
            ]

            explanation = self.generate_explanation(
                input_data=input_data,
                output_data=result_data,
                calculation_steps=calculation_steps
            )
            result_data["explanation"] = explanation

            # =================================================================
            # STEP 3: AI-POWERED RECOMMENDATIONS
            # =================================================================
            recommendations = self.generate_recommendations(
                analysis={
                    "total_annual_demand_gwh": total_annual_demand_gwh,
                    "delta_t": delta_t,
                    "peak_demand_kw": peak_demand_kw,
                    "load_factor": result_data["load_factor"],
                },
                max_recommendations=5,
                focus_areas=["load_optimization", "peak_shaving", "efficiency", "scheduling"]
            )
            result_data["recommendations"] = recommendations

            # =================================================================
            # STEP 4: ADD INTELLIGENCE METADATA
            # =================================================================
            result_data["intelligence_level"] = self.get_intelligence_level().value

            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent": "LoadProfileAgent",
                    "hours_processed": len(df),
                    "intelligence_metrics": self.get_intelligence_metrics(),
                    "regulatory_context": "ISO 50001, ASHRAE"
                },
            )
        except Exception as e:
            return AgentResult(success=False, error=str(e))
