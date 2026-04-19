# -*- coding: utf-8 -*-
"""
EnergyBalanceAgent - AI-Native Hourly Energy Balance Simulation

This agent performs hourly energy balance simulation for solar fields with
FULL LLM INTELLIGENCE capabilities via IntelligenceMixin.

Intelligence Level: ADVANCED
Capabilities: Explanations, Recommendations, Anomaly Detection, Reasoning

Regulatory Context: ISO 50001, IEC 62862
"""

try:
    import pandas as pd
except ImportError:
    raise ImportError(
        "pandas is required for the EnergyBalanceAgent. "
        "Install it with: pip install greenlang[analytics]"
    )

from typing import Dict, Any
from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from greenlang.agents.intelligence_mixin import IntelligenceMixin
from greenlang.agents.intelligence_interface import IntelligenceCapabilities, IntelligenceLevel

# Performance constants for a typical parabolic trough collector
OPTICAL_EFFICIENCY = 0.75  # Includes reflectivity, intercept factor, etc.
THERMAL_LOSS_COEFF_U1 = 0.2  # W/m^2-K
THERMAL_LOSS_COEFF_U2 = 0.003  # W/m^2-K^2


class EnergyBalanceAgent(IntelligenceMixin, BaseAgent):
    """
    AI-Native Hourly Energy Balance Simulation Agent.

    This agent performs solar field energy balance simulation with FULL LLM INTELLIGENCE:

    1. DETERMINISTIC CALCULATIONS (Zero-Hallucination):
       - Incident energy calculation
       - Absorbed energy calculation
       - Thermal loss calculation
       - Solar fraction determination

    2. AI-POWERED INTELLIGENCE:
       - Natural language explanations of simulation results
       - Storage and optimization recommendations
       - Anomaly detection for unusual performance
       - Reasoning about system improvements
    """

    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="EnergyBalanceAgent",
                description="AI-native hourly energy balance simulation with LLM intelligence",
            )
        super().__init__(config)
        # Intelligence auto-initializes on first use

    def get_intelligence_level(self) -> IntelligenceLevel:
        """Return the agent's intelligence level."""
        return IntelligenceLevel.ADVANCED

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

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data."""
        return all([
            input_data.get("solar_resource_df_json"),
            input_data.get("load_profile_df_json"),
            input_data.get("required_aperture_area_m2") is not None
        ])

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Execute energy balance simulation with AI-powered insights."""
        try:
            solar_resource_df_json = input_data.get("solar_resource_df_json")
            load_profile_df_json = input_data.get("load_profile_df_json")
            required_aperture_area_m2 = input_data.get("required_aperture_area_m2")

            if not all([
                solar_resource_df_json,
                load_profile_df_json,
                required_aperture_area_m2 is not None,
            ]):
                return AgentResult(success=False, error="Missing required inputs")

            # =================================================================
            # STEP 1: DETERMINISTIC CALCULATION (Zero-Hallucination)
            # =================================================================
            solar_df = pd.read_json(solar_resource_df_json, orient="split")
            load_df = pd.read_json(load_profile_df_json, orient="split")

            # Merge the two dataframes on their hourly index
            df = solar_df.join(load_df, how="inner")

            # Assume average fluid temperature for thermal loss calculation
            avg_fluid_temp_C = 100  # A reasonable average for this process

            # Calculate hourly performance
            def calculate_hourly_yield(row):
                dni = row["dni_w_per_m2"]
                ambient_temp = row["temp_c"]

                # 1. Calculate incident energy
                incident_energy_w = dni * required_aperture_area_m2

                # 2. Calculate absorbed energy
                absorbed_energy_w = incident_energy_w * OPTICAL_EFFICIENCY

                # 3. Calculate thermal losses
                delta_t = avg_fluid_temp_C - ambient_temp
                if delta_t < 0:
                    delta_t = 0

                thermal_loss_w = (
                    THERMAL_LOSS_COEFF_U1 * delta_t + THERMAL_LOSS_COEFF_U2 * delta_t**2
                ) * required_aperture_area_m2

                # 4. Calculate useful energy generated by the field
                useful_energy_w = absorbed_energy_w - thermal_loss_w
                if useful_energy_w < 0:
                    useful_energy_w = 0

                # Convert from W to kWh
                return useful_energy_w / 1000

            df["solar_yield_kWh"] = df.apply(calculate_hourly_yield, axis=1)

            # The actual heat delivered is the minimum of what's generated and what's demanded
            df["heat_delivered_kWh"] = df[["solar_yield_kWh", "demand_kWh"]].min(axis=1)

            # Calculate the final solar fraction
            total_solar_yield = df["solar_yield_kWh"].sum()
            total_heat_delivered = df["heat_delivered_kWh"].sum()
            total_demand = df["demand_kWh"].sum()

            solar_fraction = (
                total_heat_delivered / total_demand if total_demand > 0 else 0
            )

            # Calculate excess solar
            excess_solar = total_solar_yield - total_heat_delivered

            # Build deterministic result
            result_data = {
                "solar_fraction": round(solar_fraction, 4),
                "performance_df_json": df.to_json(orient="split"),
                "total_solar_yield_gwh": round(total_solar_yield / 1e6, 4),
                "total_heat_delivered_gwh": round(total_heat_delivered / 1e6, 4),
                "total_demand_gwh": round(total_demand / 1e6, 4),
                "excess_solar_gwh": round(excess_solar / 1e6, 4),
            }

            # =================================================================
            # STEP 2: AI-POWERED EXPLANATION
            # =================================================================
            calculation_steps = [
                f"Aperture area: {required_aperture_area_m2:,.0f} m2",
                f"Optical efficiency: {OPTICAL_EFFICIENCY * 100}%",
                f"Thermal loss coefficients: U1={THERMAL_LOSS_COEFF_U1}, U2={THERMAL_LOSS_COEFF_U2}",
                f"Total solar yield: {total_solar_yield/1e6:.4f} GWh",
                f"Total heat delivered: {total_heat_delivered/1e6:.4f} GWh",
                f"Total demand: {total_demand/1e6:.4f} GWh",
                f"Solar fraction achieved: {solar_fraction*100:.1f}%",
                f"Excess solar energy: {excess_solar/1e6:.4f} GWh"
            ]

            explanation = self.generate_explanation(
                input_data={"aperture_area_m2": required_aperture_area_m2},
                output_data=result_data,
                calculation_steps=calculation_steps
            )
            result_data["explanation"] = explanation

            # =================================================================
            # STEP 3: AI-POWERED RECOMMENDATIONS
            # =================================================================
            recommendations = self.generate_recommendations(
                analysis={
                    "solar_fraction": solar_fraction,
                    "excess_solar_gwh": excess_solar / 1e6,
                    "aperture_area_m2": required_aperture_area_m2,
                    "shortfall_gwh": max(0, (total_demand - total_heat_delivered) / 1e6),
                },
                max_recommendations=5,
                focus_areas=["solar_optimization", "thermal_storage", "efficiency", "sizing"]
            )
            result_data["recommendations"] = recommendations

            # =================================================================
            # STEP 4: ANOMALY DETECTION
            # =================================================================
            anomalies = self.detect_anomalies(
                data={
                    "solar_fraction": solar_fraction,
                    "excess_ratio": excess_solar / total_solar_yield if total_solar_yield > 0 else 0,
                },
                expected_ranges={
                    "solar_fraction": (0.30, 0.80),
                    "excess_ratio": (0.0, 0.30),
                }
            )
            result_data["anomalies"] = anomalies

            # =================================================================
            # STEP 5: ADD INTELLIGENCE METADATA
            # =================================================================
            result_data["intelligence_level"] = self.get_intelligence_level().value

            return AgentResult(
                success=True,
                data=result_data,
                metadata={
                    "agent": "EnergyBalanceAgent",
                    "hours_simulated": len(df),
                    "intelligence_metrics": self.get_intelligence_metrics(),
                    "regulatory_context": "ISO 50001, IEC 62862"
                },
            )
        except Exception as e:
            return AgentResult(success=False, error=str(e))
