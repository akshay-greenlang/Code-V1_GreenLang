"""
Process Heat Causal Inference Examples

This module demonstrates practical applications of causal inference
for Process Heat agents in GreenLang. It includes real-world scenarios
with what-if analysis, counterfactual predictions, and sensitivity analysis.

Example scenarios:
- Analyzing excess air impact on boiler efficiency
- Maintenance frequency effects on equipment failure
- Load changes driving emissions
- Fouling effects on heat transfer
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

from greenlang.ml.explainability.causal_inference import (
    CausalInference,
    CausalInferenceConfig,
    ProcessHeatCausalModels,
    IdentificationMethod,
    EstimationMethod
)

logger = logging.getLogger(__name__)


class ProcessHeatCausalAnalyzer:
    """
    Comprehensive causal analyzer for Process Heat applications.

    Provides methods for:
    - Causal effect estimation
    - Counterfactual "what-if" analysis
    - Sensitivity analysis
    - Confounding diagnosis
    """

    def __init__(self, data: pd.DataFrame):
        """Initialize analyzer with historical data."""
        self.data = data.copy()
        logger.info(f"ProcessHeatCausalAnalyzer initialized with {len(data)} records")

    def analyze_excess_air_efficiency(self) -> Dict[str, Any]:
        """
        Analyze how excess air ratio affects boiler efficiency.

        Returns:
            Dictionary with causal analysis results

        What-if scenarios:
        - Reducing excess air from 25% to 15%
        - Impact on thermal efficiency
        - Cost savings potential
        """
        logger.info("Starting excess air efficiency analysis")

        # Create causal model
        model = ProcessHeatCausalModels.excess_air_efficiency_model(self.data)

        # Estimate causal effect
        result = model.estimate_causal_effect()

        logger.info(f"ATE: {result.average_treatment_effect:.4f} (CI: {result.confidence_interval})")

        # What-if analysis: reduce excess air from 25% to 15%
        current_state = {
            "excess_air_ratio": 0.25,
            "efficiency": 0.78,
            "fuel_type": "gas",
            "burner_age": 5,
            "ambient_temp": 15
        }

        target_o2 = 0.15
        counterfactual = model.estimate_counterfactual(current_state, target_o2)

        efficiency_gain = counterfactual.counterfactual_outcome - current_state["efficiency"]

        return {
            "causal_effect": {
                "ate": result.average_treatment_effect,
                "confidence_interval": result.confidence_interval,
                "is_robust": result.is_robust,
                "refutation_results": result.refutation_results
            },
            "what_if_scenario": {
                "current_excess_air": current_state["excess_air_ratio"],
                "target_excess_air": target_o2,
                "current_efficiency": current_state["efficiency"],
                "predicted_efficiency": counterfactual.counterfactual_outcome,
                "efficiency_gain": efficiency_gain,
                "efficiency_gain_pct": (efficiency_gain / current_state["efficiency"]) * 100
            },
            "confounders": model.get_confounders(),
            "causal_graph": model.get_causal_graph()
        }

    def analyze_maintenance_failure(self) -> Dict[str, Any]:
        """
        Analyze how maintenance frequency affects failure probability.

        Returns:
            Dictionary with maintenance impact analysis

        Scenarios:
        - Doubling maintenance visits (3 -> 6 per year)
        - Expected reduction in failure probability
        - ROI on maintenance investment
        """
        logger.info("Starting maintenance failure analysis")

        model = ProcessHeatCausalModels.maintenance_failure_model(self.data)
        result = model.estimate_causal_effect()

        logger.info(f"Maintenance effect: {result.average_treatment_effect:.4f}")

        # What-if: increase maintenance from 3 to 6 times per year
        current_state = {
            "maintenance_frequency": 3,
            "failure_probability": 0.08,
            "equipment_age": 10,
            "utilization": 0.75,
            "maintenance_cost": 5000
        }

        target_maintenance = 6
        counterfactual = model.estimate_counterfactual(current_state, target_maintenance)

        failure_reduction = current_state["failure_probability"] - counterfactual.counterfactual_outcome

        return {
            "causal_effect": {
                "ate": result.average_treatment_effect,
                "confidence_interval": result.confidence_interval,
                "is_robust": result.is_robust
            },
            "what_if_scenario": {
                "current_maintenance_visits": current_state["maintenance_frequency"],
                "target_maintenance_visits": target_maintenance,
                "current_failure_prob": current_state["failure_probability"],
                "predicted_failure_prob": counterfactual.counterfactual_outcome,
                "failure_probability_reduction": failure_reduction,
                "failure_reduction_pct": (failure_reduction / current_state["failure_probability"]) * 100,
                "potential_downtime_avoided_hours": failure_reduction * 24 * 365  # Hypothetical
            },
            "confounders": model.get_confounders()
        }

    def analyze_load_emissions(self) -> Dict[str, Any]:
        """
        Analyze relationship between steam load changes and emissions.

        Returns:
            Dictionary with load-emissions analysis

        Use cases:
        - Demand elasticity of emissions
        - Load shifting impact on carbon footprint
        - Optimal load point identification
        """
        logger.info("Starting load-emissions analysis")

        model = ProcessHeatCausalModels.load_changes_emissions_model(self.data)
        result = model.estimate_causal_effect()

        logger.info(f"Load-emissions ATE: {result.average_treatment_effect:.4f}")

        # What-if: reduce steam load by 20% during peak hours
        current_state = {
            "steam_load_change": 0,
            "co2_emissions": 250,  # kg/h
            "demand_pattern": "peak",
            "weather_temp": 15,
            "fuel_type": "gas",
            "boiler_efficiency": 0.82
        }

        load_reduction = -20  # -20% load
        counterfactual = model.estimate_counterfactual(current_state, load_reduction)

        emission_reduction = current_state["co2_emissions"] - counterfactual.counterfactual_outcome
        annual_savings = emission_reduction * 24 * 365 / 1000  # Convert to tonnes

        return {
            "causal_effect": {
                "ate": result.average_treatment_effect,
                "confidence_interval": result.confidence_interval,
                "unit": "kg CO2 per unit load change"
            },
            "what_if_scenario": {
                "current_load_change": current_state["steam_load_change"],
                "target_load_change": load_reduction,
                "current_emissions": current_state["co2_emissions"],
                "predicted_emissions": counterfactual.counterfactual_outcome,
                "emission_reduction_kg_per_h": emission_reduction,
                "annual_emission_savings_tonnes": annual_savings
            },
            "confounders": model.get_confounders()
        }

    def analyze_fouling_degradation(self) -> Dict[str, Any]:
        """
        Analyze fouling impact on heat transfer coefficient.

        Returns:
            Dictionary with fouling degradation analysis

        Key insights:
        - Heat transfer degradation per mm of fouling
        - Cleaning schedule optimization
        - Energy penalty quantification
        """
        logger.info("Starting fouling degradation analysis")

        model = ProcessHeatCausalModels.fouling_heat_transfer_model(self.data)
        result = model.estimate_causal_effect()

        logger.info(f"Fouling effect on U: {result.average_treatment_effect:.2f} W/m2K per mm")

        # What-if: clean heat exchanger to remove 2mm of fouling
        current_state = {
            "fouling_mm": 2.5,
            "heat_transfer_coef": 950,  # W/m2K
            "water_quality": "fair",
            "operating_hours": 25000,
            "temperature": 85,
            "fluid_velocity": 1.5
        }

        clean_fouling = 0.5  # After cleaning
        counterfactual = model.estimate_counterfactual(current_state, clean_fouling)

        u_improvement = counterfactual.counterfactual_outcome - current_state["heat_transfer_coef"]
        u_improvement_pct = (u_improvement / current_state["heat_transfer_coef"]) * 100

        return {
            "causal_effect": {
                "ate": result.average_treatment_effect,
                "confidence_interval": result.confidence_interval,
                "unit": "W/m2K per mm fouling"
            },
            "what_if_scenario": {
                "current_fouling_mm": current_state["fouling_mm"],
                "fouling_after_cleaning_mm": clean_fouling,
                "current_heat_transfer_coef": current_state["heat_transfer_coef"],
                "predicted_heat_transfer_coef": counterfactual.counterfactual_outcome,
                "u_improvement": u_improvement,
                "u_improvement_pct": u_improvement_pct,
                "fouling_removed_mm": current_state["fouling_mm"] - clean_fouling
            },
            "confounders": model.get_confounders()
        }

    def generate_sensitivity_analysis(
        self,
        model_type: str = "excess_air_efficiency",
        variable_range: List[float] = None
    ) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on causal estimates.

        Args:
            model_type: Type of causal model to analyze
            variable_range: Range of treatment values to explore

        Returns:
            Sensitivity analysis results with effect curves
        """
        logger.info(f"Starting sensitivity analysis for {model_type}")

        if variable_range is None:
            variable_range = [x * 0.1 for x in range(1, 11)]

        # Select model factory
        model_factories = {
            "excess_air_efficiency": ProcessHeatCausalModels.excess_air_efficiency_model,
            "maintenance_failure": ProcessHeatCausalModels.maintenance_failure_model,
            "load_emissions": ProcessHeatCausalModels.load_changes_emissions_model,
            "fouling_heat_transfer": ProcessHeatCausalModels.fouling_heat_transfer_model
        }

        if model_type not in model_factories:
            raise ValueError(f"Unknown model type: {model_type}")

        model = model_factories[model_type](self.data)
        result = model.estimate_causal_effect()

        # Generate sensitivity curves
        sensitivity_curve = []
        for treatment_val in variable_range:
            # Use base instance depending on model type
            if model_type == "excess_air_efficiency":
                instance = {
                    "excess_air_ratio": 0.2,
                    "efficiency": 0.8,
                    "fuel_type": "gas",
                    "burner_age": 5,
                    "ambient_temp": 15
                }
            elif model_type == "maintenance_failure":
                instance = {
                    "maintenance_frequency": 3,
                    "failure_probability": 0.08,
                    "equipment_age": 10,
                    "utilization": 0.75,
                    "maintenance_cost": 5000
                }
            else:
                instance = {}

            try:
                cf = model.estimate_counterfactual(instance, treatment_val)
                sensitivity_curve.append({
                    "treatment_value": treatment_val,
                    "predicted_outcome": cf.counterfactual_outcome,
                    "ite": cf.individual_treatment_effect
                })
            except Exception as e:
                logger.warning(f"Sensitivity analysis failed at {treatment_val}: {e}")

        return {
            "model_type": model_type,
            "primary_ate": result.average_treatment_effect,
            "sensitivity_curve": sensitivity_curve,
            "identification_method": result.identification_method,
            "estimation_method": result.estimation_method
        }


def generate_synthetic_process_heat_data(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate synthetic process heat data for demonstrations.

    Args:
        n_samples: Number of records to generate

    Returns:
        DataFrame with realistic process heat variables
    """
    np.random.seed(42)

    # Excess air and efficiency relationship
    excess_air = np.random.uniform(0.15, 0.35, n_samples)
    burner_age = np.random.uniform(1, 15, n_samples)
    fuel_type_vals = np.random.choice(["gas", "oil", "coal"], n_samples, p=[0.5, 0.3, 0.2])
    ambient_temp = np.random.uniform(10, 30, n_samples)

    # Efficiency model: decreases with excess air, burner age
    efficiency = (
        0.90 -
        (excess_air - 0.15) * 0.3 -
        burner_age * 0.01 +
        (fuel_type_vals == "gas").astype(float) * 0.05 -
        (ambient_temp - 20) * 0.002 +
        np.random.normal(0, 0.02, n_samples)
    )
    efficiency = np.clip(efficiency, 0.70, 0.92)

    # Maintenance and failure relationship
    maintenance_freq = np.random.uniform(1, 8, n_samples)
    equipment_age = np.random.uniform(1, 20, n_samples)
    utilization = np.random.uniform(0.5, 0.95, n_samples)
    maintenance_cost = maintenance_freq * 2000

    # Failure probability decreases with maintenance
    failure_prob = (
        0.15 +
        (equipment_age * 0.005) +
        (utilization - 0.5) * 0.1 -
        maintenance_freq * 0.02 +
        np.random.normal(0, 0.02, n_samples)
    )
    failure_prob = np.clip(failure_prob, 0.01, 0.25)

    # Load and emissions relationship
    steam_load_change = np.random.uniform(-30, 30, n_samples)
    demand_pattern = np.random.choice(["peak", "off-peak", "medium"], n_samples, p=[0.4, 0.3, 0.3])
    weather_temp = np.random.uniform(10, 35, n_samples)
    boiler_efficiency = np.random.uniform(0.78, 0.88, n_samples)

    # Emissions increase with load, decrease with efficiency
    co2_emissions = (
        100 +
        steam_load_change * 1.5 +
        (demand_pattern == "peak").astype(float) * 30 -
        boiler_efficiency * 100 +
        np.random.normal(0, 5, n_samples)
    )
    co2_emissions = np.clip(co2_emissions, 50, 300)

    # Fouling and heat transfer relationship
    fouling_mm = np.random.uniform(0.5, 3.5, n_samples)
    water_quality = np.random.choice(["good", "fair", "poor"], n_samples, p=[0.4, 0.4, 0.2])
    operating_hours = np.random.uniform(1000, 50000, n_samples)
    temperature = np.random.uniform(75, 100, n_samples)
    fluid_velocity = np.random.uniform(1.0, 2.5, n_samples)

    # Heat transfer coefficient decreases with fouling
    heat_transfer_coef = (
        1800 -
        fouling_mm * 200 -
        (operating_hours / 10000) * 50 +
        (water_quality == "good").astype(float) * 100 -
        (water_quality == "poor").astype(float) * 150 +
        temperature * 2 +
        fluid_velocity * 100 +
        np.random.normal(0, 30, n_samples)
    )
    heat_transfer_coef = np.clip(heat_transfer_coef, 700, 1800)

    return pd.DataFrame({
        # Excess air - efficiency
        "excess_air_ratio": excess_air,
        "efficiency": efficiency,
        "fuel_type": fuel_type_vals,
        "burner_age": burner_age,
        "ambient_temp": ambient_temp,
        # Maintenance - failure
        "maintenance_frequency": maintenance_freq,
        "failure_probability": failure_prob,
        "equipment_age": equipment_age,
        "utilization": utilization,
        "maintenance_cost": maintenance_cost,
        # Load - emissions
        "steam_load_change": steam_load_change,
        "co2_emissions": co2_emissions,
        "demand_pattern": demand_pattern,
        "weather_temp": weather_temp,
        "boiler_efficiency": boiler_efficiency,
        # Fouling - heat transfer
        "fouling_mm": fouling_mm,
        "heat_transfer_coef": heat_transfer_coef,
        "water_quality": water_quality,
        "operating_hours": operating_hours,
        "temperature": temperature,
        "fluid_velocity": fluid_velocity
    })


def run_comprehensive_demo():
    """Run comprehensive demonstration of causal inference for process heat."""
    logger.info("Starting comprehensive process heat causal inference demonstration")

    # Generate synthetic data
    data = generate_synthetic_process_heat_data(n_samples=500)
    logger.info(f"Generated synthetic dataset: {data.shape}")

    # Create analyzer
    analyzer = ProcessHeatCausalAnalyzer(data)

    # Run all analyses
    print("\n" + "="*80)
    print("PROCESS HEAT CAUSAL INFERENCE ANALYSIS")
    print("="*80)

    # 1. Excess air analysis
    print("\n1. EXCESS AIR IMPACT ON EFFICIENCY")
    print("-" * 80)
    excess_air_results = analyzer.analyze_excess_air_efficiency()
    print(f"Causal Effect (ATE): {excess_air_results['causal_effect']['ate']:.4f}")
    print(f"Confidence Interval: {excess_air_results['causal_effect']['confidence_interval']}")
    print(f"Is Robust: {excess_air_results['causal_effect']['is_robust']}")
    print(f"\nWhat-If Scenario: Reduce O2 from {excess_air_results['what_if_scenario']['current_excess_air']:.2%} to {excess_air_results['what_if_scenario']['target_excess_air']:.2%}")
    print(f"Predicted Efficiency Gain: {excess_air_results['what_if_scenario']['efficiency_gain_pct']:.2f}%")
    print(f"Confounders: {excess_air_results['confounders']}")

    # 2. Maintenance analysis
    print("\n2. MAINTENANCE FREQUENCY IMPACT ON FAILURE PROBABILITY")
    print("-" * 80)
    maintenance_results = analyzer.analyze_maintenance_failure()
    print(f"Causal Effect (ATE): {maintenance_results['causal_effect']['ate']:.4f}")
    print(f"Is Robust: {maintenance_results['causal_effect']['is_robust']}")
    print(f"\nWhat-If Scenario: Increase maintenance from {maintenance_results['what_if_scenario']['current_maintenance_visits']} to {maintenance_results['what_if_scenario']['target_maintenance_visits']} visits/year")
    print(f"Predicted Failure Reduction: {maintenance_results['what_if_scenario']['failure_reduction_pct']:.2f}%")

    # 3. Load-emissions analysis
    print("\n3. STEAM LOAD IMPACT ON EMISSIONS")
    print("-" * 80)
    load_results = analyzer.analyze_load_emissions()
    print(f"Causal Effect (ATE): {load_results['causal_effect']['ate']:.4f} kg CO2 per unit load change")
    print(f"Current Emissions: {load_results['what_if_scenario']['current_emissions']:.1f} kg CO2/h")
    print(f"Predicted Emissions (20% load reduction): {load_results['what_if_scenario']['predicted_emissions']:.1f} kg CO2/h")
    print(f"Annual Savings: {load_results['what_if_scenario']['annual_emission_savings_tonnes']:.1f} tonnes CO2")

    # 4. Fouling analysis
    print("\n4. FOULING IMPACT ON HEAT TRANSFER COEFFICIENT")
    print("-" * 80)
    fouling_results = analyzer.analyze_fouling_degradation()
    print(f"Causal Effect (ATE): {fouling_results['causal_effect']['ate']:.2f} W/m2K per mm fouling")
    print(f"Current Fouling: {fouling_results['what_if_scenario']['current_fouling_mm']:.2f} mm")
    print(f"Current U: {fouling_results['what_if_scenario']['current_heat_transfer_coef']:.0f} W/m2K")
    print(f"U Improvement After Cleaning: {fouling_results['what_if_scenario']['u_improvement_pct']:.2f}%")

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    run_comprehensive_demo()
