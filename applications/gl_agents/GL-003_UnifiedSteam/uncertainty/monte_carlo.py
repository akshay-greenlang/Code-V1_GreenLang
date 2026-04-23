"""
Steam-Specific Monte Carlo Functions for GL-003 UNIFIEDSTEAM SteamSystemOptimizer.

This module provides specialized Monte Carlo uncertainty propagation functions
for steam system calculations, including:
- IAPWS-IF97 thermodynamic property uncertainty
- Enthalpy balance uncertainty calculations
- Mass/energy balance closure analysis
- Visualization data generation

Zero-Hallucination Guarantee:
- All calculations are deterministic with seeded random generators
- Complete provenance tracking with SHA-256 hashes
- No LLM inference in any calculation path
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Any
import hashlib
import json
import logging
import math
import time

import numpy as np
from numpy.random import Generator, PCG64

from .uncertainty_models import (
    UncertainValue,
    PropagatedUncertainty,
    MonteCarloResult,
    Distribution,
    DistributionType,
    ConfidenceLevel
)
from .propagation import (
    MonteCarloEngine,
    MonteCarloConfig,
    ConvergenceDiagnostic,
    ExtendedMonteCarloResult,
    CorrelationMatrix
)


logger = logging.getLogger(__name__)


# =============================================================================
# STEAM PROPERTY UNCERTAINTY MODELS
# =============================================================================

@dataclass
class SteamPropertyUncertainty:
    """
    Uncertainty result for a steam thermodynamic property.

    Attributes:
        property_name: Name of the property (e.g., enthalpy, entropy)
        value: Computed property value
        uncertainty: Absolute uncertainty (1-sigma)
        uncertainty_percent: Relative uncertainty (%)
        confidence_interval: 95% confidence interval (lower, upper)
        input_contributions: Contribution of each input to uncertainty
        dominant_input: Input contributing most to uncertainty
        iapws_region: IAPWS-IF97 region where calculation was performed
        provenance_hash: SHA-256 hash for audit trail
    """
    property_name: str
    value: float
    uncertainty: float
    uncertainty_percent: float
    confidence_interval: Tuple[float, float]
    input_contributions: Dict[str, float]
    dominant_input: str
    iapws_region: int
    computation_time_ms: float = 0.0
    provenance_hash: str = ""

    def __post_init__(self):
        """Compute provenance hash if not provided."""
        if not self.provenance_hash:
            hash_data = {
                "property_name": self.property_name,
                "value": self.value,
                "uncertainty": self.uncertainty,
                "iapws_region": self.iapws_region,
                "dominant_input": self.dominant_input
            }
            hash_str = json.dumps(hash_data, sort_keys=True)
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()


@dataclass
class EnthalpyBalanceResult:
    """
    Result of enthalpy balance uncertainty analysis.

    Attributes:
        inlet_enthalpy: Total inlet enthalpy with uncertainty
        outlet_enthalpy: Total outlet enthalpy with uncertainty
        heat_input: Heat input with uncertainty
        imbalance: Energy imbalance (should be ~0)
        imbalance_percent: Imbalance as percentage of throughput
        closure_achieved: Whether balance closes within uncertainty
        uncertainty_budget: Breakdown of uncertainty contributions
        provenance_hash: SHA-256 hash for audit trail
    """
    inlet_enthalpy: UncertainValue
    outlet_enthalpy: UncertainValue
    heat_input: UncertainValue
    imbalance: UncertainValue
    imbalance_percent: float
    closure_achieved: bool
    uncertainty_budget: Dict[str, float]
    monte_carlo_samples: Optional[np.ndarray] = None
    computation_time_ms: float = 0.0
    provenance_hash: str = ""

    def __post_init__(self):
        """Compute provenance hash if not provided."""
        if not self.provenance_hash:
            hash_data = {
                "inlet_enthalpy": self.inlet_enthalpy.mean,
                "outlet_enthalpy": self.outlet_enthalpy.mean,
                "imbalance": self.imbalance.mean,
                "closure_achieved": self.closure_achieved
            }
            hash_str = json.dumps(hash_data, sort_keys=True)
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()


@dataclass
class VisualizationData:
    """
    Data for uncertainty visualization.

    Attributes:
        histogram: Histogram data for distribution plot
        scatter_input_output: Scatter data for input-output correlation
        sensitivity_tornado: Data for tornado chart
        cumulative_distribution: CDF data
        percentile_markers: Key percentile values
    """
    histogram: Dict[str, Any]
    scatter_input_output: Optional[Dict[str, Any]] = None
    sensitivity_tornado: Optional[Dict[str, Any]] = None
    cumulative_distribution: Optional[Dict[str, Any]] = None
    percentile_markers: Optional[Dict[float, float]] = None


# =============================================================================
# STEAM-SPECIFIC MONTE CARLO ENGINE
# =============================================================================

class SteamMonteCarloEngine:
    """
    Specialized Monte Carlo engine for steam thermodynamic calculations.

    Provides uncertainty propagation through IAPWS-IF97 steam property
    calculations with proper handling of thermodynamic correlations
    and region transitions.

    Features:
    - IAPWS-IF97 property uncertainty propagation
    - Enthalpy balance analysis
    - Mass balance closure checking
    - Visualization data generation
    - Complete audit trail

    Example:
        engine = SteamMonteCarloEngine(seed=42)

        # Propagate uncertainty through enthalpy calculation
        result = engine.propagate_through_iapws(
            temperature=UncertainValue.from_measurement(450.0, 1.0, "C"),
            pressure=UncertainValue.from_measurement(10.0, 0.05, "MPa"),
            property_name="enthalpy"
        )

        print(f"Enthalpy: {result.value:.1f} +/- {result.uncertainty:.1f} kJ/kg")
    """

    def __init__(
        self,
        seed: int = 42,
        n_samples: int = 10000,
        use_lhs: bool = True
    ):
        """
        Initialize Steam Monte Carlo engine.

        Args:
            seed: Random seed for reproducibility
            n_samples: Number of Monte Carlo samples
            use_lhs: Use Latin Hypercube Sampling
        """
        self.config = MonteCarloConfig(
            seed=seed,
            n_samples=n_samples,
            use_lhs=use_lhs,
            cache_samples=True
        )
        self.engine = MonteCarloEngine(self.config)

    def propagate_through_iapws(
        self,
        temperature: UncertainValue,
        pressure: UncertainValue,
        property_name: str,
        correlation: Optional[float] = None,
        iapws_function: Optional[Callable] = None
    ) -> SteamPropertyUncertainty:
        """
        Propagate uncertainty through IAPWS-IF97 steam property calculation.

        Args:
            temperature: Temperature with uncertainty (C or K)
            pressure: Pressure with uncertainty (MPa)
            property_name: Property to compute (enthalpy, entropy, density, etc.)
            correlation: Correlation coefficient between T and P measurements
            iapws_function: Custom IAPWS function (default: use iapws module)

        Returns:
            SteamPropertyUncertainty with complete uncertainty analysis
        """
        start_time = time.perf_counter()

        # Create distributions from uncertain values
        inputs = {
            "temperature": Distribution.normal(temperature.mean, temperature.std),
            "pressure": Distribution.normal(pressure.mean, pressure.std)
        }

        # Create correlation matrix if specified
        corr_matrix = None
        if correlation is not None and correlation != 0.0:
            corr_matrix = CorrelationMatrix(
                variable_names=["temperature", "pressure"],
                matrix=np.array([[1.0, correlation], [correlation, 1.0]])
            )

        # Define IAPWS property function
        if iapws_function is None:
            iapws_function = self._get_iapws_function(property_name)

        def property_function(vals: Dict[str, float]) -> float:
            return iapws_function(vals["temperature"], vals["pressure"])

        # Run Monte Carlo propagation
        mc_result = self.engine.propagate(
            inputs=inputs,
            function=property_function,
            correlation=corr_matrix,
            output_name=property_name,
            adaptive=True
        )

        # Compute input contributions via sensitivity analysis
        mean_values = {
            "temperature": temperature.mean,
            "pressure": pressure.mean
        }

        # Numerical sensitivities
        h = 1e-6
        base_value = property_function(mean_values)

        dF_dT = (
            property_function({"temperature": temperature.mean + h, "pressure": pressure.mean}) -
            property_function({"temperature": temperature.mean - h, "pressure": pressure.mean})
        ) / (2 * h)

        dF_dP = (
            property_function({"temperature": temperature.mean, "pressure": pressure.mean + h}) -
            property_function({"temperature": temperature.mean, "pressure": pressure.mean - h})
        ) / (2 * h)

        # Variance contributions
        var_T_contrib = (dF_dT * temperature.std) ** 2
        var_P_contrib = (dF_dP * pressure.std) ** 2
        total_var = var_T_contrib + var_P_contrib

        if total_var > 0:
            input_contributions = {
                "temperature": (var_T_contrib / total_var) * 100,
                "pressure": (var_P_contrib / total_var) * 100
            }
            dominant_input = max(input_contributions, key=input_contributions.get)
        else:
            input_contributions = {"temperature": 50.0, "pressure": 50.0}
            dominant_input = "temperature"

        # Determine IAPWS region
        iapws_region = self._determine_iapws_region(temperature.mean, pressure.mean)

        computation_time = (time.perf_counter() - start_time) * 1000

        # Compute relative uncertainty
        uncertainty_percent = (mc_result.std / abs(mc_result.mean) * 100) if abs(mc_result.mean) > 1e-10 else 0.0

        return SteamPropertyUncertainty(
            property_name=property_name,
            value=mc_result.mean,
            uncertainty=mc_result.std,
            uncertainty_percent=uncertainty_percent,
            confidence_interval=mc_result.confidence_interval_95,
            input_contributions=input_contributions,
            dominant_input=dominant_input,
            iapws_region=iapws_region,
            computation_time_ms=computation_time
        )

    def enthalpy_balance_uncertainty(
        self,
        inlet_streams: List[Dict[str, UncertainValue]],
        outlet_streams: List[Dict[str, UncertainValue]],
        heat_input: Optional[UncertainValue] = None,
        heat_output: Optional[UncertainValue] = None,
        closure_tolerance_percent: float = 2.0
    ) -> EnthalpyBalanceResult:
        """
        Calculate uncertainty in enthalpy balance for a steam system component.

        Performs Monte Carlo analysis of energy balance:
        sum(m_in * h_in) + Q_in = sum(m_out * h_out) + Q_out

        Args:
            inlet_streams: List of inlet streams, each with mass_flow and enthalpy
            outlet_streams: List of outlet streams, each with mass_flow and enthalpy
            heat_input: Heat input to system (optional)
            heat_output: Heat output from system (optional)
            closure_tolerance_percent: Acceptable imbalance (%)

        Returns:
            EnthalpyBalanceResult with complete uncertainty analysis

        Example:
            result = engine.enthalpy_balance_uncertainty(
                inlet_streams=[
                    {"mass_flow": m1, "enthalpy": h1},
                    {"mass_flow": m2, "enthalpy": h2}
                ],
                outlet_streams=[
                    {"mass_flow": m3, "enthalpy": h3}
                ],
                heat_input=Q_in
            )
        """
        start_time = time.perf_counter()

        # Build input distributions
        inputs = {}
        input_index = 0

        for i, stream in enumerate(inlet_streams):
            m_name = f"inlet_{i}_mass_flow"
            h_name = f"inlet_{i}_enthalpy"
            inputs[m_name] = Distribution.normal(
                stream["mass_flow"].mean,
                stream["mass_flow"].std
            )
            inputs[h_name] = Distribution.normal(
                stream["enthalpy"].mean,
                stream["enthalpy"].std
            )
            input_index += 2

        for i, stream in enumerate(outlet_streams):
            m_name = f"outlet_{i}_mass_flow"
            h_name = f"outlet_{i}_enthalpy"
            inputs[m_name] = Distribution.normal(
                stream["mass_flow"].mean,
                stream["mass_flow"].std
            )
            inputs[h_name] = Distribution.normal(
                stream["enthalpy"].mean,
                stream["enthalpy"].std
            )

        if heat_input is not None:
            inputs["heat_input"] = Distribution.normal(heat_input.mean, heat_input.std)

        if heat_output is not None:
            inputs["heat_output"] = Distribution.normal(heat_output.mean, heat_output.std)

        # Define balance function
        n_inlet = len(inlet_streams)
        n_outlet = len(outlet_streams)

        def balance_function(vals: Dict[str, float]) -> float:
            # Inlet energy
            inlet_energy = sum(
                vals[f"inlet_{i}_mass_flow"] * vals[f"inlet_{i}_enthalpy"]
                for i in range(n_inlet)
            )

            # Add heat input
            if "heat_input" in vals:
                inlet_energy += vals["heat_input"]

            # Outlet energy
            outlet_energy = sum(
                vals[f"outlet_{i}_mass_flow"] * vals[f"outlet_{i}_enthalpy"]
                for i in range(n_outlet)
            )

            # Add heat output
            if "heat_output" in vals:
                outlet_energy += vals["heat_output"]

            # Return imbalance
            return inlet_energy - outlet_energy

        # Run Monte Carlo
        mc_result = self.engine.propagate(
            inputs=inputs,
            function=balance_function,
            output_name="energy_imbalance",
            adaptive=True
        )

        # Compute individual energy flows for reporting
        def inlet_energy_func(vals: Dict[str, float]) -> float:
            energy = sum(
                vals[f"inlet_{i}_mass_flow"] * vals[f"inlet_{i}_enthalpy"]
                for i in range(n_inlet)
            )
            if "heat_input" in vals:
                energy += vals["heat_input"]
            return energy

        def outlet_energy_func(vals: Dict[str, float]) -> float:
            energy = sum(
                vals[f"outlet_{i}_mass_flow"] * vals[f"outlet_{i}_enthalpy"]
                for i in range(n_outlet)
            )
            if "heat_output" in vals:
                energy += vals["heat_output"]
            return energy

        inlet_result = self.engine.propagate(inputs, inlet_energy_func, output_name="inlet_energy")
        outlet_result = self.engine.propagate(inputs, outlet_energy_func, output_name="outlet_energy")

        # Create UncertainValues for results
        inlet_uv = inlet_result.to_uncertain_value()
        outlet_uv = outlet_result.to_uncertain_value()
        imbalance_uv = mc_result.to_uncertain_value()

        # Heat input UncertainValue
        if heat_input is not None:
            heat_uv = heat_input
        else:
            heat_uv = UncertainValue(
                mean=0.0, std=0.0, lower_95=0.0, upper_95=0.0
            )

        # Compute imbalance percentage
        throughput = max(abs(inlet_uv.mean), abs(outlet_uv.mean))
        imbalance_percent = (abs(imbalance_uv.mean) / throughput * 100) if throughput > 0 else 0.0

        # Check closure
        # Balance closes if zero is within the uncertainty bounds
        closure_achieved = (
            abs(imbalance_uv.mean) <= closure_tolerance_percent * throughput / 100 or
            (imbalance_uv.lower_95 <= 0 <= imbalance_uv.upper_95)
        )

        # Compute uncertainty budget (contribution of each stream)
        uncertainty_budget = {}
        total_var = mc_result.std ** 2

        # Estimate contributions from streams
        for i in range(n_inlet):
            m_name = f"inlet_{i}_mass_flow"
            h_name = f"inlet_{i}_enthalpy"
            # Approximate contribution
            stream_var = (
                inlet_streams[i]["mass_flow"].std * inlet_streams[i]["enthalpy"].mean
            ) ** 2 + (
                inlet_streams[i]["mass_flow"].mean * inlet_streams[i]["enthalpy"].std
            ) ** 2

            budget_key = f"inlet_stream_{i}"
            uncertainty_budget[budget_key] = (stream_var / total_var * 100) if total_var > 0 else 0.0

        for i in range(n_outlet):
            stream_var = (
                outlet_streams[i]["mass_flow"].std * outlet_streams[i]["enthalpy"].mean
            ) ** 2 + (
                outlet_streams[i]["mass_flow"].mean * outlet_streams[i]["enthalpy"].std
            ) ** 2

            budget_key = f"outlet_stream_{i}"
            uncertainty_budget[budget_key] = (stream_var / total_var * 100) if total_var > 0 else 0.0

        computation_time = (time.perf_counter() - start_time) * 1000

        return EnthalpyBalanceResult(
            inlet_enthalpy=inlet_uv,
            outlet_enthalpy=outlet_uv,
            heat_input=heat_uv,
            imbalance=imbalance_uv,
            imbalance_percent=imbalance_percent,
            closure_achieved=closure_achieved,
            uncertainty_budget=uncertainty_budget,
            monte_carlo_samples=mc_result.samples,
            computation_time_ms=computation_time
        )

    def generate_visualization_data(
        self,
        mc_result: ExtendedMonteCarloResult,
        input_samples: Optional[Dict[str, np.ndarray]] = None,
        n_bins: int = 50
    ) -> VisualizationData:
        """
        Generate visualization data for uncertainty analysis results.

        Creates data structures suitable for plotting:
        - Histogram of output distribution
        - Scatter plots for input-output correlation
        - Tornado chart data for sensitivity
        - Cumulative distribution function

        Args:
            mc_result: Monte Carlo result to visualize
            input_samples: Optional input sample arrays for scatter plots
            n_bins: Number of histogram bins

        Returns:
            VisualizationData with all visualization datasets
        """
        # Histogram data
        if mc_result.samples is not None:
            histogram_data = self.engine.compute_histogram_data(mc_result.samples, n_bins)
        else:
            histogram_data = {
                "counts": [],
                "bin_edges": [],
                "bin_centers": [],
                "n_samples": mc_result.n_samples,
                "mean": mc_result.mean,
                "std": mc_result.std
            }

        # Scatter data for input-output correlation
        scatter_data = None
        if input_samples is not None and mc_result.samples is not None:
            scatter_data = {}
            for input_name, input_arr in input_samples.items():
                scatter_data[input_name] = self.engine.compute_scatter_data(
                    input_arr,
                    mc_result.samples,
                    max_points=1000
                )

        # Sensitivity tornado chart data
        tornado_data = None
        if mc_result.sensitivity_indices is not None:
            sorted_inputs = sorted(
                mc_result.sensitivity_indices.items(),
                key=lambda x: x[1],
                reverse=True
            )
            tornado_data = {
                "inputs": [x[0] for x in sorted_inputs],
                "sensitivities": [x[1] for x in sorted_inputs]
            }

        # Cumulative distribution function
        cdf_data = None
        if mc_result.samples is not None:
            sorted_samples = np.sort(mc_result.samples)
            n = len(sorted_samples)
            cdf_data = {
                "x": sorted_samples.tolist(),
                "y": (np.arange(1, n + 1) / n).tolist()
            }

        # Percentile markers
        percentile_markers = mc_result.percentiles

        return VisualizationData(
            histogram=histogram_data,
            scatter_input_output=scatter_data,
            sensitivity_tornado=tornado_data,
            cumulative_distribution=cdf_data,
            percentile_markers=percentile_markers
        )

    def _get_iapws_function(self, property_name: str) -> Callable:
        """
        Get IAPWS-IF97 property function.

        Args:
            property_name: Name of property to compute

        Returns:
            Function that computes property from T, P
        """
        # Try to import iapws
        try:
            from iapws import IAPWS97
        except ImportError:
            logger.warning("iapws module not available, using simplified steam functions")
            return self._get_simplified_steam_function(property_name)

        def iapws_property(temperature_c: float, pressure_mpa: float) -> float:
            """Compute IAPWS-IF97 property."""
            # Convert to K if needed (assuming input is Celsius)
            temperature_k = temperature_c + 273.15

            # Create IAPWS97 object
            steam = IAPWS97(T=temperature_k, P=pressure_mpa)

            # Get requested property
            if property_name == "enthalpy" or property_name == "h":
                return steam.h  # kJ/kg
            elif property_name == "entropy" or property_name == "s":
                return steam.s  # kJ/(kg*K)
            elif property_name == "density" or property_name == "rho":
                return steam.rho  # kg/m3
            elif property_name == "specific_volume" or property_name == "v":
                return steam.v  # m3/kg
            elif property_name == "internal_energy" or property_name == "u":
                return steam.u  # kJ/kg
            elif property_name == "cp":
                return steam.cp  # kJ/(kg*K)
            elif property_name == "cv":
                return steam.cv  # kJ/(kg*K)
            else:
                raise ValueError(f"Unknown property: {property_name}")

        return iapws_property

    def _get_simplified_steam_function(self, property_name: str) -> Callable:
        """
        Get simplified steam property function when iapws not available.

        Uses polynomial approximations valid for superheated steam.
        """
        def simplified_enthalpy(temperature_c: float, pressure_mpa: float) -> float:
            """Simplified enthalpy calculation for superheated steam."""
            # Approximate: h = 2500 + 2.0*T + 50*ln(P/0.1)
            # Valid for superheated steam roughly 150-550C, 0.1-20 MPa
            h = 2500 + 2.0 * temperature_c + 50 * math.log(max(pressure_mpa, 0.001) / 0.1)
            return h

        def simplified_entropy(temperature_c: float, pressure_mpa: float) -> float:
            """Simplified entropy calculation for superheated steam."""
            # Approximate: s = 6.5 + 0.005*T - 0.3*ln(P/0.1)
            s = 6.5 + 0.005 * temperature_c - 0.3 * math.log(max(pressure_mpa, 0.001) / 0.1)
            return s

        def simplified_density(temperature_c: float, pressure_mpa: float) -> float:
            """Simplified density calculation for superheated steam."""
            # Ideal gas approximation: rho = P*M/(R*T)
            # R_specific = 461.5 J/(kg*K) for water
            temperature_k = temperature_c + 273.15
            rho = (pressure_mpa * 1e6) / (461.5 * temperature_k)
            return rho

        functions = {
            "enthalpy": simplified_enthalpy,
            "h": simplified_enthalpy,
            "entropy": simplified_entropy,
            "s": simplified_entropy,
            "density": simplified_density,
            "rho": simplified_density
        }

        if property_name in functions:
            return functions[property_name]
        else:
            raise ValueError(f"Simplified function not available for: {property_name}")

    def _determine_iapws_region(self, temperature_c: float, pressure_mpa: float) -> int:
        """
        Determine IAPWS-IF97 region for given T, P.

        Regions:
        1: Subcooled liquid
        2: Superheated vapor
        3: Supercritical
        4: Two-phase (saturation)
        5: High-temperature steam (>800C)

        Args:
            temperature_c: Temperature in Celsius
            pressure_mpa: Pressure in MPa

        Returns:
            IAPWS-IF97 region number (1-5)
        """
        T_K = temperature_c + 273.15

        # Critical point: Tc = 647.096 K (373.946 C), Pc = 22.064 MPa
        Tc_K = 647.096
        Pc_MPa = 22.064

        # Simplified region determination
        if pressure_mpa > Pc_MPa:
            if T_K > Tc_K:
                return 3  # Supercritical
            else:
                return 1  # Compressed liquid

        # Estimate saturation temperature at pressure (simplified)
        # Tsat = 373.15 * (P/0.101325)^0.25 approximately
        Tsat_K = 373.15 * (pressure_mpa / 0.101325) ** 0.25

        if T_K < Tsat_K - 5:
            return 1  # Subcooled liquid
        elif T_K > Tsat_K + 5:
            if T_K > 1073.15:  # 800 C
                return 5  # High temperature
            else:
                return 2  # Superheated vapor
        else:
            return 4  # Two-phase region


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def propagate_steam_property(
    temperature: UncertainValue,
    pressure: UncertainValue,
    property_name: str,
    seed: int = 42,
    n_samples: int = 10000
) -> SteamPropertyUncertainty:
    """
    Convenience function to propagate uncertainty through steam property calculation.

    Args:
        temperature: Temperature with uncertainty (C)
        pressure: Pressure with uncertainty (MPa)
        property_name: Property to compute
        seed: Random seed
        n_samples: Number of Monte Carlo samples

    Returns:
        SteamPropertyUncertainty result
    """
    engine = SteamMonteCarloEngine(seed=seed, n_samples=n_samples)
    return engine.propagate_through_iapws(temperature, pressure, property_name)


def analyze_enthalpy_balance(
    inlet_streams: List[Dict[str, UncertainValue]],
    outlet_streams: List[Dict[str, UncertainValue]],
    heat_input: Optional[UncertainValue] = None,
    seed: int = 42,
    n_samples: int = 10000
) -> EnthalpyBalanceResult:
    """
    Convenience function for enthalpy balance uncertainty analysis.

    Args:
        inlet_streams: List of inlet streams
        outlet_streams: List of outlet streams
        heat_input: Optional heat input
        seed: Random seed
        n_samples: Number of Monte Carlo samples

    Returns:
        EnthalpyBalanceResult
    """
    engine = SteamMonteCarloEngine(seed=seed, n_samples=n_samples)
    return engine.enthalpy_balance_uncertainty(
        inlet_streams, outlet_streams, heat_input
    )
