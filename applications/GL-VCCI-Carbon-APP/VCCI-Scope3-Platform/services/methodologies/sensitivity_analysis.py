# -*- coding: utf-8 -*-
"""
Sensitivity Analysis Engine

This module provides advanced sensitivity analysis methods for carbon emissions
uncertainty quantification, supporting Sobol variance-based decomposition,
Morris elementary effects screening, tornado diagram generation, and
convergence assessment.

All calculations are ZERO-HALLUCINATION deterministic (NumPy/SciPy only).

Key Components:
- SobolAnalyzer: First-order and total-order Sobol sensitivity indices
  via Saltelli's sampling scheme.
- MorrisAnalyzer: Elementary effects screening for parameter importance
  classification (OAT trajectory design).
- TornadoDiagramGenerator: One-way sensitivity sweeps and conversion from
  Sobol / Monte Carlo results.
- ConvergenceAnalyzer: Running-mean convergence diagnostics.

References:
- Saltelli et al. (2010): Variance based sensitivity analysis of model output.
- Morris (1991): Factorial sampling plans for preliminary computational experiments.
- IPCC Guidelines for National Greenhouse Gas Inventories, Volume 1, Chapter 3.
- GHG Protocol: Corporate Value Chain (Scope 3) Standard, Chapter 7.

Version: 1.1.0
Date: 2026-03-01
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Callable, Any, Tuple
from pydantic import BaseModel, Field

from .constants import DistributionType

logger = logging.getLogger(__name__)


# ============================================================================
# PYDANTIC RESULT MODELS
# ============================================================================

class SobolResult(BaseModel):
    """Result of a Sobol variance-based sensitivity analysis."""

    parameters: List[str] = Field(
        ..., description="Parameter names in analysis order"
    )
    first_order_indices: Dict[str, float] = Field(
        ..., description="First-order Sobol indices (Si) per parameter"
    )
    total_order_indices: Dict[str, float] = Field(
        ..., description="Total-order Sobol indices (STi) per parameter"
    )
    interaction_effects: Dict[str, float] = Field(
        ..., description="Interaction effects (STi - Si) per parameter"
    )
    convergence_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Convergence diagnostics for the Sobol estimates",
    )
    computation_time: float = Field(
        default=0.0, ge=0.0, description="Wall-clock time in seconds"
    )
    sample_size: int = Field(
        default=0, ge=0, description="Base sample size N used"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "parameters": ["activity_data", "emission_factor"],
                    "first_order_indices": {"activity_data": 0.45, "emission_factor": 0.50},
                    "total_order_indices": {"activity_data": 0.48, "emission_factor": 0.52},
                    "interaction_effects": {"activity_data": 0.03, "emission_factor": 0.02},
                    "convergence_info": {"sum_first_order": 0.95, "sum_total_order": 1.0},
                    "computation_time": 1.25,
                    "sample_size": 1024,
                }
            ]
        }
    }


class MorrisResult(BaseModel):
    """Result of a Morris elementary effects screening analysis."""

    parameters: List[str] = Field(
        ..., description="Parameter names in analysis order"
    )
    mu_star: Dict[str, float] = Field(
        ..., description="Mean of absolute elementary effects (mu*) per parameter"
    )
    sigma: Dict[str, float] = Field(
        ..., description="Standard deviation of elementary effects per parameter"
    )
    mu_star_conf: Dict[str, float] = Field(
        default_factory=dict,
        description="95% confidence interval half-width for mu* per parameter",
    )
    classification: Dict[str, str] = Field(
        default_factory=dict,
        description="Parameter classification: important / non-important / interactive",
    )
    computation_time: float = Field(
        default=0.0, ge=0.0, description="Wall-clock time in seconds"
    )
    num_trajectories: int = Field(
        default=0, ge=0, description="Number of trajectories used"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "parameters": ["activity_data", "emission_factor"],
                    "mu_star": {"activity_data": 120.5, "emission_factor": 180.3},
                    "sigma": {"activity_data": 30.1, "emission_factor": 25.7},
                    "mu_star_conf": {"activity_data": 15.2, "emission_factor": 12.8},
                    "classification": {
                        "activity_data": "important",
                        "emission_factor": "important",
                    },
                    "computation_time": 0.45,
                    "num_trajectories": 10,
                }
            ]
        }
    }


class TornadoParameter(BaseModel):
    """Sensitivity data for a single parameter in a tornado diagram."""

    name: str = Field(..., description="Parameter name")
    low_value: float = Field(..., description="Output value when parameter is at low end")
    high_value: float = Field(..., description="Output value when parameter is at high end")
    impact: float = Field(
        ..., description="Absolute impact (high_value - low_value)"
    )
    relative_impact: float = Field(
        default=0.0, description="Impact relative to baseline output"
    )


class TornadoData(BaseModel):
    """Complete tornado diagram data set."""

    parameters: List[TornadoParameter] = Field(
        ..., description="Parameters sorted by descending absolute impact"
    )
    baseline_output: float = Field(
        ..., description="Output at baseline (nominal) parameter values"
    )
    computation_time: float = Field(
        default=0.0, ge=0.0, description="Wall-clock time in seconds"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "parameters": [
                        {
                            "name": "emission_factor",
                            "low_value": 2250.0,
                            "high_value": 2750.0,
                            "impact": 500.0,
                            "relative_impact": 0.20,
                        }
                    ],
                    "baseline_output": 2500.0,
                    "computation_time": 0.02,
                }
            ]
        }
    }


class ConvergenceResult(BaseModel):
    """Result of Monte Carlo convergence assessment."""

    is_converged: bool = Field(
        ..., description="Whether the simulation has converged"
    )
    recommended_iterations: int = Field(
        ..., ge=0, description="Recommended number of iterations for convergence"
    )
    running_means: List[float] = Field(
        ..., description="Running mean at each window checkpoint"
    )
    running_stds: List[float] = Field(
        ..., description="Running std dev at each window checkpoint"
    )
    window_sizes: List[int] = Field(
        ..., description="Window sizes evaluated"
    )
    relative_changes: List[float] = Field(
        default_factory=list,
        description="Relative change in mean between consecutive windows",
    )
    convergence_threshold: float = Field(
        default=0.01, description="Threshold used for convergence check"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "is_converged": True,
                    "recommended_iterations": 5000,
                    "running_means": [250.1, 249.8, 250.0],
                    "running_stds": [50.2, 49.9, 50.0],
                    "window_sizes": [1000, 5000, 10000],
                    "relative_changes": [0.0012, 0.0004],
                    "convergence_threshold": 0.01,
                }
            ]
        }
    }


# ============================================================================
# SOBOL ANALYZER
# ============================================================================

class SobolAnalyzer:
    """
    Variance-based global sensitivity analysis using Sobol indices.

    Implements Saltelli's quasi-random sampling scheme to estimate
    first-order (Si) and total-order (STi) sensitivity indices.

    First-order index Si measures the direct contribution of parameter Xi
    to the variance of the output. Total-order index STi measures the
    contribution including all interactions involving Xi.

    Reference:
        Saltelli, A. et al. (2010). "Variance based sensitivity analysis
        of model output. Design and estimator for the total sensitivity
        index." Computer Physics Communications, 181(2), 259-270.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize SobolAnalyzer.

        Args:
            seed: Random seed for reproducibility.
        """
        self._rng = np.random.RandomState(seed)
        logger.info(f"Initialized SobolAnalyzer with seed={seed}")

    # ------------------------------------------------------------------
    # Sample generation helpers
    # ------------------------------------------------------------------

    def _sample_parameter(
        self, param: Dict[str, Any], uniform_samples: np.ndarray
    ) -> np.ndarray:
        """
        Transform uniform [0,1] samples to the target distribution.

        Args:
            param: Parameter descriptor with keys name, distribution,
                   mean, std_dev, and optionally min, max.
            uniform_samples: Array of uniform(0,1) samples.

        Returns:
            Transformed samples in the target distribution.
        """
        dist = param.get("distribution", "normal")
        mean = param["mean"]
        std_dev = param.get("std_dev", 0.0)

        if dist == DistributionType.NORMAL or dist == "normal":
            from scipy.stats import norm
            return norm.ppf(uniform_samples, loc=mean, scale=std_dev)

        if dist == DistributionType.LOGNORMAL or dist == "lognormal":
            if mean <= 0:
                mean = abs(mean) or 1e-10
            cv = std_dev / mean if mean != 0 else 0
            sigma = np.sqrt(np.log(1 + cv ** 2))
            mu = np.log(mean) - 0.5 * sigma ** 2
            from scipy.stats import lognorm
            return lognorm.ppf(uniform_samples, s=sigma, scale=np.exp(mu))

        if dist == DistributionType.UNIFORM or dist == "uniform":
            lo = param.get("min", mean - std_dev * np.sqrt(3))
            hi = param.get("max", mean + std_dev * np.sqrt(3))
            return lo + uniform_samples * (hi - lo)

        if dist == DistributionType.TRIANGULAR or dist == "triangular":
            lo = param.get("min", mean - std_dev * np.sqrt(6))
            hi = param.get("max", mean + std_dev * np.sqrt(6))
            c_norm = (mean - lo) / (hi - lo) if hi != lo else 0.5
            from scipy.stats import triang
            return triang.ppf(uniform_samples, c=c_norm, loc=lo, scale=(hi - lo))

        # Fallback to normal
        from scipy.stats import norm
        return norm.ppf(uniform_samples, loc=mean, scale=std_dev)

    # ------------------------------------------------------------------
    # Saltelli sampling
    # ------------------------------------------------------------------

    def generate_saltelli_samples(
        self, parameters: List[Dict[str, Any]], N: int = 1024
    ) -> Dict[str, np.ndarray]:
        """
        Generate Saltelli quasi-random sample matrices A, B, and AB_i.

        The Saltelli scheme requires N*(2k+2) model evaluations, where k
        is the number of parameters. This method builds two independent
        base matrices A and B of shape (N, k), then constructs k AB
        matrices where AB_i equals A with the i-th column replaced by
        the corresponding column from B.

        Args:
            parameters: List of parameter descriptors. Each dict must
                contain 'name', 'mean', 'std_dev', and optionally
                'distribution' (default 'normal'), 'min', 'max'.
            N: Base sample size (power of 2 recommended).

        Returns:
            Dictionary with keys 'A', 'B', 'AB' (list of arrays),
            and 'param_names'.
        """
        k = len(parameters)
        if k == 0:
            raise ValueError("At least one parameter is required")

        logger.info(
            f"Generating Saltelli samples: N={N}, k={k}, "
            f"total evaluations={N * (2 * k + 2)}"
        )

        # Generate two independent uniform (0,1) base matrices
        A_uniform = self._rng.rand(N, k)
        B_uniform = self._rng.rand(N, k)

        # Transform to target distributions
        A = np.empty_like(A_uniform)
        B = np.empty_like(B_uniform)
        for j, param in enumerate(parameters):
            A[:, j] = self._sample_parameter(param, A_uniform[:, j])
            B[:, j] = self._sample_parameter(param, B_uniform[:, j])

        # Build AB matrices (one per parameter)
        AB_list: List[np.ndarray] = []
        for j in range(k):
            AB_j = A.copy()
            AB_j[:, j] = B[:, j]
            AB_list.append(AB_j)

        param_names = [p["name"] for p in parameters]

        return {
            "A": A,
            "B": B,
            "AB": AB_list,
            "param_names": param_names,
        }

    # ------------------------------------------------------------------
    # Index estimators
    # ------------------------------------------------------------------

    def calculate_first_order(
        self,
        A_results: np.ndarray,
        AB_results: Dict[str, np.ndarray],
        B_results: np.ndarray,
        N: int,
    ) -> Dict[str, float]:
        """
        Estimate first-order Sobol indices using the Jansen (1999) estimator.

        Si = 1 - (1 / (2N)) * sum((B_results - AB_i_results)^2) / Var(Y)

        where Var(Y) is estimated from the full set of A and B evaluations.

        Args:
            A_results: Model outputs from matrix A, shape (N,).
            AB_results: Dict mapping parameter name to model outputs
                from the corresponding AB matrix, each shape (N,).
            B_results: Model outputs from matrix B, shape (N,).
            N: Base sample size.

        Returns:
            Dict mapping parameter name to first-order index.
        """
        # Total variance from combined A and B evaluations
        all_results = np.concatenate([A_results, B_results])
        var_y = float(np.var(all_results, ddof=1))

        if var_y == 0:
            logger.warning("Total variance is zero; all first-order indices set to 0")
            return {name: 0.0 for name in AB_results}

        first_order: Dict[str, float] = {}
        for name, ab_res in AB_results.items():
            # Jansen estimator
            si = 1.0 - float(np.mean((B_results - ab_res) ** 2)) / (2.0 * var_y)
            # Clamp to [0, 1] to handle numerical noise
            si = max(0.0, min(1.0, si))
            first_order[name] = round(si, 6)

        return first_order

    def calculate_total_order(
        self,
        A_results: np.ndarray,
        AB_results: Dict[str, np.ndarray],
        B_results: np.ndarray,
        N: int,
    ) -> Dict[str, float]:
        """
        Estimate total-order Sobol indices using the Jansen (1999) estimator.

        STi = (1 / (2N)) * sum((A_results - AB_i_results)^2) / Var(Y)

        Args:
            A_results: Model outputs from matrix A, shape (N,).
            AB_results: Dict mapping parameter name to model outputs
                from the corresponding AB matrix, each shape (N,).
            B_results: Model outputs from matrix B, shape (N,).
            N: Base sample size.

        Returns:
            Dict mapping parameter name to total-order index.
        """
        all_results = np.concatenate([A_results, B_results])
        var_y = float(np.var(all_results, ddof=1))

        if var_y == 0:
            logger.warning("Total variance is zero; all total-order indices set to 0")
            return {name: 0.0 for name in AB_results}

        total_order: Dict[str, float] = {}
        for name, ab_res in AB_results.items():
            sti = float(np.mean((A_results - ab_res) ** 2)) / (2.0 * var_y)
            sti = max(0.0, min(1.0, sti))
            total_order[name] = round(sti, 6)

        return total_order

    # ------------------------------------------------------------------
    # Full analysis
    # ------------------------------------------------------------------

    def run_analysis(
        self,
        calculation_func: Callable[[Dict[str, float]], float],
        parameters: List[Dict[str, Any]],
        N: int = 1024,
    ) -> SobolResult:
        """
        Run a complete Sobol sensitivity analysis.

        Generates Saltelli samples, evaluates the model, and computes
        first-order and total-order indices with interaction effects.

        Args:
            calculation_func: Deterministic model f(params) -> scalar.
                Accepts a dict {param_name: value}.
            parameters: List of parameter descriptors (see
                generate_saltelli_samples for expected keys).
            N: Base sample size.

        Returns:
            SobolResult with first-order, total-order, and interaction
            effect indices.
        """
        start = time.time()
        param_names = [p["name"] for p in parameters]
        k = len(parameters)

        logger.info(
            f"Running Sobol analysis: {k} parameters, N={N}, "
            f"total evaluations={N * (2 * k + 2)}"
        )

        # Generate sample matrices
        samples = self.generate_saltelli_samples(parameters, N)
        A = samples["A"]
        B = samples["B"]
        AB_list = samples["AB"]

        # Evaluate model on A
        A_results = np.array([
            calculation_func({param_names[j]: A[i, j] for j in range(k)})
            for i in range(N)
        ])

        # Evaluate model on B
        B_results = np.array([
            calculation_func({param_names[j]: B[i, j] for j in range(k)})
            for i in range(N)
        ])

        # Evaluate model on each AB_i
        AB_results: Dict[str, np.ndarray] = {}
        for j, name in enumerate(param_names):
            AB_j = AB_list[j]
            AB_results[name] = np.array([
                calculation_func({param_names[c]: AB_j[i, c] for c in range(k)})
                for i in range(N)
            ])

        # Calculate indices
        first_order = self.calculate_first_order(A_results, AB_results, B_results, N)
        total_order = self.calculate_total_order(A_results, AB_results, B_results, N)

        # Interaction effects = STi - Si
        interaction: Dict[str, float] = {}
        for name in param_names:
            diff = total_order[name] - first_order[name]
            interaction[name] = round(max(0.0, diff), 6)

        # Convergence diagnostics
        sum_si = sum(first_order.values())
        sum_sti = sum(total_order.values())
        convergence_info = {
            "sum_first_order": round(sum_si, 4),
            "sum_total_order": round(sum_sti, 4),
            "note": (
                "sum(Si) should be near 1.0 for additive models; "
                "sum(STi) >= 1.0 always."
            ),
        }

        elapsed = time.time() - start
        logger.info(
            f"Sobol analysis completed in {elapsed:.3f}s: "
            f"sum(Si)={sum_si:.4f}, sum(STi)={sum_sti:.4f}"
        )

        return SobolResult(
            parameters=param_names,
            first_order_indices=first_order,
            total_order_indices=total_order,
            interaction_effects=interaction,
            convergence_info=convergence_info,
            computation_time=round(elapsed, 4),
            sample_size=N,
        )


# ============================================================================
# MORRIS ANALYZER
# ============================================================================

class MorrisAnalyzer:
    """
    Elementary effects (Morris) screening method.

    Generates OAT (One-At-a-Time) trajectories through the parameter space
    and computes the mean absolute elementary effect (mu*) and its standard
    deviation (sigma) for each parameter. These are used to classify
    parameters as important, non-important, or interactive.

    Reference:
        Morris, M.D. (1991). "Factorial sampling plans for preliminary
        computational experiments." Technometrics, 33(2), 161-174.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize MorrisAnalyzer.

        Args:
            seed: Random seed for reproducibility.
        """
        self._rng = np.random.RandomState(seed)
        logger.info(f"Initialized MorrisAnalyzer with seed={seed}")

    def _to_physical(
        self, normalized: float, param: Dict[str, Any]
    ) -> float:
        """
        Map a [0,1] normalised value to the physical parameter space.

        Uses the parameter's mean +/- 3*std_dev range if min/max are not
        provided.

        Args:
            normalized: Value in [0, 1].
            param: Parameter descriptor.

        Returns:
            Physical-space value.
        """
        lo = param.get("min", param["mean"] - 3 * param.get("std_dev", 0))
        hi = param.get("max", param["mean"] + 3 * param.get("std_dev", 0))
        return lo + normalized * (hi - lo)

    def generate_trajectories(
        self,
        parameters: List[Dict[str, Any]],
        r: int = 10,
        levels: int = 4,
    ) -> List[np.ndarray]:
        """
        Generate r OAT trajectories through k-dimensional unit hypercube.

        Each trajectory consists of k+1 rows where each successive row
        differs from the previous in exactly one coordinate by a step
        of delta = levels / (2 * (levels - 1)).

        Args:
            parameters: List of parameter descriptors.
            r: Number of trajectories.
            levels: Number of grid levels (even integer recommended).

        Returns:
            List of r trajectories, each an ndarray of shape (k+1, k).
        """
        k = len(parameters)
        delta = levels / (2.0 * (levels - 1))

        trajectories: List[np.ndarray] = []

        for _ in range(r):
            # Random base point on the grid
            base = np.zeros(k)
            for j in range(k):
                grid_vals = np.linspace(0, 1 - delta, levels)
                base[j] = self._rng.choice(grid_vals)

            # Random permutation of coordinate indices
            order = self._rng.permutation(k)

            # Build trajectory
            trajectory = np.zeros((k + 1, k))
            trajectory[0] = base.copy()

            for step_idx, j in enumerate(order):
                trajectory[step_idx + 1] = trajectory[step_idx].copy()
                # Decide direction
                if trajectory[step_idx, j] + delta <= 1.0:
                    trajectory[step_idx + 1, j] += delta
                else:
                    trajectory[step_idx + 1, j] -= delta

            trajectories.append(trajectory)

        logger.debug(
            f"Generated {r} Morris trajectories: k={k}, levels={levels}, "
            f"delta={delta:.4f}"
        )
        return trajectories

    def calculate_elementary_effects(
        self,
        calculation_func: Callable[[Dict[str, float]], float],
        trajectories: List[np.ndarray],
        parameters: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute elementary effects from trajectories.

        For each trajectory and each parameter, the elementary effect is:
            EE_i = (f(x + delta*e_i) - f(x)) / delta

        Args:
            calculation_func: Model f(params) -> scalar.
            trajectories: OAT trajectories (from generate_trajectories).
            parameters: Parameter descriptors.

        Returns:
            Dict mapping parameter name to
            {'mu_star': float, 'sigma': float, 'effects': List[float]}.
        """
        k = len(parameters)
        param_names = [p["name"] for p in parameters]
        effects: Dict[str, List[float]] = {name: [] for name in param_names}

        for traj in trajectories:
            # Evaluate at each row
            outputs = []
            for row_idx in range(traj.shape[0]):
                physical_vals: Dict[str, float] = {}
                for j, param in enumerate(parameters):
                    physical_vals[param["name"]] = self._to_physical(
                        float(traj[row_idx, j]), param
                    )
                outputs.append(calculation_func(physical_vals))

            # Identify which parameter changed at each step
            for step in range(k):
                diff_mask = traj[step + 1] - traj[step]
                changed_idx = int(np.argmax(np.abs(diff_mask)))
                delta_val = diff_mask[changed_idx]

                if abs(delta_val) > 1e-15:
                    ee = (outputs[step + 1] - outputs[step]) / delta_val
                    effects[param_names[changed_idx]].append(float(ee))

        # Summarize
        result: Dict[str, Dict[str, float]] = {}
        for name in param_names:
            ees = np.array(effects[name]) if effects[name] else np.array([0.0])
            mu_star = float(np.mean(np.abs(ees)))
            sigma = float(np.std(ees, ddof=1)) if len(ees) > 1 else 0.0
            result[name] = {
                "mu_star": round(mu_star, 6),
                "sigma": round(sigma, 6),
                "n_effects": len(effects[name]),
            }

        return result

    def _classify_parameter(self, mu_star: float, sigma: float) -> str:
        """
        Classify a parameter based on mu* and sigma.

        Classification rules (after Morris 1991):
        - If mu* is low -> non-important
        - If mu* is high and sigma is low -> important (linear / additive)
        - If mu* is high and sigma is high -> interactive (non-linear /
          interacting)

        We use sigma / mu_star ratio to determine interaction:
        ratio > 0.5 -> interactive, else important.

        Args:
            mu_star: Mean absolute elementary effect.
            sigma: Std dev of elementary effects.

        Returns:
            Classification string.
        """
        if mu_star < 1e-10:
            return "non-important"

        ratio = sigma / mu_star if mu_star > 0 else 0.0
        if ratio > 0.5:
            return "interactive"
        return "important"

    def run_screening(
        self,
        calculation_func: Callable[[Dict[str, float]], float],
        parameters: List[Dict[str, Any]],
        r: int = 10,
        levels: int = 4,
    ) -> MorrisResult:
        """
        Run a complete Morris screening analysis.

        Args:
            calculation_func: Model f(params) -> scalar.
            parameters: Parameter descriptors.
            r: Number of trajectories.
            levels: Number of grid levels.

        Returns:
            MorrisResult with mu*, sigma, confidence, and classification.
        """
        start = time.time()
        param_names = [p["name"] for p in parameters]

        logger.info(
            f"Running Morris screening: {len(parameters)} parameters, "
            f"r={r}, levels={levels}"
        )

        trajectories = self.generate_trajectories(parameters, r, levels)
        ee_results = self.calculate_elementary_effects(
            calculation_func, trajectories, parameters
        )

        mu_star: Dict[str, float] = {}
        sigma: Dict[str, float] = {}
        mu_star_conf: Dict[str, float] = {}
        classification: Dict[str, str] = {}

        for name in param_names:
            info = ee_results[name]
            ms = info["mu_star"]
            sg = info["sigma"]
            n_eff = info["n_effects"]

            mu_star[name] = ms
            sigma[name] = sg

            # 95% confidence interval for mu* (approximation)
            if n_eff > 1:
                conf = 1.96 * sg / np.sqrt(n_eff)
            else:
                conf = 0.0
            mu_star_conf[name] = round(float(conf), 6)

            classification[name] = self._classify_parameter(ms, sg)

        elapsed = time.time() - start
        logger.info(
            f"Morris screening completed in {elapsed:.3f}s: "
            f"classifications={classification}"
        )

        return MorrisResult(
            parameters=param_names,
            mu_star=mu_star,
            sigma=sigma,
            mu_star_conf=mu_star_conf,
            classification=classification,
            computation_time=round(elapsed, 4),
            num_trajectories=r,
        )


# ============================================================================
# TORNADO DIAGRAM GENERATOR
# ============================================================================

class TornadoDiagramGenerator:
    """
    Generator for tornado (one-way sensitivity) diagrams.

    Supports three modes of generation:
    1. Direct one-way sweep around baseline values.
    2. Conversion from Sobol indices.
    3. Conversion from Monte Carlo Pearson correlations.
    """

    def __init__(self) -> None:
        """Initialize TornadoDiagramGenerator."""
        logger.info("Initialized TornadoDiagramGenerator")

    def generate_one_way(
        self,
        calculation_func: Callable[[Dict[str, float]], float],
        parameters: List[Dict[str, Any]],
        baseline_values: Dict[str, float],
        variation_pct: float = 0.1,
    ) -> TornadoData:
        """
        Generate a tornado diagram via one-way parameter sweeps.

        For each parameter, the output is evaluated at
        baseline * (1 - variation_pct) and baseline * (1 + variation_pct)
        while all other parameters remain at their baseline values.

        Args:
            calculation_func: Model f(params) -> scalar.
            parameters: Parameter descriptors (used for names).
            baseline_values: Nominal parameter values.
            variation_pct: Fractional variation (e.g. 0.10 = +/-10%).

        Returns:
            TornadoData sorted by descending absolute impact.
        """
        start = time.time()
        baseline_output = float(calculation_func(baseline_values))

        tornado_params: List[TornadoParameter] = []

        for param in parameters:
            name = param["name"]
            base_val = baseline_values.get(name, param["mean"])

            # Low and high perturbations
            low_params = baseline_values.copy()
            high_params = baseline_values.copy()
            low_params[name] = base_val * (1.0 - variation_pct)
            high_params[name] = base_val * (1.0 + variation_pct)

            low_output = float(calculation_func(low_params))
            high_output = float(calculation_func(high_params))

            impact = abs(high_output - low_output)
            rel_impact = impact / abs(baseline_output) if baseline_output != 0 else 0.0

            tornado_params.append(
                TornadoParameter(
                    name=name,
                    low_value=round(low_output, 6),
                    high_value=round(high_output, 6),
                    impact=round(impact, 6),
                    relative_impact=round(rel_impact, 6),
                )
            )

        # Sort by descending impact
        tornado_params.sort(key=lambda p: p.impact, reverse=True)

        elapsed = time.time() - start
        logger.info(
            f"Tornado one-way sweep completed in {elapsed:.3f}s: "
            f"{len(tornado_params)} parameters"
        )

        return TornadoData(
            parameters=tornado_params,
            baseline_output=round(baseline_output, 6),
            computation_time=round(elapsed, 4),
        )

    def generate_from_sobol(
        self, sobol_result: SobolResult, baseline_output: float
    ) -> TornadoData:
        """
        Convert Sobol first-order indices into tornado-style impact data.

        The impact for each parameter is approximated as
        Si * Var(Y) ** 0.5, scaled relative to the baseline output.

        Since Sobol indices are normalised by total variance, we
        reconstruct a proxy impact as Si * baseline_output.

        Args:
            sobol_result: Completed Sobol analysis result.
            baseline_output: Baseline (nominal) model output.

        Returns:
            TornadoData sorted by descending impact.
        """
        start = time.time()
        tornado_params: List[TornadoParameter] = []

        for name in sobol_result.parameters:
            si = sobol_result.first_order_indices.get(name, 0.0)
            # Proxy impact proportional to Si
            proxy_impact = abs(si * baseline_output)
            half = proxy_impact / 2.0

            tornado_params.append(
                TornadoParameter(
                    name=name,
                    low_value=round(baseline_output - half, 6),
                    high_value=round(baseline_output + half, 6),
                    impact=round(proxy_impact, 6),
                    relative_impact=round(si, 6),
                )
            )

        tornado_params.sort(key=lambda p: p.impact, reverse=True)

        elapsed = time.time() - start
        return TornadoData(
            parameters=tornado_params,
            baseline_output=round(baseline_output, 6),
            computation_time=round(elapsed, 4),
        )

    def generate_from_monte_carlo(
        self,
        mc_sensitivity: Dict[str, float],
        baseline_output: float,
        mc_std: float,
    ) -> TornadoData:
        """
        Convert Monte Carlo Pearson correlations into tornado data.

        Uses the squared Pearson correlation (r^2) as a proxy for
        the fraction of variance explained, then converts to an
        impact magnitude using the MC standard deviation.

        Args:
            mc_sensitivity: Dict of {param_name: pearson_correlation}
                from MonteCarloSimulator._calculate_sensitivity_indices.
            baseline_output: Baseline model output (mean or nominal).
            mc_std: Standard deviation of the Monte Carlo output
                distribution.

        Returns:
            TornadoData sorted by descending impact.
        """
        start = time.time()
        tornado_params: List[TornadoParameter] = []

        for name, corr in mc_sensitivity.items():
            r_squared = corr ** 2
            # Impact proxy: the standard deviation contribution
            impact = abs(corr) * mc_std * 2.0  # +/- 1 sigma swing
            half = impact / 2.0

            rel_impact = (
                impact / abs(baseline_output) if baseline_output != 0 else 0.0
            )

            tornado_params.append(
                TornadoParameter(
                    name=name,
                    low_value=round(baseline_output - half, 6),
                    high_value=round(baseline_output + half, 6),
                    impact=round(impact, 6),
                    relative_impact=round(rel_impact, 6),
                )
            )

        tornado_params.sort(key=lambda p: p.impact, reverse=True)

        elapsed = time.time() - start
        return TornadoData(
            parameters=tornado_params,
            baseline_output=round(baseline_output, 6),
            computation_time=round(elapsed, 4),
        )


# ============================================================================
# CONVERGENCE ANALYZER
# ============================================================================

class ConvergenceAnalyzer:
    """
    Assess convergence of Monte Carlo simulation results.

    Evaluates running statistics at specified window sizes and determines
    whether the mean estimate has stabilised within a given threshold.
    """

    def assess_convergence(
        self,
        samples: np.ndarray,
        window_sizes: Optional[List[int]] = None,
        threshold: float = 0.01,
    ) -> ConvergenceResult:
        """
        Assess convergence of a Monte Carlo sample array.

        At each window size, the cumulative mean and standard deviation
        are computed. Convergence is declared if the relative change
        in the running mean between the last two windows is below the
        threshold.

        Args:
            samples: 1-D array of Monte Carlo output samples.
            window_sizes: List of window sizes to evaluate.
                Default: [100, 500, 1000, 5000, 10000].
            threshold: Relative change threshold for convergence
                (default 0.01 = 1%).

        Returns:
            ConvergenceResult with running statistics and recommendation.
        """
        if window_sizes is None:
            window_sizes = [100, 500, 1000, 5000, 10000]

        n_total = len(samples)

        # Filter out window sizes larger than the sample
        valid_windows = [w for w in window_sizes if w <= n_total]
        if not valid_windows:
            valid_windows = [n_total]

        running_means: List[float] = []
        running_stds: List[float] = []

        for w in valid_windows:
            subset = samples[:w]
            running_means.append(round(float(np.mean(subset)), 6))
            running_stds.append(round(float(np.std(subset, ddof=1)), 6))

        # Relative changes between consecutive windows
        relative_changes: List[float] = []
        for i in range(1, len(running_means)):
            prev = running_means[i - 1]
            curr = running_means[i]
            if abs(prev) > 1e-15:
                rel = abs((curr - prev) / prev)
            else:
                rel = abs(curr - prev)
            relative_changes.append(round(rel, 6))

        # Convergence check: last relative change below threshold
        is_converged = False
        if relative_changes:
            is_converged = relative_changes[-1] < threshold

        # Recommend iterations
        recommended = valid_windows[-1]
        if not is_converged:
            # Suggest doubling the largest evaluated window
            recommended = min(valid_windows[-1] * 2, 1_000_000)
        else:
            # Find the smallest window where convergence holds
            for i, rc in enumerate(relative_changes):
                if rc < threshold:
                    recommended = valid_windows[i + 1]
                    break

        logger.info(
            f"Convergence assessment: converged={is_converged}, "
            f"recommended={recommended}, last_change="
            f"{relative_changes[-1] if relative_changes else 'N/A'}"
        )

        return ConvergenceResult(
            is_converged=is_converged,
            recommended_iterations=recommended,
            running_means=running_means,
            running_stds=running_stds,
            window_sizes=valid_windows,
            relative_changes=relative_changes,
            convergence_threshold=threshold,
        )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Analyzers
    "SobolAnalyzer",
    "MorrisAnalyzer",
    "TornadoDiagramGenerator",
    "ConvergenceAnalyzer",
    # Models
    "SobolResult",
    "MorrisResult",
    "TornadoParameter",
    "TornadoData",
    "ConvergenceResult",
]
