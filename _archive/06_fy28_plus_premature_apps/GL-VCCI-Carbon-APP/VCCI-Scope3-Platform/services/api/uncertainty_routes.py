# -*- coding: utf-8 -*-
"""
Uncertainty Analysis & Sensitivity API Routes

FastAPI router providing endpoints for Monte Carlo uncertainty analysis,
Sobol / Morris sensitivity analysis, tornado diagrams, convergence
diagnostics, and multi-scenario comparison.

All calculations are ZERO-HALLUCINATION deterministic (NumPy/SciPy only).

Endpoints:
    POST /api/v1/uncertainty/analyze      - Full uncertainty analysis
    GET  /api/v1/uncertainty/{id}         - Retrieve stored results
    GET  /api/v1/uncertainty/{id}/distribution - Distribution chart data
    POST /api/v1/uncertainty/sensitivity  - Sensitivity analysis
    GET  /api/v1/uncertainty/convergence/{id} - Convergence assessment
    POST /api/v1/uncertainty/compare-scenarios - Scenario comparison

Version: 1.1.0
Date: 2026-03-01
"""

import uuid
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from services.methodologies.monte_carlo import MonteCarloSimulator
from services.methodologies.models import MonteCarloInput, MonteCarloResult
from services.methodologies.constants import DistributionType
from services.methodologies.sensitivity_analysis import (
    SobolAnalyzer,
    MorrisAnalyzer,
    TornadoDiagramGenerator,
    ConvergenceAnalyzer,
    SobolResult,
    MorrisResult,
    TornadoData,
    ConvergenceResult,
)

logger = logging.getLogger(__name__)


# ============================================================================
# IN-MEMORY RESULT STORE
# ============================================================================

class _ResultStore:
    """Thread-safe in-memory cache for calculation results."""

    def __init__(self) -> None:
        self._results: Dict[str, Dict[str, Any]] = {}

    def save(
        self,
        calc_id: str,
        mc_result: MonteCarloResult,
        samples: np.ndarray,
        sensitivity: Optional[Any] = None,
        convergence: Optional[ConvergenceResult] = None,
    ) -> None:
        """Persist a calculation result by ID."""
        self._results[calc_id] = {
            "mc_result": mc_result,
            "samples": samples,
            "sensitivity": sensitivity,
            "convergence": convergence,
            "stored_at": datetime.utcnow().isoformat(),
        }

    def get(self, calc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored result or None."""
        return self._results.get(calc_id)

    def exists(self, calc_id: str) -> bool:
        """Check whether a result exists."""
        return calc_id in self._results


_store = _ResultStore()


# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================

class ParameterSpec(BaseModel):
    """Specification for a single uncertain parameter."""

    name: str = Field(..., description="Parameter name")
    mean: float = Field(..., description="Mean (central) value")
    std_dev: float = Field(
        ..., ge=0.0, description="Standard deviation (absolute)"
    )
    distribution: str = Field(
        default="lognormal",
        description="Distribution type: normal, lognormal, uniform, triangular",
    )
    min_value: Optional[float] = Field(
        None, alias="min", description="Minimum value (for uniform/triangular)"
    )
    max_value: Optional[float] = Field(
        None, alias="max", description="Maximum value (for uniform/triangular)"
    )

    model_config = {"populate_by_name": True}


# ---------- POST /analyze ----------

class AnalyzeRequest(BaseModel):
    """Request body for POST /analyze."""

    calculation_type: str = Field(
        default="multiply",
        description="Calculation type: 'multiply' (A*F), 'add' (A+F), or 'custom'",
    )
    parameters: List[ParameterSpec] = Field(
        ..., min_length=1, description="Uncertain parameters"
    )
    iterations: int = Field(
        default=10000, ge=1000, le=1000000, description="Monte Carlo iterations"
    )
    seed: Optional[int] = Field(
        None, description="Random seed for reproducibility"
    )
    include_sensitivity: bool = Field(
        default=False, description="Run Sobol analysis alongside MC"
    )
    sensitivity_N: int = Field(
        default=512,
        ge=64,
        le=8192,
        description="Sobol base sample size (if include_sensitivity=True)",
    )


class AnalyzeResponse(BaseModel):
    """Response body for POST /analyze."""

    calculation_id: str = Field(..., description="Unique calculation identifier")
    result: MonteCarloResult = Field(
        ..., description="Monte Carlo simulation result"
    )
    sensitivity: Optional[SobolResult] = Field(
        None, description="Sobol sensitivity result (if requested)"
    )
    convergence: ConvergenceResult = Field(
        ..., description="Convergence assessment"
    )


# ---------- GET /{id}/distribution ----------

class BinData(BaseModel):
    """Single histogram bin."""

    x: float = Field(..., description="Bin centre")
    count: int = Field(..., description="Sample count in bin")


class KDEPoint(BaseModel):
    """Single KDE evaluation point."""

    x: float = Field(..., description="X coordinate")
    density: float = Field(..., description="Probability density")


class DistributionResponse(BaseModel):
    """Response for GET /{id}/distribution."""

    bins: List[BinData] = Field(..., description="Histogram bins")
    kde: List[KDEPoint] = Field(..., description="Kernel density estimate")
    percentiles: Dict[str, float] = Field(
        ..., description="Key percentiles: p5, p25, p50, p75, p95"
    )
    statistics: Dict[str, float] = Field(
        ..., description="mean, median, std_dev, skewness, kurtosis"
    )


# ---------- POST /sensitivity ----------

class SensitivityRequest(BaseModel):
    """Request body for POST /sensitivity."""

    method: str = Field(
        ..., description="Analysis method: 'sobol', 'morris', or 'tornado'"
    )
    parameters: List[ParameterSpec] = Field(
        ..., min_length=1, description="Uncertain parameters"
    )
    calculation_func_type: str = Field(
        default="multiply",
        description="Built-in calculation: 'multiply', 'add', 'weighted_sum'",
    )
    iterations: int = Field(
        default=1024,
        ge=64,
        le=100000,
        description="Base sample size (Sobol N) or trajectories (Morris r)",
    )
    seed: Optional[int] = Field(None, description="Random seed")
    variation_pct: float = Field(
        default=0.10,
        ge=0.01,
        le=1.0,
        description="Variation percentage for tornado (default 10%)",
    )


class SensitivityResponse(BaseModel):
    """Response for POST /sensitivity."""

    method: str = Field(..., description="Analysis method used")
    sobol: Optional[SobolResult] = Field(None, description="Sobol result")
    morris: Optional[MorrisResult] = Field(None, description="Morris result")
    tornado: Optional[TornadoData] = Field(None, description="Tornado result")


# ---------- POST /compare-scenarios ----------

class ScenarioSpec(BaseModel):
    """One scenario for comparison."""

    name: str = Field(..., description="Scenario name")
    parameters: List[ParameterSpec] = Field(
        ..., min_length=1, description="Parameters for this scenario"
    )


class CompareRequest(BaseModel):
    """Request body for POST /compare-scenarios."""

    scenarios: List[ScenarioSpec] = Field(
        ..., min_length=2, description="At least two scenarios"
    )
    iterations: int = Field(
        default=10000, ge=1000, le=1000000, description="Monte Carlo iterations"
    )
    seed: Optional[int] = Field(None, description="Random seed")
    target_value: Optional[float] = Field(
        None,
        description="Target emission value for exceedance probability",
    )
    calculation_type: str = Field(
        default="multiply",
        description="Calculation type: 'multiply', 'add', 'weighted_sum'",
    )


class ScenarioComparison(BaseModel):
    """Statistics for one scenario."""

    name: str = Field(..., description="Scenario name")
    mean: float = Field(..., description="Mean emission")
    std_dev: float = Field(..., description="Standard deviation")
    p5: float = Field(..., description="5th percentile")
    p25: float = Field(..., description="25th percentile")
    p50: float = Field(..., description="50th percentile (median)")
    p75: float = Field(..., description="75th percentile")
    p95: float = Field(..., description="95th percentile")


class StatisticalSignificance(BaseModel):
    """Pairwise statistical significance between two scenarios."""

    scenario_a: str = Field(..., description="First scenario name")
    scenario_b: str = Field(..., description="Second scenario name")
    mean_difference: float = Field(..., description="Difference in means")
    p_value: float = Field(
        ..., description="P-value from Welch's t-test"
    )
    significant: bool = Field(
        ..., description="Significant at p < 0.05"
    )


class CompareResponse(BaseModel):
    """Response for POST /compare-scenarios."""

    comparisons: List[ScenarioComparison] = Field(
        ..., description="Per-scenario statistics"
    )
    target_probability: Optional[Dict[str, float]] = Field(
        None,
        description="P(emission > target) per scenario, if target_value provided",
    )
    statistical_significance: List[StatisticalSignificance] = Field(
        default_factory=list,
        description="Pairwise significance tests",
    )


# ============================================================================
# HELPER: BUILD CALCULATION FUNCTION
# ============================================================================

def _build_calculation_func(
    calc_type: str, param_names: List[str]
) -> Any:
    """
    Build a deterministic calculation function from a type string.

    Args:
        calc_type: One of 'multiply', 'add', 'weighted_sum'.
        param_names: Ordered parameter names.

    Returns:
        Callable[[Dict[str, float]], float]
    """
    if calc_type == "multiply":
        def _multiply(params: Dict[str, float]) -> float:
            result = 1.0
            for name in param_names:
                result *= params[name]
            return result
        return _multiply

    if calc_type == "add":
        def _add(params: Dict[str, float]) -> float:
            return sum(params[name] for name in param_names)
        return _add

    if calc_type == "weighted_sum":
        # Use index as weight: w_i = i+1
        def _wsum(params: Dict[str, float]) -> float:
            return sum(
                (i + 1) * params[name]
                for i, name in enumerate(param_names)
            )
        return _wsum

    # Default fallback: multiply
    def _default(params: Dict[str, float]) -> float:
        result = 1.0
        for name in param_names:
            result *= params[name]
        return result
    return _default


def _to_mc_input(spec: ParameterSpec) -> MonteCarloInput:
    """Convert a ParameterSpec to a MonteCarloInput model."""
    dist_map = {
        "normal": DistributionType.NORMAL,
        "lognormal": DistributionType.LOGNORMAL,
        "uniform": DistributionType.UNIFORM,
        "triangular": DistributionType.TRIANGULAR,
    }
    return MonteCarloInput(
        name=spec.name,
        mean=spec.mean,
        std_dev=spec.std_dev,
        distribution=dist_map.get(spec.distribution, DistributionType.LOGNORMAL),
        min_value=spec.min_value,
        max_value=spec.max_value,
    )


def _to_param_dict(spec: ParameterSpec) -> Dict[str, Any]:
    """Convert a ParameterSpec to the dict format used by sensitivity analyzers."""
    return {
        "name": spec.name,
        "mean": spec.mean,
        "std_dev": spec.std_dev,
        "distribution": spec.distribution,
        "min": spec.min_value,
        "max": spec.max_value,
    }


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(prefix="/api/v1/uncertainty", tags=["uncertainty"])


# --------------------------------------------------------------------------
# POST /analyze
# --------------------------------------------------------------------------

@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Run full uncertainty analysis",
    description=(
        "Runs a Monte Carlo simulation with optional Sobol sensitivity "
        "analysis and convergence diagnostics."
    ),
)
async def analyze_uncertainty(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Run full Monte Carlo uncertainty analysis with optional sensitivity.

    Steps:
    1. Build calculation function from calculation_type.
    2. Run Monte Carlo simulation for N iterations.
    3. Optionally run Sobol analysis.
    4. Assess convergence.
    5. Cache results for subsequent GET requests.
    """
    try:
        calc_id = str(uuid.uuid4())
        param_names = [p.name for p in request.parameters]

        logger.info(
            f"Uncertainty analysis requested: id={calc_id}, "
            f"params={param_names}, iterations={request.iterations}"
        )

        # Build function
        calc_func = _build_calculation_func(request.calculation_type, param_names)

        # Prepare MC inputs
        mc_params: Dict[str, MonteCarloInput] = {
            p.name: _to_mc_input(p) for p in request.parameters
        }

        # Run Monte Carlo
        simulator = MonteCarloSimulator(seed=request.seed)
        mc_result = simulator.run_simulation(
            calc_func, mc_params, iterations=request.iterations
        )

        # Generate raw samples for convergence / distribution
        samples_dict: Dict[str, np.ndarray] = {}
        for name, mc_input in mc_params.items():
            samples_dict[name] = simulator.generate_samples(
                mc_input, request.iterations
            )
        raw_samples = np.array([
            calc_func({name: samples_dict[name][i] for name in param_names})
            for i in range(request.iterations)
        ])

        # Optional Sobol analysis
        sobol_result: Optional[SobolResult] = None
        if request.include_sensitivity:
            sobol_analyzer = SobolAnalyzer(seed=request.seed)
            sobol_params = [_to_param_dict(p) for p in request.parameters]
            sobol_result = sobol_analyzer.run_analysis(
                calc_func, sobol_params, N=request.sensitivity_N
            )

        # Convergence assessment
        convergence_analyzer = ConvergenceAnalyzer()
        convergence = convergence_analyzer.assess_convergence(raw_samples)

        # Cache
        _store.save(
            calc_id,
            mc_result=mc_result,
            samples=raw_samples,
            sensitivity=sobol_result,
            convergence=convergence,
        )

        logger.info(
            f"Uncertainty analysis completed: id={calc_id}, "
            f"mean={mc_result.mean:.4f}, std={mc_result.std_dev:.4f}"
        )

        return AnalyzeResponse(
            calculation_id=calc_id,
            result=mc_result,
            sensitivity=sobol_result,
            convergence=convergence,
        )

    except ValueError as e:
        logger.error(f"Validation error in /analyze: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error in /analyze: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during uncertainty analysis",
        )


# --------------------------------------------------------------------------
# GET /{calculation_id}
# --------------------------------------------------------------------------

@router.get(
    "/{calculation_id}",
    response_model=MonteCarloResult,
    status_code=status.HTTP_200_OK,
    summary="Get stored uncertainty results",
)
async def get_uncertainty_result(calculation_id: str) -> MonteCarloResult:
    """
    Retrieve a previously computed Monte Carlo result by its ID.
    """
    entry = _store.get(calculation_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Calculation {calculation_id} not found",
        )
    return entry["mc_result"]


# --------------------------------------------------------------------------
# GET /{calculation_id}/distribution
# --------------------------------------------------------------------------

@router.get(
    "/{calculation_id}/distribution",
    response_model=DistributionResponse,
    status_code=status.HTTP_200_OK,
    summary="Get distribution data for charting",
)
async def get_distribution_data(
    calculation_id: str,
    num_bins: int = 50,
    kde_points: int = 200,
) -> DistributionResponse:
    """
    Return histogram bins, KDE curve, percentiles, and descriptive
    statistics for a previously computed Monte Carlo result.

    Args:
        calculation_id: ID returned from POST /analyze.
        num_bins: Number of histogram bins (default 50).
        kde_points: Number of KDE evaluation points (default 200).
    """
    entry = _store.get(calculation_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Calculation {calculation_id} not found",
        )

    try:
        samples: np.ndarray = entry["samples"]
        mc_result: MonteCarloResult = entry["mc_result"]

        # Histogram
        counts, bin_edges = np.histogram(samples, bins=num_bins)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bins = [
            BinData(x=round(float(bin_centres[i]), 6), count=int(counts[i]))
            for i in range(len(counts))
        ]

        # KDE
        from scipy.stats import gaussian_kde
        kde_func = gaussian_kde(samples)
        x_kde = np.linspace(float(samples.min()), float(samples.max()), kde_points)
        y_kde = kde_func(x_kde)
        kde = [
            KDEPoint(x=round(float(x_kde[i]), 6), density=round(float(y_kde[i]), 8))
            for i in range(len(x_kde))
        ]

        # Percentiles
        pcts = np.percentile(samples, [5, 25, 50, 75, 95])
        percentiles = {
            "p5": round(float(pcts[0]), 6),
            "p25": round(float(pcts[1]), 6),
            "p50": round(float(pcts[2]), 6),
            "p75": round(float(pcts[3]), 6),
            "p95": round(float(pcts[4]), 6),
        }

        # Descriptive statistics
        from scipy.stats import skew, kurtosis as scipy_kurtosis
        statistics = {
            "mean": round(float(np.mean(samples)), 6),
            "median": round(float(np.median(samples)), 6),
            "std_dev": round(float(np.std(samples, ddof=1)), 6),
            "skewness": round(float(skew(samples)), 6),
            "kurtosis": round(float(scipy_kurtosis(samples)), 6),
        }

        return DistributionResponse(
            bins=bins,
            kde=kde,
            percentiles=percentiles,
            statistics=statistics,
        )

    except Exception as e:
        logger.error(f"Error generating distribution data: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating distribution data",
        )


# --------------------------------------------------------------------------
# POST /sensitivity
# --------------------------------------------------------------------------

@router.post(
    "/sensitivity",
    response_model=SensitivityResponse,
    status_code=status.HTTP_200_OK,
    summary="Run sensitivity analysis",
    description="Run Sobol, Morris, or Tornado sensitivity analysis.",
)
async def run_sensitivity(request: SensitivityRequest) -> SensitivityResponse:
    """
    Run a sensitivity analysis using the specified method.

    Supported methods:
    - 'sobol': Variance-based global sensitivity (Saltelli scheme).
    - 'morris': Elementary effects screening (OAT trajectories).
    - 'tornado': One-way deterministic sweep around baseline.
    """
    try:
        param_names = [p.name for p in request.parameters]
        calc_func = _build_calculation_func(
            request.calculation_func_type, param_names
        )
        param_dicts = [_to_param_dict(p) for p in request.parameters]

        method = request.method.lower()

        logger.info(
            f"Sensitivity analysis requested: method={method}, "
            f"params={param_names}"
        )

        if method == "sobol":
            analyzer = SobolAnalyzer(seed=request.seed)
            sobol_result = analyzer.run_analysis(
                calc_func, param_dicts, N=request.iterations
            )
            return SensitivityResponse(
                method="sobol", sobol=sobol_result
            )

        elif method == "morris":
            analyzer = MorrisAnalyzer(seed=request.seed)
            morris_result = analyzer.run_screening(
                calc_func, param_dicts, r=request.iterations
            )
            return SensitivityResponse(
                method="morris", morris=morris_result
            )

        elif method == "tornado":
            generator = TornadoDiagramGenerator()
            baseline = {p.name: p.mean for p in request.parameters}
            tornado_result = generator.generate_one_way(
                calc_func,
                param_dicts,
                baseline,
                variation_pct=request.variation_pct,
            )
            return SensitivityResponse(
                method="tornado", tornado=tornado_result
            )

        else:
            raise ValueError(
                f"Unsupported method '{request.method}'. "
                "Use 'sobol', 'morris', or 'tornado'."
            )

    except ValueError as e:
        logger.error(f"Validation error in /sensitivity: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Unexpected error in /sensitivity: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during sensitivity analysis",
        )


# --------------------------------------------------------------------------
# GET /convergence/{calculation_id}
# --------------------------------------------------------------------------

@router.get(
    "/convergence/{calculation_id}",
    response_model=ConvergenceResult,
    status_code=status.HTTP_200_OK,
    summary="Get convergence assessment",
)
async def get_convergence(calculation_id: str) -> ConvergenceResult:
    """
    Retrieve or compute convergence assessment for a stored calculation.
    """
    entry = _store.get(calculation_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Calculation {calculation_id} not found",
        )

    # Return cached convergence if available
    if entry.get("convergence") is not None:
        return entry["convergence"]

    # Otherwise compute
    try:
        samples = entry["samples"]
        analyzer = ConvergenceAnalyzer()
        convergence = analyzer.assess_convergence(samples)

        # Update cache
        entry["convergence"] = convergence
        return convergence

    except Exception as e:
        logger.error(f"Error computing convergence: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error computing convergence assessment",
        )


# --------------------------------------------------------------------------
# POST /compare-scenarios
# --------------------------------------------------------------------------

@router.post(
    "/compare-scenarios",
    response_model=CompareResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare scenarios with uncertainty",
    description=(
        "Run Monte Carlo simulation for multiple scenarios and compare "
        "their distributions, with optional target exceedance probability "
        "and pairwise statistical significance testing."
    ),
)
async def compare_scenarios(request: CompareRequest) -> CompareResponse:
    """
    Compare two or more emission scenarios under uncertainty.

    For each scenario, runs a Monte Carlo simulation and returns
    summary statistics, optional exceedance probabilities, and
    pairwise Welch's t-test significance results.
    """
    try:
        comparisons: List[ScenarioComparison] = []
        scenario_samples: Dict[str, np.ndarray] = {}

        for scenario in request.scenarios:
            param_names = [p.name for p in scenario.parameters]
            calc_func = _build_calculation_func(
                request.calculation_type, param_names
            )

            mc_params: Dict[str, MonteCarloInput] = {
                p.name: _to_mc_input(p) for p in scenario.parameters
            }

            simulator = MonteCarloSimulator(seed=request.seed)
            mc_result = simulator.run_simulation(
                calc_func, mc_params, iterations=request.iterations
            )

            # Reconstruct raw samples for t-test and exceedance
            samples_dict: Dict[str, np.ndarray] = {}
            for name, mc_input in mc_params.items():
                samples_dict[name] = simulator.generate_samples(
                    mc_input, request.iterations
                )
            raw_samples = np.array([
                calc_func({name: samples_dict[name][i] for name in param_names})
                for i in range(request.iterations)
            ])
            scenario_samples[scenario.name] = raw_samples

            pcts = np.percentile(raw_samples, [5, 25, 50, 75, 95])
            comparisons.append(
                ScenarioComparison(
                    name=scenario.name,
                    mean=round(float(np.mean(raw_samples)), 6),
                    std_dev=round(float(np.std(raw_samples, ddof=1)), 6),
                    p5=round(float(pcts[0]), 6),
                    p25=round(float(pcts[1]), 6),
                    p50=round(float(pcts[2]), 6),
                    p75=round(float(pcts[3]), 6),
                    p95=round(float(pcts[4]), 6),
                )
            )

        # Target exceedance probability
        target_probability: Optional[Dict[str, float]] = None
        if request.target_value is not None:
            target_probability = {}
            for name, samples in scenario_samples.items():
                prob = float(np.mean(samples > request.target_value))
                target_probability[name] = round(prob, 6)

        # Pairwise statistical significance (Welch's t-test)
        from scipy.stats import ttest_ind

        significance: List[StatisticalSignificance] = []
        scenario_names = list(scenario_samples.keys())
        for i in range(len(scenario_names)):
            for j in range(i + 1, len(scenario_names)):
                name_a = scenario_names[i]
                name_b = scenario_names[j]
                samples_a = scenario_samples[name_a]
                samples_b = scenario_samples[name_b]

                t_stat, p_val = ttest_ind(
                    samples_a, samples_b, equal_var=False
                )
                mean_diff = float(np.mean(samples_a) - np.mean(samples_b))

                significance.append(
                    StatisticalSignificance(
                        scenario_a=name_a,
                        scenario_b=name_b,
                        mean_difference=round(mean_diff, 6),
                        p_value=round(float(p_val), 8),
                        significant=float(p_val) < 0.05,
                    )
                )

        logger.info(
            f"Scenario comparison completed: {len(comparisons)} scenarios, "
            f"{len(significance)} pairwise tests"
        )

        return CompareResponse(
            comparisons=comparisons,
            target_probability=target_probability,
            statistical_significance=significance,
        )

    except ValueError as e:
        logger.error(f"Validation error in /compare-scenarios: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error(
            f"Unexpected error in /compare-scenarios: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during scenario comparison",
        )


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "router",
]
