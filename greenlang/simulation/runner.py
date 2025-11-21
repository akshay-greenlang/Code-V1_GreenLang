# -*- coding: utf-8 -*-
"""
Scenario Runner - Execute scenarios with deterministic RNG

This module provides the runtime execution engine for GreenLang scenarios.
Supports parameter sweeps, Monte Carlo sampling, and provenance tracking.

Author: GreenLang Framework Team
Date: October 2025
Spec: SIM-401 (Scenario Spec & Seeded RNG)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ..specs.scenariospec_v1 import ScenarioSpecV1, from_yaml
from ..intelligence.glrng import GLRNG
from ..provenance.utils import ProvenanceContext, record_seed_info

logger = logging.getLogger(__name__)


class ScenarioRunner:
    """
    Execute scenarios with deterministic parameter sampling.

    Provides runtime execution for scenario specifications with:
    - Parameter sweep generation (grid search)
    - Monte Carlo sampling (stochastic parameters)
    - Deterministic RNG with hierarchical substreams
    - Provenance tracking for reproducibility

    Example:
        >>> runner = ScenarioRunner("scenarios/baseline_sweep.yaml")
        >>> results = []
        >>> for params in runner.generate_samples():
        ...     result = my_model(**params)
        ...     results.append(result)
        >>> runner.save_results(results, "output/results.json")
    """

    def __init__(
        self,
        scenario_path: Optional[str | Path] = None,
        spec: Optional[ScenarioSpecV1] = None
    ):
        """
        Initialize scenario runner.

        Args:
            scenario_path: Path to scenario YAML file
            spec: ScenarioSpecV1 instance (alternative to path)

        Raises:
            ValueError: If neither path nor spec provided
        """
        if scenario_path is None and spec is None:
            raise ValueError("Must provide either scenario_path or spec")

        if scenario_path is not None:
            self.spec = from_yaml(scenario_path)
            self.scenario_path = Path(scenario_path)
        else:
            self.spec = spec
            self.scenario_path = None

        # Initialize root RNG
        self.rng = GLRNG(seed=self.spec.seed)

        # Initialize provenance context
        self.provenance_ctx = ProvenanceContext(name=self.spec.name)

        # Record scenario spec in provenance
        spec_dict = self.spec.model_dump()
        record_seed_info(
            ctx=self.provenance_ctx,
            spec=spec_dict,
            seed_root=self.spec.seed,
            seed_path=f"scenario:{self.spec.name}",
            spec_type="scenario"
        )

        logger.info(f"Initialized ScenarioRunner for '{self.spec.name}' (seed={self.spec.seed})")

    def generate_samples(self) -> Iterator[Dict[str, Any]]:
        """
        Generate parameter samples according to scenario specification.

        For sweep parameters: iterates through discrete values.
        For distribution parameters: samples from Monte Carlo trials.

        Yields:
            Dictionary of parameter name → sampled value

        Example:
            >>> runner = ScenarioRunner("scenario.yaml")
            >>> for i, params in enumerate(runner.generate_samples()):
            ...     print(f"Trial {i}: {params}")
            Trial 0: {'temperature': 20.5, 'pressure': 101325, ...}
            Trial 1: {'temperature': 25.3, 'pressure': 101325, ...}
        """
        # Separate sweep and distribution parameters
        sweep_params = [p for p in self.spec.parameters if p.type == "sweep"]
        dist_params = [p for p in self.spec.parameters if p.type == "distribution"]

        if not dist_params:
            # Pure grid sweep (deterministic)
            yield from self._generate_grid_sweep(sweep_params)
        else:
            # Monte Carlo with optional sweep dimensions
            if self.spec.monte_carlo is None:
                raise ValueError("Monte Carlo config required for distribution parameters")

            yield from self._generate_monte_carlo(sweep_params, dist_params)

    def _generate_grid_sweep(self, sweep_params: List) -> Iterator[Dict[str, Any]]:
        """
        Generate parameter combinations for grid sweep.

        Args:
            sweep_params: List of sweep parameter specs

        Yields:
            Parameter dictionaries
        """
        if not sweep_params:
            yield {}
            return

        # Build parameter grid (Cartesian product)
        import itertools

        param_names = [p.id for p in sweep_params]
        param_values = [p.values for p in sweep_params]

        for combination in itertools.product(*param_values):
            yield dict(zip(param_names, combination))

    def _generate_monte_carlo(
        self,
        sweep_params: List,
        dist_params: List
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate Monte Carlo samples with optional sweep dimensions.

        Args:
            sweep_params: List of sweep parameter specs
            dist_params: List of distribution parameter specs

        Yields:
            Parameter dictionaries
        """
        num_trials = self.spec.monte_carlo.trials

        # For each sweep combination
        for sweep_values in self._generate_grid_sweep(sweep_params):
            # Generate Monte Carlo trials
            for trial_idx in range(num_trials):
                params = sweep_values.copy()

                # Sample each distribution parameter
                for param_spec in dist_params:
                    # Create substream for this parameter and trial
                    param_path = f"scenario:{self.spec.name}|param:{param_spec.id}|trial:{trial_idx}"
                    param_rng = self.rng.spawn(param_path)

                    # Sample from distribution
                    dist = param_spec.distribution
                    if dist.kind == "uniform":
                        value = param_rng.uniform(dist.low, dist.high)
                    elif dist.kind == "normal":
                        value = param_rng.normal(dist.mean, dist.std)
                    elif dist.kind == "lognormal":
                        value = param_rng.lognormal(dist.mean, dist.sigma)
                    elif dist.kind == "triangular":
                        value = param_rng.triangular(dist.low, dist.mode, dist.high)
                    else:
                        raise ValueError(f"Unsupported distribution: {dist.kind}")

                    params[param_spec.id] = value

                yield params

    def get_rng(self, path: str = "") -> GLRNG:
        """
        Get RNG for custom sampling.

        Provides access to scenario's root RNG with optional substream derivation.

        Args:
            path: Optional path for substream (e.g., "custom:my_sampler")

        Returns:
            GLRNG instance

        Example:
            >>> runner = ScenarioRunner("scenario.yaml")
            >>> custom_rng = runner.get_rng("custom:sensitivity_analysis")
            >>> samples = [custom_rng.normal(0, 1) for _ in range(100)]
        """
        if path:
            return self.rng.spawn(path)
        return self.rng

    def finalize(self) -> Path:
        """
        Finalize scenario execution and write provenance.

        Returns:
            Path to provenance ledger

        Example:
            >>> runner = ScenarioRunner("scenario.yaml")
            >>> # ... run scenario ...
            >>> ledger_path = runner.finalize()
            >>> print(f"Provenance written to: {ledger_path}")
        """
        self.provenance_ctx.status = "success"
        ledger_path = self.provenance_ctx.finalize()

        logger.info(f"Scenario '{self.spec.name}' completed. Provenance: {ledger_path}")
        return ledger_path


def run_scenario(
    scenario_path: str | Path,
    model_fn: Callable[[Dict[str, Any]], Any],
    output_path: Optional[str | Path] = None
) -> List[Any]:
    """
    Convenience function to run scenario with a model function.

    Args:
        scenario_path: Path to scenario YAML file
        model_fn: Function that takes parameter dict and returns result
        output_path: Optional path to save results

    Returns:
        List of results from model_fn for each parameter sample

    Example:
        >>> def my_model(temperature, pressure, **kwargs):
        ...     return {"energy": temperature * pressure}
        >>>
        >>> results = run_scenario(
        ...     "scenarios/sensitivity.yaml",
        ...     my_model,
        ...     "output/results.json"
        ... )
        >>> len(results)
        1000
    """
    import json

    runner = ScenarioRunner(scenario_path)
    results = []

    logger.info(f"Running scenario: {runner.spec.name}")

    for i, params in enumerate(runner.generate_samples()):
        if i % 100 == 0:
            logger.info(f"Sample {i}/{runner.spec.monte_carlo.trials if runner.spec.monte_carlo else '?'}...")

        result = model_fn(**params)
        results.append({
            "sample_id": i,
            "parameters": params,
            "result": result
        })

    # Save results if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    # Finalize provenance
    runner.finalize()

    return results
