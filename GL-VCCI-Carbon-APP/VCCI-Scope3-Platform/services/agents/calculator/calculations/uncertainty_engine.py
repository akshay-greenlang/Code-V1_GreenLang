"""
Uncertainty Propagation Engine
Monte Carlo simulation for emissions calculations
"""

import logging
import numpy as np
from typing import Optional, Dict, Any

from ..models import UncertaintyResult
from ...methodologies.monte_carlo import MonteCarloSimulator

logger = logging.getLogger(__name__)


class UncertaintyEngine:
    """
    Uncertainty propagation engine using Monte Carlo simulation.

    Wraps the methodologies Monte Carlo module for calculator use.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize uncertainty engine."""
        self.simulator = MonteCarloSimulator(seed=seed)
        logger.info("Initialized UncertaintyEngine")

    async def propagate(
        self,
        quantity: float,
        quantity_uncertainty: float,
        emission_factor: float,
        factor_uncertainty: float,
        iterations: int = 10000
    ) -> UncertaintyResult:
        """
        Propagate uncertainty for simple multiplication.

        Formula: emissions = quantity × emission_factor

        Args:
            quantity: Activity quantity
            quantity_uncertainty: Relative uncertainty (CV)
            emission_factor: Emission factor
            factor_uncertainty: Factor uncertainty (CV)
            iterations: Monte Carlo iterations

        Returns:
            UncertaintyResult
        """
        result = self.simulator.simple_propagation(
            activity_data=quantity,
            activity_uncertainty=quantity_uncertainty,
            emission_factor=emission_factor,
            factor_uncertainty=factor_uncertainty,
            iterations=iterations
        )

        # Convert to our model
        uncertainty_pct = (result.std_dev / result.mean * 100) if result.mean > 0 else 0

        return UncertaintyResult(
            mean=result.mean,
            std_dev=result.std_dev,
            p5=result.p5,
            p50=result.p50,
            p95=result.p95,
            min_value=result.min_value,
            max_value=result.max_value,
            uncertainty_range=f"±{uncertainty_pct:.1f}%",
            coefficient_of_variation=result.coefficient_of_variation,
            iterations=iterations
        )

    async def propagate_logistics(
        self,
        distance: float,
        distance_uncertainty: float,
        weight: float,
        weight_uncertainty: float,
        emission_factor: float,
        factor_uncertainty: float,
        load_factor: float = 1.0,
        iterations: int = 10000
    ) -> UncertaintyResult:
        """
        Propagate uncertainty for logistics calculation.

        Formula: emissions = distance × weight × EF / load_factor

        Args:
            distance: Distance in km
            distance_uncertainty: Distance uncertainty (CV)
            weight: Weight in tonnes
            weight_uncertainty: Weight uncertainty (CV)
            emission_factor: Emission factor
            factor_uncertainty: Factor uncertainty (CV)
            load_factor: Load factor
            iterations: Monte Carlo iterations

        Returns:
            UncertaintyResult
        """
        def calc(params: Dict[str, float]) -> float:
            return params['distance'] * params['weight'] * params['ef'] / params['load_factor']

        mean_values = {
            'distance': distance,
            'weight': weight,
            'ef': emission_factor,
            'load_factor': load_factor
        }

        uncertainties = {
            'distance': distance_uncertainty,
            'weight': weight_uncertainty,
            'ef': factor_uncertainty,
            'load_factor': 0.05  # Assume 5% uncertainty on load factor
        }

        result = self.simulator.propagate_uncertainty(
            mean_values=mean_values,
            uncertainties=uncertainties,
            calculation_func=calc,
            iterations=iterations
        )

        uncertainty_pct = (result.std_dev / result.mean * 100) if result.mean > 0 else 0

        return UncertaintyResult(
            mean=result.mean,
            std_dev=result.std_dev,
            p5=result.p5,
            p50=result.p50,
            p95=result.p95,
            min_value=result.min_value,
            max_value=result.max_value,
            uncertainty_range=f"±{uncertainty_pct:.1f}%",
            coefficient_of_variation=result.coefficient_of_variation,
            iterations=iterations
        )


__all__ = ["UncertaintyEngine"]
