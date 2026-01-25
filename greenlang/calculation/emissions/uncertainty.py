# -*- coding: utf-8 -*-
"""
Uncertainty Quantification

Monte Carlo simulation for propagating uncertainty through calculations.

Uncertainty Sources:
1. Activity Data Uncertainty (measurement/estimation errors)
2. Emission Factor Uncertainty (inherent variability)
3. Model Uncertainty (calculation methodology)

Uses: ISO 14064-1 guidance on uncertainty assessment
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from decimal import Decimal
from greenlang.determinism import FinancialDecimal


@dataclass
class UncertaintyResult:
    """
    Result of uncertainty analysis.

    Attributes:
        mean_kg_co2e: Mean emissions from Monte Carlo simulation
        std_kg_co2e: Standard deviation
        confidence_interval_95: 95% confidence interval (lower, upper)
        confidence_interval_90: 90% confidence interval (lower, upper)
        relative_uncertainty_pct: Relative uncertainty as percentage
        n_simulations: Number of Monte Carlo simulations
        activity_uncertainty_pct: Input activity data uncertainty (%)
        factor_uncertainty_pct: Emission factor uncertainty (%)
    """
    mean_kg_co2e: float
    std_kg_co2e: float
    confidence_interval_95: Tuple[float, float]
    confidence_interval_90: Tuple[float, float]
    relative_uncertainty_pct: float
    n_simulations: int
    activity_uncertainty_pct: float
    factor_uncertainty_pct: float

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'mean_kg_co2e': self.mean_kg_co2e,
            'std_kg_co2e': self.std_kg_co2e,
            'confidence_interval_95': {
                'lower': self.confidence_interval_95[0],
                'upper': self.confidence_interval_95[1],
            },
            'confidence_interval_90': {
                'lower': self.confidence_interval_90[0],
                'upper': self.confidence_interval_90[1],
            },
            'relative_uncertainty_pct': self.relative_uncertainty_pct,
            'n_simulations': self.n_simulations,
            'activity_uncertainty_pct': self.activity_uncertainty_pct,
            'factor_uncertainty_pct': self.factor_uncertainty_pct,
        }


class UncertaintyCalculator:
    """
    Monte Carlo uncertainty quantification.

    DETERMINISTIC: While using random sampling, results are reproducible
    with same random seed (default: seed=42 for reproducibility).

    Method: Error propagation using Monte Carlo simulation
    - Activity data: Normal distribution
    - Emission factors: Log-normal distribution (more realistic for positive values)
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize uncertainty calculator.

        Args:
            random_seed: Random seed for reproducibility (default: 42)
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)

    def propagate_uncertainty(
        self,
        activity_data: float,
        activity_uncertainty_pct: float,
        emission_factor: float,
        factor_uncertainty_pct: float,
        n_simulations: int = 10000,
    ) -> UncertaintyResult:
        """
        Propagate uncertainty through calculation using Monte Carlo.

        Formula: Emissions = Activity × EmissionFactor

        Uncertainty propagation:
        - Activity data: Normal distribution (mean=activity_data, std=activity_data * uncertainty/100)
        - Emission factor: Log-normal distribution (for positive-only values)

        Args:
            activity_data: Activity amount (e.g., 100 gallons)
            activity_uncertainty_pct: Activity uncertainty (e.g., 5 = ±5%)
            emission_factor: Emission factor (e.g., 10.21 kg CO2e/gallon)
            factor_uncertainty_pct: Factor uncertainty (e.g., 10 = ±10%)
            n_simulations: Number of Monte Carlo simulations (default: 10000)

        Returns:
            UncertaintyResult with mean, std, and confidence intervals

        Example:
            >>> from greenlang.calculation.uncertainty import UncertaintyCalculator
            >>> calc = UncertaintyCalculator()
            >>> result = calc.propagate_uncertainty(
            ...     activity_data=100,  # 100 gallons
            ...     activity_uncertainty_pct=5,  # ±5%
            ...     emission_factor=10.21,  # kg CO2e/gallon
            ...     factor_uncertainty_pct=10,  # ±10%
            ...     n_simulations=10000
            ... )
            >>> print(f"Emissions: {result.mean_kg_co2e:.1f} ± {result.std_kg_co2e:.1f} kg CO2e")
            >>> print(f"95% CI: [{result.confidence_interval_95[0]:.1f}, {result.confidence_interval_95[1]:.1f}]")
        """
        # Seed for reproducibility
        np.random.seed(self.random_seed)

        # Activity data: Normal distribution
        activity_std = activity_data * (activity_uncertainty_pct / 100)
        activity_samples = np.random.normal(
            loc=activity_data,
            scale=activity_std,
            size=n_simulations
        )

        # Emission factor: Log-normal distribution
        # Log-normal ensures positive values (emission factors can't be negative)
        factor_std = emission_factor * (factor_uncertainty_pct / 100)

        # Convert to log-normal parameters
        mu = np.log(emission_factor)
        sigma = np.sqrt(np.log(1 + (factor_std / emission_factor) ** 2))

        factor_samples = np.random.lognormal(
            mean=mu,
            sigma=sigma,
            size=n_simulations
        )

        # Calculate emissions for each simulation
        # Emissions = Activity × Factor
        emissions_samples = activity_samples * factor_samples

        # Calculate statistics
        mean_emissions = FinancialDecimal.from_string(np.mean(emissions_samples))
        std_emissions = FinancialDecimal.from_string(np.std(emissions_samples))

        # Confidence intervals
        ci_95_lower, ci_95_upper = np.percentile(emissions_samples, [2.5, 97.5])
        ci_90_lower, ci_90_upper = np.percentile(emissions_samples, [5, 95])

        # Relative uncertainty
        relative_uncertainty = (std_emissions / mean_emissions * 100) if mean_emissions > 0 else 0

        return UncertaintyResult(
            mean_kg_co2e=mean_emissions,
            std_kg_co2e=std_emissions,
            confidence_interval_95=(float(ci_95_lower), float(ci_95_upper)),
            confidence_interval_90=(float(ci_90_lower), float(ci_90_upper)),
            relative_uncertainty_pct=relative_uncertainty,
            n_simulations=n_simulations,
            activity_uncertainty_pct=activity_uncertainty_pct,
            factor_uncertainty_pct=factor_uncertainty_pct,
        )

    def combine_uncertainties(
        self,
        uncertainties: list[float],
        correlation: float = 0.0,
    ) -> float:
        """
        Combine multiple uncertainties (error propagation).

        For independent uncertainties: σ_total = sqrt(σ1² + σ2² + ... + σn²)
        For correlated uncertainties: includes covariance terms

        Args:
            uncertainties: List of uncertainty values (standard deviations)
            correlation: Correlation coefficient between uncertainties (-1 to 1)

        Returns:
            Combined uncertainty (standard deviation)

        Example:
            >>> calc = UncertaintyCalculator()
            >>> # Combine uncertainties from 3 sources
            >>> combined = calc.combine_uncertainties([10.5, 8.2, 5.1])
            >>> print(f"Combined uncertainty: {combined:.2f}")
        """
        if not uncertainties:
            return 0.0

        # Independent uncertainties (root sum of squares)
        if correlation == 0.0:
            variance_sum = sum(u ** 2 for u in uncertainties)
            return np.sqrt(variance_sum)

        # Correlated uncertainties
        # σ_total² = Σσi² + 2ρΣΣσiσj (for i<j)
        n = len(uncertainties)
        variance_sum = sum(u ** 2 for u in uncertainties)

        # Add covariance terms
        covariance_sum = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                covariance_sum += 2 * correlation * uncertainties[i] * uncertainties[j]

        total_variance = variance_sum + covariance_sum
        return np.sqrt(max(0, total_variance))  # Ensure non-negative

    def estimate_minimum_uncertainty(
        self,
        activity_data: float,
        emission_factor: float,
        data_quality_tier: str = 'tier_1',
    ) -> Tuple[float, float]:
        """
        Estimate minimum uncertainty based on data quality tier.

        ISO 14064-1 guidance:
        - Tier 1: National averages (±10-20%)
        - Tier 2: Regional/technology-specific (±5-10%)
        - Tier 3: Facility-specific measurements (±2-5%)

        Args:
            activity_data: Activity amount
            emission_factor: Emission factor
            data_quality_tier: 'tier_1', 'tier_2', or 'tier_3'

        Returns:
            Tuple of (activity_uncertainty_pct, factor_uncertainty_pct)

        Example:
            >>> calc = UncertaintyCalculator()
            >>> activity_unc, factor_unc = calc.estimate_minimum_uncertainty(
            ...     activity_data=100,
            ...     emission_factor=10.21,
            ...     data_quality_tier='tier_2'
            ... )
            >>> print(f"Estimated uncertainties: activity={activity_unc}%, factor={factor_unc}%")
        """
        tier_uncertainties = {
            'tier_1': (15.0, 15.0),  # ±15% for both
            'tier_2': (7.5, 7.5),    # ±7.5% for both
            'tier_3': (3.5, 3.5),    # ±3.5% for both
        }

        return tier_uncertainties.get(data_quality_tier.lower(), (15.0, 15.0))

    def calculate_discrepancy(
        self,
        reported_value: float,
        verified_value: float,
    ) -> dict:
        """
        Calculate discrepancy between reported and verified emissions.

        Used in third-party verification and audits.

        Args:
            reported_value: Reported emissions (kg CO2e)
            verified_value: Verified emissions (kg CO2e)

        Returns:
            Dictionary with discrepancy metrics

        Example:
            >>> calc = UncertaintyCalculator()
            >>> discrepancy = calc.calculate_discrepancy(
            ...     reported_value=1000,
            ...     verified_value=1050
            ... )
            >>> print(f"Discrepancy: {discrepancy['percentage']:.2f}%")
        """
        absolute_diff = verified_value - reported_value
        percentage_diff = (absolute_diff / reported_value * 100) if reported_value > 0 else 0

        return {
            'reported_value': reported_value,
            'verified_value': verified_value,
            'absolute_difference': absolute_diff,
            'percentage_difference': percentage_diff,
            'within_5_percent': abs(percentage_diff) <= 5.0,
            'within_10_percent': abs(percentage_diff) <= 10.0,
            'materiality_threshold': abs(percentage_diff),
        }
