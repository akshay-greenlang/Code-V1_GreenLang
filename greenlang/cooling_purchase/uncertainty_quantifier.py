"""
UncertaintyQuantifierEngine - Quantifies uncertainty in cooling purchase emissions calculations.

This module implements Monte Carlo simulation and analytical error propagation for cooling
emissions calculations. Supports tier-specific uncertainties for electric chillers, absorption
chillers, free cooling, thermal energy storage, and district cooling systems.

Implements IPCC/GHG Protocol uncertainty quantification standards with 10,000 Monte Carlo
iterations (default) and 95% confidence intervals.

Example:
    >>> engine = get_uncertainty_quantifier()
    >>> result = engine.quantify_uncertainty(request)
    >>> assert result.relative_uncertainty_pct < 50.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

# Try numpy for Monte Carlo, fallback to stdlib random
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    import random
    HAS_NUMPY = False

# Import models
from greenlang.cooling_purchase.models import (
    UncertaintyRequest,
    UncertaintyResult,
    CalculationResult,
    DataQualityTier,
    CoolingTechnology,
    GWPSource,
    EmissionGas,
)

# Try config, metrics, provenance (graceful degradation)
try:
    from greenlang.cooling_purchase.config import CoolingPurchaseConfig
except ImportError:
    CoolingPurchaseConfig = None

try:
    from greenlang.cooling_purchase.metrics import CoolingPurchaseMetrics
except ImportError:
    CoolingPurchaseMetrics = None

try:
    from greenlang.cooling_purchase.provenance import CoolingPurchaseProvenance
except ImportError:
    CoolingPurchaseProvenance = None

logger = logging.getLogger(__name__)


class UncertaintyQuantifierEngine:
    """
    Thread-safe singleton for uncertainty quantification in cooling purchase emissions.

    Implements Monte Carlo simulation and analytical error propagation following
    IPCC Good Practice Guidance and GHG Protocol uncertainty requirements.

    Attributes:
        _initialized: Singleton initialization flag
        config: Optional configuration instance
        metrics: Optional metrics instance
        provenance: Optional provenance instance
    """

    _instance: Optional[UncertaintyQuantifierEngine] = None
    _lock = threading.RLock()

    def __new__(cls) -> UncertaintyQuantifierEngine:
        """Create or return singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        """Initialize engine (once per singleton lifecycle)."""
        with self._lock:
            if self._initialized:
                return

            self.config = CoolingPurchaseConfig() if CoolingPurchaseConfig else None
            self.metrics = CoolingPurchaseMetrics() if CoolingPurchaseMetrics else None
            self.provenance = CoolingPurchaseProvenance() if CoolingPurchaseProvenance else None

            # Tier-specific uncertainties (relative %)
            self._tier_uncertainties = {
                DataQualityTier.TIER_1: {
                    "cop": Decimal("0.30"),  # ±30%
                    "grid_ef": Decimal("0.15"),  # ±15%
                    "cooling_output": Decimal("0.05"),  # ±5%
                    "heat_input": Decimal("0.10"),  # ±10%
                    "overall": Decimal("0.40"),  # ~40% combined
                },
                DataQualityTier.TIER_2: {
                    "cop": Decimal("0.15"),  # ±15%
                    "grid_ef": Decimal("0.10"),  # ±10%
                    "cooling_output": Decimal("0.03"),  # ±3%
                    "heat_input": Decimal("0.05"),  # ±5%
                    "overall": Decimal("0.20"),  # ~20% combined
                },
                DataQualityTier.TIER_3: {
                    "cop": Decimal("0.05"),  # ±5%
                    "grid_ef": Decimal("0.05"),  # ±5%
                    "cooling_output": Decimal("0.02"),  # ±2%
                    "heat_input": Decimal("0.03"),  # ±3%
                    "overall": Decimal("0.10"),  # ~10% combined
                },
            }

            # Technology-specific COP uncertainties (additional)
            self._tech_cop_adjustments = {
                CoolingTechnology.ELECTRIC_CHILLER: Decimal("0.00"),
                CoolingTechnology.ABSORPTION_CHILLER: Decimal("0.05"),  # +5% for absorption
                CoolingTechnology.FREE_COOLING: Decimal("0.10"),  # +10% for free cooling
                CoolingTechnology.THERMAL_ENERGY_STORAGE: Decimal("0.03"),  # +3% for TES
                CoolingTechnology.DISTRICT_COOLING: Decimal("0.08"),  # +8% for district
            }

            # Monte Carlo defaults
            self._default_mc_iterations = 10000
            self._default_confidence = Decimal("0.95")

            self._initialized = True
            logger.info("UncertaintyQuantifierEngine initialized (singleton)")

    @classmethod
    def reset(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None

    # -------------------------------------------------------------------------
    # Core Uncertainty Quantification
    # -------------------------------------------------------------------------

    def quantify_uncertainty(
        self,
        request: UncertaintyRequest,
    ) -> UncertaintyResult:
        """
        Quantify uncertainty for cooling purchase emissions calculation.

        Args:
            request: Uncertainty quantification request with calculation result

        Returns:
            Uncertainty result with Monte Carlo samples and statistics

        Raises:
            ValueError: If request is invalid
        """
        start_time = time.time()

        try:
            # Validate request
            if not request.calculation_result:
                raise ValueError("calculation_result is required")

            calc_result = request.calculation_result
            method = request.method or "monte_carlo"
            iterations = request.iterations or self._default_mc_iterations
            confidence = request.confidence_level or self._default_confidence

            logger.info(
                f"Quantifying uncertainty using {method} for "
                f"{calc_result.cooling_technology.value} Tier {calc_result.tier.value}"
            )

            # Route to appropriate method
            if method.lower() == "monte_carlo":
                result = self.run_monte_carlo(
                    calc_result,
                    iterations=iterations,
                    confidence=float(confidence),
                    seed=request.seed,
                )
            elif method.lower() == "analytical":
                result = self.run_analytical(
                    calc_result,
                    confidence=float(confidence),
                )
            else:
                raise ValueError(f"Unknown uncertainty method: {method}")

            # Track metrics
            elapsed = (time.time() - start_time) * 1000
            if self.metrics:
                self.metrics.record_uncertainty_quantification(
                    technology=calc_result.cooling_technology.value,
                    method=method,
                    duration_ms=elapsed,
                    success=True,
                )

            logger.info(
                f"Uncertainty quantification completed in {elapsed:.2f}ms: "
                f"±{result.relative_uncertainty_pct:.2f}%"
            )

            return result

        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            if self.metrics:
                self.metrics.record_uncertainty_quantification(
                    technology=getattr(request.calculation_result, "cooling_technology", "unknown"),
                    method=request.method or "monte_carlo",
                    duration_ms=elapsed,
                    success=False,
                )
            logger.error(f"Uncertainty quantification failed: {e}", exc_info=True)
            raise

    def run_monte_carlo(
        self,
        calc_result: CalculationResult,
        iterations: int = 10000,
        confidence: float = 0.95,
        seed: Optional[int] = None,
    ) -> UncertaintyResult:
        """
        Run Monte Carlo simulation for uncertainty quantification.

        Args:
            calc_result: Calculation result to quantify uncertainty for
            iterations: Number of Monte Carlo iterations (default 10,000)
            confidence: Confidence level (default 0.95 for 95% CI)
            seed: Random seed for reproducibility (optional)

        Returns:
            Uncertainty result with samples and statistics
        """
        # Set random seed
        if seed is not None:
            if HAS_NUMPY:
                np.random.seed(seed)
            else:
                random.seed(seed)

        # Route to technology-specific quantification
        technology = calc_result.cooling_technology

        if technology == CoolingTechnology.ELECTRIC_CHILLER:
            return self.quantify_electric_chiller_uncertainty(
                calc_result, calc_result.tier, iterations
            )
        elif technology == CoolingTechnology.ABSORPTION_CHILLER:
            return self.quantify_absorption_uncertainty(
                calc_result, calc_result.tier, iterations
            )
        elif technology == CoolingTechnology.FREE_COOLING:
            return self.quantify_free_cooling_uncertainty(
                calc_result, calc_result.tier, iterations
            )
        elif technology == CoolingTechnology.THERMAL_ENERGY_STORAGE:
            return self.quantify_tes_uncertainty(
                calc_result, calc_result.tier, iterations
            )
        elif technology == CoolingTechnology.DISTRICT_COOLING:
            return self.quantify_district_cooling_uncertainty(
                calc_result, calc_result.tier, iterations
            )
        else:
            raise ValueError(f"Unsupported cooling technology: {technology}")

    def run_analytical(
        self,
        calc_result: CalculationResult,
        confidence: float = 0.95,
    ) -> UncertaintyResult:
        """
        Run analytical error propagation for uncertainty quantification.

        Uses first-order Taylor series approximation:
        σ_E/E = sqrt((σ_x1/x1)² + (σ_x2/x2)² + ...)

        Args:
            calc_result: Calculation result to quantify uncertainty for
            confidence: Confidence level (default 0.95)

        Returns:
            Uncertainty result with analytical uncertainty estimate
        """
        tier = calc_result.tier
        technology = calc_result.cooling_technology

        # Get tier uncertainties
        tier_unc = self._tier_uncertainties[tier]

        # Build variable uncertainties list
        variables: List[Tuple[Decimal, Decimal]] = []

        # Cooling output (if applicable)
        if hasattr(calc_result, "cooling_output_kwh") and calc_result.cooling_output_kwh:
            cooling_unc = tier_unc["cooling_output"]
            variables.append((calc_result.cooling_output_kwh, cooling_unc))

        # COP uncertainty (technology-adjusted)
        cop_unc = tier_unc["cop"] + self._tech_cop_adjustments.get(technology, Decimal("0"))
        if hasattr(calc_result, "cop_used") and calc_result.cop_used:
            variables.append((calc_result.cop_used, cop_unc))

        # Grid EF uncertainty (for electric technologies)
        if technology in [CoolingTechnology.ELECTRIC_CHILLER, CoolingTechnology.DISTRICT_COOLING]:
            grid_ef_unc = tier_unc["grid_ef"]
            # Assume grid EF ~ 0.5 kgCO2e/kWh (placeholder)
            variables.append((Decimal("0.5"), grid_ef_unc))

        # Heat input uncertainty (for absorption chillers)
        if technology == CoolingTechnology.ABSORPTION_CHILLER:
            heat_unc = tier_unc["heat_input"]
            variables.append((Decimal("1.0"), heat_unc))

        # Analytical error propagation
        relative_unc = self.analytical_error_propagation(variables)

        # Calculate absolute uncertainty
        total_emissions = calc_result.total_emissions_co2e
        absolute_unc = total_emissions * relative_unc

        # Confidence interval (assume normal distribution)
        # For 95% CI, z = 1.96
        z_score = Decimal("1.96") if confidence >= 0.95 else Decimal("1.645")
        ci_half_width = absolute_unc * z_score

        ci_lower = total_emissions - ci_half_width
        ci_upper = total_emissions + ci_half_width

        # Create result
        result = UncertaintyResult(
            uncertainty_id=str(uuid.uuid4()),
            calculation_id=calc_result.calculation_id,
            method="analytical",
            iterations=0,
            confidence_level=Decimal(str(confidence)),
            mean_emissions_co2e=total_emissions,
            std_dev_co2e=absolute_unc / z_score,  # Back-calculate std dev
            relative_uncertainty_pct=relative_unc * Decimal("100"),
            confidence_interval_lower=ci_lower,
            confidence_interval_upper=ci_upper,
            percentile_5=total_emissions - ci_half_width * Decimal("1.2"),
            percentile_25=total_emissions - ci_half_width * Decimal("0.6"),
            percentile_50=total_emissions,
            percentile_75=total_emissions + ci_half_width * Decimal("0.6"),
            percentile_95=total_emissions + ci_half_width * Decimal("1.2"),
            coefficient_of_variation=relative_unc,
            dominant_uncertainty_source="combined",
            parameter_uncertainties={
                "cooling_output": float(tier_unc["cooling_output"] * 100),
                "cop": float(cop_unc * 100),
                "grid_ef": float(tier_unc.get("grid_ef", Decimal("0")) * 100),
            },
            provenance_hash=self._calculate_provenance_hash(calc_result, "analytical"),
            timestamp=datetime.now(timezone.utc),
        )

        return result

    # -------------------------------------------------------------------------
    # Technology-Specific Uncertainty Quantification
    # -------------------------------------------------------------------------

    def quantify_electric_chiller_uncertainty(
        self,
        result: CalculationResult,
        tier: DataQualityTier,
        iterations: int,
    ) -> UncertaintyResult:
        """
        Quantify uncertainty for electric chiller emissions.

        Formula: E = (Cooling / COP) × Grid_EF + Auxiliary × Grid_EF

        Args:
            result: Calculation result for electric chiller
            tier: Data quality tier
            iterations: Number of Monte Carlo iterations

        Returns:
            Uncertainty result with Monte Carlo statistics
        """
        samples: List[Decimal] = []

        # Get distributions
        cooling_dist = self.get_cooling_output_distribution(
            result.cooling_output_kwh or Decimal("0"), tier
        )
        cop_dist = self.get_cop_distribution(
            result.cop_used or Decimal("3.5"), tier, CoolingTechnology.ELECTRIC_CHILLER
        )
        # Assume grid EF ~ 0.5 kgCO2e/kWh
        grid_ef_dist = self.get_grid_ef_distribution(Decimal("0.5"), tier)
        aux_dist = self.get_auxiliary_distribution(
            result.auxiliary_load_pct or Decimal("0.05")
        )

        # Monte Carlo simulation
        for _ in range(iterations):
            cooling = self._sample_normal(
                cooling_dist["mean"], cooling_dist["std_dev"]
            )
            cop = self._sample_normal(cop_dist["mean"], cop_dist["std_dev"])
            grid_ef = self._sample_normal(grid_ef_dist["mean"], grid_ef_dist["std_dev"])
            aux_pct = self._sample_normal(aux_dist["mean"], aux_dist["std_dev"])

            # Avoid division by zero
            if cop <= Decimal("0"):
                cop = Decimal("0.01")

            # Calculate emissions
            electricity = cooling / cop
            auxiliary_kwh = electricity * aux_pct
            total_kwh = electricity + auxiliary_kwh
            emissions = total_kwh * grid_ef

            samples.append(emissions)

        # Compute statistics
        stats = self.compute_statistics(samples)

        # Sensitivity analysis
        sensitivity = self.run_sensitivity_analysis(
            result,
            ["cooling_output", "cop", "grid_ef", "auxiliary"],
        )
        dominant = self.identify_dominant_uncertainty(sensitivity)

        # Create result
        uncertainty_result = UncertaintyResult(
            uncertainty_id=str(uuid.uuid4()),
            calculation_id=result.calculation_id,
            method="monte_carlo",
            iterations=iterations,
            confidence_level=Decimal("0.95"),
            mean_emissions_co2e=stats["mean"],
            std_dev_co2e=stats["std_dev"],
            relative_uncertainty_pct=(stats["std_dev"] / stats["mean"] * Decimal("100"))
            if stats["mean"] > 0
            else Decimal("0"),
            confidence_interval_lower=stats["ci_lower"],
            confidence_interval_upper=stats["ci_upper"],
            percentile_5=stats["p5"],
            percentile_25=stats["p25"],
            percentile_50=stats["p50"],
            percentile_75=stats["p75"],
            percentile_95=stats["p95"],
            coefficient_of_variation=stats["cv"],
            dominant_uncertainty_source=dominant,
            parameter_uncertainties=sensitivity,
            provenance_hash=self._calculate_provenance_hash(result, "monte_carlo"),
            timestamp=datetime.now(timezone.utc),
        )

        return uncertainty_result

    def quantify_absorption_uncertainty(
        self,
        result: CalculationResult,
        tier: DataQualityTier,
        iterations: int,
    ) -> UncertaintyResult:
        """
        Quantify uncertainty for absorption chiller emissions.

        Formula: E = Heat_Input × Fuel_EF (+ Electricity for auxiliaries)

        Args:
            result: Calculation result for absorption chiller
            tier: Data quality tier
            iterations: Number of Monte Carlo iterations

        Returns:
            Uncertainty result with Monte Carlo statistics
        """
        samples: List[Decimal] = []

        # Get distributions
        heat_dist = {
            "mean": result.cooling_output_kwh or Decimal("0"),
            "std_dev": (result.cooling_output_kwh or Decimal("0"))
            * self._tier_uncertainties[tier]["heat_input"],
        }
        cop_dist = self.get_cop_distribution(
            result.cop_used or Decimal("0.7"), tier, CoolingTechnology.ABSORPTION_CHILLER
        )
        # Assume fuel EF ~ 0.2 kgCO2e/kWh (natural gas)
        fuel_ef = Decimal("0.2")
        fuel_ef_unc = Decimal("0.05")  # ±5%

        # Monte Carlo simulation
        for _ in range(iterations):
            cooling = self._sample_normal(heat_dist["mean"], heat_dist["std_dev"])
            cop = self._sample_normal(cop_dist["mean"], cop_dist["std_dev"])
            ef = self._sample_normal(fuel_ef, fuel_ef * fuel_ef_unc)

            if cop <= Decimal("0"):
                cop = Decimal("0.01")

            heat_input = cooling / cop
            emissions = heat_input * ef

            samples.append(emissions)

        # Compute statistics
        stats = self.compute_statistics(samples)

        # Sensitivity analysis
        sensitivity = self.run_sensitivity_analysis(
            result,
            ["heat_input", "cop", "fuel_ef"],
        )
        dominant = self.identify_dominant_uncertainty(sensitivity)

        uncertainty_result = UncertaintyResult(
            uncertainty_id=str(uuid.uuid4()),
            calculation_id=result.calculation_id,
            method="monte_carlo",
            iterations=iterations,
            confidence_level=Decimal("0.95"),
            mean_emissions_co2e=stats["mean"],
            std_dev_co2e=stats["std_dev"],
            relative_uncertainty_pct=(stats["std_dev"] / stats["mean"] * Decimal("100"))
            if stats["mean"] > 0
            else Decimal("0"),
            confidence_interval_lower=stats["ci_lower"],
            confidence_interval_upper=stats["ci_upper"],
            percentile_5=stats["p5"],
            percentile_25=stats["p25"],
            percentile_50=stats["p50"],
            percentile_75=stats["p75"],
            percentile_95=stats["p95"],
            coefficient_of_variation=stats["cv"],
            dominant_uncertainty_source=dominant,
            parameter_uncertainties=sensitivity,
            provenance_hash=self._calculate_provenance_hash(result, "monte_carlo"),
            timestamp=datetime.now(timezone.utc),
        )

        return uncertainty_result

    def quantify_free_cooling_uncertainty(
        self,
        result: CalculationResult,
        tier: DataQualityTier,
        iterations: int,
    ) -> UncertaintyResult:
        """
        Quantify uncertainty for free cooling emissions.

        Formula: E = Parasitic_Electricity × Grid_EF (pumps, fans)

        Args:
            result: Calculation result for free cooling
            tier: Data quality tier
            iterations: Number of Monte Carlo iterations

        Returns:
            Uncertainty result with Monte Carlo statistics
        """
        samples: List[Decimal] = []

        # Get distributions
        cooling_dist = self.get_cooling_output_distribution(
            result.cooling_output_kwh or Decimal("0"), tier
        )
        parasitic_dist = self.get_parasitic_distribution(
            result.parasitic_load_ratio or Decimal("0.02")
        )
        grid_ef_dist = self.get_grid_ef_distribution(Decimal("0.5"), tier)

        # Monte Carlo simulation
        for _ in range(iterations):
            cooling = self._sample_normal(
                cooling_dist["mean"], cooling_dist["std_dev"]
            )
            parasitic_ratio = self._sample_normal(
                parasitic_dist["mean"], parasitic_dist["std_dev"]
            )
            grid_ef = self._sample_normal(grid_ef_dist["mean"], grid_ef_dist["std_dev"])

            parasitic_kwh = cooling * parasitic_ratio
            emissions = parasitic_kwh * grid_ef

            samples.append(emissions)

        # Compute statistics
        stats = self.compute_statistics(samples)

        sensitivity = self.run_sensitivity_analysis(
            result,
            ["cooling_output", "parasitic_load", "grid_ef"],
        )
        dominant = self.identify_dominant_uncertainty(sensitivity)

        uncertainty_result = UncertaintyResult(
            uncertainty_id=str(uuid.uuid4()),
            calculation_id=result.calculation_id,
            method="monte_carlo",
            iterations=iterations,
            confidence_level=Decimal("0.95"),
            mean_emissions_co2e=stats["mean"],
            std_dev_co2e=stats["std_dev"],
            relative_uncertainty_pct=(stats["std_dev"] / stats["mean"] * Decimal("100"))
            if stats["mean"] > 0
            else Decimal("0"),
            confidence_interval_lower=stats["ci_lower"],
            confidence_interval_upper=stats["ci_upper"],
            percentile_5=stats["p5"],
            percentile_25=stats["p25"],
            percentile_50=stats["p50"],
            percentile_75=stats["p75"],
            percentile_95=stats["p95"],
            coefficient_of_variation=stats["cv"],
            dominant_uncertainty_source=dominant,
            parameter_uncertainties=sensitivity,
            provenance_hash=self._calculate_provenance_hash(result, "monte_carlo"),
            timestamp=datetime.now(timezone.utc),
        )

        return uncertainty_result

    def quantify_tes_uncertainty(
        self,
        result: CalculationResult,
        tier: DataQualityTier,
        iterations: int,
    ) -> UncertaintyResult:
        """
        Quantify uncertainty for thermal energy storage emissions.

        Formula: E = (Cooling / (COP × RTE)) × Grid_EF + Losses × Grid_EF

        Args:
            result: Calculation result for TES
            tier: Data quality tier
            iterations: Number of Monte Carlo iterations

        Returns:
            Uncertainty result with Monte Carlo statistics
        """
        samples: List[Decimal] = []

        # Get distributions
        cooling_dist = self.get_cooling_output_distribution(
            result.cooling_output_kwh or Decimal("0"), tier
        )
        cop_dist = self.get_cop_distribution(
            result.cop_used or Decimal("3.5"), tier, CoolingTechnology.THERMAL_ENERGY_STORAGE
        )
        rte_dist = self.get_round_trip_efficiency_distribution(
            Decimal("0.90"), "ice"  # Assume ice storage
        )
        grid_ef_dist = self.get_grid_ef_distribution(Decimal("0.5"), tier)

        # Monte Carlo simulation
        for _ in range(iterations):
            cooling = self._sample_normal(
                cooling_dist["mean"], cooling_dist["std_dev"]
            )
            cop = self._sample_normal(cop_dist["mean"], cop_dist["std_dev"])
            rte = self._sample_normal(rte_dist["mean"], rte_dist["std_dev"])
            grid_ef = self._sample_normal(grid_ef_dist["mean"], grid_ef_dist["std_dev"])

            if cop <= Decimal("0"):
                cop = Decimal("0.01")
            if rte <= Decimal("0"):
                rte = Decimal("0.01")

            electricity = cooling / (cop * rte)
            emissions = electricity * grid_ef

            samples.append(emissions)

        # Compute statistics
        stats = self.compute_statistics(samples)

        sensitivity = self.run_sensitivity_analysis(
            result,
            ["cooling_output", "cop", "round_trip_efficiency", "grid_ef"],
        )
        dominant = self.identify_dominant_uncertainty(sensitivity)

        uncertainty_result = UncertaintyResult(
            uncertainty_id=str(uuid.uuid4()),
            calculation_id=result.calculation_id,
            method="monte_carlo",
            iterations=iterations,
            confidence_level=Decimal("0.95"),
            mean_emissions_co2e=stats["mean"],
            std_dev_co2e=stats["std_dev"],
            relative_uncertainty_pct=(stats["std_dev"] / stats["mean"] * Decimal("100"))
            if stats["mean"] > 0
            else Decimal("0"),
            confidence_interval_lower=stats["ci_lower"],
            confidence_interval_upper=stats["ci_upper"],
            percentile_5=stats["p5"],
            percentile_25=stats["p25"],
            percentile_50=stats["p50"],
            percentile_75=stats["p75"],
            percentile_95=stats["p95"],
            coefficient_of_variation=stats["cv"],
            dominant_uncertainty_source=dominant,
            parameter_uncertainties=sensitivity,
            provenance_hash=self._calculate_provenance_hash(result, "monte_carlo"),
            timestamp=datetime.now(timezone.utc),
        )

        return uncertainty_result

    def quantify_district_cooling_uncertainty(
        self,
        result: CalculationResult,
        tier: DataQualityTier,
        iterations: int,
    ) -> UncertaintyResult:
        """
        Quantify uncertainty for district cooling emissions.

        Formula: E = Cooling_Delivered × District_EF + Distribution_Losses × EF

        Args:
            result: Calculation result for district cooling
            tier: Data quality tier
            iterations: Number of Monte Carlo iterations

        Returns:
            Uncertainty result with Monte Carlo statistics
        """
        samples: List[Decimal] = []

        # Get distributions
        cooling_dist = self.get_cooling_output_distribution(
            result.cooling_output_kwh or Decimal("0"), tier
        )
        district_ef = Decimal("0.15")  # Assume district EF ~ 0.15 kgCO2e/kWh
        district_ef_unc = Decimal("0.10")  # ±10%
        loss_dist = self.get_distribution_loss_distribution(Decimal("0.05"))

        # Monte Carlo simulation
        for _ in range(iterations):
            cooling = self._sample_normal(
                cooling_dist["mean"], cooling_dist["std_dev"]
            )
            ef = self._sample_normal(district_ef, district_ef * district_ef_unc)
            loss_pct = self._sample_normal(loss_dist["mean"], loss_dist["std_dev"])

            cooling_with_losses = cooling * (Decimal("1") + loss_pct)
            emissions = cooling_with_losses * ef

            samples.append(emissions)

        # Compute statistics
        stats = self.compute_statistics(samples)

        sensitivity = self.run_sensitivity_analysis(
            result,
            ["cooling_delivered", "district_ef", "distribution_losses"],
        )
        dominant = self.identify_dominant_uncertainty(sensitivity)

        uncertainty_result = UncertaintyResult(
            uncertainty_id=str(uuid.uuid4()),
            calculation_id=result.calculation_id,
            method="monte_carlo",
            iterations=iterations,
            confidence_level=Decimal("0.95"),
            mean_emissions_co2e=stats["mean"],
            std_dev_co2e=stats["std_dev"],
            relative_uncertainty_pct=(stats["std_dev"] / stats["mean"] * Decimal("100"))
            if stats["mean"] > 0
            else Decimal("0"),
            confidence_interval_lower=stats["ci_lower"],
            confidence_interval_upper=stats["ci_upper"],
            percentile_5=stats["p5"],
            percentile_25=stats["p25"],
            percentile_50=stats["p50"],
            percentile_75=stats["p75"],
            percentile_95=stats["p95"],
            coefficient_of_variation=stats["cv"],
            dominant_uncertainty_source=dominant,
            parameter_uncertainties=sensitivity,
            provenance_hash=self._calculate_provenance_hash(result, "monte_carlo"),
            timestamp=datetime.now(timezone.utc),
        )

        return uncertainty_result

    # -------------------------------------------------------------------------
    # Distribution Methods
    # -------------------------------------------------------------------------

    def get_cop_distribution(
        self,
        cop: Decimal,
        tier: DataQualityTier,
        technology: CoolingTechnology,
    ) -> Dict[str, Decimal]:
        """
        Get COP uncertainty distribution (normal).

        Args:
            cop: Mean COP value
            tier: Data quality tier
            technology: Cooling technology type

        Returns:
            Dict with mean and std_dev
        """
        base_unc = self._tier_uncertainties[tier]["cop"]
        tech_adj = self._tech_cop_adjustments.get(technology, Decimal("0"))
        total_unc = base_unc + tech_adj

        return {
            "mean": cop,
            "std_dev": cop * total_unc,
        }

    def get_grid_ef_distribution(
        self,
        ef: Decimal,
        tier: DataQualityTier,
    ) -> Dict[str, Decimal]:
        """
        Get grid emission factor uncertainty distribution (normal).

        Args:
            ef: Mean grid EF (kgCO2e/kWh)
            tier: Data quality tier

        Returns:
            Dict with mean and std_dev
        """
        unc = self._tier_uncertainties[tier]["grid_ef"]

        return {
            "mean": ef,
            "std_dev": ef * unc,
        }

    def get_cooling_output_distribution(
        self,
        output: Decimal,
        tier: DataQualityTier,
    ) -> Dict[str, Decimal]:
        """
        Get cooling output uncertainty distribution (normal).

        Args:
            output: Mean cooling output (kWh)
            tier: Data quality tier

        Returns:
            Dict with mean and std_dev
        """
        unc = self._tier_uncertainties[tier]["cooling_output"]

        return {
            "mean": output,
            "std_dev": output * unc,
        }

    def get_auxiliary_distribution(self, pct: Decimal) -> Dict[str, Decimal]:
        """
        Get auxiliary load percentage uncertainty distribution.

        Args:
            pct: Mean auxiliary load percentage (e.g., 0.05 for 5%)

        Returns:
            Dict with mean and std_dev
        """
        # Assume ±20% uncertainty on auxiliary loads
        return {
            "mean": pct,
            "std_dev": pct * Decimal("0.20"),
        }

    def get_parasitic_distribution(self, ratio: Decimal) -> Dict[str, Decimal]:
        """
        Get parasitic load ratio uncertainty distribution.

        Args:
            ratio: Mean parasitic load ratio (e.g., 0.02 for 2%)

        Returns:
            Dict with mean and std_dev
        """
        # Assume ±30% uncertainty on parasitic loads
        return {
            "mean": ratio,
            "std_dev": ratio * Decimal("0.30"),
        }

    def get_distribution_loss_distribution(self, loss: Decimal) -> Dict[str, Decimal]:
        """
        Get distribution loss percentage uncertainty distribution.

        Args:
            loss: Mean distribution loss (e.g., 0.05 for 5%)

        Returns:
            Dict with mean and std_dev
        """
        # Assume ±25% uncertainty on distribution losses
        return {
            "mean": loss,
            "std_dev": loss * Decimal("0.25"),
        }

    def get_round_trip_efficiency_distribution(
        self,
        rte: Decimal,
        tes_type: str,
    ) -> Dict[str, Decimal]:
        """
        Get round-trip efficiency uncertainty distribution for TES.

        Args:
            rte: Mean round-trip efficiency (e.g., 0.90 for 90%)
            tes_type: TES type (ice, chilled_water, pcm)

        Returns:
            Dict with mean and std_dev
        """
        # TES-type specific uncertainties
        tes_uncertainties = {
            "ice": Decimal("0.05"),  # ±5% for ice storage
            "chilled_water": Decimal("0.03"),  # ±3% for chilled water
            "pcm": Decimal("0.08"),  # ±8% for PCM
        }

        unc = tes_uncertainties.get(tes_type, Decimal("0.05"))

        return {
            "mean": rte,
            "std_dev": rte * unc,
        }

    # -------------------------------------------------------------------------
    # Statistics Methods
    # -------------------------------------------------------------------------

    def compute_statistics(self, samples: List[Decimal]) -> Dict[str, Decimal]:
        """
        Compute statistical summary of Monte Carlo samples.

        Args:
            samples: List of Monte Carlo sample emissions

        Returns:
            Dict with mean, std, percentiles, CI, CV
        """
        if not samples:
            return {
                "mean": Decimal("0"),
                "std_dev": Decimal("0"),
                "p5": Decimal("0"),
                "p25": Decimal("0"),
                "p50": Decimal("0"),
                "p75": Decimal("0"),
                "p95": Decimal("0"),
                "ci_lower": Decimal("0"),
                "ci_upper": Decimal("0"),
                "cv": Decimal("0"),
            }

        # Sort samples
        sorted_samples = sorted(samples)
        n = len(sorted_samples)

        # Mean
        mean = sum(sorted_samples) / Decimal(n)

        # Standard deviation
        variance = sum((x - mean) ** 2 for x in sorted_samples) / Decimal(n - 1 if n > 1 else 1)
        std_dev = Decimal(math.sqrt(float(variance)))

        # Percentiles
        p5 = sorted_samples[int(n * 0.05)]
        p25 = sorted_samples[int(n * 0.25)]
        p50 = sorted_samples[int(n * 0.50)]
        p75 = sorted_samples[int(n * 0.75)]
        p95 = sorted_samples[int(n * 0.95)]

        # 95% Confidence interval
        ci_lower, ci_upper = self.compute_confidence_interval(sorted_samples, 0.95)

        # Coefficient of variation
        cv = self.compute_coefficient_of_variation(mean, std_dev)

        return {
            "mean": mean,
            "std_dev": std_dev,
            "p5": p5,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p95": p95,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "cv": cv,
        }

    def compute_confidence_interval(
        self,
        samples: List[Decimal],
        level: float,
    ) -> Tuple[Decimal, Decimal]:
        """
        Compute confidence interval from samples.

        Args:
            samples: Sorted Monte Carlo samples
            level: Confidence level (e.g., 0.95 for 95% CI)

        Returns:
            Tuple of (lower, upper) bounds
        """
        if not samples:
            return (Decimal("0"), Decimal("0"))

        n = len(samples)
        alpha = 1.0 - level
        lower_idx = int(n * alpha / 2.0)
        upper_idx = int(n * (1.0 - alpha / 2.0))

        lower_idx = max(0, min(lower_idx, n - 1))
        upper_idx = max(0, min(upper_idx, n - 1))

        return (samples[lower_idx], samples[upper_idx])

    def compute_coefficient_of_variation(
        self,
        mean: Decimal,
        std: Decimal,
    ) -> Decimal:
        """
        Compute coefficient of variation (CV = std / mean).

        Args:
            mean: Mean value
            std: Standard deviation

        Returns:
            Coefficient of variation
        """
        if mean == Decimal("0"):
            return Decimal("0")

        return std / mean

    # -------------------------------------------------------------------------
    # Analytical Error Propagation
    # -------------------------------------------------------------------------

    def analytical_error_propagation(
        self,
        variables: List[Tuple[Decimal, Decimal]],
    ) -> Decimal:
        """
        Analytical error propagation using first-order Taylor series.

        For E = f(x1, x2, ..., xn):
        σ_E/E = sqrt((σ_x1/x1)² + (σ_x2/x2)² + ...)

        Args:
            variables: List of (value, relative_uncertainty) tuples

        Returns:
            Combined relative uncertainty
        """
        squared_sum = Decimal("0")

        for value, rel_unc in variables:
            if value != Decimal("0"):
                squared_sum += rel_unc ** 2

        return Decimal(math.sqrt(float(squared_sum)))

    def compute_combined_uncertainty(
        self,
        uncertainties: List[Decimal],
    ) -> Decimal:
        """
        Compute combined uncertainty from independent sources.

        Args:
            uncertainties: List of relative uncertainties

        Returns:
            Combined relative uncertainty (quadrature sum)
        """
        squared_sum = sum(u ** 2 for u in uncertainties)
        return Decimal(math.sqrt(float(squared_sum)))

    # -------------------------------------------------------------------------
    # Sensitivity Analysis
    # -------------------------------------------------------------------------

    def run_sensitivity_analysis(
        self,
        result: CalculationResult,
        parameters: List[str],
    ) -> Dict[str, float]:
        """
        Run sensitivity analysis on parameters.

        Args:
            result: Calculation result
            parameters: List of parameter names to analyze

        Returns:
            Dict mapping parameter to relative uncertainty contribution (%)
        """
        tier = result.tier
        technology = result.cooling_technology

        sensitivity: Dict[str, float] = {}

        for param in parameters:
            if param == "cooling_output":
                unc = self._tier_uncertainties[tier]["cooling_output"]
            elif param == "cop":
                base = self._tier_uncertainties[tier]["cop"]
                adj = self._tech_cop_adjustments.get(technology, Decimal("0"))
                unc = base + adj
            elif param == "grid_ef":
                unc = self._tier_uncertainties[tier].get("grid_ef", Decimal("0.10"))
            elif param == "heat_input":
                unc = self._tier_uncertainties[tier].get("heat_input", Decimal("0.05"))
            elif param in ["auxiliary", "parasitic_load"]:
                unc = Decimal("0.20")
            elif param in ["round_trip_efficiency", "distribution_losses"]:
                unc = Decimal("0.05")
            elif param in ["district_ef", "fuel_ef"]:
                unc = Decimal("0.10")
            else:
                unc = Decimal("0.05")  # Default

            sensitivity[param] = float(unc * 100)

        return sensitivity

    def identify_dominant_uncertainty(
        self,
        param_uncertainties: Dict[str, float],
    ) -> str:
        """
        Identify the dominant uncertainty source.

        Args:
            param_uncertainties: Dict of parameter uncertainties (%)

        Returns:
            Name of parameter with highest uncertainty
        """
        if not param_uncertainties:
            return "unknown"

        return max(param_uncertainties.items(), key=lambda x: x[1])[0]

    # -------------------------------------------------------------------------
    # Cooling-Specific Uncertainty Methods
    # -------------------------------------------------------------------------

    def get_iplv_weighting_uncertainty(self) -> Decimal:
        """
        Get uncertainty in IPLV weighting factors.

        Returns:
            Relative uncertainty in IPLV (±5% typical)
        """
        return Decimal("0.05")

    def get_condenser_fouling_uncertainty(self) -> Decimal:
        """
        Get uncertainty from condenser fouling impact on COP.

        Returns:
            Relative uncertainty from fouling (±8% typical)
        """
        return Decimal("0.08")

    def get_seasonal_cop_variation(self, source: str) -> Decimal:
        """
        Get seasonal COP variation uncertainty.

        Args:
            source: Data source (manufacturer, measured, estimated)

        Returns:
            Relative uncertainty from seasonal variation
        """
        variations = {
            "manufacturer": Decimal("0.15"),  # ±15% for rated values
            "measured": Decimal("0.08"),  # ±8% for metered data
            "estimated": Decimal("0.25"),  # ±25% for estimates
        }

        return variations.get(source, Decimal("0.15"))

    def get_temporal_grid_ef_uncertainty(self) -> Decimal:
        """
        Get temporal variation in grid emission factor.

        Returns:
            Relative uncertainty from grid EF temporal variation (±10% typical)
        """
        return Decimal("0.10")

    def get_heat_input_measurement_uncertainty(self, tier: DataQualityTier) -> Decimal:
        """
        Get heat input measurement uncertainty (for absorption chillers).

        Args:
            tier: Data quality tier

        Returns:
            Relative uncertainty in heat input measurement
        """
        return self._tier_uncertainties[tier]["heat_input"]

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _sample_normal(self, mean: Decimal, std_dev: Decimal) -> Decimal:
        """
        Sample from normal distribution.

        Args:
            mean: Mean value
            std_dev: Standard deviation

        Returns:
            Sampled value (non-negative)
        """
        if HAS_NUMPY:
            sample = float(np.random.normal(float(mean), float(std_dev)))
        else:
            # Box-Muller transform
            u1 = random.random()
            u2 = random.random()
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
            sample = float(mean) + float(std_dev) * z

        # Ensure non-negative
        return max(Decimal("0"), Decimal(str(sample)))

    def _calculate_provenance_hash(
        self,
        result: CalculationResult,
        method: str,
    ) -> str:
        """
        Calculate SHA-256 provenance hash.

        Args:
            result: Calculation result
            method: Uncertainty method

        Returns:
            SHA-256 hash (hex)
        """
        data = {
            "calculation_id": result.calculation_id,
            "method": method,
            "tier": result.tier.value,
            "technology": result.cooling_technology.value,
            "total_emissions": str(result.total_emissions_co2e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()


# -----------------------------------------------------------------------------
# Module-Level Functions
# -----------------------------------------------------------------------------

_global_engine: Optional[UncertaintyQuantifierEngine] = None
_global_lock = threading.RLock()


def get_uncertainty_quantifier() -> UncertaintyQuantifierEngine:
    """
    Get or create global UncertaintyQuantifierEngine instance.

    Returns:
        Singleton UncertaintyQuantifierEngine instance
    """
    global _global_engine

    with _global_lock:
        if _global_engine is None:
            _global_engine = UncertaintyQuantifierEngine()
        return _global_engine


def reset_uncertainty_quantifier() -> None:
    """Reset global UncertaintyQuantifierEngine instance (for testing)."""
    global _global_engine

    with _global_lock:
        _global_engine = None
        UncertaintyQuantifierEngine.reset()
