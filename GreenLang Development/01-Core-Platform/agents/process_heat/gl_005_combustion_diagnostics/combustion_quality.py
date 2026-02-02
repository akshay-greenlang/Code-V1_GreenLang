# -*- coding: utf-8 -*-
"""
GL-005 Combustion Quality Index (CQI) Calculator
================================================

This module implements the proprietary Combustion Quality Index (CQI) calculation
engine for the GL-005 COMBUSENSE agent. The CQI provides a single, normalized
score (0-100) that characterizes overall combustion quality.

CQI Components:
    - Oxygen (O2): Measures excess air control
    - Carbon Monoxide (CO): Indicates incomplete combustion
    - Carbon Dioxide (CO2): Validates carbon balance
    - Nitrogen Oxides (NOx): Environmental compliance indicator
    - Combustibles: Unburned fuel detection

ZERO-HALLUCINATION GUARANTEE:
    All calculations are deterministic using documented formulas.
    No LLM/AI is used in the calculation path.
    Full provenance tracking with SHA-256 hashes.

Author: GreenLang Process Heat Team
Version: 1.0.0
Status: Production Ready
"""

import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from greenlang.agents.process_heat.gl_005_combustion_diagnostics.config import (
    CQIConfig,
    CQIThresholds,
    CQIWeights,
)
from greenlang.agents.process_heat.gl_005_combustion_diagnostics.schemas import (
    AnalysisStatus,
    CQIComponentScore,
    CQIRating,
    CQIResult,
    FlueGasReading,
    TrendDirection,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Standard atmospheric oxygen percentage
ATMOSPHERIC_O2_PCT = 20.95

# Minimum oxygen for safe calculations
MIN_SAFE_O2_PCT = 0.5

# CO2 reference for natural gas (theoretical maximum)
NATURAL_GAS_MAX_CO2_PCT = 11.8


# =============================================================================
# COMBUSTION QUALITY INDEX CALCULATOR
# =============================================================================

class CombustionQualityCalculator:
    """
    Combustion Quality Index (CQI) Calculator.

    This class calculates the proprietary CQI metric that provides a single
    normalized score (0-100) representing overall combustion quality.

    The CQI is calculated using a weighted combination of component scores,
    where each component is normalized based on optimal ranges and thresholds.

    DETERMINISTIC: All calculations use documented formulas with no randomness.
    AUDITABLE: Full calculation trace captured for compliance reporting.

    Example:
        >>> config = CQIConfig()
        >>> calculator = CombustionQualityCalculator(config)
        >>> result = calculator.calculate(flue_gas_reading)
        >>> print(f"CQI Score: {result.cqi_score}")
    """

    def __init__(self, config: CQIConfig) -> None:
        """
        Initialize CQI calculator.

        Args:
            config: CQI configuration with weights and thresholds
        """
        self.config = config
        self.weights = config.weights
        self.thresholds = config.thresholds
        self._audit_trail: List[Dict[str, Any]] = []

        logger.info(
            f"CQI Calculator initialized with scoring method: {config.scoring_method}"
        )

    def calculate(
        self,
        flue_gas: FlueGasReading,
        baseline_cqi: Optional[float] = None,
    ) -> CQIResult:
        """
        Calculate Combustion Quality Index from flue gas reading.

        This is the main entry point for CQI calculation. It performs:
        1. Input validation
        2. O2 correction for emissions
        3. Component scoring
        4. Weighted aggregation
        5. Rating assignment
        6. Provenance hash generation

        Args:
            flue_gas: Validated flue gas reading
            baseline_cqi: Optional baseline CQI for trend comparison

        Returns:
            CQIResult with complete analysis

        Raises:
            ValueError: If input data is invalid for calculation
        """
        start_time = datetime.now(timezone.utc)
        self._audit_trail = []

        # Step 1: Validate inputs
        self._validate_inputs(flue_gas)
        self._add_audit_entry("input_validation", {"status": "passed"})

        # Step 2: Calculate corrected emissions (to reference O2)
        co_corrected = self._correct_to_reference_o2(
            flue_gas.co_ppm,
            flue_gas.oxygen_pct,
            self.config.o2_reference_pct
        )
        nox_corrected = self._correct_to_reference_o2(
            flue_gas.nox_ppm,
            flue_gas.oxygen_pct,
            self.config.o2_reference_pct
        )
        self._add_audit_entry("o2_correction", {
            "co_raw": flue_gas.co_ppm,
            "co_corrected": co_corrected,
            "nox_raw": flue_gas.nox_ppm,
            "nox_corrected": nox_corrected,
            "measured_o2": flue_gas.oxygen_pct,
            "reference_o2": self.config.o2_reference_pct,
        })

        # Step 3: Calculate excess air
        excess_air_pct = self._calculate_excess_air(flue_gas.oxygen_pct)
        self._add_audit_entry("excess_air", {
            "oxygen_pct": flue_gas.oxygen_pct,
            "excess_air_pct": excess_air_pct,
        })

        # Step 4: Calculate combustion efficiency (simplified)
        combustion_efficiency = self._calculate_combustion_efficiency(
            flue_gas.oxygen_pct,
            flue_gas.flue_gas_temp_c,
            flue_gas.ambient_temp_c or 25.0,
        )
        self._add_audit_entry("combustion_efficiency", {
            "efficiency_pct": combustion_efficiency,
        })

        # Step 5: Calculate component scores
        components = self._calculate_component_scores(
            flue_gas, co_corrected, nox_corrected
        )

        # Step 6: Calculate weighted CQI score
        cqi_score = self._calculate_weighted_score(components)
        self._add_audit_entry("cqi_calculation", {
            "components": [c.dict() for c in components],
            "final_score": cqi_score,
        })

        # Step 7: Determine rating
        cqi_rating = self._determine_rating(cqi_score)

        # Step 8: Determine trend
        trend = TrendDirection.UNKNOWN
        if baseline_cqi is not None:
            if cqi_score > baseline_cqi + 2:
                trend = TrendDirection.IMPROVING
            elif cqi_score < baseline_cqi - 2:
                trend = TrendDirection.DEGRADING
            else:
                trend = TrendDirection.STABLE

        # Step 9: Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            flue_gas, cqi_score, components
        )

        processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        result = CQIResult(
            cqi_score=round(cqi_score, 2),
            cqi_rating=cqi_rating,
            components=components,
            co_corrected_ppm=round(co_corrected, 2),
            nox_corrected_ppm=round(nox_corrected, 2),
            o2_reference_pct=self.config.o2_reference_pct,
            excess_air_pct=round(excess_air_pct, 2),
            combustion_efficiency_pct=round(combustion_efficiency, 2),
            trend_vs_baseline=trend,
            baseline_cqi=baseline_cqi,
            calculation_timestamp=start_time,
            provenance_hash=provenance_hash,
        )

        logger.info(
            f"CQI calculated: {cqi_score:.1f} ({cqi_rating.value}) "
            f"in {processing_time_ms:.1f}ms"
        )

        return result

    def _validate_inputs(self, flue_gas: FlueGasReading) -> None:
        """Validate input data for CQI calculation."""
        if flue_gas.oxygen_pct < MIN_SAFE_O2_PCT:
            raise ValueError(
                f"Oxygen level ({flue_gas.oxygen_pct}%) too low for safe calculation"
            )

        if flue_gas.oxygen_pct >= ATMOSPHERIC_O2_PCT:
            raise ValueError(
                f"Oxygen level ({flue_gas.oxygen_pct}%) indicates no combustion"
            )

        if flue_gas.co_ppm < 0:
            raise ValueError(f"CO cannot be negative: {flue_gas.co_ppm}")

        if flue_gas.co2_pct < 0:
            raise ValueError(f"CO2 cannot be negative: {flue_gas.co2_pct}")

    def _correct_to_reference_o2(
        self,
        measured_value: float,
        measured_o2: float,
        reference_o2: float,
    ) -> float:
        """
        Correct emission measurement to reference O2 level.

        Uses the standard correction formula:
        Corrected = Measured * (20.95 - Reference O2) / (20.95 - Measured O2)

        This normalizes emissions for comparison regardless of excess air level.

        Args:
            measured_value: Measured emission value (ppm)
            measured_o2: Measured O2 percentage
            reference_o2: Reference O2 percentage (typically 3%)

        Returns:
            Corrected emission value (ppm)
        """
        # Prevent division by zero
        if measured_o2 >= ATMOSPHERIC_O2_PCT:
            return measured_value

        correction_factor = (
            (ATMOSPHERIC_O2_PCT - reference_o2) /
            (ATMOSPHERIC_O2_PCT - measured_o2)
        )

        return measured_value * correction_factor

    def _calculate_excess_air(self, oxygen_pct: float) -> float:
        """
        Calculate excess air percentage from flue gas oxygen.

        Formula: Excess Air (%) = (O2 / (20.95 - O2)) * 100

        This is a simplified calculation assuming complete combustion.

        Args:
            oxygen_pct: Measured oxygen percentage (dry basis)

        Returns:
            Excess air percentage
        """
        if oxygen_pct >= ATMOSPHERIC_O2_PCT:
            return 0.0

        excess_air = (oxygen_pct / (ATMOSPHERIC_O2_PCT - oxygen_pct)) * 100
        return max(0.0, excess_air)

    def _calculate_combustion_efficiency(
        self,
        oxygen_pct: float,
        stack_temp_c: float,
        ambient_temp_c: float,
    ) -> float:
        """
        Calculate simplified combustion efficiency.

        Uses the indirect method (heat loss method):
        Efficiency = 100 - Stack Losses

        Stack losses are estimated from stack temperature and excess air.
        This is a simplified calculation; precise values require fuel analysis.

        Args:
            oxygen_pct: Measured oxygen percentage
            stack_temp_c: Stack/flue gas temperature (Celsius)
            ambient_temp_c: Ambient air temperature (Celsius)

        Returns:
            Estimated combustion efficiency (%)
        """
        # Temperature difference (net stack temperature)
        delta_t = stack_temp_c - ambient_temp_c

        # Excess air factor
        excess_air_pct = self._calculate_excess_air(oxygen_pct)

        # Simplified stack loss calculation
        # Based on natural gas: ~0.05% loss per degree C per 1% excess air
        # Plus ~0.035% base loss per degree C
        base_loss_per_degree = 0.035
        excess_air_loss_factor = 0.0003 * excess_air_pct

        stack_loss = delta_t * (base_loss_per_degree + excess_air_loss_factor)

        # Other losses (radiation, blowdown, etc.) - estimated at 2%
        other_losses = 2.0

        efficiency = 100.0 - stack_loss - other_losses

        # Clamp to reasonable range
        return max(70.0, min(99.0, efficiency))

    def _calculate_component_scores(
        self,
        flue_gas: FlueGasReading,
        co_corrected: float,
        nox_corrected: float,
    ) -> List[CQIComponentScore]:
        """
        Calculate normalized scores for each CQI component.

        Each component is scored 0-100 based on its proximity to optimal values
        and distance from warning/critical thresholds.

        Args:
            flue_gas: Flue gas reading
            co_corrected: O2-corrected CO value
            nox_corrected: O2-corrected NOx value

        Returns:
            List of component scores
        """
        components = []

        # 1. Oxygen Score
        o2_score, o2_status = self._score_oxygen(flue_gas.oxygen_pct)
        components.append(CQIComponentScore(
            component="oxygen",
            raw_value=flue_gas.oxygen_pct,
            normalized_score=o2_score,
            weight=self.weights.oxygen,
            weighted_score=o2_score * self.weights.oxygen,
            status=o2_status,
        ))

        # 2. CO Score
        co_score, co_status = self._score_co(co_corrected)
        components.append(CQIComponentScore(
            component="carbon_monoxide",
            raw_value=co_corrected,
            normalized_score=co_score,
            weight=self.weights.carbon_monoxide,
            weighted_score=co_score * self.weights.carbon_monoxide,
            status=co_status,
        ))

        # 3. CO2 Score (validates carbon balance)
        co2_score, co2_status = self._score_co2(flue_gas.co2_pct, flue_gas.oxygen_pct)
        components.append(CQIComponentScore(
            component="carbon_dioxide",
            raw_value=flue_gas.co2_pct,
            normalized_score=co2_score,
            weight=self.weights.carbon_dioxide,
            weighted_score=co2_score * self.weights.carbon_dioxide,
            status=co2_status,
        ))

        # 4. NOx Score
        nox_score, nox_status = self._score_nox(nox_corrected)
        components.append(CQIComponentScore(
            component="nox",
            raw_value=nox_corrected,
            normalized_score=nox_score,
            weight=self.weights.nox,
            weighted_score=nox_score * self.weights.nox,
            status=nox_status,
        ))

        # 5. Combustibles Score
        combustibles = flue_gas.combustibles_pct or 0.0
        comb_score, comb_status = self._score_combustibles(combustibles)
        components.append(CQIComponentScore(
            component="combustibles",
            raw_value=combustibles,
            normalized_score=comb_score,
            weight=self.weights.combustibles,
            weighted_score=comb_score * self.weights.combustibles,
            status=comb_status,
        ))

        return components

    def _score_oxygen(self, o2_pct: float) -> Tuple[float, str]:
        """
        Score oxygen level (0-100).

        Optimal O2 is typically 2-4% for natural gas.
        Too low = incomplete combustion risk
        Too high = excessive heat loss

        Args:
            o2_pct: Measured oxygen percentage

        Returns:
            Tuple of (score, status)
        """
        t = self.thresholds

        if t.o2_optimal_min <= o2_pct <= t.o2_optimal_max:
            # Optimal range: 100 points
            score = 100.0
            status = "optimal"
        elif o2_pct < t.o2_optimal_min:
            # Below optimal: risk of incomplete combustion
            # Score decreases rapidly below 1% O2
            if o2_pct < 1.0:
                score = max(0, o2_pct * 30)  # 0-30 for 0-1% O2
                status = "critical"
            else:
                # Linear interpolation from 1% to optimal_min
                score = 30 + 70 * (o2_pct - 1.0) / (t.o2_optimal_min - 1.0)
                status = "warning"
        elif o2_pct <= t.o2_acceptable_max:
            # Acceptable but not optimal
            score = 100 - 25 * (o2_pct - t.o2_optimal_max) / (t.o2_acceptable_max - t.o2_optimal_max)
            status = "acceptable"
        elif o2_pct <= t.o2_warning_max:
            # Warning zone
            score = 75 - 35 * (o2_pct - t.o2_acceptable_max) / (t.o2_warning_max - t.o2_acceptable_max)
            status = "warning"
        else:
            # Critical - very high excess air
            score = max(0, 40 - 4 * (o2_pct - t.o2_warning_max))
            status = "critical"

        return round(max(0, min(100, score)), 1), status

    def _score_co(self, co_ppm: float) -> Tuple[float, str]:
        """
        Score CO level (0-100).

        Lower CO is better. High CO indicates incomplete combustion.

        Args:
            co_ppm: CO concentration (ppm, corrected to reference O2)

        Returns:
            Tuple of (score, status)
        """
        t = self.thresholds

        if co_ppm <= t.co_excellent:
            score = 100.0
            status = "optimal"
        elif co_ppm <= t.co_good:
            score = 90 - 15 * (co_ppm - t.co_excellent) / (t.co_good - t.co_excellent)
            status = "optimal"
        elif co_ppm <= t.co_acceptable:
            score = 75 - 15 * (co_ppm - t.co_good) / (t.co_acceptable - t.co_good)
            status = "acceptable"
        elif co_ppm <= t.co_warning:
            score = 60 - 20 * (co_ppm - t.co_acceptable) / (t.co_warning - t.co_acceptable)
            status = "warning"
        else:
            # Above warning threshold
            score = max(0, 40 - 0.05 * (co_ppm - t.co_warning))
            status = "critical"

        return round(max(0, min(100, score)), 1), status

    def _score_co2(self, co2_pct: float, o2_pct: float) -> Tuple[float, str]:
        """
        Score CO2 level (0-100).

        CO2 should be consistent with O2 for proper carbon balance.
        For natural gas, theoretical max CO2 is ~11.8% at stoichiometric.

        Args:
            co2_pct: CO2 concentration (%)
            o2_pct: O2 concentration (%)

        Returns:
            Tuple of (score, status)
        """
        # Calculate expected CO2 based on O2 (for natural gas)
        # CO2 + O2 should roughly equal 11.8 + 9.15 = 20.95% for ideal case
        expected_co2 = NATURAL_GAS_MAX_CO2_PCT * (ATMOSPHERIC_O2_PCT - o2_pct) / ATMOSPHERIC_O2_PCT

        if expected_co2 <= 0:
            return 0.0, "critical"

        # Calculate deviation from expected
        deviation_pct = abs(co2_pct - expected_co2) / expected_co2 * 100

        if deviation_pct <= 5:
            score = 100.0
            status = "optimal"
        elif deviation_pct <= 10:
            score = 90 - 2 * (deviation_pct - 5)
            status = "acceptable"
        elif deviation_pct <= 20:
            score = 80 - 2 * (deviation_pct - 10)
            status = "warning"
        else:
            score = max(0, 60 - 2 * (deviation_pct - 20))
            status = "critical"

        return round(max(0, min(100, score)), 1), status

    def _score_nox(self, nox_ppm: float) -> Tuple[float, str]:
        """
        Score NOx level (0-100).

        Lower NOx is better for environmental compliance.

        Args:
            nox_ppm: NOx concentration (ppm, corrected to reference O2)

        Returns:
            Tuple of (score, status)
        """
        t = self.thresholds

        if nox_ppm <= t.nox_excellent:
            score = 100.0
            status = "optimal"
        elif nox_ppm <= t.nox_good:
            score = 90 - 15 * (nox_ppm - t.nox_excellent) / (t.nox_good - t.nox_excellent)
            status = "optimal"
        elif nox_ppm <= t.nox_acceptable:
            score = 75 - 15 * (nox_ppm - t.nox_good) / (t.nox_acceptable - t.nox_good)
            status = "acceptable"
        elif nox_ppm <= t.nox_warning:
            score = 60 - 20 * (nox_ppm - t.nox_acceptable) / (t.nox_warning - t.nox_acceptable)
            status = "warning"
        else:
            score = max(0, 40 - 0.1 * (nox_ppm - t.nox_warning))
            status = "critical"

        return round(max(0, min(100, score)), 1), status

    def _score_combustibles(self, combustibles_pct: float) -> Tuple[float, str]:
        """
        Score unburned combustibles level (0-100).

        Lower is better. High combustibles indicate incomplete combustion.

        Args:
            combustibles_pct: Unburned combustibles (%)

        Returns:
            Tuple of (score, status)
        """
        t = self.thresholds

        if combustibles_pct <= t.combustibles_excellent:
            score = 100.0
            status = "optimal"
        elif combustibles_pct <= t.combustibles_good:
            score = 90 - 15 * (combustibles_pct - t.combustibles_excellent) / (t.combustibles_good - t.combustibles_excellent)
            status = "optimal"
        elif combustibles_pct <= t.combustibles_acceptable:
            score = 75 - 15 * (combustibles_pct - t.combustibles_good) / (t.combustibles_acceptable - t.combustibles_good)
            status = "acceptable"
        elif combustibles_pct <= t.combustibles_warning:
            score = 60 - 20 * (combustibles_pct - t.combustibles_acceptable) / (t.combustibles_warning - t.combustibles_acceptable)
            status = "warning"
        else:
            score = max(0, 40 - 40 * (combustibles_pct - t.combustibles_warning))
            status = "critical"

        return round(max(0, min(100, score)), 1), status

    def _calculate_weighted_score(
        self,
        components: List[CQIComponentScore],
    ) -> float:
        """
        Calculate final weighted CQI score.

        Args:
            components: List of component scores

        Returns:
            Final CQI score (0-100)
        """
        total_score = sum(c.weighted_score for c in components)
        return round(total_score, 2)

    def _determine_rating(self, cqi_score: float) -> CQIRating:
        """
        Determine CQI rating from score.

        Args:
            cqi_score: Calculated CQI score

        Returns:
            CQI rating enum
        """
        t = self.thresholds

        if cqi_score >= t.cqi_excellent:
            return CQIRating.EXCELLENT
        elif cqi_score >= t.cqi_good:
            return CQIRating.GOOD
        elif cqi_score >= t.cqi_acceptable:
            return CQIRating.ACCEPTABLE
        elif cqi_score >= t.cqi_poor:
            return CQIRating.POOR
        else:
            return CQIRating.CRITICAL

    def _calculate_provenance_hash(
        self,
        flue_gas: FlueGasReading,
        cqi_score: float,
        components: List[CQIComponentScore],
    ) -> str:
        """
        Calculate SHA-256 provenance hash for audit trail.

        Args:
            flue_gas: Input flue gas reading
            cqi_score: Calculated CQI score
            components: Component scores

        Returns:
            SHA-256 hash string
        """
        provenance_data = {
            "input": {
                "oxygen_pct": flue_gas.oxygen_pct,
                "co2_pct": flue_gas.co2_pct,
                "co_ppm": flue_gas.co_ppm,
                "nox_ppm": flue_gas.nox_ppm,
                "combustibles_pct": flue_gas.combustibles_pct,
                "timestamp": flue_gas.timestamp.isoformat(),
            },
            "output": {
                "cqi_score": cqi_score,
                "components": [
                    {"name": c.component, "score": c.normalized_score}
                    for c in components
                ],
            },
            "config": {
                "weights": self.weights.dict(),
                "o2_reference": self.config.o2_reference_pct,
            },
        }

        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _add_audit_entry(self, operation: str, data: Dict[str, Any]) -> None:
        """Add entry to audit trail."""
        self._audit_trail.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "data": data,
        })

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get calculation audit trail."""
        return self._audit_trail.copy()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_default_cqi_calculator() -> CombustionQualityCalculator:
    """
    Create CQI calculator with default configuration.

    Returns:
        CombustionQualityCalculator with default settings
    """
    return CombustionQualityCalculator(CQIConfig())


def calculate_cqi_quick(
    oxygen_pct: float,
    co_ppm: float,
    co2_pct: float,
    nox_ppm: float = 0.0,
    combustibles_pct: float = 0.0,
    flue_gas_temp_c: float = 200.0,
) -> float:
    """
    Quick CQI calculation with minimal inputs.

    Convenience function for simple CQI calculations without full schema setup.

    Args:
        oxygen_pct: Oxygen percentage
        co_ppm: CO concentration (ppm)
        co2_pct: CO2 percentage
        nox_ppm: NOx concentration (ppm)
        combustibles_pct: Unburned combustibles (%)
        flue_gas_temp_c: Flue gas temperature (C)

    Returns:
        CQI score (0-100)
    """
    flue_gas = FlueGasReading(
        timestamp=datetime.now(timezone.utc),
        oxygen_pct=oxygen_pct,
        co2_pct=co2_pct,
        co_ppm=co_ppm,
        nox_ppm=nox_ppm,
        combustibles_pct=combustibles_pct,
        flue_gas_temp_c=flue_gas_temp_c,
    )

    calculator = create_default_cqi_calculator()
    result = calculator.calculate(flue_gas)
    return result.cqi_score
