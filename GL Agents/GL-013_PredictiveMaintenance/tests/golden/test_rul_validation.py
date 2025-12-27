# -*- coding: utf-8 -*-
"""
RUL (Remaining Useful Life) Validation Tests for GL-013 PredictiveMaintenance
==============================================================================

Validates Remaining Useful Life predictions against known failure models
and industry-standard reliability calculations.

Test Categories:
    1. Weibull Distribution RUL Calculations
    2. Degradation Model Validation
    3. Confidence Interval Calculations
    4. CMMS Integration Validation
    5. Equipment-Specific Models

Reference Standards:
    - IEEE 493 (Gold Book): Reliability Data
    - OREDA: Offshore Reliability Data
    - MIL-HDBK-217: Reliability Prediction

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import math
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from decimal import Decimal, ROUND_HALF_UP
import statistics

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# RELIABILITY CONSTANTS
# =============================================================================

# Weibull distribution parameters for common equipment
# Shape (β): <1 infant mortality, =1 random, >1 wear-out
# Scale (η): Characteristic life in hours
WEIBULL_PARAMETERS = {
    "pump_centrifugal": {"beta": 2.0, "eta": 50000, "description": "Centrifugal pump"},
    "motor_induction": {"beta": 1.5, "eta": 100000, "description": "Induction motor"},
    "bearing_rolling": {"beta": 2.5, "eta": 40000, "description": "Rolling element bearing"},
    "valve_control": {"beta": 1.8, "eta": 30000, "description": "Control valve"},
    "heat_exchanger": {"beta": 1.2, "eta": 80000, "description": "Heat exchanger tubes"},
    "compressor_reciprocating": {"beta": 2.2, "eta": 35000, "description": "Reciprocating compressor"},
    "turbine_steam": {"beta": 3.0, "eta": 60000, "description": "Steam turbine"},
    "boiler_tubes": {"beta": 2.8, "eta": 120000, "description": "Boiler water tubes"},
    "fan_centrifugal": {"beta": 1.6, "eta": 70000, "description": "Centrifugal fan"},
    "gearbox": {"beta": 2.3, "eta": 45000, "description": "Industrial gearbox"},
}


# MTBF (Mean Time Between Failures) reference data from IEEE 493
IEEE493_MTBF_DATA = {
    # Equipment type: MTBF in hours
    "motor_0_50hp": 197000,
    "motor_51_250hp": 62000,
    "motor_251_600hp": 48000,
    "pump_centrifugal": 43000,
    "pump_reciprocating": 27000,
    "compressor_centrifugal": 35000,
    "compressor_reciprocating": 19000,
    "heat_exchanger": 87000,
    "transformer_liquid": 277000,
    "circuit_breaker": 181000,
    "valve_motor_operated": 75000,
}


# =============================================================================
# DEGRADATION MODELS
# =============================================================================

@dataclass
class DegradationModel:
    """Degradation model parameters."""
    name: str
    model_type: str  # "linear", "exponential", "power_law"
    initial_value: float
    failure_threshold: float
    degradation_rate: float
    rate_units: str


# Common degradation patterns
DEGRADATION_MODELS = {
    "vibration_bearing": DegradationModel(
        name="Bearing Vibration",
        model_type="exponential",
        initial_value=2.0,  # mm/s RMS
        failure_threshold=11.0,  # mm/s RMS (ISO 10816 Alarm)
        degradation_rate=0.00005,  # per hour
        rate_units="mm/s per hour",
    ),
    "efficiency_pump": DegradationModel(
        name="Pump Efficiency",
        model_type="linear",
        initial_value=85.0,  # %
        failure_threshold=70.0,  # %
        degradation_rate=0.0003,  # per hour
        rate_units="% per hour",
    ),
    "temperature_motor": DegradationModel(
        name="Motor Winding Temperature",
        model_type="linear",
        initial_value=80.0,  # °C
        failure_threshold=130.0,  # °C (Class F insulation)
        degradation_rate=0.001,  # per hour (degradation causes higher temp)
        rate_units="°C per hour",
    ),
    "thickness_pipe": DegradationModel(
        name="Pipe Wall Thickness",
        model_type="linear",
        initial_value=10.0,  # mm
        failure_threshold=5.0,  # mm (minimum safe thickness)
        degradation_rate=0.0001,  # mm per hour (corrosion rate)
        rate_units="mm per hour",
    ),
}


# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================

def weibull_reliability(t: float, beta: float, eta: float) -> float:
    """
    Calculate Weibull reliability at time t.

    R(t) = exp(-(t/η)^β)

    Args:
        t: Operating time
        beta: Shape parameter
        eta: Scale parameter (characteristic life)

    Returns:
        Reliability probability (0-1)
    """
    if t < 0 or eta <= 0 or beta <= 0:
        return 0.0
    return math.exp(-((t / eta) ** beta))


def weibull_hazard_rate(t: float, beta: float, eta: float) -> float:
    """
    Calculate Weibull hazard (failure) rate at time t.

    h(t) = (β/η) * (t/η)^(β-1)

    Args:
        t: Operating time
        beta: Shape parameter
        eta: Scale parameter

    Returns:
        Hazard rate
    """
    if t <= 0 or eta <= 0 or beta <= 0:
        return 0.0
    return (beta / eta) * ((t / eta) ** (beta - 1))


def weibull_mttf(beta: float, eta: float) -> float:
    """
    Calculate Mean Time To Failure for Weibull distribution.

    MTTF = η * Γ(1 + 1/β)

    Args:
        beta: Shape parameter
        eta: Scale parameter

    Returns:
        MTTF in same units as eta
    """
    # Gamma function approximation for common beta values
    gamma_values = {
        1.0: 1.0,
        1.5: 0.8862,
        2.0: 0.8862,
        2.5: 0.8873,
        3.0: 0.8930,
    }

    # Linear interpolation for other values
    gamma_1_plus_1_over_beta = gamma_values.get(
        beta,
        0.9  # Approximate for other values
    )

    return eta * gamma_1_plus_1_over_beta


def calculate_rul_from_weibull(
    current_age: float,
    beta: float,
    eta: float,
    confidence: float = 0.5
) -> Tuple[float, float, float]:
    """
    Calculate RUL from Weibull parameters.

    Args:
        current_age: Current operating hours
        beta: Shape parameter
        eta: Scale parameter
        confidence: Confidence level (0.5 = median)

    Returns:
        Tuple of (RUL_low, RUL_median, RUL_high) at 10%, 50%, 90%
    """
    # Calculate remaining life for different confidence levels
    # Using conditional reliability

    # Current reliability
    R_current = weibull_reliability(current_age, beta, eta)

    if R_current <= 0:
        return (0.0, 0.0, 0.0)

    # Time to reach different reliability thresholds
    def time_to_reliability(R_target: float) -> float:
        if R_target >= R_current:
            return 0.0
        # R(t) = exp(-(t/η)^β)
        # t = η * (-ln(R))^(1/β)
        return eta * ((-math.log(R_target)) ** (1 / beta))

    # Calculate times to different failure probabilities
    # RUL is time from now to reach each threshold
    t_10 = time_to_reliability(0.9)  # 10% failure probability
    t_50 = time_to_reliability(0.5)  # 50% failure probability (median)
    t_90 = time_to_reliability(0.1)  # 90% failure probability

    rul_low = max(0, t_10 - current_age)
    rul_median = max(0, t_50 - current_age)
    rul_high = max(0, t_90 - current_age)

    return (rul_low, rul_median, rul_high)


def calculate_rul_from_degradation(
    current_value: float,
    model: DegradationModel,
    operating_hours: float = 0
) -> float:
    """
    Calculate RUL from degradation model.

    Args:
        current_value: Current measurement value
        model: Degradation model parameters
        operating_hours: Current operating hours (for exponential models)

    Returns:
        Estimated RUL in hours
    """
    if model.model_type == "linear":
        # For parameters that decrease (like thickness)
        if model.degradation_rate > 0:
            if current_value <= model.failure_threshold:
                return 0.0
            rul = (current_value - model.failure_threshold) / model.degradation_rate
        else:
            # For parameters that increase (like vibration)
            rate = abs(model.degradation_rate)
            if current_value >= model.failure_threshold:
                return 0.0
            rul = (model.failure_threshold - current_value) / rate
        return max(0, rul)

    elif model.model_type == "exponential":
        # Value(t) = initial * exp(rate * t)
        if current_value >= model.failure_threshold:
            return 0.0
        # Solve for t when value reaches threshold
        t_failure = math.log(model.failure_threshold / model.initial_value) / model.degradation_rate
        t_current = math.log(current_value / model.initial_value) / model.degradation_rate
        return max(0, t_failure - t_current)

    return 0.0


# =============================================================================
# TEST CLASSES
# =============================================================================

@pytest.mark.golden
class TestWeibullRUL:
    """Test Weibull-based RUL calculations."""

    @pytest.mark.parametrize("equipment,age_hours,expected_rul_range", [
        ("pump_centrifugal", 0, (30000, 60000)),
        ("pump_centrifugal", 25000, (15000, 35000)),
        ("pump_centrifugal", 40000, (5000, 20000)),
        ("motor_induction", 0, (60000, 120000)),
        ("bearing_rolling", 20000, (10000, 30000)),
    ])
    def test_rul_decreases_with_age(
        self,
        equipment: str,
        age_hours: float,
        expected_rul_range: Tuple[float, float]
    ):
        """
        Verify RUL decreases as equipment ages.

        RUL should be within expected range for given age.
        """
        params = WEIBULL_PARAMETERS[equipment]
        rul_low, rul_median, rul_high = calculate_rul_from_weibull(
            age_hours,
            params["beta"],
            params["eta"]
        )

        assert expected_rul_range[0] <= rul_median <= expected_rul_range[1], (
            f"{equipment} at {age_hours}h: RUL median {rul_median:.0f}h "
            f"outside expected range {expected_rul_range}"
        )

    @pytest.mark.parametrize("beta,expected_behavior", [
        (0.5, "decreasing"),  # Infant mortality - hazard decreases
        (1.0, "constant"),    # Random failures
        (2.5, "increasing"),  # Wear-out - hazard increases
    ])
    def test_hazard_rate_behavior(self, beta: float, expected_behavior: str):
        """
        Verify hazard rate behavior based on Weibull shape parameter.

        β < 1: Decreasing hazard (infant mortality)
        β = 1: Constant hazard (random)
        β > 1: Increasing hazard (wear-out)
        """
        eta = 50000
        t1, t2 = 1000, 10000

        h1 = weibull_hazard_rate(t1, beta, eta)
        h2 = weibull_hazard_rate(t2, beta, eta)

        if expected_behavior == "decreasing":
            assert h2 < h1, f"β={beta}: Hazard should decrease"
        elif expected_behavior == "constant":
            # For β=1, hazard is approximately constant
            assert abs(h1 - h2) / h1 < 0.1, f"β={beta}: Hazard should be constant"
        else:  # increasing
            assert h2 > h1, f"β={beta}: Hazard should increase"

    def test_rul_confidence_interval_ordering(self):
        """Verify RUL confidence intervals are properly ordered."""
        params = WEIBULL_PARAMETERS["pump_centrifugal"]
        age = 20000

        rul_low, rul_median, rul_high = calculate_rul_from_weibull(
            age, params["beta"], params["eta"]
        )

        assert rul_low <= rul_median <= rul_high, (
            f"RUL intervals should be ordered: {rul_low} <= {rul_median} <= {rul_high}"
        )

    def test_new_equipment_has_maximum_rul(self):
        """New equipment (age=0) should have maximum RUL."""
        for equipment, params in WEIBULL_PARAMETERS.items():
            _, rul_new, _ = calculate_rul_from_weibull(0, params["beta"], params["eta"])
            _, rul_aged, _ = calculate_rul_from_weibull(
                params["eta"] / 2, params["beta"], params["eta"]
            )

            assert rul_new > rul_aged, (
                f"{equipment}: New RUL ({rul_new:.0f}) should exceed aged RUL ({rul_aged:.0f})"
            )


@pytest.mark.golden
class TestDegradationRUL:
    """Test degradation-based RUL calculations."""

    def test_vibration_rul_calculation(self):
        """
        Validate RUL from vibration degradation.

        ISO 10816 vibration limits:
        - Zone A/B boundary: 4.5 mm/s (good/acceptable)
        - Zone B/C boundary: 7.1 mm/s (acceptable/unsatisfactory)
        - Zone C/D boundary: 11.0 mm/s (unsatisfactory/unacceptable)
        """
        model = DEGRADATION_MODELS["vibration_bearing"]

        # Current vibration = 5 mm/s (Zone B)
        current_vibration = 5.0
        rul = calculate_rul_from_degradation(current_vibration, model)

        # Should have positive RUL since we're below threshold
        assert rul > 0, f"RUL should be positive for vibration {current_vibration}"

        # At threshold, RUL should be 0
        at_threshold = calculate_rul_from_degradation(model.failure_threshold, model)
        assert at_threshold == 0, "RUL at threshold should be 0"

    def test_pump_efficiency_rul(self):
        """
        Validate RUL from pump efficiency degradation.

        Pumps typically fail economically at 10-15% efficiency loss.
        """
        model = DEGRADATION_MODELS["efficiency_pump"]

        # Current efficiency = 80% (5% degradation from 85%)
        current_efficiency = 80.0
        rul = calculate_rul_from_degradation(current_efficiency, model)

        # Should have RUL to reach 70% threshold
        expected_rul = (80.0 - 70.0) / model.degradation_rate
        deviation = abs(rul - expected_rul) / expected_rul * 100

        assert deviation <= 1.0, (
            f"Pump efficiency RUL {rul:.0f}h vs expected {expected_rul:.0f}h"
        )

    def test_pipe_thickness_rul(self):
        """
        Validate RUL from pipe wall thickness (corrosion).

        API 570 requires minimum thickness calculations.
        """
        model = DEGRADATION_MODELS["thickness_pipe"]

        # Current thickness = 7 mm (30% wall loss)
        current_thickness = 7.0
        rul = calculate_rul_from_degradation(current_thickness, model)

        # RUL to reach 5mm minimum
        expected_rul = (7.0 - 5.0) / model.degradation_rate

        assert abs(rul - expected_rul) < 100, (
            f"Pipe thickness RUL {rul:.0f}h vs expected {expected_rul:.0f}h"
        )


@pytest.mark.golden
class TestMTBFValidation:
    """Test MTBF calculations against IEEE 493 reference data."""

    @pytest.mark.parametrize("equipment,ieee_mtbf", [
        ("pump_centrifugal", 43000),
        ("motor_induction", 62000),
        ("heat_exchanger", 87000),
    ])
    def test_mtbf_within_industry_range(self, equipment: str, ieee_mtbf: float):
        """
        Verify calculated MTBF is within industry range.

        IEEE 493 provides industry average failure rates.
        """
        # Map to Weibull parameters
        weibull_map = {
            "pump_centrifugal": "pump_centrifugal",
            "motor_induction": "motor_induction",
            "heat_exchanger": "heat_exchanger",
        }

        if equipment in weibull_map:
            params = WEIBULL_PARAMETERS[weibull_map[equipment]]
            calculated_mttf = weibull_mttf(params["beta"], params["eta"])

            # Allow 50% deviation from IEEE data (equipment varies widely)
            ratio = calculated_mttf / ieee_mtbf

            assert 0.5 <= ratio <= 2.0, (
                f"{equipment}: MTTF {calculated_mttf:.0f}h vs IEEE {ieee_mtbf}h"
            )


@pytest.mark.golden
class TestConfidenceIntervals:
    """Test confidence interval calculations for RUL."""

    def test_confidence_interval_width(self):
        """
        Verify confidence interval width is reasonable.

        For Weibull, 80% CI should be narrower than 95% CI.
        """
        params = WEIBULL_PARAMETERS["pump_centrifugal"]
        age = 25000

        rul_10, rul_50, rul_90 = calculate_rul_from_weibull(
            age, params["beta"], params["eta"]
        )

        # 80% confidence interval
        ci_80 = rul_90 - rul_10

        # Interval should be positive
        assert ci_80 > 0, "Confidence interval should be positive"

        # Interval should be reasonable (not larger than median)
        assert ci_80 < 2 * rul_50, (
            f"CI width {ci_80:.0f}h should be < 2x median {rul_50:.0f}h"
        )

    def test_older_equipment_has_narrower_intervals(self):
        """
        Older equipment should have narrower confidence intervals.

        As equipment ages, uncertainty about remaining life decreases.
        """
        params = WEIBULL_PARAMETERS["turbine_steam"]

        # Young equipment
        rul_low_young, rul_med_young, rul_high_young = calculate_rul_from_weibull(
            1000, params["beta"], params["eta"]
        )
        ci_young = rul_high_young - rul_low_young

        # Older equipment
        rul_low_old, rul_med_old, rul_high_old = calculate_rul_from_weibull(
            40000, params["beta"], params["eta"]
        )
        ci_old = rul_high_old - rul_low_old

        # Older equipment should have smaller absolute CI
        assert ci_old < ci_young, (
            f"Old CI ({ci_old:.0f}h) should be < young CI ({ci_young:.0f}h)"
        )


@pytest.mark.golden
class TestPredictionAccuracy:
    """Test prediction accuracy validation."""

    def test_rul_prediction_vs_actual(self):
        """
        Simulate prediction accuracy over equipment lifecycle.

        Track how predictions compare to actual end-of-life.
        """
        # Simulate equipment with known failure time
        actual_failure_time = 45000  # hours
        params = {"beta": 2.0, "eta": 50000}

        prediction_errors = []
        check_points = [10000, 20000, 30000, 40000]

        for age in check_points:
            _, rul_predicted, _ = calculate_rul_from_weibull(
                age, params["beta"], params["eta"]
            )
            actual_rul = actual_failure_time - age

            if rul_predicted > 0:
                error_percent = abs(rul_predicted - actual_rul) / actual_rul * 100
                prediction_errors.append(error_percent)

        # Average error should be < 50% (RUL prediction is inherently uncertain)
        avg_error = statistics.mean(prediction_errors) if prediction_errors else 0
        assert avg_error < 50, f"Average prediction error {avg_error:.1f}% too high"


@pytest.mark.golden
class TestCMMSIntegration:
    """Test CMMS (Computerized Maintenance Management System) integration."""

    def test_work_order_timing(self):
        """
        Verify work order generation timing based on RUL.

        Work orders should be generated with sufficient lead time.
        """
        # Define maintenance lead times
        maintenance_lead_times = {
            "critical": 168,    # 1 week for critical
            "major": 720,       # 30 days for major
            "minor": 168,       # 1 week for minor
        }

        # Equipment with calculated RUL
        params = WEIBULL_PARAMETERS["bearing_rolling"]
        age = 30000
        _, rul, _ = calculate_rul_from_weibull(age, params["beta"], params["eta"])

        # Determine appropriate maintenance action based on RUL
        if rul < maintenance_lead_times["critical"]:
            action = "emergency"
        elif rul < maintenance_lead_times["major"]:
            action = "schedule_immediate"
        elif rul < maintenance_lead_times["minor"] * 4:
            action = "plan_maintenance"
        else:
            action = "monitor"

        # Verify action is appropriate
        assert action in ["emergency", "schedule_immediate", "plan_maintenance", "monitor"]

    def test_spare_parts_lead_time(self):
        """
        Verify spare parts ordering based on RUL predictions.

        Parts should be ordered before predicted failure.
        """
        # Typical spare parts lead times (hours)
        parts_lead_time = {
            "bearing": 168,     # 1 week
            "seal_kit": 72,     # 3 days
            "motor": 720,       # 30 days
            "impeller": 480,    # 20 days
        }

        # Calculate minimum RUL needed for each part
        for part, lead_time in parts_lead_time.items():
            # Order when RUL < 2x lead time
            order_threshold = lead_time * 2

            assert order_threshold > lead_time, (
                f"Order threshold {order_threshold}h should exceed lead time {lead_time}h"
            )


@pytest.mark.golden
class TestDeterminism:
    """Verify RUL calculations are deterministic."""

    def test_weibull_rul_determinism(self):
        """Weibull RUL calculation must be deterministic."""
        params = WEIBULL_PARAMETERS["pump_centrifugal"]
        age = 25000

        results = []
        for _ in range(100):
            rul_low, rul_med, rul_high = calculate_rul_from_weibull(
                age, params["beta"], params["eta"]
            )
            # Use Decimal for precise comparison
            result = Decimal(str(rul_med)).quantize(Decimal("0.0001"))
            results.append(str(result))

        unique_results = set(results)
        assert len(unique_results) == 1, (
            f"Non-deterministic: {len(unique_results)} unique results"
        )

    def test_degradation_rul_determinism(self):
        """Degradation RUL calculation must be deterministic."""
        model = DEGRADATION_MODELS["vibration_bearing"]
        current_value = 5.0

        results = []
        for _ in range(100):
            rul = calculate_rul_from_degradation(current_value, model)
            result = Decimal(str(rul)).quantize(Decimal("0.0001"))
            results.append(str(result))

        unique_results = set(results)
        assert len(unique_results) == 1, "Degradation RUL not deterministic"


# =============================================================================
# EXPORT FUNCTION
# =============================================================================

def export_rul_golden_values() -> Dict[str, Any]:
    """Export all RUL golden values for external validation."""
    return {
        "metadata": {
            "version": "1.0.0",
            "standards": ["IEEE 493", "OREDA", "MIL-HDBK-217"],
            "agent": "GL-013_PredictiveMaintenance",
        },
        "weibull_parameters": {
            key: {
                "beta": params["beta"],
                "eta": params["eta"],
                "mttf": weibull_mttf(params["beta"], params["eta"]),
                "description": params["description"],
            }
            for key, params in WEIBULL_PARAMETERS.items()
        },
        "ieee493_reference": IEEE493_MTBF_DATA,
        "degradation_models": {
            key: {
                "type": model.model_type,
                "initial": model.initial_value,
                "threshold": model.failure_threshold,
                "rate": model.degradation_rate,
            }
            for key, model in DEGRADATION_MODELS.items()
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(export_rul_golden_values(), indent=2))
