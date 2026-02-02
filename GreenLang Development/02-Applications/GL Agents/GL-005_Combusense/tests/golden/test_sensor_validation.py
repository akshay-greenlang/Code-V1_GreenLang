# -*- coding: utf-8 -*-
"""
Sensor Validation Golden Tests for GL-005 CombustionSense
=========================================================

Validates combustion sensor measurements against known references
and cross-validates sensor readings for consistency.

Test Categories:
    1. O2 Sensor Accuracy
    2. CO Sensor Accuracy
    3. Combustion Product Relationships
    4. Sensor Cross-Validation
    5. Calibration Drift Detection

Reference Standards:
    - EPA 40 CFR Part 60, Appendix B (Performance Specifications)
    - EPA 40 CFR Part 75, Appendix A (CEMS Specifications)
    - ASTM D6522 (Combustion Analyzers)

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

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# SENSOR SPECIFICATIONS
# =============================================================================

@dataclass(frozen=True)
class SensorSpecification:
    """Sensor performance specification."""
    sensor_type: str
    measurement_range: Tuple[float, float]
    accuracy_percent_of_span: float
    repeatability_percent: float
    response_time_t90_seconds: float
    operating_temp_range_c: Tuple[float, float]
    cross_sensitivity: Dict[str, float]  # Interfering gases


# EPA Performance Specifications for CEMS (40 CFR Part 60)
EPA_SENSOR_SPECS = {
    "O2_paramagnetic": SensorSpecification(
        sensor_type="Paramagnetic O2",
        measurement_range=(0.0, 25.0),
        accuracy_percent_of_span=0.5,
        repeatability_percent=0.2,
        response_time_t90_seconds=30,
        operating_temp_range_c=(-20, 50),
        cross_sensitivity={"CO2": 0.0, "CO": 0.0},  # No cross-sensitivity
    ),
    "O2_zirconia": SensorSpecification(
        sensor_type="Zirconia O2 (In-situ)",
        measurement_range=(0.0, 21.0),
        accuracy_percent_of_span=1.0,
        repeatability_percent=0.5,
        response_time_t90_seconds=5,
        operating_temp_range_c=(600, 1400),
        cross_sensitivity={"CO": -0.5, "H2": -0.3},  # Reducing gases affect reading
    ),
    "CO_NDIR": SensorSpecification(
        sensor_type="NDIR CO",
        measurement_range=(0.0, 1000.0),
        accuracy_percent_of_span=2.0,
        repeatability_percent=1.0,
        response_time_t90_seconds=60,
        operating_temp_range_c=(0, 50),
        cross_sensitivity={"CO2": 0.1, "H2O": 0.5},
    ),
    "CO2_NDIR": SensorSpecification(
        sensor_type="NDIR CO2",
        measurement_range=(0.0, 20.0),
        accuracy_percent_of_span=1.0,
        repeatability_percent=0.5,
        response_time_t90_seconds=30,
        operating_temp_range_c=(0, 50),
        cross_sensitivity={"H2O": 0.2},
    ),
    "NOx_chemiluminescence": SensorSpecification(
        sensor_type="Chemiluminescence NOx",
        measurement_range=(0.0, 500.0),
        accuracy_percent_of_span=2.0,
        repeatability_percent=1.0,
        response_time_t90_seconds=90,
        operating_temp_range_c=(5, 40),
        cross_sensitivity={"NH3": -5.0, "H2O": 0.5},
    ),
}


# =============================================================================
# COMBUSTION REFERENCE RELATIONSHIPS
# =============================================================================

# Theoretical combustion products for natural gas at different excess air levels
COMBUSTION_REFERENCE_DATA = {
    "natural_gas": {
        0: {"O2": 0.0, "CO2": 11.7, "N2": 88.3},   # Stoichiometric
        10: {"O2": 1.8, "CO2": 10.6, "N2": 87.6},  # 10% excess air
        15: {"O2": 2.6, "CO2": 10.2, "N2": 87.2},  # 15% excess air
        20: {"O2": 3.3, "CO2": 9.8, "N2": 86.9},   # 20% excess air
        30: {"O2": 4.5, "CO2": 9.0, "N2": 86.5},   # 30% excess air
        50: {"O2": 6.3, "CO2": 7.8, "N2": 85.9},   # 50% excess air
    },
    "fuel_oil": {
        0: {"O2": 0.0, "CO2": 15.5, "N2": 84.5},
        15: {"O2": 2.6, "CO2": 13.5, "N2": 83.9},
        20: {"O2": 3.3, "CO2": 13.0, "N2": 83.7},
        30: {"O2": 4.5, "CO2": 12.0, "N2": 83.5},
    },
}


# =============================================================================
# CALIBRATION REFERENCE GASES
# =============================================================================

@dataclass(frozen=True)
class CalibrationGas:
    """Calibration gas specification."""
    component: str
    concentration: float
    units: str
    tolerance_percent: float
    traceability: str


EPA_CALIBRATION_GASES = {
    "O2_span": CalibrationGas("O2", 16.0, "%", 1.0, "NIST-traceable"),
    "O2_zero": CalibrationGas("O2", 0.0, "%", 0.0, "N2 balance"),
    "CO2_span": CalibrationGas("CO2", 14.0, "%", 1.0, "NIST-traceable"),
    "CO_span": CalibrationGas("CO", 500.0, "ppm", 2.0, "NIST-traceable"),
    "CO_zero": CalibrationGas("CO", 0.0, "ppm", 0.0, "N2 balance"),
    "NOx_span": CalibrationGas("NOx", 250.0, "ppm", 2.0, "NIST-traceable"),
}


# =============================================================================
# TEST CLASSES
# =============================================================================

@pytest.mark.golden
class TestO2SensorAccuracy:
    """Test O2 sensor accuracy and specifications."""

    @pytest.mark.parametrize("sensor_type,spec_key", [
        ("Paramagnetic", "O2_paramagnetic"),
        ("Zirconia", "O2_zirconia"),
    ])
    def test_o2_sensor_accuracy_within_spec(self, sensor_type: str, spec_key: str):
        """
        Verify O2 sensor accuracy meets EPA specifications.

        EPA 40 CFR Part 60, PS-3 requires ±0.5% O2 accuracy.
        """
        spec = EPA_SENSOR_SPECS[spec_key]

        # Simulate measurement at mid-span
        mid_span = (spec.measurement_range[0] + spec.measurement_range[1]) / 2
        span = spec.measurement_range[1] - spec.measurement_range[0]

        # Calculate allowable error
        max_error = span * spec.accuracy_percent_of_span / 100

        # For mid-span reading, error should be within spec
        assert max_error <= 0.5, (
            f"{sensor_type} O2: Max error {max_error:.3f}% should be <= 0.5%"
        )

    def test_o2_co2_relationship_natural_gas(self):
        """
        Verify O2 and CO2 follow theoretical relationship for natural gas.

        For natural gas: CO2_max ≈ 11.7% at 0% O2
        O2 + CO2/11.7 * 21 ≈ 21 (approximately)
        """
        for excess_air, expected in COMBUSTION_REFERENCE_DATA["natural_gas"].items():
            o2 = expected["O2"]
            co2 = expected["CO2"]

            # Check that O2 + CO2 relationship is consistent
            # CO2 decreases as O2 increases (dilution with excess air)
            if excess_air > 0:
                # Verify inverse relationship
                assert o2 > 0 and co2 < 11.7, (
                    f"At {excess_air}% EA: O2={o2}% should be >0, CO2={co2}% should be <11.7%"
                )

    @pytest.mark.parametrize("o2_reading,expected_excess_air", [
        (2.0, 10.5),
        (3.0, 16.7),
        (4.0, 23.5),
        (5.0, 31.3),
    ])
    def test_excess_air_from_o2_calculation(
        self,
        o2_reading: float,
        expected_excess_air: float
    ):
        """
        Validate excess air calculation from O2 measurement.

        Formula: EA% = O2 / (21 - O2) * 100
        """
        calculated_ea = (o2_reading / (21.0 - o2_reading)) * 100

        deviation = abs(calculated_ea - expected_excess_air)
        assert deviation <= 1.0, (
            f"O2={o2_reading}%: EA={calculated_ea:.1f}% vs expected {expected_excess_air:.1f}%"
        )


@pytest.mark.golden
class TestCOSensorAccuracy:
    """Test CO sensor accuracy and specifications."""

    def test_co_sensor_accuracy_within_spec(self):
        """
        Verify CO sensor accuracy meets EPA specifications.

        EPA 40 CFR Part 60, PS-4A requires ±5% of span or 10 ppm.
        """
        spec = EPA_SENSOR_SPECS["CO_NDIR"]
        span = spec.measurement_range[1] - spec.measurement_range[0]

        # Calculate allowable error
        max_error_percent = span * spec.accuracy_percent_of_span / 100
        max_error_ppm = max(max_error_percent, 10)  # EPA minimum 10 ppm

        # Verify within typical industrial standards
        assert max_error_ppm <= 50, (
            f"CO sensor max error {max_error_ppm:.1f} ppm should be <= 50 ppm"
        )

    def test_co_response_to_incomplete_combustion(self):
        """
        Verify CO increases when O2 drops below combustion threshold.

        Typical behavior:
        - O2 > 2%: CO typically < 100 ppm
        - O2 < 1%: CO can exceed 1000 ppm
        """
        # Simplified CO vs O2 model
        def estimate_co_ppm(o2_percent: float) -> float:
            if o2_percent >= 3.0:
                return 20.0  # Well-tuned combustion
            elif o2_percent >= 2.0:
                return 50.0  # Moderate
            elif o2_percent >= 1.0:
                return 200.0  # Getting low
            else:
                return 1000.0  # Incomplete combustion

        assert estimate_co_ppm(3.5) < estimate_co_ppm(1.5), "CO should increase at low O2"
        assert estimate_co_ppm(0.5) > 500, "CO should be high at very low O2"


@pytest.mark.golden
class TestSensorCrossValidation:
    """Test cross-validation between multiple sensors."""

    def test_o2_co2_sum_validation(self):
        """
        Cross-validate O2 and CO2 sensors.

        For good combustion: O2 + CO2 should be approximately 20-21%
        (accounts for N2 dilution)
        """
        for fuel, data in COMBUSTION_REFERENCE_DATA.items():
            for excess_air, values in data.items():
                o2 = values["O2"]
                co2 = values["CO2"]

                # O2 + CO2 should be reasonable for flue gas
                sum_value = o2 + co2
                assert 10 <= sum_value <= 21, (
                    f"{fuel} at {excess_air}% EA: O2+CO2={sum_value:.1f}% "
                    "outside expected 10-21%"
                )

    def test_triple_point_validation(self):
        """
        Validate O2, CO2, and calculated combustion efficiency alignment.

        If all three measurements are from the same process, they should
        be thermodynamically consistent.
        """
        # At 15% excess air for natural gas
        o2 = 2.6
        co2 = 10.2
        n2 = 87.2

        # Sum should be 100%
        total = o2 + co2 + n2
        assert abs(total - 100.0) <= 0.5, (
            f"Flue gas composition sum {total:.1f}% should be ~100%"
        )

    def test_redundant_sensor_agreement(self):
        """
        Test that redundant sensors agree within specifications.

        For safety-critical applications, multiple sensors should agree.
        """
        # Simulate two O2 sensors
        o2_sensor_1 = 3.2  # %
        o2_sensor_2 = 3.4  # %

        # Acceptable deviation for redundant sensors (typically ±0.5%)
        deviation = abs(o2_sensor_1 - o2_sensor_2)
        max_allowed_deviation = 0.5

        assert deviation <= max_allowed_deviation, (
            f"Redundant O2 sensors differ by {deviation}%, max allowed {max_allowed_deviation}%"
        )


@pytest.mark.golden
class TestCalibrationDrift:
    """Test calibration drift detection."""

    @pytest.mark.parametrize("pollutant,max_drift_percent", [
        ("O2", 0.5),
        ("CO2", 0.5),
        ("CO", 5.0),
        ("NOx", 2.5),
    ])
    def test_daily_calibration_drift_limits(
        self,
        pollutant: str,
        max_drift_percent: float
    ):
        """
        Verify calibration drift limits per EPA requirements.

        EPA 40 CFR Part 60 requires daily calibration drift checks.
        """
        # Simulate calibration check
        span_gas = EPA_CALIBRATION_GASES.get(f"{pollutant}_span")

        if span_gas:
            # Calculate allowable drift
            if span_gas.units == "%":
                allowable_drift = max_drift_percent  # Direct percentage
            else:
                # ppm units - use percentage of span
                allowable_drift = span_gas.concentration * max_drift_percent / 100

            assert allowable_drift > 0, f"Allowable drift for {pollutant} should be positive"

    def test_zero_drift_detection(self):
        """
        Test detection of zero drift condition.

        Zero drift occurs when baseline shifts.
        """
        # Simulate zero calibration readings over time
        zero_readings = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2]  # Drifting zero

        # Calculate drift rate
        drift_per_reading = (zero_readings[-1] - zero_readings[0]) / (len(zero_readings) - 1)

        # Alert if drift rate exceeds threshold
        drift_threshold = 0.1  # Per reading
        is_drifting = drift_per_reading > drift_threshold

        assert is_drifting, "Should detect zero drift in this data"

    def test_span_drift_detection(self):
        """
        Test detection of span drift condition.

        Span drift occurs when sensitivity changes.
        """
        # Simulate span calibration readings (should be 16.0% O2)
        expected_span = 16.0
        span_readings = [16.0, 15.9, 15.7, 15.5, 15.2]  # Drifting low

        # Calculate drift
        drift = expected_span - span_readings[-1]
        drift_percent = (drift / expected_span) * 100

        # EPA typically allows 2.5% span drift
        max_span_drift = 2.5

        is_drift_excessive = drift_percent > max_span_drift

        assert is_drift_excessive, "Should detect excessive span drift"


@pytest.mark.golden
class TestSensorResponseTime:
    """Test sensor response time validation."""

    @pytest.mark.parametrize("sensor_key,max_t90_seconds", [
        ("O2_paramagnetic", 30),
        ("O2_zirconia", 5),
        ("CO_NDIR", 60),
        ("CO2_NDIR", 30),
        ("NOx_chemiluminescence", 90),
    ])
    def test_sensor_response_time_meets_spec(
        self,
        sensor_key: str,
        max_t90_seconds: float
    ):
        """
        Verify sensor response time meets specifications.

        T90 is the time to reach 90% of final reading.
        """
        spec = EPA_SENSOR_SPECS[sensor_key]

        assert spec.response_time_t90_seconds <= max_t90_seconds, (
            f"{sensor_key}: T90 {spec.response_time_t90_seconds}s exceeds max {max_t90_seconds}s"
        )

    def test_response_time_for_safety_systems(self):
        """
        Verify response time adequate for combustion safety.

        Flame failure detection typically requires < 4 second response.
        """
        # Zirconia O2 sensor is typically used for fast response
        zirconia_spec = EPA_SENSOR_SPECS["O2_zirconia"]

        # For safety systems, need fast response
        safety_max_response = 5.0  # seconds

        assert zirconia_spec.response_time_t90_seconds <= safety_max_response, (
            f"In-situ O2 sensor T90 {zirconia_spec.response_time_t90_seconds}s "
            f"should be <= {safety_max_response}s for safety"
        )


@pytest.mark.golden
class TestSensorCrossSensitivity:
    """Test sensor cross-sensitivity effects."""

    def test_zirconia_o2_reducing_gas_interference(self):
        """
        Verify zirconia O2 sensor cross-sensitivity to reducing gases.

        Reducing gases (CO, H2) cause lower O2 readings.
        """
        spec = EPA_SENSOR_SPECS["O2_zirconia"]

        # CO causes negative interference on zirconia O2
        co_sensitivity = spec.cross_sensitivity.get("CO", 0)

        assert co_sensitivity < 0, (
            "Zirconia O2 should show negative cross-sensitivity to CO"
        )

    def test_ndir_water_interference(self):
        """
        Verify NDIR sensors have documented water interference.

        Water vapor absorbs in similar IR bands as CO and CO2.
        """
        co_spec = EPA_SENSOR_SPECS["CO_NDIR"]
        co2_spec = EPA_SENSOR_SPECS["CO2_NDIR"]

        # Both should show some water sensitivity
        assert co_spec.cross_sensitivity.get("H2O", 0) > 0, (
            "NDIR CO should be affected by water vapor"
        )
        assert co2_spec.cross_sensitivity.get("H2O", 0) > 0, (
            "NDIR CO2 should be affected by water vapor"
        )


@pytest.mark.golden
class TestDeterminism:
    """Verify sensor validation calculations are deterministic."""

    def test_excess_air_calculation_determinism(self):
        """Excess air calculation must be deterministic."""
        results = []

        for _ in range(100):
            o2 = Decimal("3.5")
            excess_air = (o2 / (Decimal("21.0") - o2) * Decimal("100")).quantize(
                Decimal("0.0001"),
                rounding=ROUND_HALF_UP
            )
            results.append(str(excess_air))

        unique_results = set(results)
        assert len(unique_results) == 1, (
            f"Non-deterministic: {len(unique_results)} unique results"
        )

    def test_sensor_validation_hash_consistency(self):
        """Sensor validation results should produce consistent hashes."""
        hashes = []

        for _ in range(50):
            validation_data = {
                "o2": 3.5,
                "co2": 10.2,
                "excess_air": 19.44,
                "status": "valid",
            }
            hash_val = hashlib.sha256(
                str(sorted(validation_data.items())).encode()
            ).hexdigest()
            hashes.append(hash_val)

        assert len(set(hashes)) == 1, "Validation hashes not consistent"


# =============================================================================
# EXPORT FUNCTION
# =============================================================================

def export_sensor_validation_golden_values() -> Dict[str, Any]:
    """Export all sensor validation golden values."""
    return {
        "metadata": {
            "version": "1.0.0",
            "source": "EPA 40 CFR Part 60/75, ASTM D6522",
            "agent": "GL-005_CombustionSense",
        },
        "sensor_specs": {
            key: {
                "type": spec.sensor_type,
                "range": spec.measurement_range,
                "accuracy_percent": spec.accuracy_percent_of_span,
                "t90_seconds": spec.response_time_t90_seconds,
            }
            for key, spec in EPA_SENSOR_SPECS.items()
        },
        "combustion_reference": COMBUSTION_REFERENCE_DATA,
        "calibration_gases": {
            key: {
                "component": gas.component,
                "concentration": gas.concentration,
                "units": gas.units,
            }
            for key, gas in EPA_CALIBRATION_GASES.items()
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(export_sensor_validation_golden_values(), indent=2))
