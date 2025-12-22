"""
Sensor Failure Edge Case Tests for GL-004 BURNMASTER

Tests system behavior under various sensor failure scenarios:
- Individual sensor failures (O2, CO, NOx, flame, temperature)
- Multiple simultaneous sensor failures
- Sensor drift and degradation
- Sensor noise and spike detection
- Sensor quality flag handling
- Cascading sensor failures

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import numpy as np
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, MagicMock, patch
from enum import Enum
from dataclasses import dataclass, field
import random

# Import system modules
import sys
sys.path.insert(0, 'C:/Users/aksha/Code-V1_GreenLang/GL Agents/GL-004_Burnmaster')

from combustion.stoichiometry import (
    compute_stoichiometric_air,
    compute_lambda,
    compute_excess_o2,
    infer_lambda_from_o2,
    validate_stoichiometry_inputs,
)
from combustion.fuel_properties import FuelType
from safety.safety_envelope import SafetyEnvelope, Setpoint, EnvelopeStatus
from safety.interlock_manager import (
    InterlockManager, BMSStatus, SISStatus, BMSState, SISState,
    InterlockState, Interlock, PermissiveStatus,
)
from calculators.stability_calculator import (
    FlameStabilityCalculator, StabilityLevel, RiskLevel,
)


# ============================================================================
# DATA CLASSES FOR SENSOR SIMULATION
# ============================================================================

class SensorQuality(Enum):
    """Sensor quality levels."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    STALE = "stale"
    COMM_FAILURE = "comm_failure"


@dataclass
class SensorReading:
    """Simulated sensor reading with quality information."""
    value: float
    quality: SensorQuality
    timestamp: datetime = field(default_factory=datetime.utcnow)
    sensor_id: str = ""
    raw_value: Optional[float] = None
    last_good_value: Optional[float] = None


@dataclass
class SensorConfig:
    """Configuration for sensor simulation."""
    sensor_id: str
    sensor_type: str
    min_value: float
    max_value: float
    typical_value: float
    noise_std: float = 0.1
    drift_rate: float = 0.0
    failure_probability: float = 0.0


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def stability_calculator():
    """Create FlameStabilityCalculator instance."""
    return FlameStabilityCalculator(precision=4)


@pytest.fixture
def safety_envelope():
    """Create configured SafetyEnvelope instance."""
    envelope = SafetyEnvelope(unit_id="BLR-TEST")
    envelope.define_envelope("BLR-TEST", {
        "o2_min": 1.5,
        "o2_max": 8.0,
        "co_max": 200,
        "nox_max": 100,
        "draft_min": -0.5,
        "draft_max": -0.01,
        "flame_signal_min": 30.0,
        "steam_temp_max": 550.0,
        "steam_pressure_max": 150.0,
        "firing_rate_min": 10.0,
        "firing_rate_max": 100.0,
    })
    return envelope


@pytest.fixture
def interlock_manager():
    """Create InterlockManager instance."""
    return InterlockManager(unit_id="BLR-TEST")


@pytest.fixture
def mock_bms_interface():
    """Create mock BMS interface for testing."""
    mock = MagicMock()
    mock.read_status.return_value = {
        'state': 'run',
        'flame_proven': True,
        'purge_complete': True,
        'pilot_proven': True,
        'main_fuel_valve_open': True,
        'air_damper_proven': True,
        'lockout_active': False,
        'fault_codes': []
    }
    return mock


@pytest.fixture
def sensor_configs() -> Dict[str, SensorConfig]:
    """Create sensor configuration dictionary."""
    return {
        "O2": SensorConfig(
            sensor_id="O2-001",
            sensor_type="oxygen",
            min_value=0.0,
            max_value=21.0,
            typical_value=3.0,
            noise_std=0.1
        ),
        "CO": SensorConfig(
            sensor_id="CO-001",
            sensor_type="carbon_monoxide",
            min_value=0.0,
            max_value=2000.0,
            typical_value=50.0,
            noise_std=5.0
        ),
        "NOx": SensorConfig(
            sensor_id="NOx-001",
            sensor_type="nitrogen_oxides",
            min_value=0.0,
            max_value=500.0,
            typical_value=40.0,
            noise_std=3.0
        ),
        "FLAME": SensorConfig(
            sensor_id="FLAME-001",
            sensor_type="flame_signal",
            min_value=0.0,
            max_value=100.0,
            typical_value=80.0,
            noise_std=2.0
        ),
        "TEMP_STACK": SensorConfig(
            sensor_id="TEMP-001",
            sensor_type="stack_temperature",
            min_value=100.0,
            max_value=400.0,
            typical_value=180.0,
            noise_std=2.0
        ),
        "FUEL_FLOW": SensorConfig(
            sensor_id="FLOW-001",
            sensor_type="fuel_flow",
            min_value=0.0,
            max_value=100.0,
            typical_value=50.0,
            noise_std=0.5
        ),
    }


class SensorSimulator:
    """Simulator for sensor readings with failure injection."""

    def __init__(self, config: SensorConfig):
        self.config = config
        self.current_value = config.typical_value
        self.drift_accumulated = 0.0
        self.is_failed = False
        self.failure_type: Optional[str] = None
        self.last_good_reading: Optional[SensorReading] = None

    def read(self) -> SensorReading:
        """Generate a sensor reading."""
        if self.is_failed:
            return self._generate_failed_reading()

        # Apply drift
        self.drift_accumulated += self.config.drift_rate
        base_value = self.config.typical_value + self.drift_accumulated

        # Add noise
        noise = np.random.normal(0, self.config.noise_std)
        value = base_value + noise

        # Clamp to valid range
        value = max(self.config.min_value, min(self.config.max_value, value))

        reading = SensorReading(
            value=value,
            quality=SensorQuality.GOOD,
            sensor_id=self.config.sensor_id,
            raw_value=value,
            last_good_value=value
        )
        self.last_good_reading = reading
        self.current_value = value

        return reading

    def inject_failure(self, failure_type: str):
        """Inject a sensor failure."""
        self.is_failed = True
        self.failure_type = failure_type

    def clear_failure(self):
        """Clear any injected failure."""
        self.is_failed = False
        self.failure_type = None

    def _generate_failed_reading(self) -> SensorReading:
        """Generate a reading based on failure type."""
        last_good = self.last_good_reading.value if self.last_good_reading else self.config.typical_value

        if self.failure_type == "stuck":
            return SensorReading(
                value=last_good,
                quality=SensorQuality.STALE,
                sensor_id=self.config.sensor_id,
                last_good_value=last_good
            )
        elif self.failure_type == "zero":
            return SensorReading(
                value=0.0,
                quality=SensorQuality.BAD,
                sensor_id=self.config.sensor_id,
                last_good_value=last_good
            )
        elif self.failure_type == "max_scale":
            return SensorReading(
                value=self.config.max_value,
                quality=SensorQuality.BAD,
                sensor_id=self.config.sensor_id,
                last_good_value=last_good
            )
        elif self.failure_type == "random_noise":
            return SensorReading(
                value=random.uniform(self.config.min_value, self.config.max_value),
                quality=SensorQuality.UNCERTAIN,
                sensor_id=self.config.sensor_id,
                last_good_value=last_good
            )
        elif self.failure_type == "comm_failure":
            return SensorReading(
                value=float('nan'),
                quality=SensorQuality.COMM_FAILURE,
                sensor_id=self.config.sensor_id,
                last_good_value=last_good
            )
        elif self.failure_type == "drift":
            # Excessive drift
            drifted_value = last_good + random.uniform(5, 20) * np.sign(random.random() - 0.5)
            return SensorReading(
                value=drifted_value,
                quality=SensorQuality.UNCERTAIN,
                sensor_id=self.config.sensor_id,
                last_good_value=last_good
            )
        else:
            return SensorReading(
                value=float('nan'),
                quality=SensorQuality.BAD,
                sensor_id=self.config.sensor_id,
                last_good_value=last_good
            )


# ============================================================================
# INDIVIDUAL SENSOR FAILURE TESTS
# ============================================================================

class TestIndividualSensorFailures:
    """Test suite for individual sensor failures."""

    @pytest.mark.parametrize("failure_type", [
        "stuck",
        "zero",
        "max_scale",
        "random_noise",
        "comm_failure",
        "drift",
    ])
    def test_o2_sensor_failure_types(self, sensor_configs, failure_type: str):
        """Test O2 sensor behavior under different failure types."""
        simulator = SensorSimulator(sensor_configs["O2"])

        # Get baseline reading
        baseline = simulator.read()
        assert baseline.quality == SensorQuality.GOOD

        # Inject failure
        simulator.inject_failure(failure_type)
        failed_reading = simulator.read()

        # Verify failure is detected
        assert failed_reading.quality != SensorQuality.GOOD

        # Verify last good value is preserved
        if not np.isnan(failed_reading.value):
            assert failed_reading.last_good_value is not None

    def test_o2_sensor_stuck_at_value(self, sensor_configs):
        """Test detection of stuck O2 sensor."""
        simulator = SensorSimulator(sensor_configs["O2"])

        # Collect some good readings
        good_readings = [simulator.read() for _ in range(10)]

        # Inject stuck failure
        simulator.inject_failure("stuck")

        # All subsequent readings should be identical
        stuck_readings = [simulator.read() for _ in range(10)]

        # Verify readings are stuck
        values = [r.value for r in stuck_readings]
        assert len(set(values)) == 1, "Stuck sensor should return same value"

        # Verify quality indicates staleness
        assert all(r.quality == SensorQuality.STALE for r in stuck_readings)

    def test_flame_sensor_loss(self, sensor_configs, stability_calculator):
        """Test stability calculation when flame sensor fails."""
        simulator = SensorSimulator(sensor_configs["FLAME"])

        # Normal operation
        normal_signal = np.array([simulator.read().value for _ in range(50)])
        normal_result = stability_calculator.compute_stability_index(normal_signal, 0.1)

        assert normal_result.stability_level in [StabilityLevel.EXCELLENT, StabilityLevel.GOOD]

        # Inject zero failure (simulating flame loss)
        simulator.inject_failure("zero")
        failed_signal = np.array([simulator.read().value for _ in range(50)])

        # Very low signal should trigger warnings
        result = stability_calculator.compute_stability_index(failed_signal, 0.1)
        assert any("low flame signal" in rec.lower() for rec in result.recommendations)

    def test_co_sensor_max_scale_failure(self, sensor_configs, safety_envelope):
        """Test CO sensor max scale failure detection."""
        simulator = SensorSimulator(sensor_configs["CO"])

        # Inject max scale failure
        simulator.inject_failure("max_scale")
        reading = simulator.read()

        # Should read at maximum
        assert reading.value == sensor_configs["CO"].max_value
        assert reading.quality == SensorQuality.BAD

        # Safety envelope should block on bad quality
        # (In production, we would check quality before using value)

    def test_temperature_sensor_drift(self, sensor_configs):
        """Test detection of temperature sensor drift."""
        config = sensor_configs["TEMP_STACK"]
        config.drift_rate = 0.5  # 0.5 degree per reading

        simulator = SensorSimulator(config)

        # Collect readings over time
        readings = [simulator.read() for _ in range(100)]
        values = [r.value for r in readings]

        # Calculate drift
        first_10_avg = np.mean(values[:10])
        last_10_avg = np.mean(values[-10:])
        drift = last_10_avg - first_10_avg

        # Should detect significant drift
        assert drift > 30, "Drift should be detectable"

    def test_fuel_flow_sensor_failure_impact(self, sensor_configs):
        """Test impact of fuel flow sensor failure on calculations."""
        simulator = SensorSimulator(sensor_configs["FUEL_FLOW"])

        # Normal reading
        normal_reading = simulator.read()

        # Inject zero failure
        simulator.inject_failure("zero")
        failed_reading = simulator.read()

        assert failed_reading.value == 0.0
        assert failed_reading.quality == SensorQuality.BAD

        # Should preserve last good value
        assert failed_reading.last_good_value == normal_reading.value


# ============================================================================
# MULTIPLE SENSOR FAILURE TESTS
# ============================================================================

class TestMultipleSensorFailures:
    """Test suite for multiple simultaneous sensor failures."""

    def test_dual_sensor_failure(self, sensor_configs):
        """Test system behavior with two sensor failures."""
        o2_sim = SensorSimulator(sensor_configs["O2"])
        co_sim = SensorSimulator(sensor_configs["CO"])

        # Inject failures on both
        o2_sim.inject_failure("stuck")
        co_sim.inject_failure("max_scale")

        o2_reading = o2_sim.read()
        co_reading = co_sim.read()

        # Both should indicate failure
        assert o2_reading.quality != SensorQuality.GOOD
        assert co_reading.quality != SensorQuality.GOOD

    def test_all_emission_sensors_failure(self, sensor_configs):
        """Test when all emission sensors fail simultaneously."""
        sensors = {
            name: SensorSimulator(config)
            for name, config in sensor_configs.items()
            if name in ["O2", "CO", "NOx"]
        }

        # Inject failures on all
        for sim in sensors.values():
            sim.inject_failure("comm_failure")

        readings = {name: sim.read() for name, sim in sensors.items()}

        # All should indicate communication failure
        for name, reading in readings.items():
            assert reading.quality == SensorQuality.COMM_FAILURE
            assert np.isnan(reading.value)

    def test_flame_and_o2_combined_failure(self, sensor_configs, stability_calculator):
        """Test combined flame and O2 sensor failures."""
        flame_sim = SensorSimulator(sensor_configs["FLAME"])
        o2_sim = SensorSimulator(sensor_configs["O2"])

        # Normal operation
        normal_flame = np.array([flame_sim.read().value for _ in range(50)])
        normal_o2_variance = 0.1

        # Inject failures
        flame_sim.inject_failure("random_noise")
        o2_sim.inject_failure("stuck")

        # Create noisy flame signal
        failed_flame = np.array([flame_sim.read().value for _ in range(50)])
        high_o2_variance = 0.8  # Simulate high variance from stuck value

        result = stability_calculator.compute_stability_index(failed_flame, high_o2_variance)

        # Should indicate poor stability
        assert result.stability_level in [StabilityLevel.MARGINAL, StabilityLevel.POOR, StabilityLevel.CRITICAL]

    def test_cascading_failure_scenario(self, sensor_configs):
        """Test cascading failure where one failure causes others."""
        simulators = {
            name: SensorSimulator(config)
            for name, config in sensor_configs.items()
        }

        # Simulate power supply failure affecting multiple sensors
        affected_sensors = ["O2", "CO", "NOx"]
        for name in affected_sensors:
            simulators[name].inject_failure("comm_failure")

        # Check failure count
        failed_count = sum(
            1 for sim in simulators.values()
            if sim.read().quality in [SensorQuality.BAD, SensorQuality.COMM_FAILURE]
        )

        assert failed_count >= len(affected_sensors)


# ============================================================================
# SENSOR QUALITY HANDLING TESTS
# ============================================================================

class TestSensorQualityHandling:
    """Test suite for sensor quality flag handling."""

    @pytest.mark.parametrize("quality,should_use", [
        (SensorQuality.GOOD, True),
        (SensorQuality.UNCERTAIN, True),  # May use with caution
        (SensorQuality.BAD, False),
        (SensorQuality.STALE, False),
        (SensorQuality.COMM_FAILURE, False),
    ])
    def test_quality_based_value_usage(self, quality: SensorQuality, should_use: bool):
        """Test that values are used appropriately based on quality."""
        reading = SensorReading(
            value=3.0,
            quality=quality,
            sensor_id="O2-001"
        )

        # Simulate decision logic
        use_value = reading.quality in [SensorQuality.GOOD, SensorQuality.UNCERTAIN]

        assert use_value == should_use

    def test_fallback_to_last_good_value(self, sensor_configs):
        """Test fallback to last good value on sensor failure."""
        simulator = SensorSimulator(sensor_configs["O2"])

        # Get a good reading
        good_reading = simulator.read()
        last_good = good_reading.value

        # Inject failure
        simulator.inject_failure("comm_failure")
        failed_reading = simulator.read()

        # Should have access to last good value
        assert failed_reading.last_good_value == last_good
        assert not np.isnan(failed_reading.last_good_value)

    def test_quality_degradation_over_time(self, sensor_configs):
        """Test quality degradation when sensor becomes stale."""
        simulator = SensorSimulator(sensor_configs["O2"])

        # Get initial reading
        initial = simulator.read()
        assert initial.quality == SensorQuality.GOOD

        # Simulate stuck condition
        simulator.inject_failure("stuck")

        readings = []
        for i in range(100):
            reading = simulator.read()
            reading.timestamp = datetime.utcnow() + timedelta(seconds=i * 10)
            readings.append(reading)

        # All readings should be stale
        assert all(r.quality == SensorQuality.STALE for r in readings)


# ============================================================================
# BMS/SIS SENSOR FAILURE TESTS
# ============================================================================

class TestBMSSensorFailures:
    """Test suite for BMS/SIS sensor failure handling."""

    def test_bms_read_failure(self, interlock_manager, mock_bms_interface):
        """Test BMS status when read operation fails."""
        # Configure mock to raise exception
        mock_bms_interface.read_status.side_effect = Exception("Communication timeout")

        manager = InterlockManager(
            unit_id="BLR-TEST",
            bms_interface=mock_bms_interface
        )

        status = manager.read_bms_status("BLR-TEST")

        # Should return fault status
        assert status.state == BMSState.FAULT
        assert status.lockout_active == True
        assert "READ_FAIL" in status.fault_codes[0]

    def test_flame_proven_false_on_failure(self, interlock_manager, mock_bms_interface):
        """Test flame proven is false on sensor failure."""
        mock_bms_interface.read_status.side_effect = Exception("Flame sensor failure")

        manager = InterlockManager(
            unit_id="BLR-TEST",
            bms_interface=mock_bms_interface
        )

        status = manager.read_bms_status("BLR-TEST")

        # Flame should not be proven on failure
        assert status.flame_proven == False

    def test_sis_read_failure(self, interlock_manager):
        """Test SIS status when read operation fails."""
        mock_sis = MagicMock()
        mock_sis.read_status.side_effect = Exception("SIS communication error")

        manager = InterlockManager(
            unit_id="BLR-TEST",
            sis_interface=mock_sis
        )

        status = manager.read_sis_status("BLR-TEST")

        # Should return fault status
        assert status.state == SISState.FAULT
        assert "READ_FAIL" in status.fault_codes[0]

    def test_interlock_state_on_sensor_failure(self, interlock_manager):
        """Test interlock blocking on sensor failure."""
        interlock = Interlock(
            interlock_id="INT-001",
            name="O2 Low",
            state=InterlockState.FAULT,  # Sensor fault
            trip_point=1.5,
            actual_value=float('nan'),  # No valid reading
            description="O2 sensor fault"
        )

        result = interlock_manager.block_on_interlock(interlock)

        assert result.blocked == True
        assert "fault" in result.reason.lower()
        assert result.can_proceed_observe_only == True  # Can still observe

    def test_permissives_check_with_bad_sensors(self, interlock_manager, mock_bms_interface):
        """Test permissive checking when sensors report bad quality."""
        mock_bms_interface.read_status.return_value = {
            'state': 'fault',
            'flame_proven': False,
            'purge_complete': False,
            'pilot_proven': False,
            'main_fuel_valve_open': False,
            'air_damper_proven': False,
            'lockout_active': True,
            'fault_codes': ['SENSOR_FAULT']
        }

        manager = InterlockManager(
            unit_id="BLR-TEST",
            bms_interface=mock_bms_interface
        )

        permissives = manager.check_permissives("BLR-TEST")

        assert permissives.all_permissives_met == False
        assert len(permissives.missing_permissives) > 0


# ============================================================================
# SENSOR REDUNDANCY TESTS
# ============================================================================

class TestSensorRedundancy:
    """Test suite for sensor redundancy handling."""

    def test_triple_redundant_sensor_voting(self, sensor_configs):
        """Test 2-out-of-3 voting for redundant sensors."""
        # Create three O2 sensors
        configs = [
            SensorConfig(**{**sensor_configs["O2"].__dict__, "sensor_id": f"O2-00{i}"})
            for i in range(1, 4)
        ]
        simulators = [SensorSimulator(c) for c in configs]

        # Normal operation - all agree
        readings = [sim.read() for sim in simulators]
        values = [r.value for r in readings]

        # Should be within tolerance
        assert max(values) - min(values) < 0.5, "Redundant sensors should agree"

        # Simulate one sensor failure
        simulators[0].inject_failure("max_scale")

        readings = [sim.read() for sim in simulators]
        good_readings = [r for r in readings if r.quality == SensorQuality.GOOD]

        # Should still have 2 good readings for voting
        assert len(good_readings) >= 2

    def test_dual_redundant_sensor_mismatch(self, sensor_configs):
        """Test detection of mismatch in dual redundant sensors."""
        configs = [
            SensorConfig(**{**sensor_configs["O2"].__dict__, "sensor_id": f"O2-00{i}"})
            for i in range(1, 3)
        ]
        simulators = [SensorSimulator(c) for c in configs]

        # Both reading normally
        readings = [sim.read() for sim in simulators]
        normal_diff = abs(readings[0].value - readings[1].value)

        # Inject drift on one
        simulators[0].inject_failure("drift")
        readings = [sim.read() for sim in simulators]
        failed_diff = abs(readings[0].value - readings[1].value)

        # Difference should be larger after drift
        assert failed_diff > normal_diff * 5

    def test_all_redundant_sensors_failed(self, sensor_configs):
        """Test system behavior when all redundant sensors fail."""
        configs = [
            SensorConfig(**{**sensor_configs["O2"].__dict__, "sensor_id": f"O2-00{i}"})
            for i in range(1, 4)
        ]
        simulators = [SensorSimulator(c) for c in configs]

        # Fail all sensors
        for sim in simulators:
            sim.inject_failure("comm_failure")

        readings = [sim.read() for sim in simulators]

        # All should be bad
        assert all(r.quality == SensorQuality.COMM_FAILURE for r in readings)

        # Should use last good values
        last_good_values = [r.last_good_value for r in readings if r.last_good_value is not None]
        # At least one should have a last good value preserved
        # (First reading after failure)


# ============================================================================
# SENSOR SPIKE AND NOISE TESTS
# ============================================================================

class TestSensorSpikesAndNoise:
    """Test suite for sensor spike and noise handling."""

    def test_sensor_spike_detection(self, sensor_configs):
        """Test detection of sudden sensor spikes."""
        simulator = SensorSimulator(sensor_configs["O2"])

        # Collect baseline readings
        baseline = [simulator.read().value for _ in range(20)]
        baseline_mean = np.mean(baseline)
        baseline_std = np.std(baseline)

        # Simulate a spike by temporarily injecting max_scale
        simulator.inject_failure("max_scale")
        spike_reading = simulator.read()

        # Clear failure
        simulator.clear_failure()
        post_spike = simulator.read()

        # Spike should be detectable (> 3 sigma)
        spike_zscore = abs(spike_reading.value - baseline_mean) / max(baseline_std, 0.1)
        assert spike_zscore > 3, "Spike should be > 3 sigma from baseline"

    def test_noise_filtering_stability(self, sensor_configs, stability_calculator):
        """Test that noise filtering maintains calculation stability."""
        config = sensor_configs["FLAME"]
        config.noise_std = 5.0  # Higher noise

        simulator = SensorSimulator(config)

        # Collect noisy readings
        noisy_signal = np.array([simulator.read().value for _ in range(100)])

        result = stability_calculator.compute_stability_index(noisy_signal, 0.3)

        # Should still provide valid stability assessment
        assert result.stability_level is not None
        assert 0 <= float(result.stability_index) <= 1

    def test_intermittent_noise_bursts(self, sensor_configs):
        """Test handling of intermittent noise bursts."""
        simulator = SensorSimulator(sensor_configs["O2"])

        readings = []
        for i in range(100):
            if 30 <= i < 40:  # Noise burst window
                simulator.inject_failure("random_noise")
            else:
                simulator.clear_failure()

            readings.append(simulator.read())

        # Count good vs bad readings
        good_count = sum(1 for r in readings if r.quality == SensorQuality.GOOD)
        uncertain_count = sum(1 for r in readings if r.quality == SensorQuality.UNCERTAIN)

        # Most readings should be good
        assert good_count > 80
        # Some should be uncertain (during noise burst)
        assert uncertain_count > 0


# ============================================================================
# SENSOR RECOVERY TESTS
# ============================================================================

class TestSensorRecovery:
    """Test suite for sensor recovery after failure."""

    def test_sensor_recovery_after_comm_failure(self, sensor_configs):
        """Test sensor recovery after communication failure."""
        simulator = SensorSimulator(sensor_configs["O2"])

        # Normal operation
        normal_reading = simulator.read()
        assert normal_reading.quality == SensorQuality.GOOD

        # Fail
        simulator.inject_failure("comm_failure")
        failed_reading = simulator.read()
        assert failed_reading.quality == SensorQuality.COMM_FAILURE

        # Recover
        simulator.clear_failure()
        recovered_reading = simulator.read()
        assert recovered_reading.quality == SensorQuality.GOOD

    def test_gradual_recovery_from_drift(self, sensor_configs):
        """Test gradual recovery from sensor drift."""
        config = sensor_configs["O2"]
        config.drift_rate = 0.1
        simulator = SensorSimulator(config)

        # Accumulate drift
        for _ in range(50):
            simulator.read()

        drifted_value = simulator.current_value

        # Reset drift (simulating recalibration)
        simulator.drift_accumulated = 0.0

        # Check recovery
        recovered_reading = simulator.read()

        # Value should be closer to typical after reset
        assert abs(recovered_reading.value - config.typical_value) < abs(drifted_value - config.typical_value)

    def test_bms_recovery_after_failure(self, interlock_manager, mock_bms_interface):
        """Test BMS recovery sequence after failure."""
        # First call fails
        mock_bms_interface.read_status.side_effect = Exception("Timeout")

        manager = InterlockManager(
            unit_id="BLR-TEST",
            bms_interface=mock_bms_interface
        )

        failed_status = manager.read_bms_status("BLR-TEST")
        assert failed_status.state == BMSState.FAULT

        # Recover
        mock_bms_interface.read_status.side_effect = None
        mock_bms_interface.read_status.return_value = {
            'state': 'run',
            'flame_proven': True,
            'purge_complete': True,
            'pilot_proven': True,
            'main_fuel_valve_open': True,
            'air_damper_proven': True,
            'lockout_active': False,
            'fault_codes': []
        }

        recovered_status = manager.read_bms_status("BLR-TEST")
        assert recovered_status.state == BMSState.RUN
        assert recovered_status.flame_proven == True


# ============================================================================
# SENSOR VALIDATION AND PLAUSIBILITY TESTS
# ============================================================================

class TestSensorValidation:
    """Test suite for sensor value validation and plausibility checks."""

    @pytest.mark.parametrize("o2_value,is_plausible", [
        (3.0, True),      # Normal
        (0.5, True),      # Low but possible
        (15.0, True),     # High but possible
        (-1.0, False),    # Impossible
        (25.0, False),    # Above air (impossible)
        (float('nan'), False),
        (float('inf'), False),
    ])
    def test_o2_value_plausibility(self, o2_value: float, is_plausible: bool):
        """Test O2 value plausibility checking."""
        # Simple plausibility check
        def is_o2_plausible(value):
            if np.isnan(value) or np.isinf(value):
                return False
            return 0 <= value <= 21

        assert is_o2_plausible(o2_value) == is_plausible

    @pytest.mark.parametrize("co_value,is_plausible", [
        (50.0, True),     # Normal
        (0.0, True),      # Very clean
        (500.0, True),    # High but possible
        (-10.0, False),   # Impossible
        (50000.0, False), # Unrealistic
    ])
    def test_co_value_plausibility(self, co_value: float, is_plausible: bool):
        """Test CO value plausibility checking."""
        def is_co_plausible(value):
            if np.isnan(value) or np.isinf(value):
                return False
            return 0 <= value <= 10000  # Max realistic is ~10000 ppm

        assert is_co_plausible(co_value) == is_plausible

    def test_cross_sensor_validation(self, sensor_configs):
        """Test cross-validation between related sensors."""
        o2_sim = SensorSimulator(sensor_configs["O2"])
        co_sim = SensorSimulator(sensor_configs["CO"])

        o2_reading = o2_sim.read()
        co_reading = co_sim.read()

        # Cross-validation: high O2 with high CO is suspicious
        # (High excess air typically means complete combustion = low CO)
        if o2_reading.value > 6.0 and co_reading.value > 200:
            # This combination is suspicious
            pass  # Would flag for investigation

        # Low O2 with low CO is normal
        if o2_reading.value < 2.0 and co_reading.value < 100:
            # This is consistent
            pass

    def test_rate_of_change_validation(self, sensor_configs):
        """Test validation based on rate of change."""
        simulator = SensorSimulator(sensor_configs["O2"])

        readings = [simulator.read() for _ in range(10)]
        values = [r.value for r in readings]

        # Calculate rate of change between consecutive readings
        rates = [abs(values[i+1] - values[i]) for i in range(len(values)-1)]
        max_rate = max(rates)

        # Normal rate of change should be small
        assert max_rate < 1.0, "Normal O2 change rate should be < 1%/reading"

        # Inject sudden change
        simulator.inject_failure("max_scale")
        spike_reading = simulator.read()

        rate_to_spike = abs(spike_reading.value - values[-1])

        # Spike rate should be detectable
        assert rate_to_spike > 10.0, "Spike should cause large rate of change"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
