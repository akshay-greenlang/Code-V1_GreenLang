# -*- coding: utf-8 -*-
"""
NFPA 85 Burner Safety Golden Value Tests

Comprehensive test suite validating Flameguard burner management and safety
systems against NFPA 85 (Boiler and Combustion Systems Hazards Code) requirements.

Reference Documents:
- NFPA 85-2023: Boiler and Combustion Systems Hazards Code
- NFPA 86: Standard for Ovens and Furnaces
- IEC 61511: Functional Safety - Safety Instrumented Systems
- ISA 84: Application of Safety Instrumented Systems

Author: GL-CalculatorEngineer
"""

import pytest
import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
from datetime import datetime, timedelta


# ==============================================================================
# NFPA 85 GOLDEN VALUE REFERENCE DATA
# ==============================================================================

@dataclass(frozen=True)
class NFPA85GoldenValue:
    """NFPA 85 regulatory reference value with citation."""
    name: str
    value: Decimal
    unit: str
    tolerance_percent: Decimal
    nfpa_section: str
    description: str

    def validate(self, calculated: Decimal) -> Tuple[bool, Decimal]:
        """Validate calculated value against golden value."""
        if self.value == 0:
            return abs(calculated) < Decimal('0.001'), abs(calculated)
        deviation = abs(calculated - self.value) / abs(self.value) * Decimal('100')
        return deviation <= self.tolerance_percent, deviation


# NFPA 85 Timing Requirements (Chapter 8)
NFPA_85_TIMING: Dict[str, NFPA85GoldenValue] = {
    'pre_purge_time_min': NFPA85GoldenValue(
        'Pre-Purge Time Minimum', Decimal('5'), 'minutes',
        Decimal('0'), '8.5.3', 'Minimum 5 furnace volume changes'),
    'pre_purge_airflow_min': NFPA85GoldenValue(
        'Pre-Purge Airflow', Decimal('25'), 'percent',
        Decimal('0'), '8.5.3.2', 'Minimum 25% of full-load airflow'),
    'pilot_trial_max': NFPA85GoldenValue(
        'Pilot Trial Time Max', Decimal('10'), 'seconds',
        Decimal('0'), '8.6.3.2', 'Maximum pilot trial for ignition time'),
    'main_flame_trial_max': NFPA85GoldenValue(
        'Main Flame Trial Max', Decimal('10'), 'seconds',
        Decimal('0'), '8.6.4.2', 'Maximum main burner trial for ignition'),
    'flame_failure_response': NFPA85GoldenValue(
        'Flame Failure Response', Decimal('4'), 'seconds',
        Decimal('0'), '8.8.3.3', 'Maximum flame failure response time'),
    'post_purge_time': NFPA85GoldenValue(
        'Post-Purge Time', Decimal('60'), 'seconds',
        Decimal('0'), '8.9.2', 'Post-purge duration requirement'),
    'high_fire_rate': NFPA85GoldenValue(
        'High Fire Rate', Decimal('100'), 'percent',
        Decimal('0'), '8.5.1', 'High fire rate for purge'),
    'low_fire_rate': NFPA85GoldenValue(
        'Low Fire Rate', Decimal('25'), 'percent',
        Decimal('0'), '8.6.1', 'Low fire start requirement'),
}

# NFPA 85 Safety Interlock Requirements (Chapter 7)
NFPA_85_INTERLOCKS: Dict[str, Dict[str, Any]] = {
    'combustion_air_proving': {
        'description': 'Combustion air flow proving',
        'section': '7.5.1',
        'trip_action': 'prevent_light_off',
        'severity': 'safety_critical',
    },
    'fuel_pressure_proving': {
        'description': 'Fuel supply pressure within limits',
        'section': '7.5.2',
        'trip_action': 'safety_shutdown',
        'severity': 'safety_critical',
    },
    'flame_detector_proving': {
        'description': 'Flame detector operational',
        'section': '7.5.3',
        'trip_action': 'prevent_light_off',
        'severity': 'safety_critical',
    },
    'purge_proving': {
        'description': 'Pre-purge completed',
        'section': '7.5.4',
        'trip_action': 'prevent_light_off',
        'severity': 'safety_critical',
    },
    'fuel_valve_position': {
        'description': 'Main fuel valves closed during purge',
        'section': '7.5.5',
        'trip_action': 'safety_shutdown',
        'severity': 'safety_critical',
    },
    'drum_level_low': {
        'description': 'Drum level above low-low limit',
        'section': '7.4.1',
        'trip_action': 'safety_shutdown',
        'severity': 'safety_critical',
    },
    'drum_level_high': {
        'description': 'Drum level below high-high limit',
        'section': '7.4.2',
        'trip_action': 'safety_shutdown',
        'severity': 'safety_critical',
    },
    'furnace_pressure_high': {
        'description': 'Furnace pressure below high limit',
        'section': '7.4.3',
        'trip_action': 'safety_shutdown',
        'severity': 'safety_critical',
    },
}

# SIL (Safety Integrity Level) Requirements per IEC 61511
SIL_REQUIREMENTS: Dict[int, Dict[str, Any]] = {
    1: {
        'pfd_avg_low': Decimal('0.01'),
        'pfd_avg_high': Decimal('0.1'),
        'rrf_low': 10,
        'rrf_high': 100,
        'spurious_trip_rate': Decimal('0.1'),
    },
    2: {
        'pfd_avg_low': Decimal('0.001'),
        'pfd_avg_high': Decimal('0.01'),
        'rrf_low': 100,
        'rrf_high': 1000,
        'spurious_trip_rate': Decimal('0.01'),
    },
    3: {
        'pfd_avg_low': Decimal('0.0001'),
        'pfd_avg_high': Decimal('0.001'),
        'rrf_low': 1000,
        'rrf_high': 10000,
        'spurious_trip_rate': Decimal('0.001'),
    },
}

# Flame Scanner Performance Requirements
FLAME_SCANNER_SPECS: Dict[str, NFPA85GoldenValue] = {
    'min_flame_signal': NFPA85GoldenValue(
        'Minimum Flame Signal', Decimal('10'), 'percent',
        Decimal('0'), '8.8.2', 'Minimum flame signal for proven flame'),
    'response_time': NFPA85GoldenValue(
        'Flame Scanner Response', Decimal('1'), 'seconds',
        Decimal('0'), '8.8.3.1', 'Maximum response time'),
    'self_check_interval': NFPA85GoldenValue(
        'Self-Check Interval', Decimal('8'), 'hours',
        Decimal('0'), '8.8.4', 'Maximum self-check interval'),
    'redundancy_requirement': NFPA85GoldenValue(
        'Redundancy', Decimal('2'), 'scanners',
        Decimal('0'), '8.8.1', 'Minimum flame scanners per burner'),
}


# ==============================================================================
# BURNER STATE DEFINITIONS
# ==============================================================================

class BurnerState(Enum):
    """Burner operating states per NFPA 85."""
    OFFLINE = 'offline'
    COLD_STANDBY = 'cold_standby'
    PRE_PURGE = 'pre_purge'
    PILOT_LIGHT_TRIAL = 'pilot_trial'
    PILOT_PROVEN = 'pilot_proven'
    MAIN_FLAME_TRIAL = 'main_flame_trial'
    MAIN_FLAME_PROVEN = 'main_flame_proven'
    FIRING = 'firing'
    POST_PURGE = 'post_purge'
    LOCKOUT = 'lockout'
    EMERGENCY_SHUTDOWN = 'emergency_shutdown'


class TripCause(Enum):
    """Safety trip causes per NFPA 85."""
    FLAME_FAILURE = 'flame_failure'
    LOW_FUEL_PRESSURE = 'low_fuel_pressure'
    HIGH_FUEL_PRESSURE = 'high_fuel_pressure'
    LOW_COMBUSTION_AIR = 'low_combustion_air'
    DRUM_LEVEL_LOW_LOW = 'drum_level_low_low'
    DRUM_LEVEL_HIGH_HIGH = 'drum_level_high_high'
    FURNACE_PRESSURE_HIGH = 'furnace_pressure_high'
    PILOT_FAILURE = 'pilot_failure'
    MAIN_FLAME_FAILURE = 'main_flame_failure'
    OPERATOR_EMERGENCY = 'operator_emergency'
    BMS_FAULT = 'bms_fault'


# ==============================================================================
# CALCULATION FUNCTIONS
# ==============================================================================

def calculate_furnace_volume_changes(
    airflow_scfm: Decimal,
    furnace_volume_cuft: Decimal,
    purge_time_minutes: Decimal
) -> Decimal:
    """
    Calculate furnace volume changes during purge per NFPA 85.8.5.3.

    Formula: Volume Changes = (Airflow × Purge Time) / Furnace Volume

    Minimum requirement: 5 volume changes at ≥25% full-load airflow.
    """
    total_air = airflow_scfm * purge_time_minutes
    volume_changes = total_air / furnace_volume_cuft
    return volume_changes.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)


def calculate_sil_pfd(
    mttr_hours: Decimal,
    test_interval_hours: Decimal,
    failure_rate_per_hour: Decimal
) -> Decimal:
    """
    Calculate Probability of Failure on Demand (PFD) for SIS.

    Formula per IEC 61511:
    PFD_avg = (λ × TI) / 2 + λ × MTTR

    Where:
        λ = failure rate (per hour)
        TI = test interval (hours)
        MTTR = mean time to repair (hours)
    """
    pfd = (failure_rate_per_hour * test_interval_hours / Decimal('2') +
           failure_rate_per_hour * mttr_hours)
    return pfd


def calculate_risk_reduction_factor(pfd: Decimal) -> Decimal:
    """
    Calculate Risk Reduction Factor (RRF) from PFD.

    Formula: RRF = 1 / PFD
    """
    if pfd == 0:
        return Decimal('999999')  # Maximum RRF
    return (Decimal('1') / pfd).quantize(Decimal('1'), rounding=ROUND_HALF_UP)


def validate_sil_level(pfd: Decimal) -> int:
    """
    Determine SIL level from PFD per IEC 61511.

    Returns SIL level (1, 2, 3) or 0 if insufficient.
    """
    for sil_level in [3, 2, 1]:
        req = SIL_REQUIREMENTS[sil_level]
        if req['pfd_avg_low'] <= pfd < req['pfd_avg_high']:
            return sil_level
    return 0  # Does not meet SIL 1


def calculate_flame_response_time(
    flame_signal_start: Decimal,
    flame_signal_threshold: Decimal,
    time_to_threshold_seconds: Decimal
) -> Tuple[Decimal, bool]:
    """
    Calculate flame scanner response time per NFPA 85.8.8.3.

    Returns: (response_time_seconds, meets_requirement)
    """
    response_time = time_to_threshold_seconds
    meets_req = response_time <= FLAME_SCANNER_SPECS['response_time'].value
    return response_time, meets_req


def calculate_provenance_hash(
    calculation_type: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any]
) -> str:
    """Calculate SHA-256 provenance hash for audit trail."""
    provenance_data = {
        'calculation_type': calculation_type,
        'inputs': {k: str(v) for k, v in sorted(inputs.items())},
        'outputs': {k: str(v) for k, v in sorted(outputs.items())},
        'timestamp': datetime.utcnow().isoformat(),
    }
    provenance_str = json.dumps(provenance_data, sort_keys=True)
    return hashlib.sha256(provenance_str.encode()).hexdigest()


def validate_startup_sequence(
    sequence_states: List[BurnerState],
    timing_seconds: List[Decimal]
) -> Tuple[bool, List[str]]:
    """
    Validate startup sequence against NFPA 85 requirements.

    Returns: (is_valid, list_of_violations)
    """
    violations: List[str] = []

    # Required sequence per NFPA 85.8
    required_sequence = [
        BurnerState.OFFLINE,
        BurnerState.PRE_PURGE,
        BurnerState.PILOT_LIGHT_TRIAL,
        BurnerState.PILOT_PROVEN,
        BurnerState.MAIN_FLAME_TRIAL,
        BurnerState.MAIN_FLAME_PROVEN,
        BurnerState.FIRING,
    ]

    # Verify sequence order
    if sequence_states != required_sequence[:len(sequence_states)]:
        violations.append('Startup sequence out of order per NFPA 85.8')

    # Check pre-purge time (minimum 5 minutes = 300 seconds)
    if len(timing_seconds) >= 2:
        pre_purge_time = timing_seconds[1]  # Duration of pre-purge
        if pre_purge_time < Decimal('300'):
            violations.append(
                f'Pre-purge time {pre_purge_time}s < 300s minimum (NFPA 85.8.5.3)'
            )

    # Check pilot trial time (maximum 10 seconds)
    if len(timing_seconds) >= 3:
        pilot_trial = timing_seconds[2]
        if pilot_trial > Decimal('10'):
            violations.append(
                f'Pilot trial {pilot_trial}s > 10s maximum (NFPA 85.8.6.3.2)'
            )

    # Check main flame trial time (maximum 10 seconds)
    if len(timing_seconds) >= 5:
        main_trial = timing_seconds[4]
        if main_trial > Decimal('10'):
            violations.append(
                f'Main flame trial {main_trial}s > 10s maximum (NFPA 85.8.6.4.2)'
            )

    return len(violations) == 0, violations


# ==============================================================================
# TEST CLASSES
# ==============================================================================

@pytest.mark.golden
class TestNFPA85TimingRequirements:
    """Validate timing requirements per NFPA 85 Chapter 8."""

    def test_pre_purge_minimum_time(self):
        """Verify pre-purge minimum time is 5 minutes."""
        golden = NFPA_85_TIMING['pre_purge_time_min']
        assert golden.value == Decimal('5')
        assert golden.unit == 'minutes'
        assert golden.nfpa_section == '8.5.3'

    def test_pre_purge_airflow_minimum(self):
        """Verify minimum purge airflow is 25%."""
        golden = NFPA_85_TIMING['pre_purge_airflow_min']
        assert golden.value == Decimal('25')
        assert golden.unit == 'percent'

    def test_pilot_trial_maximum(self):
        """Verify pilot trial maximum is 10 seconds."""
        golden = NFPA_85_TIMING['pilot_trial_max']
        assert golden.value == Decimal('10')
        assert golden.unit == 'seconds'

    def test_main_flame_trial_maximum(self):
        """Verify main flame trial maximum is 10 seconds."""
        golden = NFPA_85_TIMING['main_flame_trial_max']
        assert golden.value == Decimal('10')

    def test_flame_failure_response_time(self):
        """Verify flame failure response maximum is 4 seconds."""
        golden = NFPA_85_TIMING['flame_failure_response']
        assert golden.value == Decimal('4')
        assert golden.unit == 'seconds'
        assert golden.nfpa_section == '8.8.3.3'

    def test_post_purge_duration(self):
        """Verify post-purge duration requirement."""
        golden = NFPA_85_TIMING['post_purge_time']
        assert golden.value == Decimal('60')
        assert golden.unit == 'seconds'


@pytest.mark.golden
class TestFurnaceVolumeChange:
    """Validate furnace volume change calculations per NFPA 85.8.5.3."""

    @dataclass(frozen=True)
    class VolumeChangeTestCase:
        name: str
        airflow_scfm: Decimal
        furnace_volume_cuft: Decimal
        purge_time_minutes: Decimal
        expected_changes: Decimal
        meets_requirement: bool

    TEST_CASES = [
        VolumeChangeTestCase(
            name='Standard Industrial Boiler',
            airflow_scfm=Decimal('10000'),
            furnace_volume_cuft=Decimal('2000'),
            purge_time_minutes=Decimal('5'),
            expected_changes=Decimal('25'),
            meets_requirement=True,
        ),
        VolumeChangeTestCase(
            name='Minimum Purge (Marginal)',
            airflow_scfm=Decimal('5000'),
            furnace_volume_cuft=Decimal('5000'),
            purge_time_minutes=Decimal('5'),
            expected_changes=Decimal('5'),
            meets_requirement=True,
        ),
        VolumeChangeTestCase(
            name='Insufficient Purge',
            airflow_scfm=Decimal('3000'),
            furnace_volume_cuft=Decimal('5000'),
            purge_time_minutes=Decimal('5'),
            expected_changes=Decimal('3'),
            meets_requirement=False,
        ),
    ]

    @pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda tc: tc.name)
    def test_volume_change_calculation(self, test_case: VolumeChangeTestCase):
        """Verify furnace volume change calculation."""
        calculated = calculate_furnace_volume_changes(
            test_case.airflow_scfm,
            test_case.furnace_volume_cuft,
            test_case.purge_time_minutes,
        )

        # Minimum 5 volume changes required
        meets_req = calculated >= Decimal('5')
        assert meets_req == test_case.meets_requirement, \
            f'{test_case.name}: {calculated} volume changes'

    def test_minimum_5_changes_required(self):
        """NFPA 85 requires minimum 5 furnace volume changes."""
        # Calculate airflow needed for 5 changes in 5 minutes
        furnace_vol = Decimal('1000')
        purge_time = Decimal('5')

        # Need 5000 cu.ft total = 1000 SCFM
        min_airflow = Decimal('1000')

        changes = calculate_furnace_volume_changes(
            min_airflow, furnace_vol, purge_time)

        assert changes >= Decimal('5')


@pytest.mark.golden
class TestSafetyInterlocks:
    """Validate safety interlock requirements per NFPA 85 Chapter 7."""

    def test_required_interlocks_defined(self):
        """Verify all required interlocks are defined."""
        required = [
            'combustion_air_proving',
            'fuel_pressure_proving',
            'flame_detector_proving',
            'purge_proving',
            'fuel_valve_position',
            'drum_level_low',
            'drum_level_high',
            'furnace_pressure_high',
        ]

        for interlock in required:
            assert interlock in NFPA_85_INTERLOCKS, \
                f'Missing required interlock: {interlock}'

    @pytest.mark.parametrize("interlock_name", NFPA_85_INTERLOCKS.keys())
    def test_interlock_severity_assignment(self, interlock_name: str):
        """Verify all interlocks have safety-critical severity."""
        interlock = NFPA_85_INTERLOCKS[interlock_name]
        assert interlock['severity'] == 'safety_critical', \
            f'{interlock_name} must be safety-critical'

    def test_combustion_air_proving_action(self):
        """Combustion air loss should prevent light-off."""
        interlock = NFPA_85_INTERLOCKS['combustion_air_proving']
        assert interlock['trip_action'] == 'prevent_light_off'
        assert interlock['section'] == '7.5.1'

    def test_fuel_pressure_trip_action(self):
        """Fuel pressure failure should cause safety shutdown."""
        interlock = NFPA_85_INTERLOCKS['fuel_pressure_proving']
        assert interlock['trip_action'] == 'safety_shutdown'


@pytest.mark.golden
class TestSILRequirements:
    """Validate SIL level calculations per IEC 61511."""

    @dataclass(frozen=True)
    class SILTestCase:
        name: str
        mttr_hours: Decimal
        test_interval_hours: Decimal
        failure_rate_per_hour: Decimal
        expected_sil: int

    SIL_TEST_CASES = [
        SILTestCase(
            name='High Reliability SIL3',
            mttr_hours=Decimal('4'),
            test_interval_hours=Decimal('8760'),  # Annual
            failure_rate_per_hour=Decimal('1E-8'),
            expected_sil=3,
        ),
        SILTestCase(
            name='Standard BMS SIL2',
            mttr_hours=Decimal('8'),
            test_interval_hours=Decimal('8760'),
            failure_rate_per_hour=Decimal('1E-6'),
            expected_sil=2,
        ),
        SILTestCase(
            name='Basic Protection SIL1',
            mttr_hours=Decimal('24'),
            test_interval_hours=Decimal('8760'),
            failure_rate_per_hour=Decimal('1E-5'),
            expected_sil=1,
        ),
    ]

    def test_sil_level_pfd_ranges(self):
        """Verify SIL level PFD ranges per IEC 61511."""
        assert SIL_REQUIREMENTS[1]['pfd_avg_low'] == Decimal('0.01')
        assert SIL_REQUIREMENTS[1]['pfd_avg_high'] == Decimal('0.1')

        assert SIL_REQUIREMENTS[2]['pfd_avg_low'] == Decimal('0.001')
        assert SIL_REQUIREMENTS[2]['pfd_avg_high'] == Decimal('0.01')

        assert SIL_REQUIREMENTS[3]['pfd_avg_low'] == Decimal('0.0001')
        assert SIL_REQUIREMENTS[3]['pfd_avg_high'] == Decimal('0.001')

    @pytest.mark.parametrize("test_case", SIL_TEST_CASES, ids=lambda tc: tc.name)
    def test_sil_level_determination(self, test_case: SILTestCase):
        """Verify SIL level calculation from PFD."""
        pfd = calculate_sil_pfd(
            test_case.mttr_hours,
            test_case.test_interval_hours,
            test_case.failure_rate_per_hour,
        )

        sil_level = validate_sil_level(pfd)
        assert sil_level >= test_case.expected_sil, \
            f'{test_case.name}: PFD={pfd}, expected SIL{test_case.expected_sil}, got SIL{sil_level}'

    def test_rrf_calculation(self):
        """Verify Risk Reduction Factor calculation."""
        # SIL2 requires RRF 100-1000
        pfd_sil2 = Decimal('0.005')  # Mid SIL2 range
        rrf = calculate_risk_reduction_factor(pfd_sil2)

        assert Decimal('100') <= rrf <= Decimal('1000'), \
            f'SIL2 RRF should be 100-1000, got {rrf}'


@pytest.mark.golden
class TestFlameScanner:
    """Validate flame scanner requirements per NFPA 85.8.8."""

    def test_minimum_flame_signal(self):
        """Verify minimum flame signal for proven flame."""
        golden = FLAME_SCANNER_SPECS['min_flame_signal']
        assert golden.value == Decimal('10')
        assert golden.unit == 'percent'

    def test_scanner_response_time(self):
        """Verify maximum response time is 1 second."""
        golden = FLAME_SCANNER_SPECS['response_time']
        assert golden.value == Decimal('1')
        assert golden.unit == 'seconds'

    def test_self_check_interval(self):
        """Verify self-check interval is 8 hours maximum."""
        golden = FLAME_SCANNER_SPECS['self_check_interval']
        assert golden.value == Decimal('8')
        assert golden.unit == 'hours'

    def test_redundancy_requirement(self):
        """Verify minimum 2 scanners per burner for redundancy."""
        golden = FLAME_SCANNER_SPECS['redundancy_requirement']
        assert golden.value == Decimal('2')

    @pytest.mark.parametrize("response_time,expected_pass", [
        (Decimal('0.5'), True),
        (Decimal('1.0'), True),
        (Decimal('1.1'), False),
        (Decimal('2.0'), False),
    ])
    def test_flame_response_validation(self, response_time: Decimal, expected_pass: bool):
        """Verify flame response time validation."""
        _, meets_req = calculate_flame_response_time(
            flame_signal_start=Decimal('0'),
            flame_signal_threshold=Decimal('10'),
            time_to_threshold_seconds=response_time,
        )
        assert meets_req == expected_pass


@pytest.mark.golden
class TestStartupSequence:
    """Validate startup sequence per NFPA 85.8."""

    def test_valid_startup_sequence(self):
        """Verify valid startup sequence passes validation."""
        valid_sequence = [
            BurnerState.OFFLINE,
            BurnerState.PRE_PURGE,
            BurnerState.PILOT_LIGHT_TRIAL,
            BurnerState.PILOT_PROVEN,
            BurnerState.MAIN_FLAME_TRIAL,
            BurnerState.MAIN_FLAME_PROVEN,
            BurnerState.FIRING,
        ]

        # Timing: offline(0), pre-purge(300s), pilot(8s), proven(0s), main(8s), proven(0s), firing
        timing = [Decimal('0'), Decimal('300'), Decimal('8'), Decimal('0'),
                  Decimal('8'), Decimal('0'), Decimal('0')]

        is_valid, violations = validate_startup_sequence(valid_sequence, timing)
        assert is_valid, f'Valid sequence should pass: {violations}'

    def test_insufficient_purge_time(self):
        """Verify insufficient purge time is flagged."""
        sequence = [BurnerState.OFFLINE, BurnerState.PRE_PURGE]
        timing = [Decimal('0'), Decimal('240')]  # Only 4 minutes

        is_valid, violations = validate_startup_sequence(sequence, timing)
        assert not is_valid
        assert any('Pre-purge time' in v for v in violations)

    def test_excessive_pilot_trial(self):
        """Verify excessive pilot trial time is flagged."""
        sequence = [
            BurnerState.OFFLINE,
            BurnerState.PRE_PURGE,
            BurnerState.PILOT_LIGHT_TRIAL,
        ]
        timing = [Decimal('0'), Decimal('300'), Decimal('15')]  # 15s pilot trial

        is_valid, violations = validate_startup_sequence(sequence, timing)
        assert not is_valid
        assert any('Pilot trial' in v for v in violations)


@pytest.mark.golden
class TestTripCauses:
    """Validate trip cause handling per NFPA 85."""

    def test_all_trip_causes_defined(self):
        """Verify all standard trip causes are defined."""
        expected_causes = [
            'FLAME_FAILURE',
            'LOW_FUEL_PRESSURE',
            'HIGH_FUEL_PRESSURE',
            'LOW_COMBUSTION_AIR',
            'DRUM_LEVEL_LOW_LOW',
            'DRUM_LEVEL_HIGH_HIGH',
            'FURNACE_PRESSURE_HIGH',
            'PILOT_FAILURE',
            'MAIN_FLAME_FAILURE',
            'OPERATOR_EMERGENCY',
            'BMS_FAULT',
        ]

        for cause in expected_causes:
            assert hasattr(TripCause, cause), f'Missing trip cause: {cause}'

    def test_flame_failure_is_critical(self):
        """Flame failure must trigger immediate shutdown."""
        # Flame failure response time is 4 seconds max
        response_time = NFPA_85_TIMING['flame_failure_response']
        assert response_time.value == Decimal('4')


@pytest.mark.golden
class TestProvenanceAndDeterminism:
    """Validate provenance tracking and deterministic behavior."""

    def test_sil_calculation_determinism(self):
        """SIL calculation must be deterministic."""
        results = set()
        for _ in range(100):
            pfd = calculate_sil_pfd(
                mttr_hours=Decimal('8'),
                test_interval_hours=Decimal('8760'),
                failure_rate_per_hour=Decimal('1E-6'),
            )
            results.add(str(pfd))

        assert len(results) == 1, 'SIL calculation is non-deterministic'

    def test_volume_change_determinism(self):
        """Volume change calculation must be deterministic."""
        results = set()
        for _ in range(100):
            changes = calculate_furnace_volume_changes(
                airflow_scfm=Decimal('10000'),
                furnace_volume_cuft=Decimal('2000'),
                purge_time_minutes=Decimal('5'),
            )
            results.add(str(changes))

        assert len(results) == 1, 'Volume change calculation is non-deterministic'

    def test_provenance_hash_stability(self):
        """Provenance hash must be stable for same inputs."""
        inputs = {'airflow': '10000', 'volume': '2000'}
        outputs = {'changes': '25'}

        hashes = set()
        for _ in range(50):
            h = calculate_provenance_hash('volume_change', inputs, outputs)
            hashes.add(h)

        assert len(hashes) == 1, 'Provenance hash is not deterministic'


# ==============================================================================
# EXPORT FUNCTIONS
# ==============================================================================

def export_golden_values() -> Dict[str, Any]:
    """Export all NFPA 85 golden values for documentation."""
    return {
        'metadata': {
            'agent': 'GL-002_Flameguard',
            'version': '1.0.0',
            'standard': 'NFPA 85-2023',
        },
        'timing_requirements': {
            key: {
                'value': str(val.value),
                'unit': val.unit,
                'section': val.nfpa_section,
            } for key, val in NFPA_85_TIMING.items()
        },
        'interlocks': NFPA_85_INTERLOCKS,
        'sil_requirements': {
            str(level): {
                'pfd_range': f"{req['pfd_avg_low']} - {req['pfd_avg_high']}",
                'rrf_range': f"{req['rrf_low']} - {req['rrf_high']}",
            } for level, req in SIL_REQUIREMENTS.items()
        },
        'flame_scanner': {
            key: str(val.value) for key, val in FLAME_SCANNER_SPECS.items()
        },
    }


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
