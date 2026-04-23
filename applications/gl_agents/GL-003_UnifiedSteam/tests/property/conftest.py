"""
Pytest Configuration for Property-Based Tests

This module provides Hypothesis configuration and shared fixtures
for property-based testing of GL-003 UnifiedSteam.

Profiles:
- ci: For CI/CD pipelines (200 examples, no deadline)
- dev: For local development (50 examples, verbose)
- full: For comprehensive testing (1000 examples)
- debug: For debugging failures (10 examples, debug output)

Author: GL-TestEngineer
Version: 1.0.0
"""

import os
import pytest
from typing import Dict, List, Tuple
from datetime import datetime, timezone

from hypothesis import settings, Verbosity, Phase, HealthCheck

import sys

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# =============================================================================
# HYPOTHESIS PROFILE CONFIGURATION
# =============================================================================

# CI Profile - Balanced for CI/CD pipelines
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,  # Disable deadline for CI (avoid flaky failures)
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ],
    phases=[
        Phase.explicit,
        Phase.reuse,
        Phase.generate,
        Phase.shrink,
    ],
    verbosity=Verbosity.normal,
    stateful_step_count=50,
    report_multiple_bugs=True,
    derandomize=False,  # Keep random for full coverage
    database=None,  # Use default database
)

# Development Profile - Fast iteration
settings.register_profile(
    "dev",
    max_examples=50,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
    ],
    verbosity=Verbosity.verbose,
    stateful_step_count=20,
    report_multiple_bugs=True,
    derandomize=False,
)

# Full Profile - Comprehensive testing
settings.register_profile(
    "full",
    max_examples=1000,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
        HealthCheck.large_base_example,
    ],
    phases=[
        Phase.explicit,
        Phase.reuse,
        Phase.generate,
        Phase.shrink,
    ],
    verbosity=Verbosity.normal,
    stateful_step_count=100,
    report_multiple_bugs=True,
    derandomize=False,
)

# Debug Profile - For investigating failures
settings.register_profile(
    "debug",
    max_examples=10,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ],
    verbosity=Verbosity.debug,
    stateful_step_count=10,
    report_multiple_bugs=False,
    derandomize=True,  # Make reproducible
)

# Quick Profile - For smoke tests
settings.register_profile(
    "quick",
    max_examples=10,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
    ],
    verbosity=Verbosity.quiet,
    stateful_step_count=5,
    report_multiple_bugs=False,
)

# Exhaustive Profile - For release testing
settings.register_profile(
    "exhaustive",
    max_examples=5000,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
        HealthCheck.large_base_example,
        HealthCheck.not_a_test_method,
    ],
    phases=[
        Phase.explicit,
        Phase.reuse,
        Phase.generate,
        Phase.shrink,
    ],
    verbosity=Verbosity.normal,
    stateful_step_count=200,
    report_multiple_bugs=True,
)


def pytest_configure(config):
    """Configure pytest with Hypothesis profile from environment."""
    # Get profile from environment variable or default to 'dev'
    profile = os.environ.get("HYPOTHESIS_PROFILE", "dev")

    # Load the profile
    try:
        settings.load_profile(profile)
        print(f"Hypothesis profile loaded: {profile}")
    except Exception as e:
        print(f"Warning: Could not load profile '{profile}': {e}")
        settings.load_profile("dev")

    # Add custom markers for property tests
    config.addinivalue_line(
        "markers",
        "hypothesis: marks tests as hypothesis property-based tests"
    )
    config.addinivalue_line(
        "markers",
        "slow_hypothesis: marks hypothesis tests that are slow"
    )
    config.addinivalue_line(
        "markers",
        "stateful: marks stateful hypothesis tests"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark property-based tests."""
    for item in items:
        # Auto-mark tests in property directory
        if "test_property" in str(item.fspath) or "/property/" in str(item.fspath):
            item.add_marker(pytest.mark.hypothesis)

        # Mark stateful tests
        if "StateMachine" in item.name or "stateful" in item.name.lower():
            item.add_marker(pytest.mark.stateful)


# =============================================================================
# SHARED FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def hypothesis_profile() -> str:
    """Return the current Hypothesis profile name."""
    return os.environ.get("HYPOTHESIS_PROFILE", "dev")


@pytest.fixture(scope="session")
def max_examples(hypothesis_profile: str) -> int:
    """Return the max_examples for current profile."""
    profile_examples = {
        "ci": 200,
        "dev": 50,
        "full": 1000,
        "debug": 10,
        "quick": 10,
        "exhaustive": 5000,
    }
    return profile_examples.get(hypothesis_profile, 50)


# =============================================================================
# IAPWS-IF97 REFERENCE DATA FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def iapws_constants() -> Dict[str, float]:
    """IAPWS-IF97 fundamental constants."""
    return {
        "R": 0.461526,  # kJ/(kg*K) specific gas constant
        "T_CRIT": 647.096,  # K critical temperature
        "P_CRIT": 22.064,  # MPa critical pressure
        "RHO_CRIT": 322.0,  # kg/m3 critical density
        "T_TRIPLE": 273.16,  # K triple point temperature
        "P_TRIPLE": 0.000611657,  # MPa triple point pressure
    }


@pytest.fixture(scope="module")
def region_boundaries() -> Dict[str, float]:
    """IAPWS-IF97 region boundaries."""
    return {
        "P_MIN": 0.000611657,  # MPa triple point
        "P_MAX_1_2": 100.0,  # MPa
        "P_MAX_5": 50.0,  # MPa
        "T_MIN": 273.15,  # K (0 C)
        "T_MAX_1_3": 623.15,  # K (350 C)
        "T_MAX_2": 1073.15,  # K (800 C)
        "T_MAX_5": 2273.15,  # K (2000 C)
    }


@pytest.fixture(scope="module")
def verification_points() -> Dict[str, List[Tuple]]:
    """IAPWS-IF97 verification test points."""
    return {
        "region1_table5": [
            # (P [MPa], T [K], v [m3/kg], h [kJ/kg], s [kJ/(kg*K)], u [kJ/kg])
            (3.0, 300.0, 0.00100215168e-2, 115.331273, 0.392294792, 112.324818),
            (80.0, 300.0, 0.000971180894e-2, 184.142828, 0.368563852, 106.448356),
            (80.0, 500.0, 0.00120241800e-2, 975.542239, 2.58041912, 971.934985),
        ],
        "region2_table15": [
            (0.001, 300.0, 0.394913866e2, 2549.91145, 9.15546786, 2411.69160),
            (0.001, 500.0, 0.658861816e2, 2928.62360, 9.87571394, 2862.75261),
            (3.0, 300.0, 0.00394913866, 2549.91145, 5.85296786, 2411.69160),
        ],
        "saturation_table33": [
            # (T [K], P_sat [MPa])
            (300.0, 0.00353658941),
            (500.0, 2.63889776),
            (600.0, 12.3443146),
        ],
    }


# =============================================================================
# STEAM SYSTEM STATE FIXTURES
# =============================================================================

@pytest.fixture
def typical_hp_header_state() -> Dict[str, float]:
    """Typical high pressure steam header conditions."""
    return {
        "pressure_psig": 600.0,
        "temperature_f": 750.0,
        "flow_klb_hr": 100.0,
        "saturation_temp_f": 489.0,  # Approximate for 600 psig
        "superheat_f": 261.0,
    }


@pytest.fixture
def typical_mp_header_state() -> Dict[str, float]:
    """Typical medium pressure steam header conditions."""
    return {
        "pressure_psig": 150.0,
        "temperature_f": 450.0,
        "flow_klb_hr": 50.0,
        "saturation_temp_f": 366.0,  # Approximate for 150 psig
        "superheat_f": 84.0,
    }


@pytest.fixture
def typical_lp_header_state() -> Dict[str, float]:
    """Typical low pressure steam header conditions."""
    return {
        "pressure_psig": 15.0,
        "temperature_f": 280.0,
        "flow_klb_hr": 20.0,
        "saturation_temp_f": 250.0,  # Approximate for 15 psig
        "superheat_f": 30.0,
    }


# =============================================================================
# SAFETY ENVELOPE FIXTURES
# =============================================================================

@pytest.fixture
def sample_alarm_margins():
    """Sample alarm margin configuration."""
    from safety.safety_envelope import AlarmMargins
    return AlarmMargins(
        warning_pct=10.0,
        alarm_pct=5.0,
        trip_pct=0.0,
    )


@pytest.fixture
def sample_pressure_limits(sample_alarm_margins):
    """Sample pressure limit configuration."""
    from safety.safety_envelope import SafetyEnvelope

    envelope = SafetyEnvelope()
    limits = envelope.define_pressure_limits(
        equipment_id="TEST-001",
        min_kpa=500.0,
        max_kpa=5000.0,
        alarm_margins=sample_alarm_margins,
    )
    return limits


# =============================================================================
# OPTIMIZATION FIXTURES
# =============================================================================

@pytest.fixture
def sample_desuperheater_state():
    """Sample desuperheater state for testing."""
    from optimization.desuperheater_optimizer import DesuperheaterState

    return DesuperheaterState(
        desuperheater_id="DS-001",
        inlet_temp_f=750.0,
        outlet_temp_f=600.0,
        setpoint_temp_f=580.0,
        steam_pressure_psig=600.0,
        saturation_temp_f=489.0,
        steam_flow_lb_hr=50000.0,
        spray_valve_position_pct=45.0,
        spray_flow_gpm=15.0,
        spray_water_temp_f=100.0,
        spray_water_pressure_psig=800.0,
        nozzle_delta_p_psi=100.0,
    )


@pytest.fixture
def sample_target_constraints():
    """Sample target constraints for testing."""
    from optimization.desuperheater_optimizer import TargetConstraints

    return TargetConstraints(
        min_outlet_temp_f=None,
        max_outlet_temp_f=None,
        target_superheat_f=50.0,
        min_approach_to_saturation_f=20.0,
        max_spray_valve_position_pct=90.0,
        max_spray_flow_gpm=50.0,
        max_temp_rate_f_min=50.0,
    )


# =============================================================================
# UNIT CONVERSION FIXTURES
# =============================================================================

@pytest.fixture
def pressure_conversion_factors() -> Dict[str, float]:
    """Pressure unit conversion factors to kPa."""
    return {
        "kPa": 1.0,
        "bar": 100.0,
        "psi": 6.89476,
        "MPa": 1000.0,
        "atm": 101.325,
    }


@pytest.fixture
def temperature_offsets() -> Dict[str, Tuple[float, float]]:
    """Temperature conversion parameters (offset, scale)."""
    return {
        "K": (0.0, 1.0),  # Base unit
        "C": (273.15, 1.0),  # K = C + 273.15
        "F": (459.67, 5/9),  # K = (F + 459.67) * 5/9
        "R": (0.0, 5/9),  # K = R * 5/9
    }
