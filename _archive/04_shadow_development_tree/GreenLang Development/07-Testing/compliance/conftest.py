# -*- coding: utf-8 -*-
"""
Compliance Test Configuration and Shared Fixtures

Provides regulatory reference data fixtures for compliance validation testing
of GreenLang Process Heat Agents.

Standards Covered:
    - EPA 40 CFR Part 60 (NSPS - New Source Performance Standards)
    - EPA 40 CFR Part 75 (CEMS - Continuous Emission Monitoring)
    - EPA 40 CFR Part 98 (GHG Mandatory Reporting)
    - IEC 61511 (Functional Safety - SIL Calculations)
    - NFPA 85 (Boiler and Combustion Systems Hazards Code)
"""

from dataclasses import dataclass


# =============================================================================
# PYTEST MARKER REGISTRATION
# =============================================================================


def pytest_configure(config):
    """Register custom pytest markers for compliance tests."""
    config.addinivalue_line(
        "markers", "compliance: marks tests as regulatory compliance tests"
    )
    config.addinivalue_line(
        "markers", "epa_part60: marks tests for EPA Part 60 NSPS compliance"
    )
    config.addinivalue_line(
        "markers", "epa_part75: marks tests for EPA Part 75 CEMS compliance"
    )
    config.addinivalue_line(
        "markers", "epa_part98: marks tests for EPA Part 98 GHG compliance"
    )
    config.addinivalue_line(
        "markers", "iec61511: marks tests for IEC 61511 functional safety"
    )
    config.addinivalue_line(
        "markers", "nfpa85: marks tests for NFPA 85 combustion safety"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests requiring external system integration"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance benchmark tests"
    )


from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
import math
import pytest


# =============================================================================
# EPA PART 60 - NSPS REFERENCE DATA
# =============================================================================


@pytest.fixture(scope="session")
def epa_part60_emission_limits() -> Dict[str, Dict[str, float]]:
    """
    EPA 40 CFR Part 60 NSPS emission limits by source category.

    Units:
        - NOx: lb/MMBTU
        - SO2: lb/MMBTU
        - PM (Particulate Matter): lb/MMBTU
        - CO: ppm @ 3% O2

    Source: 40 CFR Part 60, Subparts D, Da, Db, Dc
    """
    return {
        # Subpart Db - Industrial-Commercial-Institutional Steam Generating Units
        "steam_generator_natural_gas_db": {
            "nox_lb_mmbtu": 0.20,  # Low NOx requirement
            "co_ppm_3pct_o2": 400.0,
            "so2_lb_mmbtu": 0.50,
            "pm_lb_mmbtu": 0.05,
            "heat_input_threshold_mmbtu_hr": 100.0,
        },
        "steam_generator_fuel_oil_db": {
            "nox_lb_mmbtu": 0.30,
            "co_ppm_3pct_o2": 400.0,
            "so2_lb_mmbtu": 0.50,
            "pm_lb_mmbtu": 0.10,
            "heat_input_threshold_mmbtu_hr": 100.0,
        },
        "steam_generator_coal_db": {
            "nox_lb_mmbtu": 0.50,
            "co_ppm_3pct_o2": 200.0,
            "so2_lb_mmbtu": 1.20,
            "pm_lb_mmbtu": 0.05,
            "heat_input_threshold_mmbtu_hr": 100.0,
        },
        # Subpart Dc - Small Industrial-Commercial-Institutional
        # (10-100 MMBTU/hr capacity)
        "steam_generator_natural_gas_dc": {
            "nox_lb_mmbtu": 0.30,
            "co_ppm_3pct_o2": 400.0,
            "so2_lb_mmbtu": 0.50,
            "pm_lb_mmbtu": 0.10,
            "heat_input_threshold_mmbtu_hr": 10.0,
        },
        # Process heaters (Subpart Ja)
        "process_heater_refinery": {
            "nox_lb_mmbtu": 0.040,  # Very stringent for refineries
            "co_ppm_3pct_o2": 200.0,
        },
    }


@pytest.fixture(scope="session")
def epa_part60_nsps_subpart_mapping() -> Dict[str, str]:
    """Mapping of equipment types to applicable NSPS subparts."""
    return {
        "steam_boiler_large": "Subpart Db",
        "steam_boiler_small": "Subpart Dc",
        "steam_boiler_utility": "Subpart Da",
        "process_heater_refinery": "Subpart Ja",
        "combustion_turbine": "Subpart KKKK",
        "stationary_engine_ci": "Subpart IIII",
        "stationary_engine_si": "Subpart JJJJ",
    }


@pytest.fixture(scope="session")
def epa_part60_test_methods() -> Dict[str, Dict[str, Any]]:
    """EPA reference test methods for emission verification."""
    return {
        "method_1": {
            "description": "Sample and Velocity Traverses for Stationary Sources",
            "pollutant": "stack_flow",
        },
        "method_2": {
            "description": "Determination of Stack Gas Velocity and Volumetric Flow Rate",
            "pollutant": "stack_flow",
        },
        "method_3": {
            "description": "Gas Analysis for CO2, O2, Excess Air, and Dry Molecular Weight",
            "pollutant": ["co2", "o2"],
        },
        "method_5": {
            "description": "Determination of Particulate Matter Emissions",
            "pollutant": "pm",
            "minimum_sample_time_minutes": 60,
        },
        "method_7": {
            "description": "Determination of Nitrogen Oxide Emissions",
            "pollutant": "nox",
        },
        "method_10": {
            "description": "Determination of Carbon Monoxide Emissions",
            "pollutant": "co",
        },
        "method_19": {
            "description": "Determination of Sulfur Dioxide Removal Efficiency and SO2/PM Emission Rates",
            "pollutant": ["so2", "pm"],
            "o2_correction_reference": 3.0,  # 3% O2 reference
        },
    }


# =============================================================================
# EPA PART 75 - CEMS REFERENCE DATA
# =============================================================================


@pytest.fixture(scope="session")
def epa_part75_data_availability_requirements() -> Dict[str, float]:
    """
    EPA 40 CFR Part 75 CEMS data availability requirements.

    Requires minimum 90% data availability for each quarter.
    Substitute data procedures apply for missing data.
    """
    return {
        "minimum_availability_pct": 90.0,
        "quarterly_minimum_hours": 2190,  # 91.25 days * 24 hours
        "annual_minimum_hours": 8760,
        "rata_relative_accuracy_pct": 10.0,  # RATA must be within 10%
        "rata_absolute_tolerance_nox_lb_mmbtu": 0.020,  # Alternative absolute
        "rata_absolute_tolerance_so2_ppm": 15.0,
        "daily_calibration_drift_pct": 2.5,
        "cylinder_gas_audit_tolerance_pct": 5.0,
    }


@pytest.fixture(scope="session")
def epa_part75_qaqc_requirements() -> Dict[str, Any]:
    """
    EPA 40 CFR Part 75 QA/QC test requirements.

    Includes RATA, CGA, linearity checks, and calibration drift.
    """
    return {
        "rata": {
            "frequency": "annual",
            "relative_accuracy_limit_pct": 10.0,
            "minimum_test_runs": 9,
            "alternative_relative_accuracy_pct": 7.5,  # For sources < 250 MW
            "bias_adjustment_factor_limit": 1.1,
        },
        "cga": {  # Cylinder Gas Audit
            "frequency": "quarterly",
            "tolerance_pct": 5.0,
            "gas_levels": ["low", "mid", "high"],
        },
        "linearity": {
            "frequency": "quarterly",
            "tolerance_pct": 5.0,
            "gas_levels": [0.20, 0.50, 0.90],  # As fraction of span
            "required_runs_per_level": 3,
        },
        "daily_calibration": {
            "frequency": "daily",
            "zero_drift_limit_pct": 2.5,
            "span_drift_limit_pct": 2.5,
            "calibration_error_limit_pct": 2.5,
        },
        "relative_accuracy_audit": {
            "frequency": "3_years",
            "limit_pct": 15.0,
        },
    }


@pytest.fixture(scope="session")
def epa_part75_substitute_data_procedures() -> Dict[str, Any]:
    """
    EPA 40 CFR Part 75 substitute data procedures for missing CEMS data.

    Substitute values are intentionally conservative (higher emissions).
    """
    return {
        "monitor_data_availability_ranges": [
            # (min_availability, max_availability, substitute_method)
            (0.95, 1.00, "average_of_hour_before_and_after"),
            (0.90, 0.95, "90th_percentile_lookback_2160_hours"),
            (0.80, 0.90, "95th_percentile_lookback_2160_hours"),
            (0.00, 0.80, "maximum_potential_value"),
        ],
        "maximum_potential_concentration_nox_ppm": 200.0,
        "maximum_potential_concentration_so2_ppm": 500.0,
        "maximum_potential_concentration_co2_pct": 14.0,
        "maximum_potential_flow_scfh_factor": 1.25,  # 125% of design
    }


# =============================================================================
# EPA PART 98 - GHG REPORTING REFERENCE DATA
# =============================================================================


@pytest.fixture(scope="session")
def epa_part98_emission_factors() -> Dict[str, Dict[str, float]]:
    """
    EPA 40 CFR Part 98 Table C-1 emission factors.

    CO2 emission factors in kg CO2/MMBTU (HHV basis).
    CH4 and N2O factors in kg/MMBTU.
    """
    return {
        "natural_gas": {
            "co2_kg_per_mmbtu": 53.06,
            "ch4_kg_per_mmbtu": 0.001,
            "n2o_kg_per_mmbtu": 0.0001,
            "default_hhv_btu_per_scf": 1028,
        },
        "distillate_fuel_oil_no2": {
            "co2_kg_per_mmbtu": 73.16,
            "ch4_kg_per_mmbtu": 0.003,
            "n2o_kg_per_mmbtu": 0.0006,
            "default_hhv_btu_per_gallon": 138690,
        },
        "residual_fuel_oil_no6": {
            "co2_kg_per_mmbtu": 75.10,
            "ch4_kg_per_mmbtu": 0.003,
            "n2o_kg_per_mmbtu": 0.0006,
            "default_hhv_btu_per_gallon": 149690,
        },
        "propane": {
            "co2_kg_per_mmbtu": 62.87,
            "ch4_kg_per_mmbtu": 0.003,
            "n2o_kg_per_mmbtu": 0.0006,
            "default_hhv_btu_per_gallon": 91500,
        },
        "bituminous_coal": {
            "co2_kg_per_mmbtu": 93.28,
            "ch4_kg_per_mmbtu": 0.011,
            "n2o_kg_per_mmbtu": 0.0016,
            "default_hhv_btu_per_ton": 24930000,
        },
        "sub_bituminous_coal": {
            "co2_kg_per_mmbtu": 97.17,
            "ch4_kg_per_mmbtu": 0.011,
            "n2o_kg_per_mmbtu": 0.0016,
            "default_hhv_btu_per_ton": 17250000,
        },
        "lignite": {
            "co2_kg_per_mmbtu": 97.72,
            "ch4_kg_per_mmbtu": 0.011,
            "n2o_kg_per_mmbtu": 0.0016,
            "default_hhv_btu_per_ton": 14210000,
        },
        "wood_biomass": {
            "co2_kg_per_mmbtu": 93.80,  # Biogenic - may be excluded
            "ch4_kg_per_mmbtu": 0.032,
            "n2o_kg_per_mmbtu": 0.0042,
            "biogenic": True,
        },
    }


@pytest.fixture(scope="session")
def epa_part98_tier_requirements() -> Dict[str, Dict[str, Any]]:
    """
    EPA 40 CFR Part 98 Subpart C tier calculation methodology requirements.

    Tier 1: Default emission factors
    Tier 2: Site-specific HHV
    Tier 3: Carbon content analysis
    Tier 4: CEMS measurement
    """
    return {
        "tier_1": {
            "description": "Default emission factors from Table C-1",
            "data_requirements": ["fuel_quantity", "fuel_type"],
            "applicable_for_emissions_less_than": 25000,  # metric tons CO2e/yr
            "uncertainty_pct": 10.0,
        },
        "tier_2": {
            "description": "Site-specific HHV with default emission factors",
            "data_requirements": ["fuel_quantity", "fuel_type", "measured_hhv"],
            "hhv_measurement_frequency": "monthly",
            "applicable_for_emissions_range": (25000, 250000),
            "uncertainty_pct": 5.0,
        },
        "tier_3": {
            "description": "Carbon content analysis",
            "data_requirements": ["fuel_quantity", "carbon_content", "molecular_weight"],
            "carbon_analysis_frequency": "monthly",
            "applicable_for_emissions_greater_than": 250000,
            "uncertainty_pct": 2.0,
        },
        "tier_4": {
            "description": "CEMS continuous monitoring",
            "data_requirements": ["cems_co2_concentration", "stack_flow"],
            "data_availability_minimum_pct": 90.0,
            "uncertainty_pct": 1.0,
        },
    }


@pytest.fixture(scope="session")
def epa_part98_gwp_values() -> Dict[str, int]:
    """
    Global Warming Potential (GWP) values from Part 98 Table A-1.

    100-year GWP values relative to CO2 = 1.
    """
    return {
        "co2": 1,
        "ch4": 25,  # Updated from AR4
        "n2o": 298,
        "hfc_134a": 1430,
        "sf6": 22800,
        "nf3": 17200,
    }


# =============================================================================
# IEC 61511 - FUNCTIONAL SAFETY REFERENCE DATA
# =============================================================================


@pytest.fixture(scope="session")
def iec61511_sil_targets() -> Dict[int, Dict[str, float]]:
    """
    IEC 61511 Safety Integrity Level (SIL) target PFD ranges.

    PFD = Probability of Failure on Demand (average)
    RRF = Risk Reduction Factor
    """
    return {
        1: {
            "pfd_avg_min": 1e-2,
            "pfd_avg_max": 1e-1,
            "rrf_min": 10,
            "rrf_max": 100,
            "safe_failure_fraction_min": 0.60,
            "hardware_fault_tolerance": 0,
            "proof_test_coverage_min": 0.90,
        },
        2: {
            "pfd_avg_min": 1e-3,
            "pfd_avg_max": 1e-2,
            "rrf_min": 100,
            "rrf_max": 1000,
            "safe_failure_fraction_min": 0.90,
            "hardware_fault_tolerance": 1,
            "proof_test_coverage_min": 0.95,
        },
        3: {
            "pfd_avg_min": 1e-4,
            "pfd_avg_max": 1e-3,
            "rrf_min": 1000,
            "rrf_max": 10000,
            "safe_failure_fraction_min": 0.99,
            "hardware_fault_tolerance": 2,
            "proof_test_coverage_min": 0.99,
        },
        4: {
            "pfd_avg_min": 1e-5,
            "pfd_avg_max": 1e-4,
            "rrf_min": 10000,
            "rrf_max": 100000,
            "safe_failure_fraction_min": 0.999,
            "hardware_fault_tolerance": 3,
            "proof_test_coverage_min": 0.99,
        },
    }


@pytest.fixture(scope="session")
def iec61511_voting_architectures() -> Dict[str, Dict[str, Any]]:
    """
    IEC 61511 voting architecture specifications.

    Defines PFD formulas and hardware requirements for each architecture.
    """
    return {
        "1oo1": {
            "channels": 1,
            "trips_on": 1,
            "pfd_formula": "lambda_du * ti / 2",
            "hardware_fault_tolerance": 0,
            "max_sil": 1,
            "description": "Single channel - any failure causes spurious trip or failure to trip",
        },
        "1oo2": {
            "channels": 2,
            "trips_on": 1,
            "pfd_formula": "(lambda_du * ti)^2 / 3",
            "hardware_fault_tolerance": 1,
            "max_sil": 3,
            "spurious_trip_rate": "2 * lambda_s",
            "description": "Redundant - high safety, low availability",
        },
        "2oo2": {
            "channels": 2,
            "trips_on": 2,
            "pfd_formula": "lambda_du * ti",
            "hardware_fault_tolerance": 0,
            "max_sil": 1,
            "spurious_trip_rate": "lambda_s^2",
            "description": "Series - high availability, low safety",
        },
        "2oo3": {
            "channels": 3,
            "trips_on": 2,
            "pfd_formula": "(lambda_du * ti)^2",
            "hardware_fault_tolerance": 1,
            "max_sil": 3,
            "spurious_trip_rate": "3 * lambda_s^2",
            "description": "Triple modular redundancy - balanced safety and availability",
        },
        "2oo4": {
            "channels": 4,
            "trips_on": 2,
            "pfd_formula": "(lambda_du * ti)^2 * 2",
            "hardware_fault_tolerance": 2,
            "max_sil": 4,
            "description": "Quad redundancy - highest safety",
        },
    }


@pytest.fixture(scope="session")
def iec61511_typical_failure_rates() -> Dict[str, Dict[str, float]]:
    """
    Typical failure rates for SIS components per IEC 61511/61508.

    Failure rates in failures per hour (1/hr).
    Lambda_DU = Dangerous Undetected failure rate
    Lambda_DD = Dangerous Detected failure rate
    Lambda_S = Safe failure rate
    """
    return {
        "pressure_transmitter": {
            "lambda_du": 1.5e-6,
            "lambda_dd": 1.0e-6,
            "lambda_s": 3.0e-6,
            "sff": 0.91,  # Safe Failure Fraction
        },
        "temperature_transmitter": {
            "lambda_du": 2.0e-6,
            "lambda_dd": 1.5e-6,
            "lambda_s": 4.0e-6,
            "sff": 0.89,
        },
        "level_transmitter": {
            "lambda_du": 2.5e-6,
            "lambda_dd": 2.0e-6,
            "lambda_s": 5.0e-6,
            "sff": 0.87,
        },
        "flow_transmitter": {
            "lambda_du": 3.0e-6,
            "lambda_dd": 2.5e-6,
            "lambda_s": 6.0e-6,
            "sff": 0.85,
        },
        "flame_detector_uv": {
            "lambda_du": 5.0e-6,
            "lambda_dd": 3.0e-6,
            "lambda_s": 8.0e-6,
            "sff": 0.82,
        },
        "shutdown_valve": {
            "lambda_du": 5.0e-6,
            "lambda_dd": 2.0e-6,
            "lambda_s": 10.0e-6,
            "sff": 0.80,
        },
        "solenoid_valve": {
            "lambda_du": 8.0e-6,
            "lambda_dd": 4.0e-6,
            "lambda_s": 15.0e-6,
            "sff": 0.78,
        },
        "safety_plc": {
            "lambda_du": 1.0e-7,
            "lambda_dd": 5.0e-7,
            "lambda_s": 2.0e-6,
            "sff": 0.98,
        },
    }


@pytest.fixture(scope="session")
def iec61511_proof_test_intervals() -> Dict[str, float]:
    """
    Typical proof test intervals in hours for SIS components.

    Based on industry practice and manufacturer recommendations.
    """
    return {
        "transmitter_quarterly": 2190,  # 3 months
        "transmitter_semi_annual": 4380,  # 6 months
        "transmitter_annual": 8760,  # 1 year
        "final_element_monthly": 730,  # 1 month
        "final_element_quarterly": 2190,
        "final_element_annual": 8760,
        "safety_plc_annual": 8760,
        "typical_sis": 8760,  # Default annual
        "high_demand_sis": 2190,  # Quarterly for critical
    }


# =============================================================================
# NFPA 85 - COMBUSTION SAFETY REFERENCE DATA
# =============================================================================


@pytest.fixture(scope="session")
def nfpa85_timing_requirements() -> Dict[str, Dict[str, float]]:
    """
    NFPA 85 timing requirements for combustion safety sequences.

    All times in seconds unless otherwise noted.
    """
    return {
        "purge": {
            "minimum_volume_changes": 4,
            "minimum_time_s": 60,  # For some boiler types
            "air_flow_minimum_pct": 25,  # Of maximum rated air flow
            "damper_position_minimum_pct": 100,  # Dampers must be open
        },
        "pilot_lightoff": {
            "pilot_flame_establishing_period_s": 10,  # Max time to prove pilot
            "pilot_flame_stabilizing_period_s": 5,  # After proven, before main
            "igniter_energize_time_s": 10,  # Max igniter on time
            "pilot_valve_opening_time_s": 3,  # Max time from signal to open
        },
        "main_flame": {
            "main_flame_establishing_period_s": 10,  # Max time to prove main
            "main_flame_trial_for_ignition_s": 15,  # MTFI per NFPA
            "flame_failure_response_time_s": 4,  # Max time to close on loss
            "burner_safety_shutoff_time_s": 1,  # Safety valve closure time
        },
        "flame_detection": {
            "flame_signal_loss_trip_time_s": 4,  # Standard for multiple burner
            "single_burner_trip_time_s": 4,  # For single burner
            "scanner_response_time_s": 1,  # Maximum scanner latency
            "self_checking_interval_s": 10,  # For self-checking scanners
        },
        "shutdown": {
            "post_purge_time_s": 60,  # Minimum post-purge
            "post_purge_volume_changes": 4,
            "fuel_valve_closure_time_s": 1,  # Max valve closure time
        },
        "startup_sequence": {
            "pre_ignition_interlock_verify_s": 5,  # Interlock check time
            "low_fire_hold_time_s": 60,  # Minimum time at low fire
            "modulation_release_delay_s": 30,  # After main proven
        },
    }


@pytest.fixture(scope="session")
def nfpa85_interlock_requirements() -> Dict[str, Dict[str, Any]]:
    """
    NFPA 85 required interlocks for combustion systems.

    Lists required interlocks by system type.
    """
    return {
        "fuel_supply": {
            "low_fuel_pressure": {"action": "prevent_ignition", "critical": True},
            "high_fuel_pressure": {"action": "trip", "critical": True},
            "double_block_leak_test": {"action": "prevent_start", "test_frequency_days": 7},
        },
        "combustion_air": {
            "combustion_air_flow_low": {"action": "trip", "critical": True},
            "forced_draft_fan_running": {"action": "prevent_ignition", "critical": True},
            "air_damper_position": {"action": "trip_if_closed", "critical": False},
        },
        "drum_level": {
            "low_water_cutoff": {"action": "trip", "critical": True, "redundancy": "2oo3"},
            "high_water_cutoff": {"action": "trip", "critical": True},
            "feed_pump_running": {"action": "alarm", "critical": False},
        },
        "pressure": {
            "high_pressure_limit": {"action": "trip", "critical": True},
            "high_high_pressure": {"action": "emergency_shutdown", "critical": True},
            "steam_pressure_control": {"action": "modulate", "critical": False},
        },
        "flame": {
            "flame_failure": {"action": "trip", "response_time_s": 4, "critical": True},
            "flame_scanner_fault": {"action": "alarm", "critical": False},
        },
        "purge": {
            "purge_complete": {"action": "enable_ignition", "critical": True},
            "air_flow_proving": {"action": "verify", "critical": True},
        },
    }


@pytest.fixture(scope="session")
def nfpa85_bms_state_transitions() -> Dict[str, Dict[str, Any]]:
    """
    NFPA 85 BMS state machine transitions.

    Defines valid state transitions and conditions.
    """
    return {
        "IDLE": {
            "valid_transitions": ["PRE_PURGE"],
            "entry_conditions": [],
            "exit_conditions": ["start_request", "interlocks_satisfied"],
        },
        "PRE_PURGE": {
            "valid_transitions": ["PILOT_TRIAL", "LOCKOUT", "IDLE"],
            "entry_conditions": ["interlocks_satisfied", "air_flow_proven"],
            "exit_conditions": ["purge_complete", "interlock_trip", "abort"],
            "minimum_duration_s": 60,
            "air_flow_requirement_pct": 25,
        },
        "PILOT_TRIAL": {
            "valid_transitions": ["MAIN_FLAME_TRIAL", "POST_PURGE", "LOCKOUT"],
            "entry_conditions": ["purge_complete", "low_fire_position"],
            "exit_conditions": ["pilot_proven", "pilot_fail", "timeout"],
            "maximum_duration_s": 10,
        },
        "MAIN_FLAME_TRIAL": {
            "valid_transitions": ["RUNNING", "POST_PURGE", "LOCKOUT"],
            "entry_conditions": ["pilot_proven"],
            "exit_conditions": ["main_flame_proven", "flame_failure", "timeout"],
            "maximum_duration_s": 15,
        },
        "RUNNING": {
            "valid_transitions": ["POST_PURGE", "LOCKOUT"],
            "entry_conditions": ["main_flame_proven"],
            "exit_conditions": ["stop_request", "interlock_trip", "flame_failure"],
        },
        "POST_PURGE": {
            "valid_transitions": ["IDLE", "LOCKOUT"],
            "entry_conditions": ["fuel_valves_closed"],
            "exit_conditions": ["purge_complete", "interlock_trip"],
            "minimum_duration_s": 60,
        },
        "LOCKOUT": {
            "valid_transitions": ["IDLE"],
            "entry_conditions": ["safety_trip"],
            "exit_conditions": ["manual_reset", "cause_cleared"],
            "requires_manual_reset": True,
        },
    }


# =============================================================================
# CALCULATION HELPER FIXTURES
# =============================================================================


@dataclass
class PFDCalculationResult:
    """Result of PFD calculation."""
    pfd_avg: float
    rrf: float
    sil_achieved: int
    meets_target: bool
    proof_test_interval_hours: float
    lambda_du: float
    voting_architecture: str
    calculation_method: str


@pytest.fixture
def pfd_calculator():
    """
    Fixture providing PFD calculation functions per IEC 61508-6.

    Returns a calculator object with methods for different architectures.
    """
    class PFDCalculator:
        """PFD calculation helper per IEC 61508-6."""

        @staticmethod
        def calculate_1oo1(
            lambda_du: float,
            proof_test_interval_hours: float,
            diagnostic_coverage: float = 0.0,
        ) -> float:
            """
            Calculate PFD for 1oo1 architecture.

            PFD_avg = (1-DC) * lambda_DU * TI / 2

            Args:
                lambda_du: Dangerous undetected failure rate (per hour)
                proof_test_interval_hours: Proof test interval in hours
                diagnostic_coverage: Diagnostic coverage factor (0-1)

            Returns:
                Average PFD value
            """
            ti = proof_test_interval_hours
            dc = diagnostic_coverage
            return (1 - dc) * lambda_du * ti / 2

        @staticmethod
        def calculate_1oo2(
            lambda_du: float,
            proof_test_interval_hours: float,
            diagnostic_coverage: float = 0.0,
            beta_factor: float = 0.1,
        ) -> float:
            """
            Calculate PFD for 1oo2 architecture.

            PFD_avg = ((1-beta) * lambda_DU * TI)^2 / 3 + beta * lambda_DU * TI / 2

            Args:
                lambda_du: Dangerous undetected failure rate (per hour)
                proof_test_interval_hours: Proof test interval in hours
                diagnostic_coverage: Diagnostic coverage factor (0-1)
                beta_factor: Common cause failure factor (typically 0.1)

            Returns:
                Average PFD value
            """
            ti = proof_test_interval_hours
            dc = diagnostic_coverage
            beta = beta_factor
            lambda_eff = (1 - dc) * lambda_du

            # Independent failure term + common cause term
            pfd_independent = ((1 - beta) * lambda_eff * ti) ** 2 / 3
            pfd_common_cause = beta * lambda_eff * ti / 2

            return pfd_independent + pfd_common_cause

        @staticmethod
        def calculate_2oo2(
            lambda_du: float,
            proof_test_interval_hours: float,
            diagnostic_coverage: float = 0.0,
        ) -> float:
            """
            Calculate PFD for 2oo2 architecture.

            PFD_avg = lambda_DU * TI (approximately 2x of 1oo1)

            Args:
                lambda_du: Dangerous undetected failure rate (per hour)
                proof_test_interval_hours: Proof test interval in hours
                diagnostic_coverage: Diagnostic coverage factor (0-1)

            Returns:
                Average PFD value
            """
            ti = proof_test_interval_hours
            dc = diagnostic_coverage
            return (1 - dc) * lambda_du * ti

        @staticmethod
        def calculate_2oo3(
            lambda_du: float,
            proof_test_interval_hours: float,
            diagnostic_coverage: float = 0.0,
            beta_factor: float = 0.1,
        ) -> float:
            """
            Calculate PFD for 2oo3 (TMR) architecture.

            PFD_avg = 3 * ((1-beta) * lambda_DU * TI)^2 + beta * lambda_DU * TI / 2

            Args:
                lambda_du: Dangerous undetected failure rate (per hour)
                proof_test_interval_hours: Proof test interval in hours
                diagnostic_coverage: Diagnostic coverage factor (0-1)
                beta_factor: Common cause failure factor (typically 0.1)

            Returns:
                Average PFD value
            """
            ti = proof_test_interval_hours
            dc = diagnostic_coverage
            beta = beta_factor
            lambda_eff = (1 - dc) * lambda_du

            # Three pairs can fail + common cause
            pfd_independent = 3 * ((1 - beta) * lambda_eff * ti) ** 2
            pfd_common_cause = beta * lambda_eff * ti / 2

            return pfd_independent + pfd_common_cause

        @staticmethod
        def determine_sil(pfd_avg: float) -> int:
            """
            Determine SIL level from PFD value.

            Args:
                pfd_avg: Average PFD value

            Returns:
                SIL level (0-4, where 0 means below SIL 1)
            """
            if pfd_avg < 1e-5:
                return 4
            elif pfd_avg < 1e-4:
                return 4
            elif pfd_avg < 1e-3:
                return 3
            elif pfd_avg < 1e-2:
                return 2
            elif pfd_avg < 1e-1:
                return 1
            else:
                return 0

        def calculate_full(
            self,
            architecture: str,
            lambda_du: float,
            proof_test_interval_hours: float,
            target_sil: int,
            diagnostic_coverage: float = 0.0,
            beta_factor: float = 0.1,
        ) -> PFDCalculationResult:
            """
            Perform full PFD calculation with SIL verification.

            Args:
                architecture: Voting architecture (1oo1, 1oo2, 2oo2, 2oo3)
                lambda_du: Dangerous undetected failure rate (per hour)
                proof_test_interval_hours: Proof test interval in hours
                target_sil: Target SIL level
                diagnostic_coverage: Diagnostic coverage factor (0-1)
                beta_factor: Common cause failure factor

            Returns:
                PFDCalculationResult with full analysis
            """
            # Calculate PFD based on architecture
            if architecture == "1oo1":
                pfd = self.calculate_1oo1(lambda_du, proof_test_interval_hours, diagnostic_coverage)
            elif architecture == "1oo2":
                pfd = self.calculate_1oo2(lambda_du, proof_test_interval_hours, diagnostic_coverage, beta_factor)
            elif architecture == "2oo2":
                pfd = self.calculate_2oo2(lambda_du, proof_test_interval_hours, diagnostic_coverage)
            elif architecture == "2oo3":
                pfd = self.calculate_2oo3(lambda_du, proof_test_interval_hours, diagnostic_coverage, beta_factor)
            else:
                raise ValueError(f"Unknown architecture: {architecture}")

            # Calculate RRF
            rrf = 1 / pfd if pfd > 0 else float('inf')

            # Determine achieved SIL
            sil_achieved = self.determine_sil(pfd)

            # Check if target is met
            meets_target = sil_achieved >= target_sil

            return PFDCalculationResult(
                pfd_avg=pfd,
                rrf=rrf,
                sil_achieved=sil_achieved,
                meets_target=meets_target,
                proof_test_interval_hours=proof_test_interval_hours,
                lambda_du=lambda_du,
                voting_architecture=architecture,
                calculation_method="IEC_61508-6",
            )

    return PFDCalculator()


@pytest.fixture
def o2_correction_calculator():
    """
    Fixture providing O2 correction calculations per EPA Method 19.

    Returns a calculator for correcting emissions to reference O2.
    """
    class O2CorrectionCalculator:
        """O2 correction calculator per EPA Method 19."""

        @staticmethod
        def correct_to_reference(
            measured_value: float,
            measured_o2_pct: float,
            reference_o2_pct: float = 3.0,
        ) -> float:
            """
            Correct measured value to reference O2 percentage.

            Formula: Corrected = Measured * (20.9 - O2_ref) / (20.9 - O2_meas)

            Args:
                measured_value: Measured emission value (ppm or lb/MMBTU)
                measured_o2_pct: Measured O2 percentage
                reference_o2_pct: Reference O2 percentage (default 3%)

            Returns:
                Corrected emission value
            """
            if measured_o2_pct >= 20.9:
                return measured_value  # Avoid division by zero

            correction_factor = (20.9 - reference_o2_pct) / (20.9 - measured_o2_pct)
            return measured_value * correction_factor

        @staticmethod
        def calculate_excess_air(o2_pct: float) -> float:
            """
            Calculate excess air from O2 measurement.

            Formula: EA% = O2 / (21 - O2) * 100

            Args:
                o2_pct: Measured O2 percentage

            Returns:
                Excess air percentage
            """
            if o2_pct >= 21:
                return float('inf')
            return (o2_pct / (21 - o2_pct)) * 100

    return O2CorrectionCalculator()


@pytest.fixture
def ghg_calculator(epa_part98_emission_factors, epa_part98_gwp_values):
    """
    Fixture providing GHG emission calculations per EPA Part 98.

    Returns a calculator for CO2, CH4, N2O, and CO2e emissions.
    """
    class GHGCalculator:
        """GHG emission calculator per EPA Part 98."""

        def __init__(self, emission_factors, gwp_values):
            self.emission_factors = emission_factors
            self.gwp_values = gwp_values

        def calculate_co2_emissions(
            self,
            fuel_type: str,
            fuel_quantity_mmbtu: float,
        ) -> float:
            """
            Calculate CO2 emissions using Tier 1 methodology.

            Args:
                fuel_type: Fuel type identifier
                fuel_quantity_mmbtu: Fuel consumption in MMBTU

            Returns:
                CO2 emissions in metric tons
            """
            factors = self.emission_factors.get(fuel_type)
            if not factors:
                raise ValueError(f"Unknown fuel type: {fuel_type}")

            co2_kg = fuel_quantity_mmbtu * factors["co2_kg_per_mmbtu"]
            return co2_kg / 1000  # Convert to metric tons

        def calculate_co2e_emissions(
            self,
            fuel_type: str,
            fuel_quantity_mmbtu: float,
        ) -> Dict[str, float]:
            """
            Calculate CO2 equivalent emissions including CH4 and N2O.

            Args:
                fuel_type: Fuel type identifier
                fuel_quantity_mmbtu: Fuel consumption in MMBTU

            Returns:
                Dict with CO2, CH4, N2O, and total CO2e in metric tons
            """
            factors = self.emission_factors.get(fuel_type)
            if not factors:
                raise ValueError(f"Unknown fuel type: {fuel_type}")

            co2_kg = fuel_quantity_mmbtu * factors["co2_kg_per_mmbtu"]
            ch4_kg = fuel_quantity_mmbtu * factors["ch4_kg_per_mmbtu"]
            n2o_kg = fuel_quantity_mmbtu * factors["n2o_kg_per_mmbtu"]

            co2e_kg = (
                co2_kg * self.gwp_values["co2"] +
                ch4_kg * self.gwp_values["ch4"] +
                n2o_kg * self.gwp_values["n2o"]
            )

            return {
                "co2_metric_tons": co2_kg / 1000,
                "ch4_metric_tons": ch4_kg / 1000,
                "n2o_metric_tons": n2o_kg / 1000,
                "co2e_metric_tons": co2e_kg / 1000,
            }

    return GHGCalculator(epa_part98_emission_factors, epa_part98_gwp_values)


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================


@pytest.fixture
def cems_data_generator():
    """
    Fixture providing CEMS data generators for testing.

    Generates realistic CEMS data streams for compliance testing.
    """
    import random

    class CEMSDataGenerator:
        """CEMS data generator for compliance testing."""

        def __init__(self, seed: int = 42):
            self.random = random.Random(seed)

        def generate_hourly_data(
            self,
            hours: int,
            base_nox_ppm: float = 25.0,
            base_co_ppm: float = 30.0,
            base_o2_pct: float = 3.0,
            availability_pct: float = 95.0,
            variance_pct: float = 10.0,
        ) -> List[Dict[str, Any]]:
            """
            Generate hourly CEMS data.

            Args:
                hours: Number of hours to generate
                base_nox_ppm: Base NOx concentration
                base_co_ppm: Base CO concentration
                base_o2_pct: Base O2 percentage
                availability_pct: Target data availability
                variance_pct: Measurement variance

            Returns:
                List of hourly CEMS readings
            """
            data = []
            base_time = datetime.now(timezone.utc) - timedelta(hours=hours)

            for hour in range(hours):
                timestamp = base_time + timedelta(hours=hour)

                # Simulate data availability
                if self.random.random() * 100 > availability_pct:
                    data.append({
                        "timestamp": timestamp,
                        "status": "missing",
                        "nox_ppm": None,
                        "co_ppm": None,
                        "o2_pct": None,
                    })
                    continue

                # Add variance to measurements
                variance_factor = 1 + (self.random.random() - 0.5) * variance_pct / 50

                data.append({
                    "timestamp": timestamp,
                    "status": "valid",
                    "nox_ppm": base_nox_ppm * variance_factor,
                    "co_ppm": base_co_ppm * variance_factor,
                    "o2_pct": base_o2_pct + (self.random.random() - 0.5) * 0.5,
                    "stack_flow_scfh": 50000 * variance_factor,
                })

            return data

        def calculate_availability(self, data: List[Dict[str, Any]]) -> float:
            """Calculate data availability percentage."""
            if not data:
                return 0.0
            valid_count = sum(1 for d in data if d.get("status") == "valid")
            return (valid_count / len(data)) * 100

    return CEMSDataGenerator()
