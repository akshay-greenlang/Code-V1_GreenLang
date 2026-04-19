"""
NFPA 85/86 Compliance Calculator for GL-018 UNIFIEDCOMBUSTION Agent

This module implements deterministic compliance checking against
NFPA 85 (Boiler and Combustion Systems Hazards Code) and
NFPA 86 (Standard for Ovens and Furnaces).

All compliance rules are deterministic checks based on the published
NFPA standards - NO ML/LLM in the compliance determination path.

Reference Standards:
- NFPA 85: Boiler and Combustion Systems Hazards Code (2023 Edition)
- NFPA 86: Standard for Ovens and Furnaces (2023 Edition)
- FM Global Data Sheets
- API 535: Burners for Fired Heaters

Zero-hallucination: All compliance checks are deterministic rule evaluations.

Example:
    >>> from nfpa_compliance import check_nfpa_compliance
    >>> result = check_nfpa_compliance(equipment_type="boiler", ...)
    >>> print(f"Status: {result['overall_status']}")
"""

from datetime import date, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# NFPA 85 - Boiler Requirements
# =============================================================================

NFPA_85_REQUIREMENTS = {
    # Burner Management System (BMS) Requirements - Chapter 5
    "bms_sil_rating": {
        "code_ref": "NFPA 85 5.3.6",
        "requirement": "BMS shall be rated SIL 2 minimum for safety functions",
        "check_type": "document",
    },
    "flame_failure_response": {
        "code_ref": "NFPA 85 5.6.3",
        "requirement": "Flame failure shall initiate safety shutdown within 4 seconds",
        "max_response_time_s": 4.0,
        "check_type": "interlock",
    },
    "main_fuel_trip": {
        "code_ref": "NFPA 85 5.6.4",
        "requirement": "Main fuel trip shall be initiated on loss of all flames",
        "check_type": "interlock",
        "required_interlock": "flame_failure",
    },

    # Purge Requirements - Chapter 6
    "pre_purge_volume": {
        "code_ref": "NFPA 85 6.4.2",
        "requirement": "Pre-ignition purge shall be minimum 5 volume changes",
        "min_volume_changes": 5,
        "check_type": "purge",
    },
    "purge_airflow": {
        "code_ref": "NFPA 85 6.4.2.1",
        "requirement": "Purge airflow shall be minimum 25% of full load airflow",
        "min_airflow_pct": 25.0,
        "check_type": "purge",
    },
    "purge_damper_position": {
        "code_ref": "NFPA 85 6.4.2.2",
        "requirement": "Dampers shall be in purge position during purge cycle",
        "check_type": "interlock",
    },

    # Safety Interlocks - Chapter 7
    "low_fuel_pressure": {
        "code_ref": "NFPA 85 7.4.2.1",
        "requirement": "Low fuel pressure interlock shall trip burner",
        "check_type": "interlock",
        "required_interlock": "low_fuel_pressure",
    },
    "high_fuel_pressure": {
        "code_ref": "NFPA 85 7.4.2.2",
        "requirement": "High fuel pressure interlock shall trip burner",
        "check_type": "interlock",
        "required_interlock": "high_fuel_pressure",
    },
    "low_air_pressure": {
        "code_ref": "NFPA 85 7.4.3.1",
        "requirement": "Low combustion air pressure interlock required",
        "check_type": "interlock",
        "required_interlock": "low_air_flow",
    },
    "high_furnace_pressure": {
        "code_ref": "NFPA 85 7.4.4.1",
        "requirement": "High furnace pressure interlock shall trip burner",
        "check_type": "interlock",
        "required_interlock": "high_furnace_pressure",
    },
    "combustion_air_proving": {
        "code_ref": "NFPA 85 7.4.3.2",
        "requirement": "Combustion air flow proving required before ignition",
        "check_type": "interlock",
        "required_interlock": "combustion_air_proving",
    },

    # Flame Detection - Chapter 8
    "flame_detector_type": {
        "code_ref": "NFPA 85 8.4.2",
        "requirement": "Flame detector shall be self-checking type",
        "check_type": "equipment",
    },
    "flame_detector_testing": {
        "code_ref": "NFPA 85 8.4.5",
        "requirement": "Flame detector shall be tested per manufacturer requirements",
        "max_test_interval_days": 90,
        "check_type": "maintenance",
    },

    # Operating Limits
    "excess_air_limits": {
        "code_ref": "NFPA 85 7.5.1",
        "requirement": "Excess air shall be maintained within safe operating limits",
        "min_o2_pct": 1.0,  # Minimum to prevent incomplete combustion
        "max_o2_pct": 10.0,  # Maximum for efficiency
        "check_type": "operating",
    },
    "co_limit": {
        "code_ref": "NFPA 85 7.5.2",
        "requirement": "CO shall not exceed safe limits indicating incomplete combustion",
        "max_co_ppm": 400,  # Warning threshold
        "critical_co_ppm": 1000,  # Trip threshold
        "check_type": "operating",
    },
}


# =============================================================================
# NFPA 86 - Oven and Furnace Requirements
# =============================================================================

NFPA_86_REQUIREMENTS = {
    # Classification Requirements - Chapter 4
    "furnace_classification": {
        "code_ref": "NFPA 86 4.1",
        "requirement": "Furnace shall be classified per Chapter 4",
        "check_type": "document",
    },

    # Ventilation - Chapter 7
    "ventilation_rate": {
        "code_ref": "NFPA 86 7.2.1",
        "requirement": "Minimum ventilation for Class A ovens: 10,000 cfm per gallon/min solvent",
        "check_type": "ventilation",
    },
    "lel_limit": {
        "code_ref": "NFPA 86 7.2.2",
        "requirement": "Atmosphere shall not exceed 25% of LEL during normal operation",
        "max_lel_pct": 25.0,
        "check_type": "operating",
    },

    # Safety Equipment - Chapter 8
    "explosion_relief": {
        "code_ref": "NFPA 86 8.3",
        "requirement": "Explosion relief shall be provided per Chapter 8",
        "check_type": "equipment",
    },
    "emergency_shutoff": {
        "code_ref": "NFPA 86 8.4.1",
        "requirement": "Emergency fuel shutoff shall be readily accessible",
        "check_type": "equipment",
    },

    # Purge Requirements - Chapter 9
    "pre_purge_furnace": {
        "code_ref": "NFPA 86 9.3.1",
        "requirement": "Pre-ignition purge: minimum 4 volume changes",
        "min_volume_changes": 4,
        "check_type": "purge",
    },
    "post_purge": {
        "code_ref": "NFPA 86 9.3.3",
        "requirement": "Post-shutdown purge shall be performed",
        "check_type": "purge",
    },

    # Flame Safeguard - Chapter 10
    "flame_safeguard_timing": {
        "code_ref": "NFPA 86 10.4.2",
        "requirement": "Pilot flame shall be proven within trial for ignition period",
        "max_pilot_trial_s": 15.0,
        "check_type": "timing",
    },
    "main_flame_trial": {
        "code_ref": "NFPA 86 10.4.3",
        "requirement": "Main flame shall be proven within main flame trial period",
        "max_main_trial_s": 15.0,
        "check_type": "timing",
    },
    "flame_failure_response_86": {
        "code_ref": "NFPA 86 10.4.4",
        "requirement": "Flame failure response time shall not exceed 4 seconds",
        "max_response_time_s": 4.0,
        "check_type": "interlock",
    },

    # Safety Interlocks - Chapter 11
    "high_temperature": {
        "code_ref": "NFPA 86 11.3.1",
        "requirement": "High temperature limit shall shut down fuel",
        "check_type": "interlock",
        "required_interlock": "high_temperature",
    },
    "low_temperature": {
        "code_ref": "NFPA 86 11.3.2",
        "requirement": "Low temperature interlock required for heat treatment",
        "check_type": "interlock",
        "required_interlock": "low_temperature",
    },
    "exhaust_fan_interlock": {
        "code_ref": "NFPA 86 11.4.1",
        "requirement": "Exhaust fan operation shall be proven before ignition",
        "check_type": "interlock",
        "required_interlock": "exhaust_fan",
    },
}


# =============================================================================
# Required Safety Interlocks by Equipment Type
# =============================================================================

REQUIRED_INTERLOCKS = {
    "boiler": [
        "flame_failure",
        "low_fuel_pressure",
        "high_fuel_pressure",
        "low_air_flow",
        "high_furnace_pressure",
        "combustion_air_proving",
        "low_water",
        "high_steam_pressure",
    ],
    "furnace": [
        "flame_failure",
        "low_fuel_pressure",
        "high_fuel_pressure",
        "low_air_flow",
        "high_furnace_pressure",
        "high_temperature",
        "exhaust_fan",
    ],
    "oven": [
        "flame_failure",
        "low_fuel_pressure",
        "high_temperature",
        "exhaust_fan",
        "explosion_relief",
    ],
    "heater": [
        "flame_failure",
        "low_fuel_pressure",
        "high_fuel_pressure",
        "low_air_flow",
        "high_temperature",
    ],
    "kiln": [
        "flame_failure",
        "low_fuel_pressure",
        "high_temperature",
        "exhaust_fan",
    ],
    "thermal_oxidizer": [
        "flame_failure",
        "low_fuel_pressure",
        "high_temperature",
        "low_temperature",
        "exhaust_fan",
        "combustion_air_proving",
    ],
}


# =============================================================================
# Compliance Check Functions
# =============================================================================

def check_nfpa_compliance(
    equipment_type: str,
    safety_interlocks: List[Dict[str, Any]],
    o2_percent: float,
    co_ppm: float,
    nfpa_standard: str = "NFPA 85",
    flame_detector_test_date: Optional[date] = None,
    purge_volume_changes: Optional[float] = None,
    purge_airflow_pct: Optional[float] = None,
    flame_response_time_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Perform comprehensive NFPA compliance check.

    This is the main entry point for NFPA compliance assessment.
    All checks are deterministic rule evaluations - NO ML/LLM.

    Args:
        equipment_type: Type of equipment (boiler, furnace, oven, etc.)
        safety_interlocks: List of safety interlock data
        o2_percent: Current O2 reading
        co_ppm: Current CO reading
        nfpa_standard: Applicable standard (NFPA 85 or NFPA 86)
        flame_detector_test_date: Date of last flame detector test
        purge_volume_changes: Number of purge volume changes
        purge_airflow_pct: Purge airflow as percentage of full load
        flame_response_time_s: Flame failure response time in seconds

    Returns:
        Dictionary with compliance results including:
        - overall_status: COMPLIANT, NON_COMPLIANT, or WARNING
        - violations: List of violations found
        - warnings: List of warning conditions
        - interlock_status: Status of each required interlock
    """
    violations = []
    warnings = []
    interlock_assessments = []

    # Select requirements based on standard
    if "86" in nfpa_standard:
        requirements = NFPA_86_REQUIREMENTS
    else:
        requirements = NFPA_85_REQUIREMENTS

    # 1. Check required safety interlocks
    interlock_result = check_required_interlocks(
        equipment_type, safety_interlocks
    )
    interlock_assessments = interlock_result["assessments"]

    for missing in interlock_result["missing"]:
        violations.append({
            "code_reference": f"{nfpa_standard} - Safety Interlocks",
            "requirement": f"Required interlock: {missing}",
            "current_state": "NOT FOUND",
            "severity": "CRITICAL",
            "corrective_action": f"Install and commission {missing} interlock",
        })

    for bypassed in interlock_result["bypassed"]:
        violations.append({
            "code_reference": f"{nfpa_standard} - Safety Interlocks",
            "requirement": f"Interlock {bypassed} shall not be bypassed during operation",
            "current_state": "BYPASSED",
            "severity": "CRITICAL",
            "corrective_action": f"Remove bypass from {bypassed} interlock",
        })

    for faulted in interlock_result["faulted"]:
        violations.append({
            "code_reference": f"{nfpa_standard} - Safety Interlocks",
            "requirement": f"Interlock {faulted} shall be functional",
            "current_state": "FAULT",
            "severity": "HIGH",
            "corrective_action": f"Repair {faulted} interlock",
        })

    # 2. Check operating limits
    operating_result = check_operating_limits(
        o2_percent, co_ppm, requirements
    )
    violations.extend(operating_result["violations"])
    warnings.extend(operating_result["warnings"])

    # 3. Check flame detector testing
    if flame_detector_test_date is not None:
        detector_result = check_flame_detector_testing(
            flame_detector_test_date, requirements
        )
        if detector_result["overdue"]:
            violations.append({
                "code_reference": requirements.get("flame_detector_testing", {}).get(
                    "code_ref", f"{nfpa_standard} 8.4.5"
                ),
                "requirement": "Flame detector shall be tested per schedule",
                "current_state": f"Last test: {flame_detector_test_date}, "
                                f"Overdue by {detector_result['days_overdue']} days",
                "severity": "HIGH",
                "corrective_action": "Perform flame detector test immediately",
            })

    # 4. Check purge requirements
    if purge_volume_changes is not None or purge_airflow_pct is not None:
        purge_result = check_purge_requirements(
            purge_volume_changes, purge_airflow_pct, requirements, nfpa_standard
        )
        violations.extend(purge_result["violations"])
        warnings.extend(purge_result["warnings"])

    # 5. Check flame failure response time
    if flame_response_time_s is not None:
        response_result = check_flame_response_time(
            flame_response_time_s, requirements, nfpa_standard
        )
        if not response_result["compliant"]:
            violations.append(response_result["violation"])

    # Determine overall status
    critical_violations = [v for v in violations if v["severity"] == "CRITICAL"]
    high_violations = [v for v in violations if v["severity"] == "HIGH"]

    if critical_violations:
        overall_status = "NON_COMPLIANT"
    elif high_violations:
        overall_status = "NON_COMPLIANT"
    elif warnings:
        overall_status = "WARNING"
    else:
        overall_status = "COMPLIANT"

    # Determine BMS, flame safeguard, and purge status
    bms_status = "COMPLIANT" if not interlock_result["missing"] else "NON_COMPLIANT"
    flame_safeguard_status = "COMPLIANT"
    purge_status = "COMPLIANT"

    for v in violations:
        if "flame" in v["requirement"].lower():
            flame_safeguard_status = "NON_COMPLIANT"
        if "purge" in v["requirement"].lower():
            purge_status = "NON_COMPLIANT"

    logger.info(
        f"NFPA compliance check complete: {overall_status}, "
        f"{len(violations)} violations, {len(warnings)} warnings"
    )

    return {
        "overall_status": overall_status,
        "standard": nfpa_standard,
        "violations": violations,
        "warnings": warnings,
        "interlock_assessments": interlock_assessments,
        "burner_management_status": bms_status,
        "flame_safeguard_status": flame_safeguard_status,
        "purge_status": purge_status,
        "required_actions": [v["corrective_action"] for v in violations],
    }


def check_required_interlocks(
    equipment_type: str,
    safety_interlocks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Check that all required safety interlocks are present and functional.

    Args:
        equipment_type: Type of equipment
        safety_interlocks: List of interlock status data

    Returns:
        Dictionary with missing, bypassed, faulted lists and assessments
    """
    # Get required interlocks for equipment type
    equipment_key = equipment_type.lower().replace(" ", "_")
    required = REQUIRED_INTERLOCKS.get(equipment_key, REQUIRED_INTERLOCKS["boiler"])

    # Create lookup of provided interlocks
    provided = {}
    for interlock in safety_interlocks:
        name = interlock.get("interlock_name", "").lower().replace(" ", "_")
        provided[name] = interlock

    missing = []
    bypassed = []
    faulted = []
    assessments = []

    for req_interlock in required:
        # Check if interlock exists
        if req_interlock not in provided:
            missing.append(req_interlock)
            assessments.append({
                "interlock_name": req_interlock,
                "required_by": f"NFPA - {equipment_type}",
                "status": "NON_COMPLIANT",
                "test_required": True,
                "notes": "Interlock not found in system",
            })
            continue

        interlock_data = provided[req_interlock]
        status = interlock_data.get("status", "").upper()

        if status == "BYPASSED":
            bypassed.append(req_interlock)
            assessments.append({
                "interlock_name": req_interlock,
                "required_by": f"NFPA - {equipment_type}",
                "status": "NON_COMPLIANT",
                "test_required": True,
                "notes": "Interlock bypassed - safety hazard",
            })
        elif status == "FAULT":
            faulted.append(req_interlock)
            assessments.append({
                "interlock_name": req_interlock,
                "required_by": f"NFPA - {equipment_type}",
                "status": "NON_COMPLIANT",
                "test_required": True,
                "notes": "Interlock in fault state",
            })
        elif status in ["ARMED", "TRIPPED"]:
            assessments.append({
                "interlock_name": req_interlock,
                "required_by": f"NFPA - {equipment_type}",
                "status": "COMPLIANT",
                "test_required": False,
                "notes": f"Interlock functional, status: {status}",
            })
        else:
            assessments.append({
                "interlock_name": req_interlock,
                "required_by": f"NFPA - {equipment_type}",
                "status": "REQUIRES_REVIEW",
                "test_required": True,
                "notes": f"Unknown status: {status}",
            })

    return {
        "missing": missing,
        "bypassed": bypassed,
        "faulted": faulted,
        "assessments": assessments,
        "all_present": len(missing) == 0,
        "all_functional": len(bypassed) == 0 and len(faulted) == 0,
    }


def check_operating_limits(
    o2_percent: float,
    co_ppm: float,
    requirements: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check combustion operating parameters against NFPA limits.

    Args:
        o2_percent: Current O2 percentage
        co_ppm: Current CO in ppm
        requirements: NFPA requirements dictionary

    Returns:
        Dictionary with violations and warnings
    """
    violations = []
    warnings = []

    # Check O2 limits
    o2_req = requirements.get("excess_air_limits", {})
    min_o2 = o2_req.get("min_o2_pct", 1.0)
    max_o2 = o2_req.get("max_o2_pct", 10.0)

    if o2_percent < min_o2:
        violations.append({
            "code_reference": o2_req.get("code_ref", "NFPA 85 7.5.1"),
            "requirement": f"O2 shall be above {min_o2}% to ensure complete combustion",
            "current_state": f"O2 at {o2_percent}%",
            "severity": "CRITICAL",
            "corrective_action": "Increase excess air immediately to prevent incomplete combustion",
        })
    elif o2_percent > max_o2:
        warnings.append(
            f"O2 at {o2_percent}% exceeds recommended maximum of {max_o2}% - efficiency impact"
        )

    # Check CO limits
    co_req = requirements.get("co_limit", {})
    max_co = co_req.get("max_co_ppm", 400)
    critical_co = co_req.get("critical_co_ppm", 1000)

    if co_ppm > critical_co:
        violations.append({
            "code_reference": co_req.get("code_ref", "NFPA 85 7.5.2"),
            "requirement": f"CO shall not exceed {critical_co} ppm",
            "current_state": f"CO at {co_ppm} ppm",
            "severity": "CRITICAL",
            "corrective_action": "Reduce firing rate and increase combustion air immediately",
        })
    elif co_ppm > max_co:
        warnings.append(
            f"CO at {co_ppm} ppm exceeds warning threshold of {max_co} ppm - "
            "investigate combustion quality"
        )

    return {
        "violations": violations,
        "warnings": warnings,
        "o2_status": "OK" if min_o2 <= o2_percent <= max_o2 else "WARNING",
        "co_status": "OK" if co_ppm <= max_co else "WARNING" if co_ppm <= critical_co else "CRITICAL",
    }


def check_flame_detector_testing(
    last_test_date: date,
    requirements: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Check flame detector testing compliance.

    Args:
        last_test_date: Date of last flame detector test
        requirements: NFPA requirements dictionary

    Returns:
        Dictionary with test status
    """
    detector_req = requirements.get("flame_detector_testing", {})
    max_interval = detector_req.get("max_test_interval_days", 90)

    days_since_test = (date.today() - last_test_date).days
    days_overdue = max(0, days_since_test - max_interval)

    return {
        "last_test_date": last_test_date,
        "days_since_test": days_since_test,
        "max_interval_days": max_interval,
        "days_overdue": days_overdue,
        "overdue": days_overdue > 0,
        "next_test_due": last_test_date + timedelta(days=max_interval),
    }


def check_purge_requirements(
    volume_changes: Optional[float],
    airflow_pct: Optional[float],
    requirements: Dict[str, Any],
    nfpa_standard: str
) -> Dict[str, Any]:
    """
    Check purge cycle compliance.

    Args:
        volume_changes: Number of purge volume changes
        airflow_pct: Purge airflow as percentage of full load
        requirements: NFPA requirements dictionary
        nfpa_standard: NFPA 85 or 86

    Returns:
        Dictionary with violations and warnings
    """
    violations = []
    warnings = []

    # Get purge requirements
    if "86" in nfpa_standard:
        purge_req = requirements.get("pre_purge_furnace", {})
        min_volumes = purge_req.get("min_volume_changes", 4)
    else:
        purge_req = requirements.get("pre_purge_volume", {})
        min_volumes = purge_req.get("min_volume_changes", 5)

    # Check volume changes
    if volume_changes is not None and volume_changes < min_volumes:
        violations.append({
            "code_reference": purge_req.get("code_ref", f"{nfpa_standard} 6.4.2"),
            "requirement": f"Purge shall provide minimum {min_volumes} volume changes",
            "current_state": f"Purge provides {volume_changes} volume changes",
            "severity": "HIGH",
            "corrective_action": f"Increase purge time to achieve {min_volumes} volume changes",
        })

    # Check airflow percentage (NFPA 85 only)
    if airflow_pct is not None and "85" in nfpa_standard:
        airflow_req = requirements.get("purge_airflow", {})
        min_airflow = airflow_req.get("min_airflow_pct", 25.0)

        if airflow_pct < min_airflow:
            violations.append({
                "code_reference": airflow_req.get("code_ref", "NFPA 85 6.4.2.1"),
                "requirement": f"Purge airflow shall be minimum {min_airflow}% of full load",
                "current_state": f"Purge airflow at {airflow_pct}%",
                "severity": "HIGH",
                "corrective_action": f"Increase purge airflow to minimum {min_airflow}%",
            })

    return {
        "violations": violations,
        "warnings": warnings,
        "volume_changes_ok": volume_changes is None or volume_changes >= min_volumes,
        "airflow_ok": airflow_pct is None or airflow_pct >= 25.0,
    }


def check_flame_response_time(
    response_time_s: float,
    requirements: Dict[str, Any],
    nfpa_standard: str
) -> Dict[str, Any]:
    """
    Check flame failure response time compliance.

    Args:
        response_time_s: Flame failure response time in seconds
        requirements: NFPA requirements dictionary
        nfpa_standard: NFPA 85 or 86

    Returns:
        Dictionary with compliance status
    """
    if "86" in nfpa_standard:
        flame_req = requirements.get("flame_failure_response_86", {})
    else:
        flame_req = requirements.get("flame_failure_response", {})

    max_response = flame_req.get("max_response_time_s", 4.0)
    compliant = response_time_s <= max_response

    result = {
        "compliant": compliant,
        "response_time_s": response_time_s,
        "max_allowed_s": max_response,
        "violation": None,
    }

    if not compliant:
        result["violation"] = {
            "code_reference": flame_req.get("code_ref", f"{nfpa_standard} 5.6.3"),
            "requirement": f"Flame failure response shall not exceed {max_response} seconds",
            "current_state": f"Response time: {response_time_s} seconds",
            "severity": "CRITICAL",
            "corrective_action": "Adjust flame safeguard system to meet response time requirement",
        }

    return result


def get_required_interlocks_for_equipment(equipment_type: str) -> List[str]:
    """
    Get list of required safety interlocks for equipment type.

    Args:
        equipment_type: Type of equipment

    Returns:
        List of required interlock names
    """
    equipment_key = equipment_type.lower().replace(" ", "_")
    return REQUIRED_INTERLOCKS.get(equipment_key, REQUIRED_INTERLOCKS["boiler"])


def check_interlock_certification(
    interlock_name: str,
    last_test_date: Optional[date],
    certified: bool
) -> Dict[str, Any]:
    """
    Check individual interlock certification status.

    Args:
        interlock_name: Name of the interlock
        last_test_date: Date of last certification test
        certified: Current certification status

    Returns:
        Dictionary with certification assessment
    """
    max_cert_interval = 365  # Annual certification typical

    if last_test_date is None:
        return {
            "interlock_name": interlock_name,
            "certified": False,
            "test_required": True,
            "days_overdue": None,
            "notes": "No test date on record",
        }

    days_since_test = (date.today() - last_test_date).days
    overdue = days_since_test > max_cert_interval

    return {
        "interlock_name": interlock_name,
        "certified": certified and not overdue,
        "test_required": overdue,
        "days_since_test": days_since_test,
        "days_overdue": max(0, days_since_test - max_cert_interval) if overdue else 0,
        "next_test_due": last_test_date + timedelta(days=max_cert_interval),
        "notes": "Certification current" if not overdue else "Recertification required",
    }


def generate_compliance_report(
    compliance_result: Dict[str, Any],
    equipment_id: str,
    equipment_type: str
) -> str:
    """
    Generate human-readable compliance report.

    Args:
        compliance_result: Result from check_nfpa_compliance
        equipment_id: Equipment identifier
        equipment_type: Type of equipment

    Returns:
        Formatted compliance report string
    """
    report_lines = [
        f"NFPA COMPLIANCE REPORT",
        f"=" * 50,
        f"Equipment ID: {equipment_id}",
        f"Equipment Type: {equipment_type}",
        f"Standard: {compliance_result['standard']}",
        f"Overall Status: {compliance_result['overall_status']}",
        f"",
        f"SUMMARY",
        f"-" * 50,
        f"BMS Status: {compliance_result['burner_management_status']}",
        f"Flame Safeguard: {compliance_result['flame_safeguard_status']}",
        f"Purge Cycle: {compliance_result['purge_status']}",
        f"",
    ]

    if compliance_result['violations']:
        report_lines.append("VIOLATIONS")
        report_lines.append("-" * 50)
        for i, v in enumerate(compliance_result['violations'], 1):
            report_lines.append(f"{i}. [{v['severity']}] {v['code_reference']}")
            report_lines.append(f"   Requirement: {v['requirement']}")
            report_lines.append(f"   Current: {v['current_state']}")
            report_lines.append(f"   Action: {v['corrective_action']}")
            report_lines.append("")

    if compliance_result['warnings']:
        report_lines.append("WARNINGS")
        report_lines.append("-" * 50)
        for w in compliance_result['warnings']:
            report_lines.append(f"- {w}")
        report_lines.append("")

    if compliance_result['required_actions']:
        report_lines.append("REQUIRED ACTIONS")
        report_lines.append("-" * 50)
        for i, action in enumerate(compliance_result['required_actions'], 1):
            report_lines.append(f"{i}. {action}")

    return "\n".join(report_lines)
