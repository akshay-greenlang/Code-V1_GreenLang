# -*- coding: utf-8 -*-
"""
Compliance Validator for GL-001 ThermalCommand
==============================================

Runtime compliance checking against regulatory requirements:
    - EPA 40 CFR Part 75 (Continuous Emissions Monitoring)
    - EPA 40 CFR Part 98 (GHG Mandatory Reporting)
    - ASME PTC 4 (Boiler Performance Testing)
    - IEC 61511 (Safety Instrumented Systems)
    - NFPA 85 (Boiler and Combustion Systems)

Author: GL-ComplianceEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime
import hashlib
import json


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class RegulatoryFramework(Enum):
    """Supported regulatory frameworks."""
    EPA_40CFR_PART75 = "epa_40cfr_part75"
    EPA_40CFR_PART98 = "epa_40cfr_part98"
    ASME_PTC4 = "asme_ptc4"
    IEC_61511 = "iec_61511"
    NFPA_85 = "nfpa_85"


class ComplianceStatus(Enum):
    """Compliance check result status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    NOT_APPLICABLE = "not_applicable"
    PENDING_REVIEW = "pending_review"


class SILLevel(Enum):
    """IEC 61511 Safety Integrity Levels."""
    SIL_1 = 1
    SIL_2 = 2
    SIL_3 = 3
    SIL_4 = 4


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ComplianceCheck:
    """Result of a single compliance check."""
    check_id: str
    framework: RegulatoryFramework
    requirement: str
    status: ComplianceStatus
    actual_value: Optional[Any] = None
    required_value: Optional[Any] = None
    deviation: Optional[float] = None
    explanation: str = ""
    remediation: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    provenance_hash: str = ""


@dataclass
class ComplianceReport:
    """Complete compliance report."""
    agent_id: str
    report_id: str
    timestamp: datetime
    frameworks_checked: List[RegulatoryFramework]
    total_checks: int
    compliant_checks: int
    non_compliant_checks: int
    warning_checks: int
    checks: List[ComplianceCheck]
    overall_status: ComplianceStatus
    provenance_hash: str = ""


@dataclass
class EPAPart75Config:
    """EPA 40 CFR Part 75 configuration."""
    unit_id: str
    monitoring_method: str  # "CEMS", "Appendix D", "LME"
    fuel_types: List[str]
    reporting_frequency: str  # "hourly", "quarterly"
    data_availability_target: float = 95.0  # Minimum 95%


@dataclass
class SafetyInterlockConfig:
    """Safety interlock configuration for IEC 61511 compliance."""
    interlock_id: str
    description: str
    sil_level: SILLevel
    trip_point: float
    trip_direction: str  # "high", "low"
    response_time_ms: int
    proof_test_interval_months: int


# =============================================================================
# EPA 40 CFR PART 75 REQUIREMENTS
# =============================================================================

EPA_PART75_LIMITS = {
    # Calibration drift limits (% of span)
    "daily_calibration_drift": {
        "SO2": 2.5,
        "NOx": 2.5,
        "CO2": 0.5,
        "O2": 0.5,
        "flow": 3.0,
    },
    # Relative accuracy requirements (%)
    "relative_accuracy": {
        "SO2": 20.0,
        "NOx": 20.0,
        "CO2": 10.0,
        "O2": 10.0,
        "flow": 15.0,
    },
    # Data availability minimum
    "data_availability_minimum": 95.0,
    # Substitute data limits (hours)
    "substitute_data_max_hours": 720,  # 30 days
}


# =============================================================================
# EPA 40 CFR PART 98 GHG REQUIREMENTS
# =============================================================================

EPA_PART98_EMISSION_FACTORS = {
    # CO2 emission factors (kg CO2/mmBtu)
    "natural_gas": 53.06,
    "sub_bituminous_coal": 97.17,
    "bituminous_coal": 93.28,
    "lignite": 97.72,
    "fuel_oil_no2": 73.96,
    "fuel_oil_no6": 75.10,
    "propane": 62.87,
}

EPA_PART98_THRESHOLDS = {
    # Reporting threshold (metric tons CO2e/year)
    "reporting_threshold": 25000,
    # Verification threshold
    "verification_threshold": 250000,
}


# =============================================================================
# IEC 61511 SAFETY REQUIREMENTS
# =============================================================================

IEC61511_SIL_PFD = {
    # Probability of Failure on Demand ranges
    SILLevel.SIL_1: (1e-2, 1e-1),   # 10^-2 to 10^-1
    SILLevel.SIL_2: (1e-3, 1e-2),   # 10^-3 to 10^-2
    SILLevel.SIL_3: (1e-4, 1e-3),   # 10^-4 to 10^-3
    SILLevel.SIL_4: (1e-5, 1e-4),   # 10^-5 to 10^-4
}

IEC61511_PROOF_TEST_INTERVALS = {
    # Maximum proof test intervals (months)
    SILLevel.SIL_1: 60,   # 5 years
    SILLevel.SIL_2: 24,   # 2 years
    SILLevel.SIL_3: 12,   # 1 year
    SILLevel.SIL_4: 6,    # 6 months
}


# =============================================================================
# NFPA 85 BOILER SAFETY REQUIREMENTS
# =============================================================================

NFPA85_REQUIREMENTS = {
    # Purge requirements
    "purge_air_changes": 4,  # Minimum 4 air changes
    "purge_airflow_percent": 25,  # Minimum 25% of full load airflow
    "purge_time_seconds": 60,  # Minimum purge time

    # Flame detection
    "flame_detection_response_ms": 4000,  # Maximum 4 seconds
    "main_flame_establish_ms": 10000,  # Maximum 10 seconds

    # Safety shutdown
    "fuel_valve_closure_ms": 1000,  # Maximum 1 second

    # Post-purge
    "post_purge_time_seconds": 30,  # Minimum post-purge
}


# =============================================================================
# COMPLIANCE VALIDATOR CLASS
# =============================================================================

class ComplianceValidator:
    """
    Validates thermal system operations against regulatory requirements.

    Provides:
        - Real-time compliance checking
        - Regulatory mapping documentation
        - Audit trail generation
        - Remediation recommendations
    """

    def __init__(self, agent_id: str = "GL-001"):
        """
        Initialize compliance validator.

        Args:
            agent_id: Agent identifier for audit trail
        """
        self.agent_id = agent_id
        self.checks: List[ComplianceCheck] = []

    def validate_all(
        self,
        data: Dict[str, Any],
        frameworks: Optional[List[RegulatoryFramework]] = None
    ) -> ComplianceReport:
        """
        Run all compliance checks for specified frameworks.

        Args:
            data: Dictionary containing system data to validate
            frameworks: List of frameworks to check (default: all)

        Returns:
            ComplianceReport with all check results
        """
        if frameworks is None:
            frameworks = list(RegulatoryFramework)

        self.checks = []

        for framework in frameworks:
            if framework == RegulatoryFramework.EPA_40CFR_PART75:
                self._validate_epa_part75(data)
            elif framework == RegulatoryFramework.EPA_40CFR_PART98:
                self._validate_epa_part98(data)
            elif framework == RegulatoryFramework.ASME_PTC4:
                self._validate_asme_ptc4(data)
            elif framework == RegulatoryFramework.IEC_61511:
                self._validate_iec_61511(data)
            elif framework == RegulatoryFramework.NFPA_85:
                self._validate_nfpa_85(data)

        return self._generate_report(frameworks)

    def _validate_epa_part75(self, data: Dict[str, Any]) -> None:
        """Validate EPA 40 CFR Part 75 compliance."""
        # Check data availability
        data_availability = data.get("data_availability_percent", 100.0)
        required = EPA_PART75_LIMITS["data_availability_minimum"]

        self.checks.append(ComplianceCheck(
            check_id="EPA75-001",
            framework=RegulatoryFramework.EPA_40CFR_PART75,
            requirement="Data availability >= 95%",
            status=ComplianceStatus.COMPLIANT if data_availability >= required else ComplianceStatus.NON_COMPLIANT,
            actual_value=data_availability,
            required_value=required,
            deviation=data_availability - required,
            explanation=(
                f"CEMS data availability is {data_availability:.1f}%. "
                f"EPA Part 75 requires minimum {required}%."
            ),
            remediation=(
                "Review CEMS maintenance procedures, calibration frequency, "
                "and backup monitoring systems."
            ) if data_availability < required else "",
        ))

        # Check calibration drift
        for pollutant in ["SO2", "NOx", "CO2", "O2"]:
            drift_key = f"{pollutant.lower()}_calibration_drift"
            actual_drift = data.get(drift_key, 0.0)
            max_drift = EPA_PART75_LIMITS["daily_calibration_drift"].get(pollutant, 2.5)

            self.checks.append(ComplianceCheck(
                check_id=f"EPA75-DRIFT-{pollutant}",
                framework=RegulatoryFramework.EPA_40CFR_PART75,
                requirement=f"{pollutant} daily calibration drift <= {max_drift}% of span",
                status=ComplianceStatus.COMPLIANT if actual_drift <= max_drift else ComplianceStatus.NON_COMPLIANT,
                actual_value=actual_drift,
                required_value=max_drift,
                deviation=actual_drift - max_drift if actual_drift > max_drift else 0,
                explanation=(
                    f"{pollutant} analyzer calibration drift is {actual_drift:.2f}% of span. "
                    f"Maximum allowed is {max_drift}%."
                ),
                remediation=(
                    f"Recalibrate {pollutant} analyzer. If drift persists, "
                    f"schedule preventive maintenance or replacement."
                ) if actual_drift > max_drift else "",
            ))

        # Check substitute data usage
        substitute_hours = data.get("substitute_data_hours", 0)
        max_hours = EPA_PART75_LIMITS["substitute_data_max_hours"]

        status = ComplianceStatus.COMPLIANT
        if substitute_hours > max_hours:
            status = ComplianceStatus.NON_COMPLIANT
        elif substitute_hours > max_hours * 0.8:
            status = ComplianceStatus.WARNING

        self.checks.append(ComplianceCheck(
            check_id="EPA75-SUBDATA",
            framework=RegulatoryFramework.EPA_40CFR_PART75,
            requirement=f"Substitute data usage <= {max_hours} hours per quarter",
            status=status,
            actual_value=substitute_hours,
            required_value=max_hours,
            explanation=(
                f"Substitute data used for {substitute_hours} hours this quarter. "
                f"Maximum allowed is {max_hours} hours."
            ),
            remediation="Reduce monitor downtime through improved maintenance." if status != ComplianceStatus.COMPLIANT else "",
        ))

    def _validate_epa_part98(self, data: Dict[str, Any]) -> None:
        """Validate EPA 40 CFR Part 98 GHG reporting compliance."""
        # Check emission factor accuracy
        fuel_type = data.get("fuel_type", "natural_gas")
        reported_ef = data.get("co2_emission_factor", 0.0)
        expected_ef = EPA_PART98_EMISSION_FACTORS.get(fuel_type, 53.06)

        deviation_percent = abs(reported_ef - expected_ef) / expected_ef * 100 if expected_ef > 0 else 0

        self.checks.append(ComplianceCheck(
            check_id="EPA98-EF",
            framework=RegulatoryFramework.EPA_40CFR_PART98,
            requirement=f"CO2 emission factor for {fuel_type} accurate within 5%",
            status=ComplianceStatus.COMPLIANT if deviation_percent <= 5.0 else ComplianceStatus.NON_COMPLIANT,
            actual_value=reported_ef,
            required_value=expected_ef,
            deviation=deviation_percent,
            explanation=(
                f"Reported CO2 emission factor is {reported_ef:.2f} kg/mmBtu. "
                f"EPA standard for {fuel_type} is {expected_ef:.2f} kg/mmBtu "
                f"({deviation_percent:.1f}% deviation)."
            ),
            remediation="Use EPA Table C-1 emission factors or conduct fuel sampling." if deviation_percent > 5.0 else "",
        ))

        # Check reporting threshold
        annual_emissions = data.get("annual_co2_emissions_mt", 0)
        threshold = EPA_PART98_THRESHOLDS["reporting_threshold"]

        self.checks.append(ComplianceCheck(
            check_id="EPA98-THRESHOLD",
            framework=RegulatoryFramework.EPA_40CFR_PART98,
            requirement=f"Determine reporting requirement based on {threshold} MT threshold",
            status=ComplianceStatus.COMPLIANT,
            actual_value=annual_emissions,
            required_value=threshold,
            explanation=(
                f"Annual CO2 emissions: {annual_emissions:,.0f} metric tons. "
                f"{'Subject to' if annual_emissions >= threshold else 'Exempt from'} "
                f"EPA Part 98 reporting (threshold: {threshold:,} MT)."
            ),
        ))

    def _validate_asme_ptc4(self, data: Dict[str, Any]) -> None:
        """Validate ASME PTC 4 boiler performance requirements."""
        # Check efficiency calculation method
        efficiency = data.get("boiler_efficiency", 0.0)

        self.checks.append(ComplianceCheck(
            check_id="PTC4-EFF",
            framework=RegulatoryFramework.ASME_PTC4,
            requirement="Boiler efficiency within expected range (75-95%)",
            status=ComplianceStatus.COMPLIANT if 75 <= efficiency <= 95 else ComplianceStatus.WARNING,
            actual_value=efficiency,
            required_value="75-95%",
            explanation=(
                f"Calculated boiler efficiency is {efficiency:.1f}%. "
                f"Typical range per ASME PTC 4 is 75-95% for fired boilers."
            ),
            remediation=(
                "Review measurement instrumentation and calculation methodology "
                "per ASME PTC 4 Section 5."
            ) if not (75 <= efficiency <= 95) else "",
        ))

        # Check heat balance closure
        heat_balance_error = data.get("heat_balance_error_percent", 0.0)

        self.checks.append(ComplianceCheck(
            check_id="PTC4-BALANCE",
            framework=RegulatoryFramework.ASME_PTC4,
            requirement="Heat balance closure within 2%",
            status=ComplianceStatus.COMPLIANT if abs(heat_balance_error) <= 2.0 else ComplianceStatus.WARNING,
            actual_value=heat_balance_error,
            required_value=2.0,
            deviation=abs(heat_balance_error) - 2.0 if abs(heat_balance_error) > 2.0 else 0,
            explanation=(
                f"Heat balance error is {heat_balance_error:.2f}%. "
                f"ASME PTC 4 recommends closure within ±2%."
            ),
            remediation=(
                "Review all input/output measurements for accuracy. "
                "Check for unmetered losses or unmeasured flows."
            ) if abs(heat_balance_error) > 2.0 else "",
        ))

    def _validate_iec_61511(self, data: Dict[str, Any]) -> None:
        """Validate IEC 61511 Safety Instrumented System requirements."""
        interlocks = data.get("safety_interlocks", [])

        for interlock in interlocks:
            interlock_id = interlock.get("id", "unknown")
            sil_level = SILLevel(interlock.get("sil_level", 1))
            pfd = interlock.get("pfd", 0.1)
            proof_test_months = interlock.get("proof_test_interval_months", 12)

            # Check PFD against SIL requirements
            pfd_range = IEC61511_SIL_PFD.get(sil_level, (1e-2, 1e-1))
            pfd_compliant = pfd_range[0] <= pfd <= pfd_range[1]

            self.checks.append(ComplianceCheck(
                check_id=f"IEC61511-PFD-{interlock_id}",
                framework=RegulatoryFramework.IEC_61511,
                requirement=f"PFD for SIL {sil_level.value} between {pfd_range[0]:.0e} and {pfd_range[1]:.0e}",
                status=ComplianceStatus.COMPLIANT if pfd_compliant else ComplianceStatus.NON_COMPLIANT,
                actual_value=pfd,
                required_value=pfd_range,
                explanation=(
                    f"Interlock {interlock_id}: PFD = {pfd:.2e}. "
                    f"SIL {sil_level.value} requires PFD between {pfd_range[0]:.0e} and {pfd_range[1]:.0e}."
                ),
                remediation=(
                    f"Review SIF design, add redundancy, or reduce proof test interval "
                    f"to achieve target PFD for SIL {sil_level.value}."
                ) if not pfd_compliant else "",
            ))

            # Check proof test interval
            max_interval = IEC61511_PROOF_TEST_INTERVALS.get(sil_level, 60)
            interval_compliant = proof_test_months <= max_interval

            self.checks.append(ComplianceCheck(
                check_id=f"IEC61511-PROOF-{interlock_id}",
                framework=RegulatoryFramework.IEC_61511,
                requirement=f"Proof test interval <= {max_interval} months for SIL {sil_level.value}",
                status=ComplianceStatus.COMPLIANT if interval_compliant else ComplianceStatus.NON_COMPLIANT,
                actual_value=proof_test_months,
                required_value=max_interval,
                explanation=(
                    f"Interlock {interlock_id}: Proof test interval = {proof_test_months} months. "
                    f"SIL {sil_level.value} requires <= {max_interval} months."
                ),
                remediation=(
                    f"Reduce proof test interval to maximum {max_interval} months "
                    f"or redesign SIF for lower SIL requirement."
                ) if not interval_compliant else "",
            ))

    def _validate_nfpa_85(self, data: Dict[str, Any]) -> None:
        """Validate NFPA 85 Boiler and Combustion Systems requirements."""
        # Check purge requirements
        purge_air_changes = data.get("purge_air_changes", 0)
        required_air_changes = NFPA85_REQUIREMENTS["purge_air_changes"]

        self.checks.append(ComplianceCheck(
            check_id="NFPA85-PURGE",
            framework=RegulatoryFramework.NFPA_85,
            requirement=f"Pre-ignition purge >= {required_air_changes} air changes",
            status=ComplianceStatus.COMPLIANT if purge_air_changes >= required_air_changes else ComplianceStatus.NON_COMPLIANT,
            actual_value=purge_air_changes,
            required_value=required_air_changes,
            explanation=(
                f"Purge sequence provides {purge_air_changes} air changes. "
                f"NFPA 85 requires minimum {required_air_changes} air changes."
            ),
            remediation=(
                f"Increase purge duration or airflow to achieve minimum "
                f"{required_air_changes} air changes before ignition."
            ) if purge_air_changes < required_air_changes else "",
        ))

        # Check flame detection response time
        flame_response = data.get("flame_detection_response_ms", 0)
        max_response = NFPA85_REQUIREMENTS["flame_detection_response_ms"]

        self.checks.append(ComplianceCheck(
            check_id="NFPA85-FLAME",
            framework=RegulatoryFramework.NFPA_85,
            requirement=f"Flame detection response <= {max_response} ms",
            status=ComplianceStatus.COMPLIANT if flame_response <= max_response else ComplianceStatus.NON_COMPLIANT,
            actual_value=flame_response,
            required_value=max_response,
            explanation=(
                f"Flame detector response time is {flame_response} ms. "
                f"NFPA 85 requires <= {max_response} ms (4 seconds)."
            ),
            remediation="Replace flame detector with faster response model." if flame_response > max_response else "",
        ))

        # Check fuel valve closure time
        valve_closure = data.get("fuel_valve_closure_ms", 0)
        max_closure = NFPA85_REQUIREMENTS["fuel_valve_closure_ms"]

        self.checks.append(ComplianceCheck(
            check_id="NFPA85-VALVE",
            framework=RegulatoryFramework.NFPA_85,
            requirement=f"Fuel valve closure <= {max_closure} ms",
            status=ComplianceStatus.COMPLIANT if valve_closure <= max_closure else ComplianceStatus.NON_COMPLIANT,
            actual_value=valve_closure,
            required_value=max_closure,
            explanation=(
                f"Fuel valve closure time is {valve_closure} ms. "
                f"NFPA 85 requires <= {max_closure} ms (1 second)."
            ),
            remediation=(
                "Install faster-acting fuel safety shutoff valves "
                "or add parallel redundant valves."
            ) if valve_closure > max_closure else "",
        ))

    def _generate_report(self, frameworks: List[RegulatoryFramework]) -> ComplianceReport:
        """Generate compliance report from checks."""
        compliant = sum(1 for c in self.checks if c.status == ComplianceStatus.COMPLIANT)
        non_compliant = sum(1 for c in self.checks if c.status == ComplianceStatus.NON_COMPLIANT)
        warnings = sum(1 for c in self.checks if c.status == ComplianceStatus.WARNING)

        # Determine overall status
        if non_compliant > 0:
            overall = ComplianceStatus.NON_COMPLIANT
        elif warnings > 0:
            overall = ComplianceStatus.WARNING
        else:
            overall = ComplianceStatus.COMPLIANT

        # Generate provenance hash
        check_data = json.dumps([
            {"id": c.check_id, "status": c.status.value, "value": str(c.actual_value)}
            for c in self.checks
        ], sort_keys=True)
        provenance_hash = hashlib.sha256(check_data.encode()).hexdigest()

        return ComplianceReport(
            agent_id=self.agent_id,
            report_id=f"CR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            frameworks_checked=frameworks,
            total_checks=len(self.checks),
            compliant_checks=compliant,
            non_compliant_checks=non_compliant,
            warning_checks=warnings,
            checks=self.checks,
            overall_status=overall,
            provenance_hash=provenance_hash,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_thermal_system(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to validate thermal system compliance.

    Args:
        data: System operational data

    Returns:
        Dictionary with compliance results
    """
    validator = ComplianceValidator()
    report = validator.validate_all(data)

    return {
        "report_id": report.report_id,
        "timestamp": report.timestamp.isoformat(),
        "overall_status": report.overall_status.value,
        "summary": {
            "total_checks": report.total_checks,
            "compliant": report.compliant_checks,
            "non_compliant": report.non_compliant_checks,
            "warnings": report.warning_checks,
        },
        "checks": [
            {
                "id": c.check_id,
                "framework": c.framework.value,
                "requirement": c.requirement,
                "status": c.status.value,
                "actual": c.actual_value,
                "required": c.required_value,
                "explanation": c.explanation,
                "remediation": c.remediation,
            }
            for c in report.checks
        ],
        "provenance_hash": report.provenance_hash,
    }


if __name__ == "__main__":
    # Example usage
    test_data = {
        "data_availability_percent": 96.5,
        "so2_calibration_drift": 1.5,
        "nox_calibration_drift": 2.0,
        "co2_calibration_drift": 0.3,
        "o2_calibration_drift": 0.4,
        "substitute_data_hours": 120,
        "fuel_type": "natural_gas",
        "co2_emission_factor": 53.06,
        "annual_co2_emissions_mt": 35000,
        "boiler_efficiency": 85.5,
        "heat_balance_error_percent": 1.2,
        "purge_air_changes": 5,
        "flame_detection_response_ms": 3500,
        "fuel_valve_closure_ms": 800,
        "safety_interlocks": [
            {"id": "HH-LEVEL", "sil_level": 2, "pfd": 0.005, "proof_test_interval_months": 12},
            {"id": "LL-FUEL", "sil_level": 2, "pfd": 0.008, "proof_test_interval_months": 12},
        ],
    }

    result = validate_thermal_system(test_data)
    print(json.dumps(result, indent=2))
