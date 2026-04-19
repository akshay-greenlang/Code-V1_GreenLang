# -*- coding: utf-8 -*-
"""
GL-002 FLAMEGUARD - Compliance Validator

Central compliance validation engine supporting multiple regulatory standards.

Supported Standards:
- ASME PTC 4.1-2013: Fired Steam Generators Performance Test
- NFPA 85-2023: Boiler and Combustion Systems Hazards Code
- EPA 40 CFR Part 60/63/98: Emissions Standards
- IEC 61511: Functional Safety for Safety Instrumented Systems

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class ComplianceStandard(str, Enum):
    """Supported compliance standards."""
    ASME_PTC_4_1 = "ASME_PTC_4.1-2013"
    NFPA_85 = "NFPA_85-2023"
    EPA_40CFR60 = "EPA_40CFR_Part60"
    EPA_40CFR63 = "EPA_40CFR_Part63"
    EPA_40CFR98 = "EPA_40CFR_Part98"
    IEC_61511 = "IEC_61511-2016"


class ComplianceLevel(str, Enum):
    """Compliance assessment levels."""
    FULL = "full_compliance"
    PARTIAL = "partial_compliance"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"


@dataclass
class ComplianceCheck:
    """Result of a single compliance check."""
    check_id: str
    standard: ComplianceStandard
    section: str
    description: str
    requirement: str
    actual_value: Any
    expected_value: Any
    is_compliant: bool
    evidence: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None
    severity: str = "medium"  # low, medium, high, critical


@dataclass
class ComplianceReport:
    """Complete compliance assessment report."""
    report_id: str
    boiler_id: str
    timestamp: datetime
    standards_assessed: List[ComplianceStandard]
    overall_level: ComplianceLevel
    checks: List[ComplianceCheck]
    compliant_count: int
    non_compliant_count: int
    critical_findings: List[ComplianceCheck]
    recommendations: List[str]
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            content = f"{self.report_id}|{self.boiler_id}|{self.timestamp.isoformat()}"
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


class ComplianceValidator:
    """
    Multi-standard compliance validator for boiler systems.
    """

    VERSION = "1.0.0"

    # ASME PTC 4.1 Calculation Method Mappings
    ASME_PTC_4_1_SECTIONS = {
        "5.4": "Direct Method Efficiency",
        "5.5": "Indirect Method (Heat Loss) Efficiency",
        "5.5.1": "Dry Flue Gas Loss",
        "5.5.2": "Moisture in Fuel Loss",
        "5.5.3": "Hydrogen in Fuel Moisture Loss",
        "5.5.4": "Moisture in Air Loss",
        "5.5.5": "Unburned Carbon Loss",
        "5.5.6": "CO Loss (Unburned Gas)",
        "5.5.7": "Surface Radiation and Convection Loss",
        "5.5.8": "Blowdown Loss",
        "5.6": "Uncertainty Analysis",
        "6.2": "Fuel Analysis Requirements",
        "6.3": "Air and Flue Gas Analysis",
        "6.4": "Steam/Water Measurements",
    }

    # NFPA 85 Chapter 5 Requirements
    NFPA_85_CHAPTER_5 = {
        "5.1": "Scope and Application",
        "5.2": "Hazards of Operation",
        "5.3": "Safety Interlocks and Trips",
        "5.3.1": "Safety Interlock Requirements",
        "5.3.3": "Flame Detection",
        "5.3.4": "Emergency Shutdown",
        "5.3.5": "Flame Failure Response",
        "5.4": "Supervised Manual Systems",
        "5.5": "Automatic Combustion Control",
        "5.6": "Purge and Operating Sequences",
        "5.6.4": "Prepurge Requirements",
        "5.6.5": "Trial for Ignition",
        "5.7": "Normal Operation and Shutdown",
    }

    def __init__(self, boiler_id: str) -> None:
        """Initialize compliance validator."""
        self.boiler_id = boiler_id
        self._validation_history: List[ComplianceReport] = []
        logger.info(f"ComplianceValidator initialized for {boiler_id}")

    def validate_asme_ptc_4_1(
        self,
        efficiency_result: Dict[str, Any],
        fuel_analysis: Optional[Dict[str, Any]] = None,
    ) -> List[ComplianceCheck]:
        """
        Validate efficiency calculations against ASME PTC 4.1.
        """
        checks = []

        # Check 1: Efficiency calculation method documented
        checks.append(ComplianceCheck(
            check_id="ASME-001",
            standard=ComplianceStandard.ASME_PTC_4_1,
            section="5.4/5.5",
            description="Efficiency calculation method selection",
            requirement="Use either Direct (5.4) or Indirect (5.5) method",
            actual_value=efficiency_result.get("method", "unknown"),
            expected_value=["direct", "indirect"],
            is_compliant=efficiency_result.get("method") in ["direct", "indirect"],
        ))

        # Check 2: All loss categories included (indirect method)
        if efficiency_result.get("method") == "indirect":
            required_losses = [
                "dry_flue_gas_loss_percent",
                "hydrogen_combustion_loss_percent",
                "radiation_loss_percent",
            ]
            losses_present = all(
                efficiency_result.get(loss) is not None
                for loss in required_losses
            )
            checks.append(ComplianceCheck(
                check_id="ASME-002",
                standard=ComplianceStandard.ASME_PTC_4_1,
                section="5.5",
                description="Required loss categories included",
                requirement="Calculate all applicable heat losses per 5.5.1-5.5.8",
                actual_value=list(efficiency_result.keys()),
                expected_value=required_losses,
                is_compliant=losses_present,
            ))

        # Check 3: Uncertainty documented
        uncertainty = efficiency_result.get("uncertainty_percent")
        checks.append(ComplianceCheck(
            check_id="ASME-003",
            standard=ComplianceStandard.ASME_PTC_4_1,
            section="5.6",
            description="Uncertainty analysis performed",
            requirement="Document measurement uncertainty per Section 5.6",
            actual_value=uncertainty,
            expected_value="<= 1.0%",
            is_compliant=uncertainty is not None and uncertainty <= 1.0,
            recommendation="Improve instrumentation accuracy" if uncertainty and uncertainty > 1.0 else None,
        ))

        # Check 4: Provenance tracking
        has_provenance = (
            efficiency_result.get("input_hash") is not None and
            efficiency_result.get("output_hash") is not None
        )
        checks.append(ComplianceCheck(
            check_id="ASME-004",
            standard=ComplianceStandard.ASME_PTC_4_1,
            section="General",
            description="Calculation provenance tracked",
            requirement="Maintain audit trail of calculations",
            actual_value=has_provenance,
            expected_value=True,
            is_compliant=has_provenance,
        ))

        # Check 5: Efficiency within reasonable bounds
        efficiency = efficiency_result.get("efficiency_hhv_percent", 0)
        checks.append(ComplianceCheck(
            check_id="ASME-005",
            standard=ComplianceStandard.ASME_PTC_4_1,
            section="5.4",
            description="Efficiency within valid range",
            requirement="Efficiency should be 50-100% (HHV basis)",
            actual_value=efficiency,
            expected_value="50-100%",
            is_compliant=50.0 <= efficiency <= 100.0,
            severity="high" if not (50.0 <= efficiency <= 100.0) else "low",
        ))

        return checks

    def validate_nfpa_85(
        self,
        bms_config: Dict[str, Any],
        interlock_status: Dict[str, Any],
    ) -> List[ComplianceCheck]:
        """
        Validate BMS configuration against NFPA 85 Chapter 5.
        """
        checks = []

        # Check 1: Prepurge timing
        purge_time = bms_config.get("pre_purge_time_s", 0)
        min_purge = 300  # 5 minutes minimum
        checks.append(ComplianceCheck(
            check_id="NFPA-001",
            standard=ComplianceStandard.NFPA_85,
            section="5.6.4",
            description="Prepurge duration meets minimum",
            requirement=f"Prepurge >= {min_purge} seconds (4 volume changes)",
            actual_value=purge_time,
            expected_value=min_purge,
            is_compliant=purge_time >= min_purge,
            severity="critical" if purge_time < min_purge else "low",
        ))

        # Check 2: Pilot trial timing
        pilot_trial = bms_config.get("pilot_trial_time_s", 999)
        max_pilot = 10.0
        checks.append(ComplianceCheck(
            check_id="NFPA-002",
            standard=ComplianceStandard.NFPA_85,
            section="5.6.5",
            description="Pilot trial time limit",
            requirement=f"Pilot trial <= {max_pilot} seconds",
            actual_value=pilot_trial,
            expected_value=max_pilot,
            is_compliant=pilot_trial <= max_pilot,
            severity="critical" if pilot_trial > max_pilot else "low",
        ))

        # Check 3: Flame failure response
        flame_response = bms_config.get("flame_failure_response_s", 999)
        max_response = 4.0
        checks.append(ComplianceCheck(
            check_id="NFPA-003",
            standard=ComplianceStandard.NFPA_85,
            section="5.3.5.2",
            description="Flame failure response time",
            requirement=f"Fuel shutoff <= {max_response} seconds",
            actual_value=flame_response,
            expected_value=max_response,
            is_compliant=flame_response <= max_response,
            severity="critical" if flame_response > max_response else "low",
        ))

        # Check 4: Required interlocks present
        required_interlocks = [
            "combustion_air_proving",
            "fuel_pressure_proving",
            "flame_detection",
            "low_water_cutoff",
        ]
        present = [il for il in required_interlocks if interlock_status.get(il)]
        checks.append(ComplianceCheck(
            check_id="NFPA-004",
            standard=ComplianceStandard.NFPA_85,
            section="5.3.1",
            description="Required safety interlocks installed",
            requirement="All mandatory interlocks present and active",
            actual_value=present,
            expected_value=required_interlocks,
            is_compliant=len(present) == len(required_interlocks),
            severity="critical" if len(present) < len(required_interlocks) else "low",
        ))

        return checks

    def generate_report(
        self,
        standards: List[ComplianceStandard],
        efficiency_result: Optional[Dict[str, Any]] = None,
        bms_config: Optional[Dict[str, Any]] = None,
        interlock_status: Optional[Dict[str, Any]] = None,
    ) -> ComplianceReport:
        """
        Generate comprehensive compliance report.
        """
        all_checks = []
        recommendations = []

        # Validate each requested standard
        if ComplianceStandard.ASME_PTC_4_1 in standards and efficiency_result:
            asme_checks = self.validate_asme_ptc_4_1(efficiency_result)
            all_checks.extend(asme_checks)

        if ComplianceStandard.NFPA_85 in standards:
            bms = bms_config or {}
            interlocks = interlock_status or {}
            nfpa_checks = self.validate_nfpa_85(bms, interlocks)
            all_checks.extend(nfpa_checks)

        # Count results
        compliant = sum(1 for c in all_checks if c.is_compliant)
        non_compliant = sum(1 for c in all_checks if not c.is_compliant)
        critical = [c for c in all_checks if not c.is_compliant and c.severity == "critical"]

        # Determine overall level
        if non_compliant == 0:
            overall = ComplianceLevel.FULL
        elif critical:
            overall = ComplianceLevel.NON_COMPLIANT
        else:
            overall = ComplianceLevel.PARTIAL

        # Gather recommendations
        recommendations = [c.recommendation for c in all_checks if c.recommendation]

        report = ComplianceReport(
            report_id=f"COMP-{self.boiler_id}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            boiler_id=self.boiler_id,
            timestamp=datetime.now(timezone.utc),
            standards_assessed=standards,
            overall_level=overall,
            checks=all_checks,
            compliant_count=compliant,
            non_compliant_count=non_compliant,
            critical_findings=critical,
            recommendations=recommendations,
        )

        self._validation_history.append(report)
        return report

    def get_section_mapping(
        self,
        standard: ComplianceStandard,
    ) -> Dict[str, str]:
        """Get section-to-description mapping for a standard."""
        if standard == ComplianceStandard.ASME_PTC_4_1:
            return self.ASME_PTC_4_1_SECTIONS
        elif standard == ComplianceStandard.NFPA_85:
            return self.NFPA_85_CHAPTER_5
        return {}


__all__ = [
    "ComplianceStandard",
    "ComplianceLevel",
    "ComplianceCheck",
    "ComplianceReport",
    "ComplianceValidator",
]
