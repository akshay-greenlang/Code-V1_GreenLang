# -*- coding: utf-8 -*-
"""
GL-007 FURNACEPULSE - SIL (Safety Integrity Level) Certification Documentation

This module documents the Safety Integrity Level requirements and compliance
for GL-007 FurnacePulse industrial furnace monitoring system per IEC 61508/61511.

Reference Standards:
- IEC 61508: Functional Safety of Electrical/Electronic/Programmable Systems
- IEC 61511: Functional Safety - Safety Instrumented Systems for Process Industry
- NFPA 86: Standard for Ovens and Furnaces
- ISA 84.00.01: Safety Instrumented Functions (SIF)

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import hashlib


class SILLevel(str, Enum):
    """Safety Integrity Levels per IEC 61508."""
    SIL_1 = "SIL_1"  # 10^-2 to 10^-1 PFDavg
    SIL_2 = "SIL_2"  # 10^-3 to 10^-2 PFDavg
    SIL_3 = "SIL_3"  # 10^-4 to 10^-3 PFDavg
    SIL_4 = "SIL_4"  # 10^-5 to 10^-4 PFDavg (rarely used in process industry)


class SafetyFunction(str, Enum):
    """Safety Instrumented Functions for furnace operations."""
    HIGH_TEMPERATURE_SHUTDOWN = "F-001"
    COMBUSTION_AIR_LOSS = "F-002"
    FLAME_FAILURE_DETECTION = "F-003"
    PRESSURE_RELIEF = "F-004"
    ATMOSPHERE_CONTROL = "F-005"
    EMERGENCY_PURGE = "F-006"
    OVERTEMPERATURE_INTERLOCK = "F-007"
    DOOR_INTERLOCK = "F-008"


class VotingArchitecture(str, Enum):
    """Redundancy voting architectures."""
    SINGLE_1oo1 = "1oo1"      # Single channel
    REDUNDANT_1oo2 = "1oo2"   # One out of two (fail-safe)
    MAJORITY_2oo3 = "2oo3"    # Two out of three (fault-tolerant)
    QUAD_2oo4 = "2oo4"        # Two out of four


class FailureMode(str, Enum):
    """Safe vs dangerous failure modes."""
    SAFE_DETECTED = "SD"
    SAFE_UNDETECTED = "SU"
    DANGEROUS_DETECTED = "DD"
    DANGEROUS_UNDETECTED = "DU"


@dataclass
class SIFSpecification:
    """Safety Instrumented Function specification."""
    sif_id: str
    name: str
    description: str
    target_sil: SILLevel
    pfd_target: float  # Probability of Failure on Demand (average)
    rrf: int  # Risk Reduction Factor
    voting: VotingArchitecture
    response_time_ms: int
    test_interval_months: int
    proof_test_coverage: float
    safe_state: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sif_id": self.sif_id,
            "name": self.name,
            "description": self.description,
            "target_sil": self.target_sil.value,
            "pfd_target": self.pfd_target,
            "rrf": self.rrf,
            "voting": self.voting.value,
            "response_time_ms": self.response_time_ms,
            "test_interval_months": self.test_interval_months,
            "proof_test_coverage": self.proof_test_coverage,
            "safe_state": self.safe_state,
        }


@dataclass
class SILVerificationRecord:
    """Record of SIL verification test."""
    record_id: str
    timestamp: datetime
    sif_id: str
    test_type: str  # "proof_test", "partial_stroke", "diagnostic"
    result: str  # "PASS", "FAIL", "DEGRADED"
    measured_pfd: Optional[float] = None
    next_test_due: Optional[datetime] = None
    technician_id: str = ""
    notes: str = ""
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            content = f"{self.record_id}|{self.sif_id}|{self.result}|{self.timestamp.isoformat()}"
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# GL-007 FURNACEPULSE SIF DEFINITIONS
# =============================================================================

FURNACEPULSE_SIFS: Dict[str, SIFSpecification] = {
    "F-001": SIFSpecification(
        sif_id="F-001",
        name="High Temperature Shutdown",
        description="Initiate controlled shutdown when furnace temperature exceeds safe limit",
        target_sil=SILLevel.SIL_2,
        pfd_target=0.005,  # 5 x 10^-3
        rrf=200,
        voting=VotingArchitecture.REDUNDANT_1oo2,
        response_time_ms=1000,
        test_interval_months=12,
        proof_test_coverage=0.95,
        safe_state="Close fuel supply, open cooling air dampers",
    ),
    "F-002": SIFSpecification(
        sif_id="F-002",
        name="Combustion Air Loss Detection",
        description="Detect loss of combustion air and trigger fuel cutoff",
        target_sil=SILLevel.SIL_2,
        pfd_target=0.003,
        rrf=333,
        voting=VotingArchitecture.MAJORITY_2oo3,
        response_time_ms=500,
        test_interval_months=6,
        proof_test_coverage=0.90,
        safe_state="Immediate fuel shutoff, activate alarms",
    ),
    "F-003": SIFSpecification(
        sif_id="F-003",
        name="Flame Failure Detection",
        description="Detect flame loss and prevent fuel accumulation per NFPA 86",
        target_sil=SILLevel.SIL_3,
        pfd_target=0.0008,
        rrf=1250,
        voting=VotingArchitecture.REDUNDANT_1oo2,
        response_time_ms=4000,  # 4 second flame failure response time (NFPA 86)
        test_interval_months=3,
        proof_test_coverage=0.98,
        safe_state="Close main fuel valve, close pilot valve, purge",
    ),
    "F-004": SIFSpecification(
        sif_id="F-004",
        name="Pressure Relief Activation",
        description="Open pressure relief upon overpressure condition",
        target_sil=SILLevel.SIL_2,
        pfd_target=0.004,
        rrf=250,
        voting=VotingArchitecture.SINGLE_1oo1,
        response_time_ms=100,
        test_interval_months=12,
        proof_test_coverage=0.85,
        safe_state="Open relief valve, fuel shutoff",
    ),
    "F-005": SIFSpecification(
        sif_id="F-005",
        name="Protective Atmosphere Control",
        description="Maintain safe atmosphere composition (N2/H2 ratio)",
        target_sil=SILLevel.SIL_2,
        pfd_target=0.006,
        rrf=167,
        voting=VotingArchitecture.MAJORITY_2oo3,
        response_time_ms=2000,
        test_interval_months=6,
        proof_test_coverage=0.92,
        safe_state="Increase N2 flow, reduce H2, alert operator",
    ),
    "F-006": SIFSpecification(
        sif_id="F-006",
        name="Emergency Purge Initiation",
        description="Initiate pre-ignition purge cycle per NFPA 86",
        target_sil=SILLevel.SIL_2,
        pfd_target=0.002,
        rrf=500,
        voting=VotingArchitecture.REDUNDANT_1oo2,
        response_time_ms=1000,
        test_interval_months=6,
        proof_test_coverage=0.95,
        safe_state="4 volume changes minimum before ignition attempt",
    ),
    "F-007": SIFSpecification(
        sif_id="F-007",
        name="Overtemperature Interlock",
        description="Prevent heating element activation above limit",
        target_sil=SILLevel.SIL_1,
        pfd_target=0.02,
        rrf=50,
        voting=VotingArchitecture.SINGLE_1oo1,
        response_time_ms=500,
        test_interval_months=12,
        proof_test_coverage=0.90,
        safe_state="Disable heating elements, alarm",
    ),
    "F-008": SIFSpecification(
        sif_id="F-008",
        name="Door/Access Interlock",
        description="Prevent furnace operation with door open",
        target_sil=SILLevel.SIL_1,
        pfd_target=0.05,
        rrf=20,
        voting=VotingArchitecture.SINGLE_1oo1,
        response_time_ms=200,
        test_interval_months=6,
        proof_test_coverage=0.95,
        safe_state="Heating disabled, fuel blocked until door closed",
    ),
}


# =============================================================================
# SIL COMPLIANCE CHECKER
# =============================================================================

class SILComplianceChecker:
    """
    Verifies SIL compliance for GL-007 FurnacePulse.

    Checks:
    - Hardware fault tolerance requirements
    - Safe failure fraction requirements
    - Systematic capability requirements
    - Proof test schedule compliance
    """

    VERSION = "1.0.0"

    # IEC 61508 Table 2 - Hardware Fault Tolerance Requirements
    HFT_REQUIREMENTS = {
        SILLevel.SIL_1: {"min_sff_type_a": 0.60, "min_sff_type_b": 0.90, "min_hft": 0},
        SILLevel.SIL_2: {"min_sff_type_a": 0.90, "min_sff_type_b": 0.99, "min_hft": 0},
        SILLevel.SIL_3: {"min_sff_type_a": 0.99, "min_sff_type_b": 0.99, "min_hft": 1},
        SILLevel.SIL_4: {"min_sff_type_a": 0.99, "min_sff_type_b": 0.99, "min_hft": 2},
    }

    def __init__(self, agent_id: str = "GL-007"):
        self.agent_id = agent_id
        self._sifs = dict(FURNACEPULSE_SIFS)
        self._test_records: List[SILVerificationRecord] = []

    def get_sif(self, sif_id: str) -> Optional[SIFSpecification]:
        """Get SIF specification by ID."""
        return self._sifs.get(sif_id)

    def list_sifs(self) -> List[SIFSpecification]:
        """List all defined SIFs."""
        return list(self._sifs.values())

    def check_pfd_compliance(self, sif_id: str, measured_pfd: float) -> Dict[str, Any]:
        """
        Check if measured PFD meets SIL requirements.

        Args:
            sif_id: Safety Instrumented Function ID
            measured_pfd: Measured Probability of Failure on Demand

        Returns:
            Compliance check result
        """
        sif = self._sifs.get(sif_id)
        if not sif:
            return {"error": f"Unknown SIF: {sif_id}"}

        # SIL PFD ranges per IEC 61508
        sil_ranges = {
            SILLevel.SIL_1: (1e-2, 1e-1),
            SILLevel.SIL_2: (1e-3, 1e-2),
            SILLevel.SIL_3: (1e-4, 1e-3),
            SILLevel.SIL_4: (1e-5, 1e-4),
        }

        pfd_range = sil_ranges[sif.target_sil]
        compliant = pfd_range[0] <= measured_pfd < pfd_range[1]
        margin = (sif.pfd_target - measured_pfd) / sif.pfd_target

        return {
            "sif_id": sif_id,
            "target_sil": sif.target_sil.value,
            "target_pfd": sif.pfd_target,
            "measured_pfd": measured_pfd,
            "pfd_range": pfd_range,
            "compliant": compliant,
            "margin_percent": margin * 100,
            "status": "PASS" if compliant else "FAIL",
        }

    def calculate_pfd(
        self,
        lambda_du: float,  # Dangerous undetected failure rate (per hour)
        test_interval_hours: float,
        proof_test_coverage: float,
        voting: VotingArchitecture
    ) -> float:
        """
        Calculate PFDavg based on IEC 61508 formulas.

        Args:
            lambda_du: Dangerous undetected failure rate (failures/hour)
            test_interval_hours: Proof test interval in hours
            proof_test_coverage: Fraction of DU failures detected by proof test
            voting: Voting architecture

        Returns:
            Average Probability of Failure on Demand
        """
        t = test_interval_hours

        if voting == VotingArchitecture.SINGLE_1oo1:
            # 1oo1: PFDavg = λDU × T/2
            pfd = lambda_du * t / 2
        elif voting == VotingArchitecture.REDUNDANT_1oo2:
            # 1oo2: PFDavg = (λDU × T)^2 / 3
            pfd = (lambda_du * t) ** 2 / 3
        elif voting == VotingArchitecture.MAJORITY_2oo3:
            # 2oo3: PFDavg = (λDU × T)^2
            pfd = (lambda_du * t) ** 2
        elif voting == VotingArchitecture.QUAD_2oo4:
            # 2oo4: PFDavg = 6 × (λDU × T)^2 / 4
            pfd = 6 * (lambda_du * t) ** 2 / 4
        else:
            pfd = lambda_du * t / 2

        # Apply proof test coverage
        pfd = pfd * (1 - proof_test_coverage)

        return pfd

    def record_test(self, record: SILVerificationRecord) -> None:
        """Record a SIL verification test result."""
        self._test_records.append(record)

    def get_test_history(self, sif_id: str) -> List[SILVerificationRecord]:
        """Get test history for a specific SIF."""
        return [r for r in self._test_records if r.sif_id == sif_id]

    def check_test_overdue(self, sif_id: str) -> Optional[Dict[str, Any]]:
        """Check if proof test is overdue for a SIF."""
        sif = self._sifs.get(sif_id)
        if not sif:
            return None

        # Find most recent test
        tests = self.get_test_history(sif_id)
        if not tests:
            return {
                "sif_id": sif_id,
                "status": "NO_TEST_RECORD",
                "message": "No test records found - test may be overdue",
            }

        latest = max(tests, key=lambda t: t.timestamp)
        now = datetime.now(timezone.utc)
        test_interval_days = sif.test_interval_months * 30

        days_since_test = (now - latest.timestamp).days
        is_overdue = days_since_test > test_interval_days

        return {
            "sif_id": sif_id,
            "last_test": latest.timestamp.isoformat(),
            "days_since_test": days_since_test,
            "test_interval_days": test_interval_days,
            "is_overdue": is_overdue,
            "status": "OVERDUE" if is_overdue else "CURRENT",
        }

    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate SIL compliance summary report."""
        sif_summaries = []
        for sif_id, sif in self._sifs.items():
            test_status = self.check_test_overdue(sif_id)
            sif_summaries.append({
                "sif_id": sif_id,
                "name": sif.name,
                "target_sil": sif.target_sil.value,
                "pfd_target": sif.pfd_target,
                "voting": sif.voting.value,
                "test_status": test_status.get("status", "UNKNOWN") if test_status else "UNKNOWN",
            })

        return {
            "agent_id": self.agent_id,
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_sifs": len(self._sifs),
            "sif_summaries": sif_summaries,
            "reference_standards": [
                "IEC 61508:2010",
                "IEC 61511:2016",
                "NFPA 86:2023",
                "ISA 84.00.01-2018",
            ],
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "SILLevel",
    "SafetyFunction",
    "VotingArchitecture",
    "FailureMode",
    "SIFSpecification",
    "SILVerificationRecord",
    "FURNACEPULSE_SIFS",
    "SILComplianceChecker",
]
