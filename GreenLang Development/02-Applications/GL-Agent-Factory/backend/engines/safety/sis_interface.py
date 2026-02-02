"""
SIS Interface Module - Safety Instrumented System Interface

This module provides the interface for interacting with Safety Instrumented
Systems (SIS) per IEC 61511. It handles configuration, status monitoring,
diagnostics, and proof testing of Safety Instrumented Functions (SIFs).

Standards:
    - IEC 61511: Safety Instrumented Systems for Process Industries
    - IEC 61508: Functional Safety of E/E/PE Systems
    - ISA 84: Application of Safety Instrumented Systems

Key Capabilities:
    - SIF configuration and management
    - Real-time SIS status monitoring
    - Diagnostic coverage calculation
    - Proof test scheduling and tracking
    - Trip cause analysis
    - Bypass management

Example:
    >>> sis = SISInterface(config)
    >>> status = sis.get_sif_status("SIF-001")
    >>> if status.is_healthy:
    ...     print(f"SIF-001 is operational, PFD: {status.current_pfd}")

CRITICAL: This module provides INTERFACE only. Actual SIS control
requires proper authorization and is handled by dedicated safety PLCs.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class SIFStatus(str, Enum):
    """Safety Instrumented Function operational states."""
    OPERATIONAL = "operational"         # Normal operation
    TRIPPED = "tripped"                 # Safety function activated
    BYPASSED = "bypassed"               # Manually bypassed
    MAINTENANCE = "maintenance"         # Under maintenance
    FAILED = "failed"                   # Detected failure
    DEGRADED = "degraded"               # Partial failure (redundant)
    TESTING = "testing"                 # Proof test in progress
    UNKNOWN = "unknown"                 # Status cannot be determined


class DiagnosticLevel(str, Enum):
    """Diagnostic coverage levels per IEC 61508."""
    NONE = "none"           # DC < 60%
    LOW = "low"             # 60% <= DC < 90%
    MEDIUM = "medium"       # 90% <= DC < 99%
    HIGH = "high"           # DC >= 99%


class TripCause(str, Enum):
    """Causes of SIF trips."""
    PROCESS_DEMAND = "process_demand"       # Actual hazard condition
    SPURIOUS = "spurious"                   # False trip
    PROOF_TEST = "proof_test"               # Test-initiated trip
    MAINTENANCE = "maintenance"             # Maintenance action
    COMPONENT_FAILURE = "component_failure"  # Detected failure
    COMMON_CAUSE = "common_cause"           # Common cause failure
    UNKNOWN = "unknown"


class BypassType(str, Enum):
    """Types of SIF bypasses."""
    MAINTENANCE = "maintenance"     # For maintenance activities
    TESTING = "testing"            # For proof testing
    PROCESS = "process"            # Process operational bypass
    STARTUP = "startup"            # Startup bypass
    EMERGENCY = "emergency"        # Emergency override


class SafetyFunction(BaseModel):
    """
    Safety Instrumented Function (SIF) definition.

    A SIF is a specific safety function implemented by the SIS
    to prevent or mitigate a hazardous event.

    Attributes:
        sif_id: Unique identifier (e.g., SIF-001)
        name: Descriptive name
        description: Detailed description
        target_sil: Required SIL level
        achieved_sil: Achieved SIL level
        proof_test_interval: Required test interval (hours)
        last_proof_test: Last proof test date
        next_proof_test: Next scheduled proof test
        pfd_achieved: Achieved PFD from SIL verification
    """
    sif_id: str = Field(..., description="Unique SIF identifier")
    name: str = Field(..., description="SIF name")
    description: Optional[str] = Field(None, description="SIF description")
    target_sil: int = Field(..., ge=1, le=4, description="Target SIL level")
    achieved_sil: Optional[int] = Field(None, ge=0, le=4, description="Achieved SIL")
    proof_test_interval_hours: float = Field(
        8760.0, gt=0, description="Proof test interval (hours)"
    )
    last_proof_test: Optional[datetime] = Field(None, description="Last proof test")
    next_proof_test: Optional[datetime] = Field(None, description="Next proof test")
    pfd_achieved: Optional[float] = Field(None, description="Achieved PFD")
    hazard_description: Optional[str] = Field(None, description="Hazard mitigated")
    trip_setpoint: Optional[float] = Field(None, description="Trip setpoint value")
    trip_setpoint_unit: Optional[str] = Field(None, description="Setpoint unit")
    voting_architecture: Optional[str] = Field(None, description="e.g., 1oo2, 2oo3")
    sensor_count: int = Field(1, ge=1, description="Number of sensors")
    final_element_count: int = Field(1, ge=1, description="Number of final elements")
    response_time_ms: Optional[float] = Field(None, description="Response time (ms)")

    @validator('achieved_sil')
    def validate_sil_achievement(cls, v, values):
        """Warn if achieved SIL is less than target."""
        target = values.get('target_sil')
        if v is not None and target is not None and v < target:
            logger.warning(
                f"SIF {values.get('sif_id')}: Achieved SIL {v} < Target SIL {target}"
            )
        return v


class DiagnosticResult(BaseModel):
    """
    Result of SIF diagnostic check.

    Diagnostics detect dangerous failures to enable timely repair.
    """
    sif_id: str = Field(..., description="SIF identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    diagnostic_coverage: float = Field(..., ge=0, le=1, description="DC (0-1)")
    diagnostic_level: DiagnosticLevel = Field(..., description="DC level")
    failures_detected: int = Field(0, ge=0, description="Detected failures")
    failures_undetected: int = Field(0, ge=0, description="Undetected failures")
    sensor_status: Dict[str, str] = Field(default_factory=dict)
    logic_solver_status: str = Field("OK", description="Logic solver status")
    final_element_status: Dict[str, str] = Field(default_factory=dict)
    overall_status: SIFStatus = Field(..., description="Overall SIF status")
    diagnostic_hash: str = Field("", description="SHA-256 hash")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.diagnostic_hash:
            self.diagnostic_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of diagnostic result."""
        hash_data = {
            "sif_id": self.sif_id,
            "timestamp": self.timestamp.isoformat(),
            "dc": self.diagnostic_coverage,
            "failures_detected": self.failures_detected,
            "status": self.overall_status.value,
        }
        return hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()


class ProofTestResult(BaseModel):
    """
    Result of SIF proof test.

    Proof tests verify that the SIF can perform its safety function.
    """
    sif_id: str = Field(..., description="SIF identifier")
    test_id: str = Field(..., description="Unique test identifier")
    test_date: datetime = Field(..., description="Test date and time")
    tester_id: str = Field(..., description="Tester identifier")
    test_type: str = Field("full", description="full, partial, online")

    # Test results
    test_passed: bool = Field(..., description="Overall pass/fail")
    sensor_tested: bool = Field(True, description="Sensors tested")
    logic_tested: bool = Field(True, description="Logic tested")
    final_element_tested: bool = Field(True, description="Final elements tested")
    response_time_measured_ms: Optional[float] = Field(None, description="Response time")

    # Findings
    failures_found: int = Field(0, ge=0, description="Failures discovered")
    failure_descriptions: List[str] = Field(default_factory=list)
    corrective_actions: List[str] = Field(default_factory=list)

    # Provenance
    test_procedure_id: Optional[str] = Field(None, description="Procedure reference")
    next_test_due: Optional[datetime] = Field(None, description="Next test date")
    provenance_hash: str = Field("", description="SHA-256 hash")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.provenance_hash:
            self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of proof test result."""
        hash_data = {
            "sif_id": self.sif_id,
            "test_id": self.test_id,
            "test_date": self.test_date.isoformat(),
            "test_passed": self.test_passed,
            "failures_found": self.failures_found,
        }
        return hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()


class TripEvent(BaseModel):
    """
    Record of a SIF trip event.

    All trips should be analyzed for cause and documented.
    """
    event_id: str = Field(..., description="Unique event identifier")
    sif_id: str = Field(..., description="SIF that tripped")
    timestamp: datetime = Field(..., description="Trip timestamp")
    trip_cause: TripCause = Field(..., description="Cause of trip")
    process_variable_value: Optional[float] = Field(None, description="PV at trip")
    setpoint_value: Optional[float] = Field(None, description="Trip setpoint")
    unit: Optional[str] = Field(None, description="PV/SP unit")
    duration_seconds: Optional[float] = Field(None, description="Trip duration")
    was_spurious: bool = Field(False, description="Identified as spurious")
    root_cause: Optional[str] = Field(None, description="Root cause analysis")
    corrective_action: Optional[str] = Field(None, description="Actions taken")


class BypassRecord(BaseModel):
    """
    Record of SIF bypass.

    All bypasses must be documented and time-limited.
    """
    bypass_id: str = Field(..., description="Unique bypass identifier")
    sif_id: str = Field(..., description="SIF being bypassed")
    bypass_type: BypassType = Field(..., description="Type of bypass")
    initiated_at: datetime = Field(..., description="Bypass start time")
    initiated_by: str = Field(..., description="Person initiating bypass")
    authorized_by: str = Field(..., description="Authorization")
    reason: str = Field(..., description="Reason for bypass")
    max_duration_hours: float = Field(8.0, gt=0, description="Max bypass duration")
    expires_at: datetime = Field(..., description="Bypass expiration")
    compensating_measures: List[str] = Field(
        default_factory=list, description="Compensating safety measures"
    )
    removed_at: Optional[datetime] = Field(None, description="Bypass removal time")
    removed_by: Optional[str] = Field(None, description="Person removing bypass")
    is_active: bool = Field(True, description="Bypass currently active")


class SISStatus(BaseModel):
    """
    Overall SIS status summary.

    Provides high-level view of SIS health.
    """
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    total_sifs: int = Field(..., description="Total SIFs configured")
    operational_sifs: int = Field(..., description="SIFs in operation")
    tripped_sifs: int = Field(..., description="SIFs currently tripped")
    bypassed_sifs: int = Field(..., description="SIFs currently bypassed")
    failed_sifs: int = Field(..., description="SIFs in failed state")
    degraded_sifs: int = Field(..., description="SIFs in degraded state")
    overdue_proof_tests: int = Field(..., description="Overdue proof tests")
    active_bypasses: int = Field(..., description="Active bypass count")
    overall_health: str = Field(..., description="HEALTHY, DEGRADED, CRITICAL")
    health_score: float = Field(..., ge=0, le=100, description="Health score (0-100)")


class SISConfig(BaseModel):
    """
    SIS Interface configuration.

    Configures connection and operational parameters.
    """
    sis_id: str = Field(..., description="SIS identifier")
    sis_name: str = Field(..., description="SIS name")
    site_id: str = Field(..., description="Site identifier")
    manufacturer: Optional[str] = Field(None, description="SIS manufacturer")
    model: Optional[str] = Field(None, description="SIS model")
    firmware_version: Optional[str] = Field(None, description="Firmware version")

    # Connection (for monitoring only)
    connection_type: str = Field("modbus_tcp", description="Connection type")
    host: Optional[str] = Field(None, description="SIS host address")
    port: Optional[int] = Field(None, description="Connection port")

    # Operational parameters
    diagnostic_interval_seconds: float = Field(60.0, description="Diagnostic interval")
    proof_test_warning_days: int = Field(30, description="Warning before due date")
    max_bypass_duration_hours: float = Field(8.0, description="Default max bypass")
    require_bypass_authorization: bool = Field(True, description="Require auth for bypass")


class SISInterface:
    """
    Safety Instrumented System Interface.

    This class provides the interface for monitoring and managing
    Safety Instrumented Systems per IEC 61511. It does NOT directly
    control safety functions - that remains with the dedicated
    safety PLC/DCS.

    Key Methods:
        get_sif_status: Get status of a specific SIF
        get_sis_status: Get overall SIS status
        run_diagnostics: Perform diagnostic check
        record_proof_test: Record proof test results
        initiate_bypass: Initiate SIF bypass (monitoring only)
        remove_bypass: Remove SIF bypass (monitoring only)

    Example:
        >>> config = SISConfig(sis_id="SIS-001", sis_name="Process Heat SIS", site_id="SITE-01")
        >>> sis = SISInterface(config)
        >>> status = sis.get_sif_status("SIF-HH-001")
        >>> if not status.is_healthy:
        ...     logger.warning(f"SIF-HH-001 status: {status.status}")

    CRITICAL: This interface is for MONITORING and DOCUMENTATION only.
    Actual safety control is performed by dedicated safety PLCs.
    """

    VERSION = "1.0.0"

    def __init__(self, config: SISConfig):
        """
        Initialize SIS Interface.

        Args:
            config: SIS configuration
        """
        self.config = config
        self._sifs: Dict[str, SafetyFunction] = {}
        self._bypasses: Dict[str, BypassRecord] = {}
        self._trip_history: List[TripEvent] = []
        self._proof_test_history: Dict[str, List[ProofTestResult]] = {}
        self._diagnostic_history: Dict[str, List[DiagnosticResult]] = {}

        logger.info(f"SIS Interface initialized: {config.sis_id}")

    def register_sif(self, sif: SafetyFunction) -> None:
        """
        Register a Safety Instrumented Function.

        Args:
            sif: SafetyFunction to register
        """
        self._sifs[sif.sif_id] = sif
        self._proof_test_history[sif.sif_id] = []
        self._diagnostic_history[sif.sif_id] = []
        logger.info(f"Registered SIF: {sif.sif_id} - {sif.name}")

    def get_sif(self, sif_id: str) -> Optional[SafetyFunction]:
        """
        Get a registered SIF by ID.

        Args:
            sif_id: SIF identifier

        Returns:
            SafetyFunction or None if not found
        """
        return self._sifs.get(sif_id)

    def get_all_sifs(self) -> List[SafetyFunction]:
        """Get all registered SIFs."""
        return list(self._sifs.values())

    def get_sif_status(self, sif_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a SIF.

        Args:
            sif_id: SIF identifier

        Returns:
            Status dictionary or None if SIF not found
        """
        sif = self._sifs.get(sif_id)
        if not sif:
            logger.warning(f"SIF not found: {sif_id}")
            return None

        # Check for active bypass
        active_bypass = None
        for bypass in self._bypasses.values():
            if bypass.sif_id == sif_id and bypass.is_active:
                active_bypass = bypass
                break

        # Determine operational status
        if active_bypass:
            status = SIFStatus.BYPASSED
        else:
            # In real implementation, this would read from SIS
            status = SIFStatus.OPERATIONAL

        # Check proof test status
        last_test = self._get_last_proof_test(sif_id)
        proof_test_overdue = False
        hours_until_due = None

        if last_test:
            next_due = last_test.test_date + timedelta(
                hours=sif.proof_test_interval_hours
            )
            hours_until_due = (next_due - datetime.utcnow()).total_seconds() / 3600
            proof_test_overdue = hours_until_due < 0

        # Get latest diagnostics
        latest_diag = self._get_latest_diagnostic(sif_id)

        # Calculate health indicators
        is_healthy = (
            status == SIFStatus.OPERATIONAL
            and not proof_test_overdue
            and (latest_diag is None or latest_diag.overall_status == SIFStatus.OPERATIONAL)
        )

        return {
            "sif_id": sif_id,
            "name": sif.name,
            "status": status.value,
            "target_sil": sif.target_sil,
            "achieved_sil": sif.achieved_sil,
            "pfd_achieved": sif.pfd_achieved,
            "is_healthy": is_healthy,
            "is_bypassed": active_bypass is not None,
            "bypass_info": active_bypass.dict() if active_bypass else None,
            "proof_test_overdue": proof_test_overdue,
            "hours_until_proof_test": hours_until_due,
            "last_proof_test": last_test.test_date.isoformat() if last_test else None,
            "latest_diagnostic": latest_diag.dict() if latest_diag else None,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def get_sis_status(self) -> SISStatus:
        """
        Get overall SIS status summary.

        Returns:
            SISStatus with aggregate health information
        """
        total = len(self._sifs)
        operational = 0
        tripped = 0
        bypassed = 0
        failed = 0
        degraded = 0
        overdue_tests = 0

        for sif_id in self._sifs:
            status = self.get_sif_status(sif_id)
            if status:
                sif_status = SIFStatus(status["status"])
                if sif_status == SIFStatus.OPERATIONAL:
                    operational += 1
                elif sif_status == SIFStatus.TRIPPED:
                    tripped += 1
                elif sif_status == SIFStatus.BYPASSED:
                    bypassed += 1
                elif sif_status == SIFStatus.FAILED:
                    failed += 1
                elif sif_status == SIFStatus.DEGRADED:
                    degraded += 1

                if status.get("proof_test_overdue"):
                    overdue_tests += 1

        # Count active bypasses
        active_bypasses = sum(1 for b in self._bypasses.values() if b.is_active)

        # Calculate health score
        if total == 0:
            health_score = 100.0
        else:
            # Deduct for non-operational SIFs
            health_score = 100.0
            health_score -= (failed / total) * 50  # Failed SIFs are critical
            health_score -= (bypassed / total) * 20  # Bypassed SIFs reduce score
            health_score -= (degraded / total) * 15  # Degraded SIFs
            health_score -= (overdue_tests / total) * 15  # Overdue tests
            health_score = max(0.0, health_score)

        # Determine overall health
        if failed > 0 or health_score < 50:
            overall_health = "CRITICAL"
        elif bypassed > 0 or degraded > 0 or overdue_tests > 0 or health_score < 80:
            overall_health = "DEGRADED"
        else:
            overall_health = "HEALTHY"

        return SISStatus(
            total_sifs=total,
            operational_sifs=operational,
            tripped_sifs=tripped,
            bypassed_sifs=bypassed,
            failed_sifs=failed,
            degraded_sifs=degraded,
            overdue_proof_tests=overdue_tests,
            active_bypasses=active_bypasses,
            overall_health=overall_health,
            health_score=health_score,
        )

    def run_diagnostics(self, sif_id: str) -> Optional[DiagnosticResult]:
        """
        Run diagnostic check on a SIF.

        In a real implementation, this would query the SIS.
        This version creates a simulated diagnostic result.

        Args:
            sif_id: SIF identifier

        Returns:
            DiagnosticResult or None if SIF not found
        """
        sif = self._sifs.get(sif_id)
        if not sif:
            logger.warning(f"Cannot run diagnostics - SIF not found: {sif_id}")
            return None

        # In real implementation, query actual SIS for diagnostic data
        # This simulates a healthy diagnostic result
        dc = 0.9  # 90% diagnostic coverage (medium)
        dc_level = self._dc_to_level(dc)

        result = DiagnosticResult(
            sif_id=sif_id,
            diagnostic_coverage=dc,
            diagnostic_level=dc_level,
            failures_detected=0,
            failures_undetected=0,
            sensor_status={f"sensor_{i}": "OK" for i in range(sif.sensor_count)},
            logic_solver_status="OK",
            final_element_status={f"fe_{i}": "OK" for i in range(sif.final_element_count)},
            overall_status=SIFStatus.OPERATIONAL,
        )

        # Store in history
        self._diagnostic_history.setdefault(sif_id, []).append(result)

        logger.info(f"Diagnostics complete for {sif_id}: DC={dc:.0%}, Status={result.overall_status.value}")
        return result

    def record_proof_test(
        self,
        sif_id: str,
        test_id: str,
        tester_id: str,
        test_passed: bool,
        test_type: str = "full",
        failures_found: int = 0,
        failure_descriptions: Optional[List[str]] = None,
        response_time_ms: Optional[float] = None,
        test_procedure_id: Optional[str] = None,
    ) -> Optional[ProofTestResult]:
        """
        Record proof test results for a SIF.

        Args:
            sif_id: SIF identifier
            test_id: Unique test identifier
            tester_id: Person performing test
            test_passed: Overall pass/fail
            test_type: full, partial, or online
            failures_found: Number of failures discovered
            failure_descriptions: Description of failures
            response_time_ms: Measured response time
            test_procedure_id: Procedure reference

        Returns:
            ProofTestResult or None if SIF not found
        """
        sif = self._sifs.get(sif_id)
        if not sif:
            logger.warning(f"Cannot record proof test - SIF not found: {sif_id}")
            return None

        test_date = datetime.utcnow()
        next_due = test_date + timedelta(hours=sif.proof_test_interval_hours)

        result = ProofTestResult(
            sif_id=sif_id,
            test_id=test_id,
            test_date=test_date,
            tester_id=tester_id,
            test_type=test_type,
            test_passed=test_passed,
            failures_found=failures_found,
            failure_descriptions=failure_descriptions or [],
            response_time_measured_ms=response_time_ms,
            test_procedure_id=test_procedure_id,
            next_test_due=next_due,
        )

        # Update SIF record
        sif.last_proof_test = test_date
        sif.next_proof_test = next_due

        # Store in history
        self._proof_test_history.setdefault(sif_id, []).append(result)

        logger.info(
            f"Proof test recorded for {sif_id}: "
            f"{'PASSED' if test_passed else 'FAILED'}, "
            f"failures={failures_found}"
        )
        return result

    def initiate_bypass(
        self,
        sif_id: str,
        bypass_type: BypassType,
        initiated_by: str,
        authorized_by: str,
        reason: str,
        duration_hours: Optional[float] = None,
        compensating_measures: Optional[List[str]] = None,
    ) -> Optional[BypassRecord]:
        """
        Record initiation of SIF bypass.

        NOTE: This does NOT actually bypass the SIF in the SIS.
        That must be done through proper authorization at the SIS.
        This records the bypass for documentation.

        Args:
            sif_id: SIF identifier
            bypass_type: Type of bypass
            initiated_by: Person initiating
            authorized_by: Person authorizing
            reason: Reason for bypass
            duration_hours: Bypass duration (uses config default if None)
            compensating_measures: Safety measures during bypass

        Returns:
            BypassRecord or None if SIF not found
        """
        sif = self._sifs.get(sif_id)
        if not sif:
            logger.warning(f"Cannot initiate bypass - SIF not found: {sif_id}")
            return None

        duration = duration_hours or self.config.max_bypass_duration_hours
        initiated_at = datetime.utcnow()
        expires_at = initiated_at + timedelta(hours=duration)

        bypass_id = f"BYP-{sif_id}-{int(time.time())}"

        record = BypassRecord(
            bypass_id=bypass_id,
            sif_id=sif_id,
            bypass_type=bypass_type,
            initiated_at=initiated_at,
            initiated_by=initiated_by,
            authorized_by=authorized_by,
            reason=reason,
            max_duration_hours=duration,
            expires_at=expires_at,
            compensating_measures=compensating_measures or [],
            is_active=True,
        )

        self._bypasses[bypass_id] = record

        logger.warning(
            f"BYPASS INITIATED: {sif_id} by {initiated_by}, "
            f"authorized by {authorized_by}, "
            f"expires {expires_at.isoformat()}"
        )
        return record

    def remove_bypass(
        self,
        bypass_id: str,
        removed_by: str,
    ) -> Optional[BypassRecord]:
        """
        Record removal of SIF bypass.

        Args:
            bypass_id: Bypass identifier
            removed_by: Person removing bypass

        Returns:
            Updated BypassRecord or None if not found
        """
        record = self._bypasses.get(bypass_id)
        if not record:
            logger.warning(f"Bypass not found: {bypass_id}")
            return None

        record.removed_at = datetime.utcnow()
        record.removed_by = removed_by
        record.is_active = False

        logger.info(f"BYPASS REMOVED: {bypass_id} by {removed_by}")
        return record

    def record_trip_event(
        self,
        sif_id: str,
        trip_cause: TripCause,
        process_variable_value: Optional[float] = None,
        setpoint_value: Optional[float] = None,
        unit: Optional[str] = None,
        was_spurious: bool = False,
        root_cause: Optional[str] = None,
    ) -> Optional[TripEvent]:
        """
        Record a SIF trip event.

        Args:
            sif_id: SIF that tripped
            trip_cause: Cause of trip
            process_variable_value: PV at time of trip
            setpoint_value: Trip setpoint
            unit: Unit for PV/SP
            was_spurious: Whether trip was spurious
            root_cause: Root cause analysis

        Returns:
            TripEvent or None if SIF not found
        """
        sif = self._sifs.get(sif_id)
        if not sif:
            logger.warning(f"Cannot record trip - SIF not found: {sif_id}")
            return None

        event_id = f"TRIP-{sif_id}-{int(time.time())}"

        event = TripEvent(
            event_id=event_id,
            sif_id=sif_id,
            timestamp=datetime.utcnow(),
            trip_cause=trip_cause,
            process_variable_value=process_variable_value,
            setpoint_value=setpoint_value or sif.trip_setpoint,
            unit=unit or sif.trip_setpoint_unit,
            was_spurious=was_spurious,
            root_cause=root_cause,
        )

        self._trip_history.append(event)

        log_level = logging.WARNING if was_spurious else logging.INFO
        logger.log(
            log_level,
            f"TRIP EVENT: {sif_id}, cause={trip_cause.value}, "
            f"spurious={was_spurious}"
        )
        return event

    def get_expired_bypasses(self) -> List[BypassRecord]:
        """Get list of bypasses that have exceeded their duration."""
        now = datetime.utcnow()
        expired = []
        for bypass in self._bypasses.values():
            if bypass.is_active and bypass.expires_at < now:
                expired.append(bypass)
        return expired

    def get_overdue_proof_tests(self) -> List[Tuple[str, float]]:
        """
        Get SIFs with overdue proof tests.

        Returns:
            List of tuples (sif_id, hours_overdue)
        """
        overdue = []
        now = datetime.utcnow()

        for sif_id, sif in self._sifs.items():
            last_test = self._get_last_proof_test(sif_id)
            if last_test:
                next_due = last_test.test_date + timedelta(
                    hours=sif.proof_test_interval_hours
                )
                if next_due < now:
                    hours_overdue = (now - next_due).total_seconds() / 3600
                    overdue.append((sif_id, hours_overdue))
            else:
                # No proof test on record - consider overdue
                overdue.append((sif_id, float('inf')))

        return sorted(overdue, key=lambda x: x[1], reverse=True)

    def get_trip_history(
        self,
        sif_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[TripEvent]:
        """
        Get trip event history.

        Args:
            sif_id: Filter by SIF (optional)
            since: Filter by date (optional)

        Returns:
            List of TripEvent records
        """
        events = self._trip_history

        if sif_id:
            events = [e for e in events if e.sif_id == sif_id]

        if since:
            events = [e for e in events if e.timestamp >= since]

        return sorted(events, key=lambda e: e.timestamp, reverse=True)

    def _get_last_proof_test(self, sif_id: str) -> Optional[ProofTestResult]:
        """Get most recent proof test for a SIF."""
        history = self._proof_test_history.get(sif_id, [])
        if not history:
            return None
        return max(history, key=lambda t: t.test_date)

    def _get_latest_diagnostic(self, sif_id: str) -> Optional[DiagnosticResult]:
        """Get most recent diagnostic for a SIF."""
        history = self._diagnostic_history.get(sif_id, [])
        if not history:
            return None
        return max(history, key=lambda d: d.timestamp)

    def _dc_to_level(self, dc: float) -> DiagnosticLevel:
        """Convert diagnostic coverage to level."""
        if dc >= 0.99:
            return DiagnosticLevel.HIGH
        elif dc >= 0.90:
            return DiagnosticLevel.MEDIUM
        elif dc >= 0.60:
            return DiagnosticLevel.LOW
        else:
            return DiagnosticLevel.NONE
