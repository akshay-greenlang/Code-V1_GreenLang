"""
ProofTestScheduler - Proof Test Interval Management

This module implements proof test scheduling and management per IEC 61511
for Safety Instrumented Systems (SIS). It handles:
- Proof test interval calculation and optimization
- Test scheduling and overdue tracking
- Test record management and audit trail
- Partial stroke testing integration

Reference: IEC 61511-1 Clause 16.3, IEC 61508-6 Annex B

Example:
    >>> from greenlang.safety.sil.proof_test_scheduler import ProofTestScheduler
    >>> scheduler = ProofTestScheduler()
    >>> schedule = scheduler.create_schedule(
    ...     equipment_id="PSV-001",
    ...     proof_test_interval_hours=8760,
    ...     target_sil=2
    ... )
    >>> print(f"Next test due: {schedule.next_test_due}")
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import logging
import uuid

logger = logging.getLogger(__name__)


class TestType(str, Enum):
    """Types of proof tests per IEC 61511."""

    FULL_PROOF_TEST = "full_proof_test"  # Complete functional test
    PARTIAL_STROKE_TEST = "partial_stroke_test"  # PST for valves
    DIAGNOSTIC_TEST = "diagnostic_test"  # Online diagnostics
    VISUAL_INSPECTION = "visual_inspection"  # Visual inspection
    CALIBRATION_CHECK = "calibration_check"  # Calibration verification


class TestResult(str, Enum):
    """Proof test result status."""

    PASS = "pass"
    FAIL = "fail"
    DEGRADED = "degraded"  # Partial functionality
    INCONCLUSIVE = "inconclusive"
    NOT_PERFORMED = "not_performed"


class ProofTestRecord(BaseModel):
    """Record of a proof test execution."""

    record_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique record identifier"
    )
    equipment_id: str = Field(
        ...,
        description="Equipment identifier"
    )
    test_type: TestType = Field(
        ...,
        description="Type of test performed"
    )
    scheduled_date: datetime = Field(
        ...,
        description="Scheduled test date"
    )
    actual_date: Optional[datetime] = Field(
        None,
        description="Actual test execution date"
    )
    result: TestResult = Field(
        default=TestResult.NOT_PERFORMED,
        description="Test result"
    )
    detected_failures: List[str] = Field(
        default_factory=list,
        description="List of detected failures"
    )
    corrective_actions: List[str] = Field(
        default_factory=list,
        description="Corrective actions taken"
    )
    test_coverage: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Test coverage factor (0-1)"
    )
    performed_by: Optional[str] = Field(
        None,
        description="Person who performed the test"
    )
    witnessed_by: Optional[str] = Field(
        None,
        description="Witness (if required)"
    )
    duration_minutes: Optional[int] = Field(
        None,
        description="Test duration in minutes"
    )
    notes: str = Field(
        default="",
        description="Additional notes"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProofTestSchedule(BaseModel):
    """Proof test schedule for a SIS component."""

    schedule_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique schedule identifier"
    )
    equipment_id: str = Field(
        ...,
        description="Equipment identifier"
    )
    equipment_description: str = Field(
        default="",
        description="Equipment description"
    )
    target_sil: int = Field(
        ...,
        ge=1,
        le=4,
        description="Target SIL level"
    )
    proof_test_interval_hours: float = Field(
        ...,
        gt=0,
        description="Proof test interval (hours)"
    )
    partial_stroke_test_interval_hours: Optional[float] = Field(
        None,
        gt=0,
        description="Partial stroke test interval if applicable"
    )
    diagnostic_test_interval_hours: float = Field(
        default=1.0,
        gt=0,
        description="Diagnostic test interval"
    )
    last_test_date: Optional[datetime] = Field(
        None,
        description="Date of last proof test"
    )
    next_test_due: datetime = Field(
        ...,
        description="Next test due date"
    )
    grace_period_days: int = Field(
        default=30,
        description="Grace period before overdue"
    )
    is_overdue: bool = Field(
        default=False,
        description="Is test overdue?"
    )
    days_until_due: int = Field(
        default=0,
        description="Days until test is due"
    )
    test_history: List[ProofTestRecord] = Field(
        default_factory=list,
        description="Test history records"
    )
    pfd_avg_current: Optional[float] = Field(
        None,
        description="Current PFDavg based on test interval"
    )
    created_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Schedule creation date"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow,
        description="Last update timestamp"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ProofTestScheduler:
    """
    Proof Test Scheduler for SIS Components.

    Manages proof test scheduling, tracking, and optimization
    per IEC 61511 requirements. Features include:
    - Test interval calculation based on target SIL
    - Overdue tracking and notifications
    - Test record management
    - PFD impact analysis
    - Partial stroke test integration

    Attributes:
        schedules: Dict of equipment_id to ProofTestSchedule
        test_records: List of all test records

    Example:
        >>> scheduler = ProofTestScheduler()
        >>> schedule = scheduler.create_schedule(
        ...     equipment_id="XV-001",
        ...     proof_test_interval_hours=8760,
        ...     target_sil=2
        ... )
    """

    # Recommended maximum proof test intervals per SIL
    MAX_INTERVALS_BY_SIL: Dict[int, float] = {
        1: 87600,  # 10 years
        2: 43800,  # 5 years
        3: 17520,  # 2 years
        4: 8760,   # 1 year
    }

    def __init__(self):
        """Initialize ProofTestScheduler."""
        self.schedules: Dict[str, ProofTestSchedule] = {}
        self.test_records: List[ProofTestRecord] = []
        logger.info("ProofTestScheduler initialized")

    def create_schedule(
        self,
        equipment_id: str,
        proof_test_interval_hours: float,
        target_sil: int,
        equipment_description: str = "",
        last_test_date: Optional[datetime] = None,
        partial_stroke_test_interval_hours: Optional[float] = None,
        grace_period_days: int = 30
    ) -> ProofTestSchedule:
        """
        Create a proof test schedule for equipment.

        Args:
            equipment_id: Unique equipment identifier
            proof_test_interval_hours: Test interval in hours
            target_sil: Target SIL level (1-4)
            equipment_description: Description of equipment
            last_test_date: Date of last test (defaults to now)
            partial_stroke_test_interval_hours: PST interval if applicable
            grace_period_days: Grace period before overdue

        Returns:
            ProofTestSchedule object

        Raises:
            ValueError: If parameters are invalid
        """
        logger.info(f"Creating proof test schedule for {equipment_id}")

        # Validate SIL-specific constraints
        max_interval = self.MAX_INTERVALS_BY_SIL.get(target_sil, 87600)
        if proof_test_interval_hours > max_interval:
            logger.warning(
                f"Proof test interval {proof_test_interval_hours}h exceeds "
                f"recommended maximum {max_interval}h for SIL {target_sil}"
            )

        # Set default last test date
        if last_test_date is None:
            last_test_date = datetime.utcnow()

        # Calculate next test due
        interval_delta = timedelta(hours=proof_test_interval_hours)
        next_test_due = last_test_date + interval_delta

        # Calculate days until due
        days_until_due = (next_test_due - datetime.utcnow()).days
        is_overdue = days_until_due < -grace_period_days

        # Create schedule
        schedule = ProofTestSchedule(
            equipment_id=equipment_id,
            equipment_description=equipment_description,
            target_sil=target_sil,
            proof_test_interval_hours=proof_test_interval_hours,
            partial_stroke_test_interval_hours=partial_stroke_test_interval_hours,
            last_test_date=last_test_date,
            next_test_due=next_test_due,
            grace_period_days=grace_period_days,
            is_overdue=is_overdue,
            days_until_due=days_until_due,
        )

        # Calculate provenance hash
        schedule.provenance_hash = self._calculate_provenance(
            equipment_id, proof_test_interval_hours, target_sil
        )

        # Store schedule
        self.schedules[equipment_id] = schedule

        logger.info(
            f"Schedule created for {equipment_id}. "
            f"Next test due: {next_test_due.isoformat()}"
        )

        return schedule

    def record_test(
        self,
        equipment_id: str,
        test_type: TestType,
        result: TestResult,
        actual_date: Optional[datetime] = None,
        detected_failures: Optional[List[str]] = None,
        corrective_actions: Optional[List[str]] = None,
        test_coverage: float = 1.0,
        performed_by: Optional[str] = None,
        witnessed_by: Optional[str] = None,
        duration_minutes: Optional[int] = None,
        notes: str = ""
    ) -> ProofTestRecord:
        """
        Record a proof test execution.

        Args:
            equipment_id: Equipment identifier
            test_type: Type of test performed
            result: Test result
            actual_date: Actual test date (defaults to now)
            detected_failures: List of detected failures
            corrective_actions: Corrective actions taken
            test_coverage: Test coverage factor (0-1)
            performed_by: Person who performed test
            witnessed_by: Witness name
            duration_minutes: Test duration
            notes: Additional notes

        Returns:
            ProofTestRecord object

        Raises:
            ValueError: If equipment_id not found in schedules
        """
        logger.info(f"Recording {test_type.value} for {equipment_id}")

        if equipment_id not in self.schedules:
            raise ValueError(f"No schedule found for equipment: {equipment_id}")

        schedule = self.schedules[equipment_id]

        if actual_date is None:
            actual_date = datetime.utcnow()

        # Create test record
        record = ProofTestRecord(
            equipment_id=equipment_id,
            test_type=test_type,
            scheduled_date=schedule.next_test_due,
            actual_date=actual_date,
            result=result,
            detected_failures=detected_failures or [],
            corrective_actions=corrective_actions or [],
            test_coverage=test_coverage,
            performed_by=performed_by,
            witnessed_by=witnessed_by,
            duration_minutes=duration_minutes,
            notes=notes,
        )

        # Calculate provenance hash
        record.provenance_hash = self._calculate_record_provenance(record)

        # Add to records
        self.test_records.append(record)
        schedule.test_history.append(record)

        # Update schedule if full proof test
        if test_type == TestType.FULL_PROOF_TEST:
            self._update_schedule_after_test(schedule, actual_date, test_coverage)

        logger.info(
            f"Test recorded for {equipment_id}. Result: {result.value}"
        )

        return record

    def _update_schedule_after_test(
        self,
        schedule: ProofTestSchedule,
        test_date: datetime,
        test_coverage: float
    ) -> None:
        """Update schedule after proof test completion."""
        schedule.last_test_date = test_date

        # Calculate next test due
        # If test coverage < 100%, may need more frequent testing
        effective_interval = schedule.proof_test_interval_hours
        if test_coverage < 1.0:
            # Reduce interval proportionally
            effective_interval *= test_coverage
            logger.warning(
                f"Test coverage {test_coverage:.0%} reduces effective interval "
                f"to {effective_interval:.0f} hours"
            )

        interval_delta = timedelta(hours=effective_interval)
        schedule.next_test_due = test_date + interval_delta

        # Update status
        schedule.days_until_due = (
            schedule.next_test_due - datetime.utcnow()
        ).days
        schedule.is_overdue = False
        schedule.last_updated = datetime.utcnow()

    def get_overdue_equipment(self) -> List[ProofTestSchedule]:
        """
        Get list of equipment with overdue proof tests.

        Returns:
            List of overdue ProofTestSchedule objects
        """
        self._update_all_schedules()
        overdue = [s for s in self.schedules.values() if s.is_overdue]

        if overdue:
            logger.warning(
                f"{len(overdue)} equipment items have overdue proof tests"
            )

        return overdue

    def get_upcoming_tests(
        self,
        days_ahead: int = 30
    ) -> List[ProofTestSchedule]:
        """
        Get list of equipment with tests due in specified period.

        Args:
            days_ahead: Number of days to look ahead

        Returns:
            List of ProofTestSchedule objects due within period
        """
        self._update_all_schedules()
        cutoff = datetime.utcnow() + timedelta(days=days_ahead)

        upcoming = [
            s for s in self.schedules.values()
            if s.next_test_due <= cutoff and not s.is_overdue
        ]

        return sorted(upcoming, key=lambda x: x.next_test_due)

    def _update_all_schedules(self) -> None:
        """Update status of all schedules."""
        now = datetime.utcnow()

        for schedule in self.schedules.values():
            schedule.days_until_due = (schedule.next_test_due - now).days
            schedule.is_overdue = (
                schedule.days_until_due < -schedule.grace_period_days
            )
            schedule.last_updated = now

    def calculate_pfd_impact(
        self,
        equipment_id: str,
        new_interval_hours: float
    ) -> Dict[str, float]:
        """
        Calculate PFD impact of changing proof test interval.

        Args:
            equipment_id: Equipment identifier
            new_interval_hours: Proposed new interval

        Returns:
            Dict with current and proposed PFD values

        Raises:
            ValueError: If equipment not found
        """
        if equipment_id not in self.schedules:
            raise ValueError(f"No schedule found for equipment: {equipment_id}")

        schedule = self.schedules[equipment_id]
        current_interval = schedule.proof_test_interval_hours

        # Simplified PFD calculation (1oo1 approximation)
        # PFDavg = lambda_DU * TI / 2
        # For comparison, assume constant lambda_DU
        # Ratio shows relative change

        pfd_ratio = new_interval_hours / current_interval

        return {
            "current_interval_hours": current_interval,
            "proposed_interval_hours": new_interval_hours,
            "pfd_change_factor": pfd_ratio,
            "pfd_increase_percent": (pfd_ratio - 1) * 100 if pfd_ratio > 1 else 0,
            "pfd_decrease_percent": (1 - pfd_ratio) * 100 if pfd_ratio < 1 else 0,
            "recommendation": (
                "Acceptable" if pfd_ratio <= 1.5
                else "Review required" if pfd_ratio <= 2.0
                else "Not recommended"
            )
        }

    def optimize_interval_for_sil(
        self,
        lambda_du: float,
        target_sil: int,
        architecture: str = "1oo1",
        beta_ccf: float = 0.1
    ) -> float:
        """
        Calculate optimal proof test interval to achieve target SIL.

        Args:
            lambda_du: Dangerous undetected failure rate (per hour)
            target_sil: Target SIL level (1-4)
            architecture: Voting architecture
            beta_ccf: Common cause failure factor

        Returns:
            Optimal proof test interval in hours
        """
        # SIL PFD upper bounds
        sil_pfd_limits = {
            1: 1e-1,
            2: 1e-2,
            3: 1e-3,
            4: 1e-4,
        }

        target_pfd = sil_pfd_limits[target_sil]

        # Apply safety margin
        target_pfd *= 0.5

        if architecture == "1oo1":
            # PFDavg = lambda_DU * TI / 2
            # TI = 2 * PFDavg / lambda_DU
            optimal_ti = 2 * target_pfd / lambda_du

        elif architecture == "1oo2":
            # Simplified: PFDavg ~ beta * lambda_DU * TI / 2 (CCF dominated)
            optimal_ti = 2 * target_pfd / (beta_ccf * lambda_du)

        elif architecture == "2oo3":
            # Similar to 1oo2 for CCF-dominated case
            optimal_ti = 2 * target_pfd / (beta_ccf * lambda_du)

        else:
            # Default to 1oo1
            optimal_ti = 2 * target_pfd / lambda_du

        # Enforce maximum intervals
        max_ti = self.MAX_INTERVALS_BY_SIL[target_sil]
        optimal_ti = min(optimal_ti, max_ti)

        # Enforce minimum practical interval (1 month)
        optimal_ti = max(optimal_ti, 720)

        logger.info(
            f"Optimal proof test interval for SIL {target_sil}: "
            f"{optimal_ti:.0f} hours ({optimal_ti/8760:.1f} years)"
        )

        return optimal_ti

    def get_compliance_report(self) -> Dict[str, Any]:
        """
        Generate compliance report for all scheduled equipment.

        Returns:
            Compliance report dictionary
        """
        self._update_all_schedules()

        total = len(self.schedules)
        overdue = sum(1 for s in self.schedules.values() if s.is_overdue)
        on_schedule = total - overdue

        # Equipment by SIL
        by_sil = {1: 0, 2: 0, 3: 0, 4: 0}
        for schedule in self.schedules.values():
            by_sil[schedule.target_sil] += 1

        # Recent test results
        recent_tests = sorted(
            self.test_records,
            key=lambda x: x.actual_date or datetime.min,
            reverse=True
        )[:10]

        recent_failures = [
            r for r in recent_tests
            if r.result == TestResult.FAIL
        ]

        return {
            "report_date": datetime.utcnow().isoformat(),
            "total_equipment": total,
            "on_schedule": on_schedule,
            "overdue": overdue,
            "compliance_rate": on_schedule / total * 100 if total > 0 else 100,
            "equipment_by_sil": by_sil,
            "recent_failures": [
                {
                    "equipment_id": r.equipment_id,
                    "test_date": r.actual_date.isoformat() if r.actual_date else None,
                    "failures": r.detected_failures
                }
                for r in recent_failures
            ],
            "overdue_equipment": [
                {
                    "equipment_id": s.equipment_id,
                    "days_overdue": -s.days_until_due - s.grace_period_days,
                    "target_sil": s.target_sil
                }
                for s in self.schedules.values()
                if s.is_overdue
            ],
            "provenance_hash": hashlib.sha256(
                f"{datetime.utcnow().isoformat()}|{total}|{overdue}".encode()
            ).hexdigest()
        }

    def export_schedule(
        self,
        equipment_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Export schedule data for external systems.

        Args:
            equipment_id: Optional specific equipment (all if None)

        Returns:
            List of schedule dictionaries
        """
        if equipment_id:
            if equipment_id not in self.schedules:
                raise ValueError(f"No schedule found for: {equipment_id}")
            schedules = [self.schedules[equipment_id]]
        else:
            schedules = list(self.schedules.values())

        return [s.model_dump() for s in schedules]

    def _calculate_provenance(
        self,
        equipment_id: str,
        interval: float,
        sil: int
    ) -> str:
        """Calculate SHA-256 provenance hash for schedule."""
        provenance_str = (
            f"{equipment_id}|{interval}|{sil}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _calculate_record_provenance(self, record: ProofTestRecord) -> str:
        """Calculate SHA-256 provenance hash for test record."""
        provenance_str = (
            f"{record.equipment_id}|"
            f"{record.test_type.value}|"
            f"{record.result.value}|"
            f"{record.actual_date.isoformat() if record.actual_date else ''}|"
            f"{datetime.utcnow().isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
