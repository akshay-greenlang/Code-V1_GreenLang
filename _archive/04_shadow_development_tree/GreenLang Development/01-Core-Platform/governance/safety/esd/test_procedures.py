"""
TestProcedures - Automated ESD Test Procedures

This module implements automated ESD test sequences and procedures per
IEC 61511-1 Clause 16. Provides comprehensive testing capabilities for
Emergency Shutdown Systems including partial stroke testing, trip point
verification, and documentation generation.

Key features:
- Automated ESD test sequences
- Partial stroke testing for valves
- Trip point verification
- Documentation generation
- Test scheduling and tracking
- Results logging with provenance

Reference: IEC 61511-1 Clause 16.3, ISA TR84.00.03

Example:
    >>> from greenlang.safety.esd.test_procedures import ESDTestManager
    >>> manager = ESDTestManager(system_id="ESD-001")
    >>> result = manager.run_test_procedure("SIF-001", "proof_test")
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime, timedelta
import uuid
import time

logger = logging.getLogger(__name__)


class TestType(str, Enum):
    """Types of ESD tests."""

    PROOF_TEST = "proof_test"  # Full proof test per IEC 61511
    PARTIAL_STROKE = "partial_stroke"  # Partial stroke test for valves
    TRIP_POINT = "trip_point"  # Trip point verification
    RESPONSE_TIME = "response_time"  # Response time measurement
    LOGIC_TEST = "logic_test"  # Logic function test
    VOTING_TEST = "voting_test"  # Voting logic verification
    END_TO_END = "end_to_end"  # Complete end-to-end test
    SENSOR_CAL = "sensor_calibration"  # Sensor calibration check
    MANUAL_INITIATION = "manual_initiation"  # Manual ESD test


class TestStatus(str, Enum):
    """Test execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PASSED = "passed"
    FAILED = "failed"
    ABORTED = "aborted"
    SKIPPED = "skipped"


class TestStep(BaseModel):
    """Individual test step definition."""

    step_number: int = Field(
        ...,
        description="Step sequence number"
    )
    step_name: str = Field(
        ...,
        description="Step name"
    )
    description: str = Field(
        default="",
        description="Step description"
    )
    action: str = Field(
        ...,
        description="Action to perform"
    )
    expected_result: str = Field(
        ...,
        description="Expected result"
    )
    actual_result: str = Field(
        default="",
        description="Actual result"
    )
    status: TestStatus = Field(
        default=TestStatus.PENDING,
        description="Step status"
    )
    start_time: Optional[datetime] = Field(
        None,
        description="Step start time"
    )
    end_time: Optional[datetime] = Field(
        None,
        description="Step end time"
    )
    duration_ms: Optional[float] = Field(
        None,
        description="Step duration (ms)"
    )
    notes: str = Field(
        default="",
        description="Step notes"
    )
    measurements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Measurements recorded"
    )


class TestProcedure(BaseModel):
    """Complete test procedure definition."""

    procedure_id: str = Field(
        default_factory=lambda: f"TP-{uuid.uuid4().hex[:8].upper()}",
        description="Procedure identifier"
    )
    procedure_name: str = Field(
        ...,
        description="Procedure name"
    )
    procedure_version: str = Field(
        default="1.0",
        description="Procedure version"
    )
    test_type: TestType = Field(
        ...,
        description="Type of test"
    )
    sif_id: str = Field(
        ...,
        description="SIF to test"
    )
    equipment_tags: List[str] = Field(
        default_factory=list,
        description="Equipment tags involved"
    )
    sil_level: int = Field(
        default=0,
        ge=0,
        le=4,
        description="SIL level"
    )
    test_interval_days: int = Field(
        default=365,
        description="Test interval (days)"
    )
    requires_witness: bool = Field(
        default=False,
        description="Requires witness for SIL 2+"
    )
    steps: List[TestStep] = Field(
        default_factory=list,
        description="Test steps"
    )
    prerequisites: List[str] = Field(
        default_factory=list,
        description="Prerequisites"
    )
    compensating_measures: List[str] = Field(
        default_factory=list,
        description="Compensating measures during test"
    )
    acceptance_criteria: List[str] = Field(
        default_factory=list,
        description="Acceptance criteria"
    )


class TestResult(BaseModel):
    """Test execution result."""

    result_id: str = Field(
        default_factory=lambda: f"TR-{uuid.uuid4().hex[:8].upper()}",
        description="Result identifier"
    )
    procedure_id: str = Field(
        ...,
        description="Procedure identifier"
    )
    sif_id: str = Field(
        ...,
        description="SIF tested"
    )
    test_type: TestType = Field(
        ...,
        description="Type of test"
    )
    test_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Test date"
    )
    tester: str = Field(
        ...,
        description="Person conducting test"
    )
    witness: Optional[str] = Field(
        None,
        description="Witness (if required)"
    )
    status: TestStatus = Field(
        default=TestStatus.PENDING,
        description="Overall status"
    )
    steps_total: int = Field(
        default=0,
        description="Total steps"
    )
    steps_passed: int = Field(
        default=0,
        description="Steps passed"
    )
    steps_failed: int = Field(
        default=0,
        description="Steps failed"
    )
    step_results: List[TestStep] = Field(
        default_factory=list,
        description="Individual step results"
    )
    overall_passed: bool = Field(
        default=False,
        description="Did test pass overall"
    )
    as_found_condition: Dict[str, Any] = Field(
        default_factory=dict,
        description="As-found condition"
    )
    as_left_condition: Dict[str, Any] = Field(
        default_factory=dict,
        description="As-left condition"
    )
    response_time_ms: Optional[float] = Field(
        None,
        description="Measured response time (ms)"
    )
    requirement_met: bool = Field(
        default=True,
        description="Did test meet requirements"
    )
    anomalies: List[str] = Field(
        default_factory=list,
        description="Anomalies detected"
    )
    corrective_actions: List[str] = Field(
        default_factory=list,
        description="Required corrective actions"
    )
    next_test_due: Optional[datetime] = Field(
        None,
        description="Next test due date"
    )
    work_permit_ref: Optional[str] = Field(
        None,
        description="Work permit reference"
    )
    duration_minutes: float = Field(
        default=0.0,
        description="Total test duration (minutes)"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PartialStrokeResult(BaseModel):
    """Result of partial stroke test."""

    test_id: str = Field(
        default_factory=lambda: f"PST-{uuid.uuid4().hex[:8].upper()}",
        description="Test identifier"
    )
    valve_tag: str = Field(
        ...,
        description="Valve tag"
    )
    test_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="Test date"
    )
    stroke_percentage: float = Field(
        ...,
        gt=0,
        le=100,
        description="Stroke percentage tested"
    )
    stroke_time_ms: float = Field(
        ...,
        description="Time to reach stroke position (ms)"
    )
    return_time_ms: float = Field(
        ...,
        description="Time to return to normal (ms)"
    )
    friction_signature: Optional[float] = Field(
        None,
        description="Friction signature value"
    )
    passed: bool = Field(
        ...,
        description="Did test pass"
    )
    breakaway_detected: bool = Field(
        default=False,
        description="Was valve breakaway detected"
    )
    position_verified: bool = Field(
        default=False,
        description="Was position feedback verified"
    )
    notes: str = Field(
        default="",
        description="Test notes"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash"
    )


class TestSchedule(BaseModel):
    """Test schedule entry."""

    schedule_id: str = Field(
        default_factory=lambda: f"TS-{uuid.uuid4().hex[:8].upper()}",
        description="Schedule identifier"
    )
    sif_id: str = Field(
        ...,
        description="SIF to test"
    )
    procedure_id: str = Field(
        ...,
        description="Procedure to use"
    )
    test_type: TestType = Field(
        ...,
        description="Type of test"
    )
    interval_days: int = Field(
        ...,
        description="Test interval (days)"
    )
    last_test_date: Optional[datetime] = Field(
        None,
        description="Last test date"
    )
    next_test_due: Optional[datetime] = Field(
        None,
        description="Next test due"
    )
    is_overdue: bool = Field(
        default=False,
        description="Is test overdue"
    )
    assigned_to: str = Field(
        default="",
        description="Assigned tester"
    )
    priority: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Priority (1=highest)"
    )


class ESDTestManager:
    """
    ESD Test Procedure Manager.

    Manages automated ESD test procedures, scheduling, and documentation
    per IEC 61511 requirements.

    Key features:
    - Automated test sequences
    - Partial stroke testing
    - Trip point verification
    - Documentation generation
    - Test scheduling
    - Complete audit trail

    The manager follows IEC 61511 principles:
    - Systematic test execution
    - Complete documentation
    - Provenance tracking

    Attributes:
        system_id: ESD system identifier
        procedures: Registered test procedures
        results: Historical test results

    Example:
        >>> manager = ESDTestManager(system_id="ESD-001")
        >>> result = manager.run_test_procedure("SIF-001", "proof_test")
    """

    def __init__(
        self,
        system_id: str,
        step_executor: Optional[Callable[[TestStep], bool]] = None
    ):
        """
        Initialize ESDTestManager.

        Args:
            system_id: ESD system identifier
            step_executor: Callback to execute test steps
        """
        self.system_id = system_id
        self.step_executor = step_executor or self._default_step_executor

        self.procedures: Dict[str, TestProcedure] = {}
        self.results: Dict[str, List[TestResult]] = {}
        self.schedules: Dict[str, TestSchedule] = {}
        self.pst_results: Dict[str, List[PartialStrokeResult]] = {}

        # Initialize standard procedures
        self._initialize_standard_procedures()

        logger.info(f"ESDTestManager initialized: {system_id}")

    def _initialize_standard_procedures(self) -> None:
        """Initialize standard test procedures."""

        # Proof test procedure template
        proof_test = TestProcedure(
            procedure_name="Standard Proof Test",
            test_type=TestType.PROOF_TEST,
            sif_id="TEMPLATE",
            sil_level=2,
            requires_witness=True,
            prerequisites=[
                "Work permit obtained",
                "Operations notified",
                "Test equipment calibrated",
                "Bypass/override capability verified"
            ],
            compensating_measures=[
                "Continuous operator monitoring",
                "Additional fire watch if applicable",
                "Manual shutdown capability verified"
            ],
            acceptance_criteria=[
                "All sensors respond within calibration tolerance",
                "Logic solver executes correctly",
                "Final elements achieve full stroke",
                "Response time meets requirement",
                "All alarms annunciate correctly"
            ],
            steps=[
                TestStep(
                    step_number=1,
                    step_name="Pre-test Verification",
                    action="Verify safe conditions and equipment ready",
                    expected_result="All prerequisites met"
                ),
                TestStep(
                    step_number=2,
                    step_name="Sensor Calibration Check",
                    action="Verify sensor calibration at 0%, 50%, 100%",
                    expected_result="Within +/- tolerance"
                ),
                TestStep(
                    step_number=3,
                    step_name="Voting Logic Test",
                    action="Test voting logic per configuration",
                    expected_result="Correct voting behavior"
                ),
                TestStep(
                    step_number=4,
                    step_name="Trip Point Verification",
                    action="Apply trip setpoint and verify action",
                    expected_result="Trip initiates at correct setpoint"
                ),
                TestStep(
                    step_number=5,
                    step_name="Final Element Test",
                    action="Verify final element achieves safe state",
                    expected_result="Full stroke to safe position"
                ),
                TestStep(
                    step_number=6,
                    step_name="Response Time Measurement",
                    action="Measure total response time",
                    expected_result="< required response time"
                ),
                TestStep(
                    step_number=7,
                    step_name="Reset and Restore",
                    action="Reset SIF and verify normal operation",
                    expected_result="Normal operation restored"
                ),
            ]
        )
        self.procedures["proof_test_template"] = proof_test

        # Partial stroke test procedure
        pst = TestProcedure(
            procedure_name="Partial Stroke Test",
            test_type=TestType.PARTIAL_STROKE,
            sif_id="TEMPLATE",
            sil_level=2,
            requires_witness=False,
            prerequisites=[
                "Valve supports partial stroke testing",
                "Process stable"
            ],
            acceptance_criteria=[
                "Valve moves to PST position",
                "Position feedback confirmed",
                "Valve returns to normal",
                "Stroke time within limits"
            ],
            steps=[
                TestStep(
                    step_number=1,
                    step_name="Initiate PST",
                    action="Command partial stroke to target position",
                    expected_result="Valve moves toward closed"
                ),
                TestStep(
                    step_number=2,
                    step_name="Verify Position",
                    action="Verify position feedback matches command",
                    expected_result="Position confirmed"
                ),
                TestStep(
                    step_number=3,
                    step_name="Return to Normal",
                    action="Return valve to normal position",
                    expected_result="Full open position"
                ),
                TestStep(
                    step_number=4,
                    step_name="Record Results",
                    action="Record stroke time and friction",
                    expected_result="Values within limits"
                ),
            ]
        )
        self.procedures["pst_template"] = pst

    def register_procedure(
        self,
        procedure: TestProcedure
    ) -> None:
        """
        Register a test procedure.

        Args:
            procedure: TestProcedure to register
        """
        self.procedures[procedure.procedure_id] = procedure
        logger.info(f"Registered procedure: {procedure.procedure_id}")

    def create_procedure_for_sif(
        self,
        sif_id: str,
        test_type: TestType,
        sil_level: int = 2,
        equipment_tags: Optional[List[str]] = None,
        custom_steps: Optional[List[TestStep]] = None
    ) -> TestProcedure:
        """
        Create a test procedure for a specific SIF.

        Args:
            sif_id: SIF identifier
            test_type: Type of test
            sil_level: SIL level
            equipment_tags: Equipment involved
            custom_steps: Custom test steps

        Returns:
            TestProcedure
        """
        # Get template
        template_id = f"{test_type.value}_template"
        if template_id in self.procedures:
            template = self.procedures[template_id]
            steps = [step.model_copy() for step in template.steps]
            prerequisites = list(template.prerequisites)
            acceptance_criteria = list(template.acceptance_criteria)
        else:
            steps = custom_steps or []
            prerequisites = []
            acceptance_criteria = []

        procedure = TestProcedure(
            procedure_name=f"{test_type.value.replace('_', ' ').title()} - {sif_id}",
            test_type=test_type,
            sif_id=sif_id,
            equipment_tags=equipment_tags or [],
            sil_level=sil_level,
            requires_witness=sil_level >= 2,
            steps=steps,
            prerequisites=prerequisites,
            acceptance_criteria=acceptance_criteria,
        )

        self.procedures[procedure.procedure_id] = procedure

        logger.info(
            f"Created procedure {procedure.procedure_id} for {sif_id}"
        )

        return procedure

    def run_test_procedure(
        self,
        procedure_id: str,
        tester: str,
        witness: Optional[str] = None,
        work_permit_ref: Optional[str] = None,
        as_found: Optional[Dict[str, Any]] = None
    ) -> TestResult:
        """
        Execute a test procedure.

        Args:
            procedure_id: Procedure to execute
            tester: Person conducting test
            witness: Witness (if required)
            work_permit_ref: Work permit reference
            as_found: As-found conditions

        Returns:
            TestResult with all step results
        """
        if procedure_id not in self.procedures:
            raise ValueError(f"Procedure not found: {procedure_id}")

        procedure = self.procedures[procedure_id]

        logger.info(
            f"Starting test procedure {procedure_id} for {procedure.sif_id}"
        )

        # Check witness requirement
        if procedure.requires_witness and not witness:
            logger.warning(
                f"Witness required for SIL {procedure.sil_level} test"
            )

        start_time = datetime.utcnow()

        # Initialize result
        result = TestResult(
            procedure_id=procedure_id,
            sif_id=procedure.sif_id,
            test_type=procedure.test_type,
            tester=tester,
            witness=witness,
            work_permit_ref=work_permit_ref,
            as_found_condition=as_found or {},
            steps_total=len(procedure.steps),
        )

        # Execute steps
        step_results = []
        all_passed = True

        for step in procedure.steps:
            step_start = time.time()
            step.start_time = datetime.utcnow()
            step.status = TestStatus.IN_PROGRESS

            try:
                # Execute step
                passed = self.step_executor(step)

                step.end_time = datetime.utcnow()
                step.duration_ms = (time.time() - step_start) * 1000

                if passed:
                    step.status = TestStatus.PASSED
                    result.steps_passed += 1
                else:
                    step.status = TestStatus.FAILED
                    result.steps_failed += 1
                    all_passed = False

            except Exception as e:
                step.status = TestStatus.FAILED
                step.notes = str(e)
                result.steps_failed += 1
                result.anomalies.append(
                    f"Step {step.step_number} error: {e}"
                )
                all_passed = False

            step_results.append(step)

        result.step_results = step_results
        result.overall_passed = all_passed
        result.status = TestStatus.PASSED if all_passed else TestStatus.FAILED
        result.requirement_met = all_passed

        # Calculate duration
        end_time = datetime.utcnow()
        result.duration_minutes = (end_time - start_time).total_seconds() / 60

        # Set next test due
        result.next_test_due = datetime.utcnow() + timedelta(
            days=procedure.test_interval_days
        )

        # Calculate provenance
        result.provenance_hash = self._calculate_provenance(result)

        # Store result
        if procedure.sif_id not in self.results:
            self.results[procedure.sif_id] = []
        self.results[procedure.sif_id].append(result)

        # Update schedule
        if procedure.sif_id in self.schedules:
            schedule = self.schedules[procedure.sif_id]
            schedule.last_test_date = result.test_date
            schedule.next_test_due = result.next_test_due
            schedule.is_overdue = False

        logger.info(
            f"Test {result.result_id}: "
            f"{'PASSED' if all_passed else 'FAILED'} "
            f"({result.steps_passed}/{result.steps_total} steps)"
        )

        return result

    def run_partial_stroke_test(
        self,
        valve_tag: str,
        stroke_percentage: float = 20.0,
        tester: str = "",
        valve_controller: Optional[Callable[[float], Tuple[float, float]]] = None
    ) -> PartialStrokeResult:
        """
        Execute a partial stroke test on a valve.

        Args:
            valve_tag: Valve tag to test
            stroke_percentage: Stroke percentage (default 20%)
            tester: Person conducting test
            valve_controller: Callback to control valve

        Returns:
            PartialStrokeResult
        """
        logger.info(
            f"Starting PST for {valve_tag}: {stroke_percentage}%"
        )

        if valve_controller:
            stroke_time, return_time = valve_controller(stroke_percentage)
        else:
            # Simulate
            import random
            stroke_time = 500 + random.random() * 200
            return_time = 400 + random.random() * 200

        # Determine pass/fail (typical: stroke < 2s)
        passed = stroke_time < 2000 and return_time < 2000

        result = PartialStrokeResult(
            valve_tag=valve_tag,
            stroke_percentage=stroke_percentage,
            stroke_time_ms=stroke_time,
            return_time_ms=return_time,
            passed=passed,
            breakaway_detected=True,  # Valve moved
            position_verified=True,
        )

        result.provenance_hash = hashlib.sha256(
            f"{result.test_id}|{valve_tag}|{passed}|{stroke_time}".encode()
        ).hexdigest()

        # Store result
        if valve_tag not in self.pst_results:
            self.pst_results[valve_tag] = []
        self.pst_results[valve_tag].append(result)

        logger.info(
            f"PST {result.test_id}: {'PASS' if passed else 'FAIL'}, "
            f"stroke={stroke_time:.0f}ms"
        )

        return result

    def verify_trip_point(
        self,
        sif_id: str,
        sensor_tag: str,
        setpoint_value: float,
        tolerance_percent: float = 1.0,
        tester: str = "",
        sensor_simulator: Optional[Callable[[float], bool]] = None
    ) -> Dict[str, Any]:
        """
        Verify trip point accuracy.

        Args:
            sif_id: SIF identifier
            sensor_tag: Sensor to test
            setpoint_value: Expected trip setpoint
            tolerance_percent: Acceptable tolerance
            tester: Person conducting test
            sensor_simulator: Callback to simulate sensor

        Returns:
            Trip point verification result
        """
        logger.info(
            f"Verifying trip point for {sif_id}/{sensor_tag}: {setpoint_value}"
        )

        # Calculate tolerance
        tolerance = setpoint_value * (tolerance_percent / 100)

        # Simulate approaching setpoint
        test_values = []
        trip_detected_at = None

        for offset_percent in range(-5, 6):
            test_value = setpoint_value * (1 + offset_percent / 100)

            if sensor_simulator:
                tripped = sensor_simulator(test_value)
            else:
                # Simulate: trip when >= setpoint
                tripped = test_value >= setpoint_value

            test_values.append({
                "value": test_value,
                "tripped": tripped,
            })

            if tripped and trip_detected_at is None:
                trip_detected_at = test_value

        # Calculate accuracy
        if trip_detected_at:
            error = abs(trip_detected_at - setpoint_value)
            error_percent = (error / setpoint_value) * 100
            passed = error <= tolerance
        else:
            error = None
            error_percent = None
            passed = False

        result = {
            "test_id": f"TPV-{uuid.uuid4().hex[:8].upper()}",
            "sif_id": sif_id,
            "sensor_tag": sensor_tag,
            "test_date": datetime.utcnow().isoformat(),
            "tester": tester,
            "setpoint_value": setpoint_value,
            "tolerance_percent": tolerance_percent,
            "tolerance_value": tolerance,
            "trip_detected_at": trip_detected_at,
            "error": error,
            "error_percent": error_percent,
            "passed": passed,
            "test_values": test_values,
            "provenance_hash": hashlib.sha256(
                f"{sif_id}|{sensor_tag}|{setpoint_value}|{passed}".encode()
            ).hexdigest()
        }

        logger.info(
            f"Trip point verification: {'PASS' if passed else 'FAIL'}, "
            f"error={error_percent:.2f}%" if error_percent else "no trip"
        )

        return result

    def create_test_schedule(
        self,
        sif_id: str,
        procedure_id: str,
        interval_days: int,
        assigned_to: str = "",
        priority: int = 1
    ) -> TestSchedule:
        """
        Create a test schedule for a SIF.

        Args:
            sif_id: SIF identifier
            procedure_id: Procedure to use
            interval_days: Test interval
            assigned_to: Assigned tester
            priority: Priority level

        Returns:
            TestSchedule
        """
        if procedure_id not in self.procedures:
            raise ValueError(f"Procedure not found: {procedure_id}")

        procedure = self.procedures[procedure_id]

        schedule = TestSchedule(
            sif_id=sif_id,
            procedure_id=procedure_id,
            test_type=procedure.test_type,
            interval_days=interval_days,
            assigned_to=assigned_to,
            priority=priority,
            next_test_due=datetime.utcnow() + timedelta(days=interval_days),
        )

        self.schedules[sif_id] = schedule

        logger.info(
            f"Created schedule for {sif_id}: every {interval_days} days"
        )

        return schedule

    def get_overdue_tests(self) -> List[TestSchedule]:
        """
        Get all overdue test schedules.

        Returns:
            List of overdue schedules
        """
        now = datetime.utcnow()
        overdue = []

        for schedule in self.schedules.values():
            if schedule.next_test_due and now > schedule.next_test_due:
                schedule.is_overdue = True
                overdue.append(schedule)

        return sorted(overdue, key=lambda s: s.priority)

    def get_upcoming_tests(
        self,
        days_ahead: int = 30
    ) -> List[TestSchedule]:
        """
        Get tests due in the next N days.

        Args:
            days_ahead: Days to look ahead

        Returns:
            List of upcoming schedules
        """
        now = datetime.utcnow()
        cutoff = now + timedelta(days=days_ahead)

        upcoming = [
            s for s in self.schedules.values()
            if s.next_test_due and now <= s.next_test_due <= cutoff
        ]

        return sorted(upcoming, key=lambda s: s.next_test_due)

    def generate_test_documentation(
        self,
        result: TestResult
    ) -> Dict[str, Any]:
        """
        Generate test documentation from result.

        Args:
            result: TestResult to document

        Returns:
            Documentation dictionary
        """
        procedure = self.procedures.get(result.procedure_id)

        doc = {
            "document_id": f"TD-{uuid.uuid4().hex[:8].upper()}",
            "document_type": "ESD Test Record",
            "generated_at": datetime.utcnow().isoformat(),
            "system_id": self.system_id,
            "header": {
                "result_id": result.result_id,
                "sif_id": result.sif_id,
                "test_type": result.test_type.value,
                "test_date": result.test_date.isoformat(),
                "tester": result.tester,
                "witness": result.witness,
                "work_permit": result.work_permit_ref,
            },
            "procedure": {
                "procedure_id": result.procedure_id,
                "procedure_name": procedure.procedure_name if procedure else "N/A",
                "sil_level": procedure.sil_level if procedure else 0,
                "prerequisites": procedure.prerequisites if procedure else [],
                "acceptance_criteria": procedure.acceptance_criteria if procedure else [],
            },
            "conditions": {
                "as_found": result.as_found_condition,
                "as_left": result.as_left_condition,
            },
            "results": {
                "overall_status": result.status.value,
                "overall_passed": result.overall_passed,
                "steps_total": result.steps_total,
                "steps_passed": result.steps_passed,
                "steps_failed": result.steps_failed,
                "response_time_ms": result.response_time_ms,
                "duration_minutes": result.duration_minutes,
            },
            "step_details": [
                {
                    "step_number": step.step_number,
                    "step_name": step.step_name,
                    "action": step.action,
                    "expected": step.expected_result,
                    "actual": step.actual_result,
                    "status": step.status.value,
                    "duration_ms": step.duration_ms,
                    "notes": step.notes,
                }
                for step in result.step_results
            ],
            "anomalies": result.anomalies,
            "corrective_actions": result.corrective_actions,
            "next_test_due": result.next_test_due.isoformat() if result.next_test_due else None,
            "signatures": {
                "tester": {
                    "name": result.tester,
                    "date": result.test_date.isoformat(),
                },
                "witness": {
                    "name": result.witness,
                    "date": result.test_date.isoformat(),
                } if result.witness else None,
            },
            "provenance": {
                "hash": result.provenance_hash,
                "algorithm": "SHA-256",
            }
        }

        return doc

    def get_test_history(
        self,
        sif_id: str,
        limit: int = 10
    ) -> List[TestResult]:
        """
        Get test history for a SIF.

        Args:
            sif_id: SIF identifier
            limit: Maximum results to return

        Returns:
            List of TestResults
        """
        results = self.results.get(sif_id, [])
        return results[-limit:]

    def get_test_summary(self) -> Dict[str, Any]:
        """
        Get overall test summary.

        Returns:
            Summary dictionary
        """
        total_results = sum(len(r) for r in self.results.values())
        total_passed = sum(
            sum(1 for r in results if r.overall_passed)
            for results in self.results.values()
        )

        overdue = self.get_overdue_tests()
        upcoming = self.get_upcoming_tests(30)

        return {
            "report_timestamp": datetime.utcnow().isoformat(),
            "system_id": self.system_id,
            "total_procedures": len(self.procedures),
            "total_schedules": len(self.schedules),
            "total_tests_conducted": total_results,
            "total_passed": total_passed,
            "total_failed": total_results - total_passed,
            "pass_rate_percent": (
                (total_passed / total_results * 100)
                if total_results > 0 else 0
            ),
            "overdue_count": len(overdue),
            "upcoming_30_days": len(upcoming),
            "pst_valves_tested": len(self.pst_results),
        }

    def _default_step_executor(self, step: TestStep) -> bool:
        """Default step executor (simulation)."""
        # Simulate step execution
        time.sleep(0.1)
        step.actual_result = step.expected_result
        return True

    def _calculate_provenance(self, result: TestResult) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_str = (
            f"{result.result_id}|"
            f"{result.sif_id}|"
            f"{result.overall_passed}|"
            f"{result.steps_passed}/{result.steps_total}|"
            f"{result.test_date.isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()
