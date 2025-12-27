"""
GL-016 Waterguard Calibration Monitor

This module provides deterministic monitoring of analyzer calibration health
for industrial water chemistry analyzers. It tracks calibration schedules,
detects calibration drift, and generates CMMS work orders when maintenance
is required.

All calculations use statistical methods (NO generative AI) for zero-hallucination
compliance. Provenance tracking via SHA-256 hashes ensures complete audit trails.

Supported Analyzers:
    - Conductivity analyzers
    - pH analyzers
    - Dissolved oxygen analyzers
    - Silica analyzers
    - Phosphate analyzers
    - Sodium analyzers

Example:
    >>> config = CalibrationMonitorConfig()
    >>> monitor = CalibrationMonitor(config)
    >>> status = monitor.check_calibration_due("COND-001", last_cal, interval_days=30)
    >>> if status.is_due:
    ...     work_order = monitor.generate_work_order(status)

Author: GreenLang Waterguard Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
import hashlib
import logging
import math
import uuid

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class AnalyzerType(Enum):
    """Types of water chemistry analyzers."""

    CONDUCTIVITY = "conductivity"
    PH = "ph"
    DISSOLVED_OXYGEN = "dissolved_oxygen"
    SILICA = "silica"
    PHOSPHATE = "phosphate"
    SODIUM = "sodium"
    ORP = "orp"
    TURBIDITY = "turbidity"
    CHLORINE = "chlorine"
    HARDNESS = "hardness"


class CalibrationStatus(Enum):
    """Calibration status levels."""

    CURRENT = "current"
    DUE_SOON = "due_soon"
    OVERDUE = "overdue"
    CRITICAL = "critical"


class DriftSeverity(Enum):
    """Severity of calibration drift."""

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class WorkOrderPriority(Enum):
    """CMMS work order priority levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EMERGENCY = "emergency"


class WorkOrderType(Enum):
    """Types of calibration work orders."""

    SCHEDULED_CALIBRATION = "scheduled_calibration"
    DRIFT_CORRECTION = "drift_correction"
    SENSOR_REPLACEMENT = "sensor_replacement"
    VERIFICATION_CHECK = "verification_check"


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CalibrationMonitorConfig:
    """
    Configuration for calibration monitoring.

    Attributes:
        default_calibration_interval_days: Default interval between calibrations
        due_soon_warning_days: Days before due to trigger warning
        overdue_grace_days: Days after due before critical status
        drift_warning_percent: Drift percentage for warning
        drift_critical_percent: Drift percentage for critical
        enable_work_order_generation: Whether to generate CMMS work orders
        cmms_system_id: Identifier for target CMMS system
        analyzer_intervals: Override intervals by analyzer type
    """

    default_calibration_interval_days: int = 30
    due_soon_warning_days: int = 7
    overdue_grace_days: int = 3
    drift_warning_percent: float = 2.0
    drift_critical_percent: float = 5.0
    enable_work_order_generation: bool = True
    cmms_system_id: str = "WATERGUARD-CMMS"
    analyzer_intervals: Dict[AnalyzerType, int] = field(default_factory=dict)

    def __post_init__(self):
        """Set default analyzer-specific intervals."""
        default_intervals = {
            AnalyzerType.CONDUCTIVITY: 30,
            AnalyzerType.PH: 14,
            AnalyzerType.DISSOLVED_OXYGEN: 14,
            AnalyzerType.SILICA: 7,
            AnalyzerType.PHOSPHATE: 14,
            AnalyzerType.SODIUM: 7,
            AnalyzerType.ORP: 30,
            AnalyzerType.TURBIDITY: 30,
            AnalyzerType.CHLORINE: 7,
            AnalyzerType.HARDNESS: 14,
        }
        for analyzer_type, interval in default_intervals.items():
            if analyzer_type not in self.analyzer_intervals:
                self.analyzer_intervals[analyzer_type] = interval


@dataclass
class AnalyzerSpec:
    """
    Specification for an individual analyzer.

    Attributes:
        analyzer_id: Unique identifier for the analyzer
        analyzer_type: Type of analyzer
        manufacturer: Equipment manufacturer
        model: Equipment model number
        expected_range: Expected measurement range (min, max)
        accuracy: Manufacturer-specified accuracy
        location: Physical location description
        calibration_points: Standard calibration points
    """

    analyzer_id: str
    analyzer_type: AnalyzerType
    manufacturer: str = ""
    model: str = ""
    expected_range: Tuple[float, float] = (0.0, 1000.0)
    accuracy: float = 1.0
    location: str = ""
    calibration_points: List[float] = field(default_factory=list)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CalibrationRecord:
    """
    Record of a calibration event.

    Attributes:
        calibration_id: Unique identifier for this calibration
        analyzer_id: Analyzer that was calibrated
        timestamp: When calibration was performed
        technician_id: Who performed the calibration
        before_readings: Readings before calibration
        after_readings: Readings after calibration
        standard_values: Standard solution values used
        adjustment_made: Description of adjustment
        passed: Whether calibration met specifications
        notes: Additional notes
        provenance_hash: SHA-256 hash for audit trail
    """

    calibration_id: str
    analyzer_id: str
    timestamp: datetime
    technician_id: str = ""
    before_readings: List[float] = field(default_factory=list)
    after_readings: List[float] = field(default_factory=list)
    standard_values: List[float] = field(default_factory=list)
    adjustment_made: str = ""
    passed: bool = True
    notes: str = ""
    provenance_hash: str = ""

    def __post_init__(self):
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        data_str = (
            f"{self.calibration_id}"
            f"{self.analyzer_id}"
            f"{self.timestamp.isoformat()}"
            f"{self.technician_id}"
            f"{','.join(str(r) for r in self.before_readings)}"
            f"{','.join(str(r) for r in self.after_readings)}"
            f"{','.join(str(v) for v in self.standard_values)}"
            f"{self.passed}"
        )
        return hashlib.sha256(data_str.encode()).hexdigest()


@dataclass
class CalibrationStatusResult:
    """
    Result of a calibration status check.

    Attributes:
        analyzer_id: Analyzer being checked
        status: Current calibration status
        last_calibration: Date of last calibration
        next_due: Date when next calibration is due
        days_until_due: Days until calibration is due (negative if overdue)
        drift_detected: Whether drift has been detected
        drift_percent: Estimated drift percentage
        drift_severity: Severity of detected drift
        recommended_action: What action should be taken
        provenance_hash: SHA-256 hash for audit trail
    """

    analyzer_id: str
    status: CalibrationStatus
    last_calibration: Optional[datetime]
    next_due: datetime
    days_until_due: int
    drift_detected: bool = False
    drift_percent: float = 0.0
    drift_severity: DriftSeverity = DriftSeverity.NONE
    recommended_action: str = ""
    provenance_hash: str = ""

    @property
    def is_due(self) -> bool:
        """Check if calibration is due or overdue."""
        return self.status in (
            CalibrationStatus.DUE_SOON,
            CalibrationStatus.OVERDUE,
            CalibrationStatus.CRITICAL
        )

    def __post_init__(self):
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        last_cal_str = (
            self.last_calibration.isoformat()
            if self.last_calibration else "none"
        )
        data_str = (
            f"{self.analyzer_id}"
            f"{self.status.value}"
            f"{last_cal_str}"
            f"{self.next_due.isoformat()}"
            f"{self.days_until_due}"
            f"{self.drift_detected}"
            f"{self.drift_percent:.6f}"
        )
        return hashlib.sha256(data_str.encode()).hexdigest()


@dataclass
class DriftAnalysisResult:
    """
    Result of calibration drift analysis.

    Attributes:
        analyzer_id: Analyzer being analyzed
        timestamp: When analysis was performed
        readings: Recent readings analyzed
        expected_value: Expected reading value
        mean_reading: Mean of recent readings
        drift_percent: Percentage drift from expected
        severity: Drift severity level
        trend_direction: Whether drift is increasing or stable
        confidence: Statistical confidence in drift detection
        recommendation: Recommended action
        provenance_hash: SHA-256 hash for audit trail
    """

    analyzer_id: str
    timestamp: datetime
    readings: List[float]
    expected_value: float
    mean_reading: float
    drift_percent: float
    severity: DriftSeverity
    trend_direction: str
    confidence: float
    recommendation: str
    provenance_hash: str = ""

    def __post_init__(self):
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        data_str = (
            f"{self.analyzer_id}"
            f"{self.timestamp.isoformat()}"
            f"{','.join(str(r) for r in self.readings)}"
            f"{self.expected_value:.6f}"
            f"{self.mean_reading:.6f}"
            f"{self.drift_percent:.6f}"
            f"{self.severity.value}"
        )
        return hashlib.sha256(data_str.encode()).hexdigest()


@dataclass
class CMMSWorkOrder:
    """
    CMMS work order for calibration maintenance.

    Attributes:
        work_order_id: Unique work order identifier
        analyzer_id: Target analyzer
        work_order_type: Type of work to be performed
        priority: Work order priority
        title: Work order title
        description: Detailed description
        due_date: When work should be completed
        estimated_duration_hours: Expected time to complete
        required_materials: List of required materials
        safety_requirements: Safety precautions
        created_timestamp: When work order was created
        status: Work order status
        provenance_hash: SHA-256 hash for audit trail
    """

    work_order_id: str
    analyzer_id: str
    work_order_type: WorkOrderType
    priority: WorkOrderPriority
    title: str
    description: str
    due_date: datetime
    estimated_duration_hours: float = 1.0
    required_materials: List[str] = field(default_factory=list)
    safety_requirements: List[str] = field(default_factory=list)
    created_timestamp: datetime = field(default_factory=datetime.now)
    status: str = "OPEN"
    provenance_hash: str = ""

    def __post_init__(self):
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            self.provenance_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash for audit trail."""
        data_str = (
            f"{self.work_order_id}"
            f"{self.analyzer_id}"
            f"{self.work_order_type.value}"
            f"{self.priority.value}"
            f"{self.due_date.isoformat()}"
            f"{self.created_timestamp.isoformat()}"
        )
        return hashlib.sha256(data_str.encode()).hexdigest()


# =============================================================================
# Main Monitor Class
# =============================================================================

class CalibrationMonitor:
    """
    Monitors analyzer calibration health and generates maintenance work orders.

    This monitor uses deterministic calculations to track calibration schedules,
    detect calibration drift, and generate CMMS work orders when maintenance
    is required. All calculations are auditable with SHA-256 provenance tracking.

    ZERO-HALLUCINATION COMPLIANCE:
    - All numeric calculations use Python arithmetic only
    - No generative AI models used for drift detection
    - Work order generation uses rule-based logic
    - Complete provenance tracking for regulatory audit

    Attributes:
        config: Monitor configuration
        analyzer_registry: Registry of monitored analyzers
        calibration_history: Historical calibration records

    Example:
        >>> config = CalibrationMonitorConfig()
        >>> monitor = CalibrationMonitor(config)
        >>> monitor.register_analyzer(spec)
        >>> status = monitor.check_calibration_due("COND-001")
        >>> if status.is_due:
        ...     work_order = monitor.generate_work_order(status)
    """

    def __init__(self, config: Optional[CalibrationMonitorConfig] = None):
        """
        Initialize the calibration monitor.

        Args:
            config: Monitor configuration. Uses defaults if not provided.
        """
        self.config = config or CalibrationMonitorConfig()
        self.analyzer_registry: Dict[str, AnalyzerSpec] = {}
        self.calibration_history: Dict[str, List[CalibrationRecord]] = {}
        self._pending_work_orders: List[CMMSWorkOrder] = []

        logger.info(
            f"CalibrationMonitor initialized with "
            f"default_interval={self.config.default_calibration_interval_days} days"
        )

    def register_analyzer(self, spec: AnalyzerSpec) -> None:
        """
        Register an analyzer for monitoring.

        Args:
            spec: Analyzer specification
        """
        self.analyzer_registry[spec.analyzer_id] = spec
        if spec.analyzer_id not in self.calibration_history:
            self.calibration_history[spec.analyzer_id] = []

        logger.info(
            f"Registered analyzer {spec.analyzer_id} "
            f"({spec.analyzer_type.value})"
        )

    def record_calibration(self, record: CalibrationRecord) -> None:
        """
        Record a calibration event.

        Args:
            record: Calibration record to store
        """
        if record.analyzer_id not in self.calibration_history:
            self.calibration_history[record.analyzer_id] = []

        self.calibration_history[record.analyzer_id].append(record)

        logger.info(
            f"Recorded calibration for {record.analyzer_id}: "
            f"passed={record.passed}"
        )

    def check_calibration_due(
        self,
        analyzer_id: str,
        last_calibration: Optional[datetime] = None,
        interval_days: Optional[int] = None,
        current_time: Optional[datetime] = None
    ) -> CalibrationStatusResult:
        """
        Check if calibration is due for an analyzer.

        Args:
            analyzer_id: Analyzer to check
            last_calibration: Override last calibration date
            interval_days: Override calibration interval
            current_time: Override current time (for testing)

        Returns:
            CalibrationStatusResult with status and recommendations
        """
        current_time = current_time or datetime.now()

        # Determine last calibration
        if last_calibration is None:
            last_calibration = self._get_last_calibration_date(analyzer_id)

        # Determine calibration interval
        if interval_days is None:
            interval_days = self._get_calibration_interval(analyzer_id)

        # Calculate next due date and days until due
        if last_calibration:
            next_due = last_calibration + timedelta(days=interval_days)
            days_until_due = (next_due - current_time).days
        else:
            # No calibration history - due immediately
            next_due = current_time
            days_until_due = 0

        # Determine status
        status = self._determine_calibration_status(days_until_due)

        # Generate recommendation
        recommended_action = self._generate_recommendation(status, days_until_due)

        result = CalibrationStatusResult(
            analyzer_id=analyzer_id,
            status=status,
            last_calibration=last_calibration,
            next_due=next_due,
            days_until_due=days_until_due,
            recommended_action=recommended_action
        )

        logger.info(
            f"Calibration check for {analyzer_id}: "
            f"status={status.value}, days_until_due={days_until_due}"
        )

        return result

    def detect_calibration_drift(
        self,
        analyzer_id: str,
        readings: List[float],
        expected_value: float,
        timestamp: Optional[datetime] = None
    ) -> DriftAnalysisResult:
        """
        Detect calibration drift from recent readings.

        Analyzes readings against expected value to determine if the
        analyzer is drifting out of calibration. Uses statistical methods
        for drift detection.

        Args:
            analyzer_id: Analyzer being analyzed
            readings: Recent readings to analyze
            expected_value: Expected reading value (from standard)
            timestamp: Analysis timestamp (defaults to now)

        Returns:
            DriftAnalysisResult with drift assessment

        Raises:
            ValueError: If readings list is empty or expected_value is zero
        """
        if not readings:
            raise ValueError("Readings list cannot be empty")
        if expected_value == 0:
            raise ValueError("Expected value cannot be zero")

        timestamp = timestamp or datetime.now()

        # Calculate statistics
        mean_reading = sum(readings) / len(readings)
        drift_percent = ((mean_reading - expected_value) / expected_value) * 100.0

        # Determine severity
        severity = self._determine_drift_severity(abs(drift_percent))

        # Analyze trend if enough readings
        trend_direction = "stable"
        if len(readings) >= 5:
            trend_direction = self._analyze_drift_trend(readings)

        # Calculate confidence based on sample size and variance
        confidence = self._calculate_drift_confidence(readings, expected_value)

        # Generate recommendation
        recommendation = self._generate_drift_recommendation(
            severity, drift_percent, trend_direction
        )

        result = DriftAnalysisResult(
            analyzer_id=analyzer_id,
            timestamp=timestamp,
            readings=readings,
            expected_value=expected_value,
            mean_reading=mean_reading,
            drift_percent=drift_percent,
            severity=severity,
            trend_direction=trend_direction,
            confidence=confidence,
            recommendation=recommendation
        )

        if severity != DriftSeverity.NONE:
            logger.warning(
                f"Drift detected for {analyzer_id}: "
                f"{drift_percent:.2f}% ({severity.value})"
            )

        return result

    def generate_work_order(
        self,
        status: CalibrationStatusResult,
        analyzer_spec: Optional[AnalyzerSpec] = None
    ) -> Optional[CMMSWorkOrder]:
        """
        Generate a CMMS work order for calibration maintenance.

        Args:
            status: Calibration status result
            analyzer_spec: Analyzer specification (looked up if not provided)

        Returns:
            CMMSWorkOrder if work order generation is enabled, None otherwise
        """
        if not self.config.enable_work_order_generation:
            logger.debug("Work order generation disabled")
            return None

        # Get analyzer spec if not provided
        if analyzer_spec is None:
            analyzer_spec = self.analyzer_registry.get(status.analyzer_id)

        # Determine work order type
        if status.drift_detected and status.drift_severity == DriftSeverity.SEVERE:
            work_order_type = WorkOrderType.SENSOR_REPLACEMENT
        elif status.drift_detected:
            work_order_type = WorkOrderType.DRIFT_CORRECTION
        else:
            work_order_type = WorkOrderType.SCHEDULED_CALIBRATION

        # Determine priority
        priority = self._determine_work_order_priority(status)

        # Generate work order ID
        work_order_id = f"WO-{uuid.uuid4().hex[:8].upper()}"

        # Build title and description
        title = self._build_work_order_title(
            status.analyzer_id, work_order_type, analyzer_spec
        )
        description = self._build_work_order_description(
            status, work_order_type, analyzer_spec
        )

        # Get required materials
        required_materials = self._get_required_materials(work_order_type, analyzer_spec)

        # Get safety requirements
        safety_requirements = self._get_safety_requirements(analyzer_spec)

        # Calculate due date
        if status.status == CalibrationStatus.CRITICAL:
            due_date = datetime.now()
        elif status.status == CalibrationStatus.OVERDUE:
            due_date = datetime.now() + timedelta(days=1)
        else:
            due_date = status.next_due

        work_order = CMMSWorkOrder(
            work_order_id=work_order_id,
            analyzer_id=status.analyzer_id,
            work_order_type=work_order_type,
            priority=priority,
            title=title,
            description=description,
            due_date=due_date,
            estimated_duration_hours=self._estimate_work_duration(work_order_type),
            required_materials=required_materials,
            safety_requirements=safety_requirements
        )

        self._pending_work_orders.append(work_order)

        logger.info(
            f"Generated work order {work_order_id} for {status.analyzer_id}: "
            f"priority={priority.value}"
        )

        return work_order

    def check_all_analyzers(
        self,
        current_time: Optional[datetime] = None
    ) -> List[CalibrationStatusResult]:
        """
        Check calibration status for all registered analyzers.

        Args:
            current_time: Override current time (for testing)

        Returns:
            List of CalibrationStatusResult for all analyzers
        """
        results = []
        for analyzer_id in self.analyzer_registry:
            result = self.check_calibration_due(
                analyzer_id, current_time=current_time
            )
            results.append(result)

        # Log summary
        due_count = sum(1 for r in results if r.is_due)
        logger.info(
            f"Checked {len(results)} analyzers: {due_count} require calibration"
        )

        return results

    def get_pending_work_orders(self) -> List[CMMSWorkOrder]:
        """Get all pending work orders."""
        return self._pending_work_orders.copy()

    def clear_pending_work_orders(self) -> None:
        """Clear the pending work orders list."""
        self._pending_work_orders.clear()

    def _get_last_calibration_date(self, analyzer_id: str) -> Optional[datetime]:
        """Get the last calibration date for an analyzer."""
        history = self.calibration_history.get(analyzer_id, [])
        if not history:
            return None

        # Return most recent calibration
        sorted_history = sorted(history, key=lambda r: r.timestamp, reverse=True)
        return sorted_history[0].timestamp

    def _get_calibration_interval(self, analyzer_id: str) -> int:
        """Get the calibration interval for an analyzer."""
        spec = self.analyzer_registry.get(analyzer_id)
        if spec and spec.analyzer_type in self.config.analyzer_intervals:
            return self.config.analyzer_intervals[spec.analyzer_type]
        return self.config.default_calibration_interval_days

    def _determine_calibration_status(self, days_until_due: int) -> CalibrationStatus:
        """Determine calibration status based on days until due."""
        if days_until_due < -self.config.overdue_grace_days:
            return CalibrationStatus.CRITICAL
        elif days_until_due < 0:
            return CalibrationStatus.OVERDUE
        elif days_until_due <= self.config.due_soon_warning_days:
            return CalibrationStatus.DUE_SOON
        else:
            return CalibrationStatus.CURRENT

    def _determine_drift_severity(self, drift_percent: float) -> DriftSeverity:
        """Determine drift severity based on percentage."""
        if drift_percent >= self.config.drift_critical_percent:
            return DriftSeverity.SEVERE
        elif drift_percent >= self.config.drift_warning_percent:
            return DriftSeverity.MODERATE
        elif drift_percent >= self.config.drift_warning_percent / 2:
            return DriftSeverity.MINOR
        else:
            return DriftSeverity.NONE

    def _analyze_drift_trend(self, readings: List[float]) -> str:
        """Analyze drift trend from readings."""
        if len(readings) < 5:
            return "stable"

        # Simple trend analysis using first and last halves
        mid = len(readings) // 2
        first_half_mean = sum(readings[:mid]) / mid
        second_half_mean = sum(readings[mid:]) / (len(readings) - mid)

        diff_percent = abs(second_half_mean - first_half_mean) / first_half_mean * 100

        if diff_percent < 1.0:
            return "stable"
        elif second_half_mean > first_half_mean:
            return "increasing"
        else:
            return "decreasing"

    def _calculate_drift_confidence(
        self,
        readings: List[float],
        expected_value: float
    ) -> float:
        """Calculate confidence in drift detection."""
        n = len(readings)

        # Base confidence on sample size
        sample_confidence = min(1.0, n / 20.0)

        # Calculate variance
        mean = sum(readings) / n
        variance = sum((r - mean) ** 2 for r in readings) / n if n > 1 else 0
        std_dev = math.sqrt(variance)

        # Lower confidence if high variance
        if std_dev > 0 and mean > 0:
            cv = std_dev / abs(mean)  # Coefficient of variation
            variance_confidence = max(0, 1.0 - cv)
        else:
            variance_confidence = 1.0

        return sample_confidence * variance_confidence

    def _generate_recommendation(
        self,
        status: CalibrationStatus,
        days_until_due: int
    ) -> str:
        """Generate recommendation based on status."""
        if status == CalibrationStatus.CRITICAL:
            return "URGENT: Calibration is critically overdue. Calibrate immediately."
        elif status == CalibrationStatus.OVERDUE:
            return "Calibration is overdue. Schedule calibration within 24 hours."
        elif status == CalibrationStatus.DUE_SOON:
            return f"Calibration due in {days_until_due} days. Schedule calibration."
        else:
            return f"Calibration current. Next due in {days_until_due} days."

    def _generate_drift_recommendation(
        self,
        severity: DriftSeverity,
        drift_percent: float,
        trend_direction: str
    ) -> str:
        """Generate recommendation based on drift analysis."""
        if severity == DriftSeverity.SEVERE:
            return "URGENT: Severe drift detected. Recalibrate immediately."
        elif severity == DriftSeverity.MODERATE:
            rec = f"Moderate drift ({drift_percent:.1f}%) detected."
            if trend_direction == "increasing":
                rec += " Drift is increasing. Schedule calibration soon."
            else:
                rec += " Monitor closely and schedule calibration."
            return rec
        elif severity == DriftSeverity.MINOR:
            return f"Minor drift ({drift_percent:.1f}%) detected. Continue monitoring."
        else:
            return "No significant drift detected."

    def _determine_work_order_priority(
        self,
        status: CalibrationStatusResult
    ) -> WorkOrderPriority:
        """Determine work order priority based on status."""
        if status.status == CalibrationStatus.CRITICAL:
            return WorkOrderPriority.EMERGENCY
        elif status.status == CalibrationStatus.OVERDUE:
            return WorkOrderPriority.HIGH
        elif status.drift_severity in (DriftSeverity.SEVERE, DriftSeverity.MODERATE):
            return WorkOrderPriority.HIGH
        elif status.status == CalibrationStatus.DUE_SOON:
            return WorkOrderPriority.MEDIUM
        else:
            return WorkOrderPriority.LOW

    def _build_work_order_title(
        self,
        analyzer_id: str,
        work_order_type: WorkOrderType,
        spec: Optional[AnalyzerSpec]
    ) -> str:
        """Build work order title."""
        analyzer_desc = ""
        if spec:
            analyzer_desc = f"{spec.analyzer_type.value.title()} Analyzer"
        else:
            analyzer_desc = "Analyzer"

        type_desc = {
            WorkOrderType.SCHEDULED_CALIBRATION: "Scheduled Calibration",
            WorkOrderType.DRIFT_CORRECTION: "Drift Correction",
            WorkOrderType.SENSOR_REPLACEMENT: "Sensor Replacement",
            WorkOrderType.VERIFICATION_CHECK: "Verification Check"
        }.get(work_order_type, "Calibration")

        return f"{type_desc} - {analyzer_desc} {analyzer_id}"

    def _build_work_order_description(
        self,
        status: CalibrationStatusResult,
        work_order_type: WorkOrderType,
        spec: Optional[AnalyzerSpec]
    ) -> str:
        """Build detailed work order description."""
        lines = []

        lines.append(f"Analyzer ID: {status.analyzer_id}")
        if spec:
            lines.append(f"Analyzer Type: {spec.analyzer_type.value}")
            lines.append(f"Location: {spec.location or 'Not specified'}")
            lines.append(f"Manufacturer: {spec.manufacturer or 'Not specified'}")
            lines.append(f"Model: {spec.model or 'Not specified'}")

        lines.append("")
        lines.append(f"Status: {status.status.value.upper()}")
        if status.last_calibration:
            lines.append(f"Last Calibration: {status.last_calibration.isoformat()}")
        lines.append(f"Next Due: {status.next_due.isoformat()}")
        lines.append(f"Days Until Due: {status.days_until_due}")

        if status.drift_detected:
            lines.append("")
            lines.append(f"Drift Detected: {status.drift_percent:.2f}%")
            lines.append(f"Drift Severity: {status.drift_severity.value}")

        lines.append("")
        lines.append(f"Recommended Action: {status.recommended_action}")

        return "\n".join(lines)

    def _get_required_materials(
        self,
        work_order_type: WorkOrderType,
        spec: Optional[AnalyzerSpec]
    ) -> List[str]:
        """Get required materials for work order."""
        materials = []

        if spec:
            if spec.analyzer_type == AnalyzerType.CONDUCTIVITY:
                materials.extend([
                    "Conductivity calibration standards (1413 uS/cm, 12880 uS/cm)",
                    "DI rinse water",
                    "Soft cloths for probe cleaning"
                ])
            elif spec.analyzer_type == AnalyzerType.PH:
                materials.extend([
                    "pH buffer solutions (4.0, 7.0, 10.0)",
                    "DI rinse water",
                    "pH electrode storage solution"
                ])
            elif spec.analyzer_type == AnalyzerType.DISSOLVED_OXYGEN:
                materials.extend([
                    "Zero oxygen solution (sodium sulfite)",
                    "Saturated air/water for span",
                    "Membrane cap and electrolyte (if replacing)"
                ])
            elif spec.analyzer_type == AnalyzerType.SILICA:
                materials.extend([
                    "Silica calibration standards",
                    "Reagent check solutions",
                    "DI rinse water"
                ])

        if work_order_type == WorkOrderType.SENSOR_REPLACEMENT:
            materials.append("Replacement sensor/probe")
            materials.append("Sensor o-rings and seals")

        return materials

    def _get_safety_requirements(
        self,
        spec: Optional[AnalyzerSpec]
    ) -> List[str]:
        """Get safety requirements for work order."""
        requirements = [
            "Wear appropriate PPE (safety glasses, gloves)",
            "Follow LOTO procedures if working on powered equipment",
            "Ensure proper ventilation when using chemicals"
        ]

        if spec:
            if spec.analyzer_type in (AnalyzerType.CHLORINE, AnalyzerType.ORP):
                requirements.append("Handle chlorine standards with care - corrosive")
            elif spec.analyzer_type == AnalyzerType.PH:
                requirements.append("pH buffers 10.0 and above are caustic")

        return requirements

    def _estimate_work_duration(self, work_order_type: WorkOrderType) -> float:
        """Estimate work duration in hours."""
        durations = {
            WorkOrderType.SCHEDULED_CALIBRATION: 1.0,
            WorkOrderType.DRIFT_CORRECTION: 1.5,
            WorkOrderType.SENSOR_REPLACEMENT: 2.0,
            WorkOrderType.VERIFICATION_CHECK: 0.5
        }
        return durations.get(work_order_type, 1.0)


# =============================================================================
# Factory Functions
# =============================================================================

def create_calibration_monitor(
    default_interval_days: int = 30,
    due_soon_warning_days: int = 7,
    drift_warning_percent: float = 2.0,
    **kwargs
) -> CalibrationMonitor:
    """
    Factory function to create a configured calibration monitor.

    Args:
        default_interval_days: Default calibration interval
        due_soon_warning_days: Days before due to warn
        drift_warning_percent: Drift percentage for warning
        **kwargs: Additional configuration options

    Returns:
        Configured CalibrationMonitor instance
    """
    config = CalibrationMonitorConfig(
        default_calibration_interval_days=default_interval_days,
        due_soon_warning_days=due_soon_warning_days,
        drift_warning_percent=drift_warning_percent,
        **kwargs
    )
    return CalibrationMonitor(config)


def create_analyzer_spec(
    analyzer_id: str,
    analyzer_type: AnalyzerType,
    **kwargs
) -> AnalyzerSpec:
    """
    Factory function to create an analyzer specification.

    Args:
        analyzer_id: Unique identifier
        analyzer_type: Type of analyzer
        **kwargs: Additional specification options

    Returns:
        Configured AnalyzerSpec instance
    """
    return AnalyzerSpec(
        analyzer_id=analyzer_id,
        analyzer_type=analyzer_type,
        **kwargs
    )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Main class
    "CalibrationMonitor",
    # Configuration
    "CalibrationMonitorConfig",
    "AnalyzerSpec",
    # Data classes
    "CalibrationRecord",
    "CalibrationStatusResult",
    "DriftAnalysisResult",
    "CMMSWorkOrder",
    # Enums
    "AnalyzerType",
    "CalibrationStatus",
    "DriftSeverity",
    "WorkOrderPriority",
    "WorkOrderType",
    # Factory functions
    "create_calibration_monitor",
    "create_analyzer_spec",
]
