"""
Calibration Tracker Module - GL-016_Waterguard Optimization Service

This module provides calibration tracking for valve curves, pump characteristics,
and other equipment models used in water treatment optimization. It includes
drift detection, CMMS integration, and calibration scheduling.

Key Components:
    - CalibrationTracker: Core tracking for equipment calibration state
    - DriftDetector: Detects calibration drift using statistical methods
    - CMMSIntegrator: Integration with Computerized Maintenance Management System
    - CalibrationScheduler: Schedules calibration activities

Example:
    >>> tracker = CalibrationTracker(config)
    >>> tracker.record_calibration("valve_001", calibration_data)
    >>> drift_status = tracker.check_drift("valve_001", current_measurement)
    >>> if drift_status.needs_recalibration:
    ...     tracker.schedule_calibration("valve_001")
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import hashlib
import logging
import math
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class EquipmentType(str, Enum):
    """Types of equipment requiring calibration."""
    VALVE = "valve"
    PUMP = "pump"
    SENSOR = "sensor"
    FLOW_METER = "flow_meter"
    LEVEL_SENSOR = "level_sensor"
    PH_SENSOR = "ph_sensor"
    CONDUCTIVITY_SENSOR = "conductivity_sensor"
    TURBIDITY_SENSOR = "turbidity_sensor"
    DOSING_PUMP = "dosing_pump"
    BLOWDOWN_VALVE = "blowdown_valve"


class CalibrationStatus(str, Enum):
    """Status of equipment calibration."""
    CALIBRATED = "calibrated"
    DRIFT_DETECTED = "drift_detected"
    NEEDS_CALIBRATION = "needs_calibration"
    OVERDUE = "overdue"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"
    UNKNOWN = "unknown"


class DriftSeverity(str, Enum):
    """Severity level of detected calibration drift."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class CMMSWorkOrderStatus(str, Enum):
    """Status of CMMS work orders."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# =============================================================================
# Data Models
# =============================================================================

class CalibrationConfig(BaseModel):
    """Configuration for calibration tracking."""

    drift_threshold_percent: float = Field(
        default=5.0,
        ge=0.1,
        le=50.0,
        description="Percentage drift threshold for recalibration trigger"
    )
    severe_drift_threshold_percent: float = Field(
        default=15.0,
        ge=1.0,
        le=100.0,
        description="Percentage drift threshold for severe drift alert"
    )
    default_calibration_interval_days: int = Field(
        default=90,
        ge=7,
        le=365,
        description="Default calibration interval in days"
    )
    drift_detection_window_samples: int = Field(
        default=20,
        ge=5,
        le=1000,
        description="Number of samples for drift detection"
    )
    cmms_enabled: bool = Field(
        default=True,
        description="Enable CMMS integration"
    )
    cmms_api_url: Optional[str] = Field(
        default=None,
        description="CMMS API endpoint URL"
    )
    auto_schedule_calibration: bool = Field(
        default=True,
        description="Automatically schedule calibration when drift detected"
    )
    provenance_enabled: bool = Field(
        default=True,
        description="Enable provenance tracking"
    )


class CalibrationPoint(BaseModel):
    """Single calibration point for curve fitting."""

    reference_value: float = Field(..., description="Reference/standard value")
    measured_value: float = Field(..., description="Measured value from equipment")
    uncertainty: Optional[float] = Field(None, ge=0.0, description="Measurement uncertainty")
    timestamp: datetime = Field(default_factory=datetime.now, description="Measurement timestamp")


class CalibrationCurve(BaseModel):
    """Calibration curve parameters."""

    curve_type: str = Field(
        default="linear",
        description="Type of curve: linear, polynomial, spline"
    )
    coefficients: List[float] = Field(
        default_factory=list,
        description="Curve coefficients (slope, intercept for linear)"
    )
    r_squared: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Coefficient of determination"
    )
    valid_range: Tuple[float, float] = Field(
        default=(0.0, 100.0),
        description="Valid input range for calibration"
    )
    calibration_points: List[CalibrationPoint] = Field(
        default_factory=list,
        description="Points used for calibration"
    )


class CalibrationRecord(BaseModel):
    """Record of a calibration event."""

    calibration_id: str = Field(..., description="Unique calibration identifier")
    equipment_id: str = Field(..., description="Equipment identifier")
    equipment_type: EquipmentType = Field(..., description="Type of equipment")
    calibration_date: datetime = Field(
        default_factory=datetime.now,
        description="Calibration date"
    )
    calibration_curve: CalibrationCurve = Field(..., description="Calibration curve")
    technician_id: Optional[str] = Field(None, description="Technician who performed calibration")
    standard_used: Optional[str] = Field(None, description="Calibration standard used")
    certificate_number: Optional[str] = Field(None, description="Calibration certificate number")
    next_calibration_due: datetime = Field(..., description="Next calibration due date")
    notes: Optional[str] = Field(None, description="Calibration notes")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")

    @validator('provenance_hash', always=True)
    def compute_provenance(cls, v, values):
        """Compute provenance hash from calibration record."""
        if v:
            return v
        data_str = (
            f"{values.get('calibration_id', '')}"
            f"{values.get('equipment_id', '')}"
            f"{values.get('calibration_date', '')}"
            f"{values.get('calibration_curve', '')}"
        )
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]


class DriftMeasurement(BaseModel):
    """Single drift measurement data point."""

    timestamp: datetime = Field(default_factory=datetime.now, description="Measurement time")
    expected_value: float = Field(..., description="Expected value from calibration")
    actual_value: float = Field(..., description="Actual measured value")
    deviation: float = Field(..., description="Absolute deviation")
    deviation_percent: float = Field(..., description="Percentage deviation")
    operating_conditions: Optional[Dict[str, float]] = Field(
        None,
        description="Operating conditions during measurement"
    )


class DriftAnalysis(BaseModel):
    """Analysis of calibration drift."""

    equipment_id: str = Field(..., description="Equipment identifier")
    analysis_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Analysis timestamp"
    )
    sample_count: int = Field(..., ge=0, description="Number of samples analyzed")
    mean_deviation_percent: float = Field(..., description="Mean deviation percentage")
    max_deviation_percent: float = Field(..., description="Maximum deviation percentage")
    trend_slope: float = Field(..., description="Drift trend slope (deviation/day)")
    drift_severity: DriftSeverity = Field(..., description="Severity of drift")
    needs_recalibration: bool = Field(..., description="Whether recalibration is needed")
    estimated_days_until_limit: Optional[float] = Field(
        None,
        description="Estimated days until drift limit exceeded"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.999,
        description="Statistical confidence level"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations based on drift analysis"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")


class CMMSWorkOrder(BaseModel):
    """CMMS work order for calibration."""

    work_order_id: str = Field(..., description="Work order identifier")
    equipment_id: str = Field(..., description="Equipment to calibrate")
    equipment_type: EquipmentType = Field(..., description="Equipment type")
    priority: int = Field(default=2, ge=1, le=5, description="Priority (1=highest)")
    status: CMMSWorkOrderStatus = Field(
        default=CMMSWorkOrderStatus.DRAFT,
        description="Work order status"
    )
    created_date: datetime = Field(default_factory=datetime.now, description="Creation date")
    scheduled_date: Optional[datetime] = Field(None, description="Scheduled execution date")
    completed_date: Optional[datetime] = Field(None, description="Completion date")
    assigned_technician: Optional[str] = Field(None, description="Assigned technician")
    estimated_duration_hours: float = Field(default=2.0, ge=0.5, description="Estimated duration")
    reason: str = Field(..., description="Reason for calibration")
    drift_analysis: Optional[DriftAnalysis] = Field(None, description="Associated drift analysis")
    notes: Optional[str] = Field(None, description="Work order notes")


class EquipmentCalibrationState(BaseModel):
    """Current calibration state for a piece of equipment."""

    equipment_id: str = Field(..., description="Equipment identifier")
    equipment_type: EquipmentType = Field(..., description="Equipment type")
    status: CalibrationStatus = Field(..., description="Current calibration status")
    last_calibration: Optional[CalibrationRecord] = Field(
        None,
        description="Most recent calibration record"
    )
    next_calibration_due: Optional[datetime] = Field(
        None,
        description="Next calibration due date"
    )
    current_drift: Optional[DriftAnalysis] = Field(
        None,
        description="Current drift analysis"
    )
    pending_work_order: Optional[CMMSWorkOrder] = Field(
        None,
        description="Pending calibration work order"
    )
    measurement_history: List[DriftMeasurement] = Field(
        default_factory=list,
        description="Recent measurement history for drift detection"
    )
    days_until_due: Optional[int] = Field(None, description="Days until calibration due")
    calibration_count: int = Field(default=0, ge=0, description="Total calibration count")


# =============================================================================
# Drift Detector
# =============================================================================

class DriftDetector:
    """
    Detects calibration drift using statistical methods.

    Monitors measurement deviations over time and detects when
    equipment is drifting out of calibration using trend analysis.

    Attributes:
        config: Calibration configuration
        measurement_buffer: Rolling buffer of measurements by equipment
    """

    def __init__(self, config: CalibrationConfig):
        """Initialize drift detector."""
        self.config = config
        self.measurement_buffer: Dict[str, List[DriftMeasurement]] = {}
        logger.info("DriftDetector initialized with threshold: %.1f%%",
                   config.drift_threshold_percent)

    def add_measurement(
        self,
        equipment_id: str,
        expected_value: float,
        actual_value: float,
        operating_conditions: Optional[Dict[str, float]] = None
    ) -> DriftMeasurement:
        """
        Add a new measurement for drift tracking.

        Args:
            equipment_id: Equipment identifier
            expected_value: Expected value from calibration curve
            actual_value: Actual measured value
            operating_conditions: Current operating conditions

        Returns:
            DriftMeasurement object
        """
        deviation = actual_value - expected_value
        deviation_percent = (deviation / expected_value * 100.0) if expected_value != 0 else 0.0

        measurement = DriftMeasurement(
            expected_value=expected_value,
            actual_value=actual_value,
            deviation=deviation,
            deviation_percent=deviation_percent,
            operating_conditions=operating_conditions
        )

        # Add to buffer
        if equipment_id not in self.measurement_buffer:
            self.measurement_buffer[equipment_id] = []

        self.measurement_buffer[equipment_id].append(measurement)

        # Trim to window size
        window = self.config.drift_detection_window_samples
        if len(self.measurement_buffer[equipment_id]) > window:
            self.measurement_buffer[equipment_id] = self.measurement_buffer[equipment_id][-window:]

        logger.debug("Added measurement for %s: deviation=%.2f%%",
                    equipment_id, deviation_percent)

        return measurement

    def analyze_drift(self, equipment_id: str) -> DriftAnalysis:
        """
        Analyze calibration drift for equipment.

        Args:
            equipment_id: Equipment identifier

        Returns:
            DriftAnalysis with drift assessment
        """
        measurements = self.measurement_buffer.get(equipment_id, [])

        if not measurements:
            logger.warning("No measurements available for drift analysis: %s", equipment_id)
            return self._create_empty_analysis(equipment_id)

        # Calculate statistics
        deviations = [abs(m.deviation_percent) for m in measurements]
        mean_deviation = statistics.mean(deviations)
        max_deviation = max(deviations)

        # Calculate trend using linear regression
        trend_slope = self._calculate_trend_slope(measurements)

        # Determine severity
        severity = self._classify_severity(mean_deviation, max_deviation)

        # Check if recalibration is needed
        needs_recal = (
            mean_deviation > self.config.drift_threshold_percent or
            max_deviation > self.config.severe_drift_threshold_percent or
            severity in (DriftSeverity.SEVERE, DriftSeverity.CRITICAL)
        )

        # Estimate days until limit (if trending)
        days_until_limit = None
        if trend_slope > 0 and mean_deviation < self.config.drift_threshold_percent:
            remaining_margin = self.config.drift_threshold_percent - mean_deviation
            days_until_limit = remaining_margin / trend_slope if trend_slope > 0.01 else None

        # Generate recommendations
        recommendations = self._generate_recommendations(
            severity, mean_deviation, trend_slope, needs_recal
        )

        # Create analysis with provenance
        analysis_data = {
            "equipment_id": equipment_id,
            "sample_count": len(measurements),
            "mean_deviation_percent": mean_deviation,
            "max_deviation_percent": max_deviation,
            "trend_slope": trend_slope,
            "drift_severity": severity,
            "needs_recalibration": needs_recal,
            "estimated_days_until_limit": days_until_limit,
            "recommendations": recommendations
        }

        provenance_str = f"{analysis_data}{datetime.now().isoformat()}"
        analysis_data["provenance_hash"] = hashlib.sha256(provenance_str.encode()).hexdigest()[:16]

        analysis = DriftAnalysis(**analysis_data)

        logger.info("Drift analysis for %s: severity=%s, needs_recal=%s",
                   equipment_id, severity.value, needs_recal)

        return analysis

    def _calculate_trend_slope(self, measurements: List[DriftMeasurement]) -> float:
        """Calculate drift trend slope using linear regression."""
        if len(measurements) < 2:
            return 0.0

        # Convert timestamps to days from first measurement
        base_time = measurements[0].timestamp
        x_values = [(m.timestamp - base_time).total_seconds() / 86400.0 for m in measurements]
        y_values = [abs(m.deviation_percent) for m in measurements]

        # Simple linear regression
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_xx = sum(x * x for x in x_values)

        denominator = n * sum_xx - sum_x * sum_x
        if abs(denominator) < 1e-10:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope  # deviation_percent per day

    def _classify_severity(self, mean_deviation: float, max_deviation: float) -> DriftSeverity:
        """Classify drift severity based on deviations."""
        threshold = self.config.drift_threshold_percent
        severe_threshold = self.config.severe_drift_threshold_percent

        if max_deviation > severe_threshold * 1.5:
            return DriftSeverity.CRITICAL
        elif max_deviation > severe_threshold or mean_deviation > threshold * 1.5:
            return DriftSeverity.SEVERE
        elif mean_deviation > threshold:
            return DriftSeverity.MODERATE
        elif mean_deviation > threshold * 0.5:
            return DriftSeverity.MINOR
        else:
            return DriftSeverity.NONE

    def _generate_recommendations(
        self,
        severity: DriftSeverity,
        mean_deviation: float,
        trend_slope: float,
        needs_recalibration: bool
    ) -> List[str]:
        """Generate recommendations based on drift analysis."""
        recommendations = []

        if severity == DriftSeverity.CRITICAL:
            recommendations.append("IMMEDIATE: Remove equipment from service for emergency calibration")
            recommendations.append("Investigate root cause of severe drift")

        elif severity == DriftSeverity.SEVERE:
            recommendations.append("Schedule urgent calibration within 24 hours")
            recommendations.append("Increase monitoring frequency until calibration")

        elif severity == DriftSeverity.MODERATE:
            recommendations.append("Schedule calibration within 1 week")
            recommendations.append("Review operating conditions for contributing factors")

        elif severity == DriftSeverity.MINOR:
            recommendations.append("Monitor drift trend closely")
            recommendations.append("Consider early calibration if trend continues")

        if trend_slope > 0.5:  # >0.5% per day
            recommendations.append(f"Rapid drift detected ({trend_slope:.2f}%/day) - check for environmental factors")

        if needs_recalibration:
            recommendations.append("Create CMMS work order for calibration")

        return recommendations

    def _create_empty_analysis(self, equipment_id: str) -> DriftAnalysis:
        """Create empty analysis when no data available."""
        return DriftAnalysis(
            equipment_id=equipment_id,
            sample_count=0,
            mean_deviation_percent=0.0,
            max_deviation_percent=0.0,
            trend_slope=0.0,
            drift_severity=DriftSeverity.NONE,
            needs_recalibration=False,
            recommendations=["Collect measurement data for drift analysis"],
            provenance_hash=hashlib.sha256(f"{equipment_id}empty".encode()).hexdigest()[:16]
        )

    def clear_history(self, equipment_id: str) -> bool:
        """Clear measurement history for equipment (e.g., after recalibration)."""
        if equipment_id in self.measurement_buffer:
            del self.measurement_buffer[equipment_id]
            logger.info("Cleared drift history for %s", equipment_id)
            return True
        return False


# =============================================================================
# CMMS Integrator
# =============================================================================

class CMMSIntegrator:
    """
    Integration with Computerized Maintenance Management System.

    Handles creation, submission, and tracking of calibration
    work orders in CMMS.

    Attributes:
        config: Calibration configuration
        work_orders: Local cache of work orders
    """

    def __init__(self, config: CalibrationConfig):
        """Initialize CMMS integrator."""
        self.config = config
        self.work_orders: Dict[str, CMMSWorkOrder] = {}
        self._order_counter = 0
        logger.info("CMMSIntegrator initialized, enabled=%s", config.cmms_enabled)

    def _generate_work_order_id(self) -> str:
        """Generate unique work order ID."""
        self._order_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"WO-CAL-{timestamp}-{self._order_counter:04d}"

    def create_work_order(
        self,
        equipment_id: str,
        equipment_type: EquipmentType,
        reason: str,
        priority: int = 2,
        drift_analysis: Optional[DriftAnalysis] = None,
        scheduled_date: Optional[datetime] = None
    ) -> CMMSWorkOrder:
        """
        Create a new calibration work order.

        Args:
            equipment_id: Equipment identifier
            equipment_type: Type of equipment
            reason: Reason for calibration
            priority: Work order priority (1-5)
            drift_analysis: Associated drift analysis
            scheduled_date: Proposed scheduled date

        Returns:
            Created work order
        """
        work_order_id = self._generate_work_order_id()

        # Adjust priority based on drift severity
        if drift_analysis:
            if drift_analysis.drift_severity == DriftSeverity.CRITICAL:
                priority = 1
            elif drift_analysis.drift_severity == DriftSeverity.SEVERE:
                priority = min(priority, 2)

        # Estimate duration based on equipment type
        duration_map = {
            EquipmentType.VALVE: 2.0,
            EquipmentType.PUMP: 3.0,
            EquipmentType.SENSOR: 1.0,
            EquipmentType.FLOW_METER: 2.5,
            EquipmentType.PH_SENSOR: 1.5,
            EquipmentType.CONDUCTIVITY_SENSOR: 1.5,
            EquipmentType.DOSING_PUMP: 2.0,
            EquipmentType.BLOWDOWN_VALVE: 2.0,
        }
        estimated_duration = duration_map.get(equipment_type, 2.0)

        work_order = CMMSWorkOrder(
            work_order_id=work_order_id,
            equipment_id=equipment_id,
            equipment_type=equipment_type,
            priority=priority,
            reason=reason,
            drift_analysis=drift_analysis,
            scheduled_date=scheduled_date,
            estimated_duration_hours=estimated_duration
        )

        self.work_orders[work_order_id] = work_order
        logger.info("Created work order %s for equipment %s (priority=%d)",
                   work_order_id, equipment_id, priority)

        return work_order

    def submit_work_order(self, work_order_id: str) -> bool:
        """Submit work order to CMMS."""
        work_order = self.work_orders.get(work_order_id)
        if not work_order:
            logger.error("Work order not found: %s", work_order_id)
            return False

        if not self.config.cmms_enabled:
            logger.warning("CMMS integration disabled, marking as submitted locally")
            work_order.status = CMMSWorkOrderStatus.SUBMITTED
            return True

        # In production, this would make API call to CMMS
        # For now, simulate submission
        try:
            work_order.status = CMMSWorkOrderStatus.SUBMITTED
            logger.info("Submitted work order %s to CMMS", work_order_id)
            return True
        except Exception as e:
            logger.error("Failed to submit work order %s: %s", work_order_id, str(e))
            return False

    def update_work_order_status(
        self,
        work_order_id: str,
        status: CMMSWorkOrderStatus,
        notes: Optional[str] = None
    ) -> bool:
        """Update work order status."""
        work_order = self.work_orders.get(work_order_id)
        if not work_order:
            logger.error("Work order not found: %s", work_order_id)
            return False

        work_order.status = status
        if notes:
            work_order.notes = notes

        if status == CMMSWorkOrderStatus.COMPLETED:
            work_order.completed_date = datetime.now()

        logger.info("Updated work order %s status to %s", work_order_id, status.value)
        return True

    def get_pending_work_orders(
        self,
        equipment_id: Optional[str] = None
    ) -> List[CMMSWorkOrder]:
        """Get pending work orders, optionally filtered by equipment."""
        pending_statuses = {
            CMMSWorkOrderStatus.DRAFT,
            CMMSWorkOrderStatus.SUBMITTED,
            CMMSWorkOrderStatus.APPROVED,
            CMMSWorkOrderStatus.SCHEDULED,
            CMMSWorkOrderStatus.IN_PROGRESS
        }

        pending = [
            wo for wo in self.work_orders.values()
            if wo.status in pending_statuses
        ]

        if equipment_id:
            pending = [wo for wo in pending if wo.equipment_id == equipment_id]

        return sorted(pending, key=lambda wo: wo.priority)

    def get_overdue_work_orders(self) -> List[CMMSWorkOrder]:
        """Get work orders that are past their scheduled date."""
        now = datetime.now()
        overdue = [
            wo for wo in self.work_orders.values()
            if wo.scheduled_date
            and wo.scheduled_date < now
            and wo.status not in (CMMSWorkOrderStatus.COMPLETED, CMMSWorkOrderStatus.CANCELLED)
        ]
        return sorted(overdue, key=lambda wo: wo.scheduled_date)


# =============================================================================
# Calibration Scheduler
# =============================================================================

class CalibrationScheduler:
    """
    Schedules calibration activities based on equipment status.

    Provides scheduling recommendations and manages calibration
    windows to minimize operational impact.

    Attributes:
        config: Calibration configuration
        equipment_intervals: Custom intervals by equipment ID
    """

    def __init__(self, config: CalibrationConfig):
        """Initialize calibration scheduler."""
        self.config = config
        self.equipment_intervals: Dict[str, int] = {}  # equipment_id -> days
        self._type_intervals: Dict[EquipmentType, int] = {
            EquipmentType.VALVE: 180,
            EquipmentType.PUMP: 180,
            EquipmentType.SENSOR: 90,
            EquipmentType.FLOW_METER: 180,
            EquipmentType.PH_SENSOR: 30,
            EquipmentType.CONDUCTIVITY_SENSOR: 60,
            EquipmentType.TURBIDITY_SENSOR: 60,
            EquipmentType.DOSING_PUMP: 120,
            EquipmentType.BLOWDOWN_VALVE: 90,
        }
        logger.info("CalibrationScheduler initialized with default interval: %d days",
                   config.default_calibration_interval_days)

    def get_calibration_interval(
        self,
        equipment_id: str,
        equipment_type: EquipmentType
    ) -> int:
        """Get calibration interval in days for equipment."""
        # Check for custom interval
        if equipment_id in self.equipment_intervals:
            return self.equipment_intervals[equipment_id]

        # Check type-specific interval
        if equipment_type in self._type_intervals:
            return self._type_intervals[equipment_type]

        # Default interval
        return self.config.default_calibration_interval_days

    def set_equipment_interval(self, equipment_id: str, interval_days: int) -> None:
        """Set custom calibration interval for equipment."""
        self.equipment_intervals[equipment_id] = interval_days
        logger.info("Set calibration interval for %s: %d days", equipment_id, interval_days)

    def calculate_next_due_date(
        self,
        last_calibration: datetime,
        equipment_id: str,
        equipment_type: EquipmentType
    ) -> datetime:
        """Calculate next calibration due date."""
        interval = self.get_calibration_interval(equipment_id, equipment_type)
        return last_calibration + timedelta(days=interval)

    def get_upcoming_calibrations(
        self,
        equipment_states: Dict[str, EquipmentCalibrationState],
        days_ahead: int = 30
    ) -> List[Tuple[str, datetime, int]]:
        """
        Get upcoming calibrations within the specified period.

        Args:
            equipment_states: Current equipment calibration states
            days_ahead: Number of days to look ahead

        Returns:
            List of (equipment_id, due_date, days_until_due)
        """
        now = datetime.now()
        cutoff = now + timedelta(days=days_ahead)

        upcoming = []
        for eq_id, state in equipment_states.items():
            if state.next_calibration_due and state.next_calibration_due <= cutoff:
                days_until = (state.next_calibration_due - now).days
                upcoming.append((eq_id, state.next_calibration_due, days_until))

        return sorted(upcoming, key=lambda x: x[1])

    def suggest_calibration_window(
        self,
        equipment_id: str,
        preferred_days: Optional[List[int]] = None,
        preferred_hours: Optional[Tuple[int, int]] = None
    ) -> datetime:
        """
        Suggest optimal calibration window.

        Args:
            equipment_id: Equipment identifier
            preferred_days: Preferred days of week (0=Monday, 6=Sunday)
            preferred_hours: Preferred hour range (start, end)

        Returns:
            Suggested datetime for calibration
        """
        # Default preferences: Tuesday-Thursday, 9AM-3PM
        if preferred_days is None:
            preferred_days = [1, 2, 3]  # Tue, Wed, Thu
        if preferred_hours is None:
            preferred_hours = (9, 15)  # 9AM to 3PM

        # Start from tomorrow
        suggested = datetime.now() + timedelta(days=1)

        # Find next preferred day
        max_iterations = 14
        for _ in range(max_iterations):
            if suggested.weekday() in preferred_days:
                break
            suggested += timedelta(days=1)

        # Set to preferred hour
        suggested = suggested.replace(
            hour=preferred_hours[0],
            minute=0,
            second=0,
            microsecond=0
        )

        logger.debug("Suggested calibration window for %s: %s",
                    equipment_id, suggested.isoformat())

        return suggested


# =============================================================================
# Calibration Tracker
# =============================================================================

class CalibrationTracker:
    """
    Core tracking class for equipment calibration.

    Provides comprehensive calibration management including:
    - Recording calibration events
    - Tracking calibration status
    - Detecting calibration drift
    - CMMS integration for work orders
    - Scheduling calibration activities

    Attributes:
        config: Calibration configuration
        drift_detector: Drift detection engine
        cmms_integrator: CMMS integration handler
        scheduler: Calibration scheduler
        equipment_states: Current state of all equipment

    Example:
        >>> config = CalibrationConfig(drift_threshold_percent=5.0)
        >>> tracker = CalibrationTracker(config)
        >>> tracker.register_equipment("valve_001", EquipmentType.BLOWDOWN_VALVE)
        >>> tracker.record_calibration("valve_001", calibration_curve)
        >>> drift = tracker.check_drift("valve_001", expected=50.0, actual=52.5)
        >>> if drift.needs_recalibration:
        ...     tracker.create_calibration_work_order("valve_001")
    """

    def __init__(self, config: Optional[CalibrationConfig] = None):
        """
        Initialize CalibrationTracker.

        Args:
            config: Calibration configuration (uses defaults if not provided)
        """
        self.config = config or CalibrationConfig()
        self.drift_detector = DriftDetector(self.config)
        self.cmms_integrator = CMMSIntegrator(self.config)
        self.scheduler = CalibrationScheduler(self.config)
        self.equipment_states: Dict[str, EquipmentCalibrationState] = {}
        self._calibration_counter = 0

        logger.info("CalibrationTracker initialized with drift threshold: %.1f%%",
                   self.config.drift_threshold_percent)

    def _generate_calibration_id(self) -> str:
        """Generate unique calibration ID."""
        self._calibration_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"CAL-{timestamp}-{self._calibration_counter:04d}"

    def register_equipment(
        self,
        equipment_id: str,
        equipment_type: EquipmentType,
        calibration_interval_days: Optional[int] = None
    ) -> EquipmentCalibrationState:
        """
        Register equipment for calibration tracking.

        Args:
            equipment_id: Unique equipment identifier
            equipment_type: Type of equipment
            calibration_interval_days: Custom calibration interval

        Returns:
            Initial calibration state for equipment
        """
        if calibration_interval_days:
            self.scheduler.set_equipment_interval(equipment_id, calibration_interval_days)

        state = EquipmentCalibrationState(
            equipment_id=equipment_id,
            equipment_type=equipment_type,
            status=CalibrationStatus.UNKNOWN
        )

        self.equipment_states[equipment_id] = state
        logger.info("Registered equipment for calibration tracking: %s (%s)",
                   equipment_id, equipment_type.value)

        return state

    def record_calibration(
        self,
        equipment_id: str,
        calibration_curve: CalibrationCurve,
        technician_id: Optional[str] = None,
        standard_used: Optional[str] = None,
        certificate_number: Optional[str] = None,
        notes: Optional[str] = None
    ) -> CalibrationRecord:
        """
        Record a calibration event.

        Args:
            equipment_id: Equipment identifier
            calibration_curve: Calibration curve from calibration
            technician_id: ID of technician who performed calibration
            standard_used: Calibration standard used
            certificate_number: Calibration certificate number
            notes: Calibration notes

        Returns:
            CalibrationRecord for the event
        """
        state = self.equipment_states.get(equipment_id)
        if not state:
            raise ValueError(f"Equipment not registered: {equipment_id}")

        calibration_id = self._generate_calibration_id()
        calibration_date = datetime.now()

        # Calculate next due date
        next_due = self.scheduler.calculate_next_due_date(
            calibration_date,
            equipment_id,
            state.equipment_type
        )

        record = CalibrationRecord(
            calibration_id=calibration_id,
            equipment_id=equipment_id,
            equipment_type=state.equipment_type,
            calibration_date=calibration_date,
            calibration_curve=calibration_curve,
            technician_id=technician_id,
            standard_used=standard_used,
            certificate_number=certificate_number,
            next_calibration_due=next_due,
            notes=notes
        )

        # Update equipment state
        state.last_calibration = record
        state.next_calibration_due = next_due
        state.status = CalibrationStatus.CALIBRATED
        state.current_drift = None
        state.calibration_count += 1
        state.days_until_due = (next_due - calibration_date).days

        # Clear drift history (fresh start after calibration)
        self.drift_detector.clear_history(equipment_id)

        # Complete any pending work orders
        pending_orders = self.cmms_integrator.get_pending_work_orders(equipment_id)
        for wo in pending_orders:
            self.cmms_integrator.update_work_order_status(
                wo.work_order_id,
                CMMSWorkOrderStatus.COMPLETED,
                f"Completed via calibration {calibration_id}"
            )

        logger.info("Recorded calibration %s for %s (next due: %s)",
                   calibration_id, equipment_id, next_due.date().isoformat())

        return record

    def add_drift_measurement(
        self,
        equipment_id: str,
        expected_value: float,
        actual_value: float,
        operating_conditions: Optional[Dict[str, float]] = None
    ) -> DriftMeasurement:
        """
        Add a measurement for drift tracking.

        Args:
            equipment_id: Equipment identifier
            expected_value: Expected value from calibration
            actual_value: Actual measured value
            operating_conditions: Current operating conditions

        Returns:
            DriftMeasurement object
        """
        state = self.equipment_states.get(equipment_id)
        if not state:
            raise ValueError(f"Equipment not registered: {equipment_id}")

        measurement = self.drift_detector.add_measurement(
            equipment_id, expected_value, actual_value, operating_conditions
        )

        # Keep recent history in state
        state.measurement_history.append(measurement)
        if len(state.measurement_history) > self.config.drift_detection_window_samples:
            state.measurement_history = state.measurement_history[-self.config.drift_detection_window_samples:]

        return measurement

    def check_drift(
        self,
        equipment_id: str,
        expected_value: Optional[float] = None,
        actual_value: Optional[float] = None
    ) -> DriftAnalysis:
        """
        Check for calibration drift.

        If expected_value and actual_value are provided, adds measurement first.

        Args:
            equipment_id: Equipment identifier
            expected_value: Optional expected value (adds measurement if provided)
            actual_value: Optional actual value (adds measurement if provided)

        Returns:
            DriftAnalysis with current drift status
        """
        state = self.equipment_states.get(equipment_id)
        if not state:
            raise ValueError(f"Equipment not registered: {equipment_id}")

        # Add measurement if provided
        if expected_value is not None and actual_value is not None:
            self.add_drift_measurement(equipment_id, expected_value, actual_value)

        # Analyze drift
        analysis = self.drift_detector.analyze_drift(equipment_id)

        # Update state
        state.current_drift = analysis
        if analysis.needs_recalibration:
            state.status = CalibrationStatus.DRIFT_DETECTED

            # Auto-schedule if enabled
            if self.config.auto_schedule_calibration and not state.pending_work_order:
                self.create_calibration_work_order(equipment_id, analysis)

        return analysis

    def create_calibration_work_order(
        self,
        equipment_id: str,
        drift_analysis: Optional[DriftAnalysis] = None,
        priority: int = 2
    ) -> CMMSWorkOrder:
        """
        Create a CMMS work order for calibration.

        Args:
            equipment_id: Equipment identifier
            drift_analysis: Optional drift analysis that triggered the order
            priority: Work order priority (1-5)

        Returns:
            Created work order
        """
        state = self.equipment_states.get(equipment_id)
        if not state:
            raise ValueError(f"Equipment not registered: {equipment_id}")

        # Determine reason
        if drift_analysis and drift_analysis.needs_recalibration:
            reason = f"Drift detected: {drift_analysis.drift_severity.value} ({drift_analysis.mean_deviation_percent:.1f}%)"
        elif state.status == CalibrationStatus.OVERDUE:
            reason = "Scheduled calibration overdue"
        elif state.status == CalibrationStatus.NEEDS_CALIBRATION:
            reason = "Scheduled calibration due"
        else:
            reason = "Routine calibration"

        # Suggest scheduling window
        scheduled_date = self.scheduler.suggest_calibration_window(equipment_id)

        work_order = self.cmms_integrator.create_work_order(
            equipment_id=equipment_id,
            equipment_type=state.equipment_type,
            reason=reason,
            priority=priority,
            drift_analysis=drift_analysis,
            scheduled_date=scheduled_date
        )

        # Update state
        state.pending_work_order = work_order

        return work_order

    def get_equipment_state(self, equipment_id: str) -> Optional[EquipmentCalibrationState]:
        """Get current calibration state for equipment."""
        return self.equipment_states.get(equipment_id)

    def get_all_equipment_states(self) -> Dict[str, EquipmentCalibrationState]:
        """Get calibration states for all equipment."""
        # Update days_until_due
        now = datetime.now()
        for state in self.equipment_states.values():
            if state.next_calibration_due:
                state.days_until_due = (state.next_calibration_due - now).days
                if state.days_until_due < 0:
                    state.status = CalibrationStatus.OVERDUE
                elif state.days_until_due < 7:
                    state.status = CalibrationStatus.NEEDS_CALIBRATION

        return self.equipment_states.copy()

    def get_calibration_summary(self) -> Dict[str, Any]:
        """
        Get summary of calibration status across all equipment.

        Returns:
            Dictionary with calibration summary statistics
        """
        states = self.get_all_equipment_states()

        if not states:
            return {
                "total_equipment": 0,
                "status_counts": {},
                "upcoming_calibrations": [],
                "overdue_count": 0,
                "provenance_hash": hashlib.sha256(b"empty").hexdigest()[:16]
            }

        status_counts = {}
        for state in states.values():
            status = state.status.value
            status_counts[status] = status_counts.get(status, 0) + 1

        upcoming = self.scheduler.get_upcoming_calibrations(states, days_ahead=30)
        overdue_orders = self.cmms_integrator.get_overdue_work_orders()

        summary = {
            "total_equipment": len(states),
            "status_counts": status_counts,
            "calibrated_count": status_counts.get(CalibrationStatus.CALIBRATED.value, 0),
            "drift_detected_count": status_counts.get(CalibrationStatus.DRIFT_DETECTED.value, 0),
            "overdue_count": status_counts.get(CalibrationStatus.OVERDUE.value, 0),
            "needs_calibration_count": status_counts.get(CalibrationStatus.NEEDS_CALIBRATION.value, 0),
            "upcoming_calibrations_30d": len(upcoming),
            "pending_work_orders": len(self.cmms_integrator.get_pending_work_orders()),
            "overdue_work_orders": len(overdue_orders),
            "timestamp": datetime.now().isoformat()
        }

        provenance_str = f"{summary}"
        summary["provenance_hash"] = hashlib.sha256(provenance_str.encode()).hexdigest()[:16]

        return summary

    def apply_calibration_curve(
        self,
        equipment_id: str,
        raw_value: float
    ) -> Tuple[float, float]:
        """
        Apply calibration curve to convert raw value to calibrated value.

        Args:
            equipment_id: Equipment identifier
            raw_value: Raw measurement value

        Returns:
            Tuple of (calibrated_value, uncertainty)
        """
        state = self.equipment_states.get(equipment_id)
        if not state or not state.last_calibration:
            raise ValueError(f"No calibration available for: {equipment_id}")

        curve = state.last_calibration.calibration_curve

        # Apply curve based on type
        if curve.curve_type == "linear" and len(curve.coefficients) >= 2:
            slope, intercept = curve.coefficients[0], curve.coefficients[1]
            calibrated = slope * raw_value + intercept
        elif curve.curve_type == "polynomial":
            calibrated = sum(
                coef * (raw_value ** i)
                for i, coef in enumerate(curve.coefficients)
            )
        else:
            # Identity transform if unknown
            calibrated = raw_value

        # Estimate uncertainty based on R-squared and point uncertainty
        base_uncertainty = 0.01 * abs(calibrated)  # 1% base
        r_squared_factor = 1.0 / max(curve.r_squared, 0.5)
        uncertainty = base_uncertainty * r_squared_factor

        return calibrated, uncertainty


# =============================================================================
# Factory Functions
# =============================================================================

def create_default_tracker() -> CalibrationTracker:
    """Create a CalibrationTracker with default configuration."""
    return CalibrationTracker(CalibrationConfig())


def create_strict_tracker() -> CalibrationTracker:
    """Create a CalibrationTracker with strict drift thresholds."""
    config = CalibrationConfig(
        drift_threshold_percent=2.0,
        severe_drift_threshold_percent=5.0,
        default_calibration_interval_days=60,
        drift_detection_window_samples=30,
        auto_schedule_calibration=True
    )
    return CalibrationTracker(config)


def create_relaxed_tracker() -> CalibrationTracker:
    """Create a CalibrationTracker with relaxed drift thresholds."""
    config = CalibrationConfig(
        drift_threshold_percent=10.0,
        severe_drift_threshold_percent=25.0,
        default_calibration_interval_days=180,
        drift_detection_window_samples=10,
        auto_schedule_calibration=False
    )
    return CalibrationTracker(config)


def create_linear_calibration_curve(
    reference_values: List[float],
    measured_values: List[float],
    uncertainties: Optional[List[float]] = None
) -> CalibrationCurve:
    """
    Create a linear calibration curve from reference and measured values.

    Args:
        reference_values: Reference/standard values
        measured_values: Measured values from equipment
        uncertainties: Optional measurement uncertainties

    Returns:
        CalibrationCurve with linear fit
    """
    if len(reference_values) != len(measured_values):
        raise ValueError("Reference and measured value lists must have same length")

    if len(reference_values) < 2:
        raise ValueError("At least 2 calibration points required")

    n = len(reference_values)
    sum_x = sum(measured_values)
    sum_y = sum(reference_values)
    sum_xy = sum(x * y for x, y in zip(measured_values, reference_values))
    sum_xx = sum(x * x for x in measured_values)

    denominator = n * sum_xx - sum_x * sum_x
    if abs(denominator) < 1e-10:
        raise ValueError("Cannot compute calibration curve: singular matrix")

    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n

    # Calculate R-squared
    mean_y = sum_y / n
    ss_tot = sum((y - mean_y) ** 2 for y in reference_values)
    ss_res = sum(
        (y - (slope * x + intercept)) ** 2
        for x, y in zip(measured_values, reference_values)
    )
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Create calibration points
    points = []
    for i in range(n):
        point = CalibrationPoint(
            reference_value=reference_values[i],
            measured_value=measured_values[i],
            uncertainty=uncertainties[i] if uncertainties else None
        )
        points.append(point)

    return CalibrationCurve(
        curve_type="linear",
        coefficients=[slope, intercept],
        r_squared=r_squared,
        valid_range=(min(measured_values), max(measured_values)),
        calibration_points=points
    )
