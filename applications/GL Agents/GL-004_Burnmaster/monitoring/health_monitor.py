"""
GL-004 BURNMASTER Health Monitor Module

This module provides comprehensive system health monitoring for combustion
optimization operations, including data quality assessment, sensor health
checks, analyzer calibration verification, model health tracking, and
control loop performance monitoring.

Example:
    >>> monitor = SystemHealthMonitor()
    >>> quality_report = monitor.check_data_quality(combustion_data)
    >>> dashboard = monitor.generate_system_health_dashboard()
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import hashlib
import logging
import statistics
import uuid

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class HealthStatus(str, Enum):
    """Health status levels."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class CalibrationState(str, Enum):
    """Analyzer calibration state."""
    CALIBRATED = "CALIBRATED"
    DUE_SOON = "DUE_SOON"
    OVERDUE = "OVERDUE"
    FAILED = "FAILED"
    IN_PROGRESS = "IN_PROGRESS"


class DataQualityLevel(str, Enum):
    """Data quality assessment level."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    INVALID = "INVALID"


# =============================================================================
# DATA MODELS
# =============================================================================

class CombustionData(BaseModel):
    """Combustion data for quality analysis."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Data timestamp"
    )
    unit_id: str = Field(..., description="Unit identifier")

    # Raw sensor readings
    o2_readings: List[float] = Field(default_factory=list, description="O2 sensor readings")
    co_readings: List[float] = Field(default_factory=list, description="CO sensor readings")
    nox_readings: List[float] = Field(default_factory=list, description="NOx sensor readings")
    temp_readings: List[float] = Field(default_factory=list, description="Temperature readings")

    # Metadata
    sample_rate_hz: float = Field(default=1.0, ge=0.1, description="Sample rate in Hz")
    data_source: str = Field(default="ANALYZER", description="Data source identifier")
    sensor_ids: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of measurement to sensor ID"
    )


class DataQualityReport(BaseModel):
    """Report on combustion data quality."""

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Report identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report timestamp"
    )
    unit_id: str = Field(..., description="Unit identifier")

    # Overall assessment
    overall_quality: DataQualityLevel = Field(..., description="Overall quality level")
    quality_score: float = Field(..., ge=0.0, le=100.0, description="Quality score 0-100")

    # Completeness
    completeness_percent: float = Field(
        ..., ge=0.0, le=100.0, description="Data completeness percentage"
    )
    missing_fields: List[str] = Field(default_factory=list, description="Missing data fields")

    # Validity
    validity_percent: float = Field(
        ..., ge=0.0, le=100.0, description="Data validity percentage"
    )
    invalid_readings: Dict[str, int] = Field(
        default_factory=dict, description="Count of invalid readings per field"
    )

    # Consistency
    consistency_score: float = Field(..., ge=0.0, le=100.0, description="Consistency score")
    outliers_detected: Dict[str, int] = Field(
        default_factory=dict, description="Outliers detected per field"
    )

    # Timeliness
    data_latency_ms: float = Field(default=0.0, ge=0.0, description="Data latency")
    stale_data: bool = Field(default=False, description="Data is stale")

    # Issues
    issues: List[str] = Field(default_factory=list, description="Quality issues found")
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )

    # Provenance
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")


class SensorHealthReport(BaseModel):
    """Report on sensor health status."""

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Report identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report timestamp"
    )

    # Overall
    overall_status: HealthStatus = Field(..., description="Overall sensor health")
    healthy_count: int = Field(default=0, ge=0, description="Healthy sensor count")
    degraded_count: int = Field(default=0, ge=0, description="Degraded sensor count")
    failed_count: int = Field(default=0, ge=0, description="Failed sensor count")

    # Per-sensor details
    sensor_statuses: Dict[str, HealthStatus] = Field(
        default_factory=dict, description="Status per sensor ID"
    )
    sensor_details: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Detailed info per sensor"
    )

    # Issues
    issues: List[str] = Field(default_factory=list, description="Sensor issues")
    maintenance_required: List[str] = Field(
        default_factory=list, description="Sensors needing maintenance"
    )


class CalibrationStatus(BaseModel):
    """Analyzer calibration status."""

    analyzer_id: str = Field(..., description="Analyzer identifier")
    analyzer_type: str = Field(..., description="Analyzer type (O2, CO, NOx, etc.)")

    # Calibration state
    state: CalibrationState = Field(..., description="Calibration state")
    last_calibration: Optional[datetime] = Field(None, description="Last calibration time")
    next_calibration_due: Optional[datetime] = Field(None, description="Next calibration due")
    days_until_due: Optional[int] = Field(None, description="Days until calibration due")

    # Quality
    span_drift_percent: float = Field(
        default=0.0, description="Span drift percentage"
    )
    zero_drift_percent: float = Field(
        default=0.0, description="Zero drift percentage"
    )
    accuracy_percent: float = Field(
        default=100.0, ge=0.0, le=100.0, description="Current accuracy"
    )

    # Standards
    reference_gas_used: Optional[str] = Field(None, description="Reference gas used")
    certification_valid: bool = Field(default=True, description="Certification valid")

    # Recommendations
    requires_attention: bool = Field(default=False, description="Requires attention")
    recommended_action: Optional[str] = Field(None, description="Recommended action")


class ModelHealthReport(BaseModel):
    """Report on ML model health status."""

    model_id: str = Field(..., description="Model identifier")
    model_type: str = Field(..., description="Model type")
    model_version: str = Field(..., description="Model version")

    # Health
    status: HealthStatus = Field(..., description="Model health status")
    last_prediction_time: Optional[datetime] = Field(
        None, description="Last prediction timestamp"
    )
    predictions_24h: int = Field(default=0, ge=0, description="Predictions in last 24h")

    # Performance
    accuracy_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Accuracy score")
    latency_p50_ms: float = Field(default=0.0, ge=0.0, description="P50 latency")
    latency_p99_ms: float = Field(default=0.0, ge=0.0, description="P99 latency")
    error_rate_percent: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Error rate"
    )

    # Drift
    data_drift_detected: bool = Field(default=False, description="Data drift detected")
    model_drift_detected: bool = Field(default=False, description="Model drift detected")
    drift_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Drift score")

    # Resources
    memory_usage_mb: float = Field(default=0.0, ge=0.0, description="Memory usage MB")
    inference_queue_depth: int = Field(default=0, ge=0, description="Inference queue depth")

    # Issues
    issues: List[str] = Field(default_factory=list, description="Model issues")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")


class LoopHealthReport(BaseModel):
    """Report on control loop health."""

    loop_id: str = Field(..., description="Control loop identifier")
    loop_type: str = Field(..., description="Loop type (PID, cascade, etc.)")

    # Health
    status: HealthStatus = Field(..., description="Loop health status")
    mode: str = Field(default="AUTO", description="Loop mode (AUTO, MANUAL, CASCADE)")

    # Performance
    setpoint: float = Field(..., description="Current setpoint")
    process_value: float = Field(..., description="Current process value")
    output_percent: float = Field(..., ge=0.0, le=100.0, description="Output percentage")
    error: float = Field(..., description="Current error (PV - SP)")

    # Stability
    oscillation_detected: bool = Field(default=False, description="Oscillation detected")
    oscillation_frequency_hz: Optional[float] = Field(None, description="Oscillation frequency")
    stability_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Stability score 0-1"
    )

    # Tuning
    kp: float = Field(default=1.0, description="Proportional gain")
    ki: float = Field(default=0.0, description="Integral gain")
    kd: float = Field(default=0.0, description="Derivative gain")
    tuning_quality: str = Field(default="OPTIMAL", description="Tuning quality")

    # Performance metrics
    rise_time_s: Optional[float] = Field(None, description="Rise time seconds")
    settling_time_s: Optional[float] = Field(None, description="Settling time seconds")
    overshoot_percent: Optional[float] = Field(None, description="Overshoot percentage")
    iae: Optional[float] = Field(None, description="Integral absolute error")

    # Issues
    issues: List[str] = Field(default_factory=list, description="Loop issues")


class HealthDashboard(BaseModel):
    """System-wide health dashboard."""

    dashboard_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Dashboard identifier"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Generation timestamp"
    )

    # Overall system health
    overall_status: HealthStatus = Field(..., description="Overall system health")
    health_score: float = Field(..., ge=0.0, le=100.0, description="Health score 0-100")

    # Component summaries
    data_quality_status: HealthStatus = Field(..., description="Data quality status")
    sensor_health_status: HealthStatus = Field(..., description="Sensor health status")
    model_health_status: HealthStatus = Field(..., description="Model health status")
    control_loop_status: HealthStatus = Field(..., description="Control loop status")

    # Counts
    total_sensors: int = Field(default=0, ge=0, description="Total sensors")
    healthy_sensors: int = Field(default=0, ge=0, description="Healthy sensors")
    total_models: int = Field(default=0, ge=0, description="Total models")
    healthy_models: int = Field(default=0, ge=0, description="Healthy models")
    total_loops: int = Field(default=0, ge=0, description="Total control loops")
    healthy_loops: int = Field(default=0, ge=0, description="Healthy control loops")

    # Critical issues
    critical_issues: List[str] = Field(default_factory=list, description="Critical issues")
    warnings: List[str] = Field(default_factory=list, description="Warnings")

    # Detailed reports
    data_quality_reports: List[DataQualityReport] = Field(
        default_factory=list, description="Data quality reports"
    )
    sensor_health_reports: List[SensorHealthReport] = Field(
        default_factory=list, description="Sensor health reports"
    )
    calibration_statuses: List[CalibrationStatus] = Field(
        default_factory=list, description="Calibration statuses"
    )
    model_health_reports: List[ModelHealthReport] = Field(
        default_factory=list, description="Model health reports"
    )
    loop_health_reports: List[LoopHealthReport] = Field(
        default_factory=list, description="Loop health reports"
    )

    # Trends
    health_trend: str = Field(
        default="STABLE", description="Health trend (IMPROVING, STABLE, DEGRADING)"
    )

    # Provenance
    provenance_hash: str = Field(default="", description="SHA-256 audit hash")


# =============================================================================
# SYSTEM HEALTH MONITOR
# =============================================================================

class SystemHealthMonitor:
    """
    Comprehensive system health monitoring for combustion optimization.

    Monitors data quality, sensor health, analyzer calibration, ML model
    health, and control loop performance to ensure reliable operations.

    Attributes:
        sensor_registry: Registered sensors being monitored
        model_registry: Registered ML models being monitored
        loop_registry: Registered control loops being monitored

    Example:
        >>> monitor = SystemHealthMonitor()
        >>> quality = monitor.check_data_quality(combustion_data)
        >>> dashboard = monitor.generate_system_health_dashboard()
    """

    # Quality thresholds
    EXCELLENT_THRESHOLD = 95.0
    GOOD_THRESHOLD = 85.0
    ACCEPTABLE_THRESHOLD = 70.0
    POOR_THRESHOLD = 50.0

    # Stale data threshold (seconds)
    STALE_DATA_THRESHOLD_S = 60.0

    # Outlier detection z-score threshold
    OUTLIER_Z_THRESHOLD = 3.0

    def __init__(self):
        """Initialize the SystemHealthMonitor."""
        self._sensor_registry: Dict[str, Dict[str, Any]] = {}
        self._model_registry: Dict[str, Dict[str, Any]] = {}
        self._loop_registry: Dict[str, Dict[str, Any]] = {}
        self._calibration_records: Dict[str, CalibrationStatus] = {}
        self._health_history: List[HealthDashboard] = []

        logger.info("SystemHealthMonitor initialized")

    def check_data_quality(self, data: CombustionData) -> DataQualityReport:
        """
        Analyze combustion data quality.

        Args:
            data: Combustion data to analyze

        Returns:
            DataQualityReport with quality assessment
        """
        issues = []
        recommendations = []

        # Calculate completeness
        expected_fields = ['o2_readings', 'co_readings', 'nox_readings', 'temp_readings']
        present_fields = [f for f in expected_fields if getattr(data, f, None)]
        completeness = (len(present_fields) / len(expected_fields)) * 100
        missing = [f for f in expected_fields if f not in present_fields]

        if missing:
            issues.append(f"Missing data fields: {missing}")
            recommendations.append("Verify sensor connections for missing data")

        # Calculate validity
        invalid_counts: Dict[str, int] = {}
        total_readings = 0
        valid_readings = 0

        for field in expected_fields:
            readings = getattr(data, field, [])
            if readings:
                total_readings += len(readings)
                invalid = sum(1 for r in readings if r is None or r < 0)
                invalid_counts[field] = invalid
                valid_readings += len(readings) - invalid

        validity = (valid_readings / total_readings * 100) if total_readings > 0 else 0

        if validity < 95:
            issues.append(f"Data validity below threshold: {validity:.1f}%")
            recommendations.append("Check sensor calibration and signal quality")

        # Calculate consistency (detect outliers)
        outliers: Dict[str, int] = {}
        consistency_scores = []

        for field in expected_fields:
            readings = getattr(data, field, [])
            if len(readings) >= 3:
                outlier_count = self._detect_outliers(readings)
                outliers[field] = outlier_count
                field_consistency = max(0, 100 - (outlier_count / len(readings) * 100))
                consistency_scores.append(field_consistency)

        consistency = statistics.mean(consistency_scores) if consistency_scores else 100.0

        if outliers and sum(outliers.values()) > 0:
            issues.append(f"Outliers detected: {outliers}")
            recommendations.append("Investigate anomalous readings")

        # Check timeliness
        now = datetime.now(timezone.utc)
        data_age_s = (now - data.timestamp).total_seconds()
        stale = data_age_s > self.STALE_DATA_THRESHOLD_S

        if stale:
            issues.append(f"Stale data: {data_age_s:.1f}s old")
            recommendations.append("Check data pipeline latency")

        # Calculate overall quality score
        quality_score = (
            completeness * 0.25 +
            validity * 0.35 +
            consistency * 0.30 +
            (100 if not stale else 50) * 0.10
        )

        # Determine quality level
        if quality_score >= self.EXCELLENT_THRESHOLD:
            quality_level = DataQualityLevel.EXCELLENT
        elif quality_score >= self.GOOD_THRESHOLD:
            quality_level = DataQualityLevel.GOOD
        elif quality_score >= self.ACCEPTABLE_THRESHOLD:
            quality_level = DataQualityLevel.ACCEPTABLE
        elif quality_score >= self.POOR_THRESHOLD:
            quality_level = DataQualityLevel.POOR
        else:
            quality_level = DataQualityLevel.INVALID

        report = DataQualityReport(
            unit_id=data.unit_id,
            overall_quality=quality_level,
            quality_score=quality_score,
            completeness_percent=completeness,
            missing_fields=missing,
            validity_percent=validity,
            invalid_readings=invalid_counts,
            consistency_score=consistency,
            outliers_detected=outliers,
            data_latency_ms=data_age_s * 1000,
            stale_data=stale,
            issues=issues,
            recommendations=recommendations,
        )

        # Compute provenance hash
        report.provenance_hash = self._compute_provenance(report)

        logger.info(
            f"Data quality check for {data.unit_id}: "
            f"score={quality_score:.1f}, level={quality_level.value}"
        )

        return report

    def _detect_outliers(self, readings: List[float]) -> int:
        """Detect outliers using z-score method."""
        if len(readings) < 3:
            return 0

        valid_readings = [r for r in readings if r is not None]
        if len(valid_readings) < 3:
            return 0

        mean = statistics.mean(valid_readings)
        stdev = statistics.stdev(valid_readings) if len(valid_readings) > 1 else 0

        if stdev == 0:
            return 0

        outliers = sum(
            1 for r in valid_readings
            if abs(r - mean) / stdev > self.OUTLIER_Z_THRESHOLD
        )
        return outliers

    def check_sensor_health(self, sensors: List[str]) -> SensorHealthReport:
        """
        Check health status of specified sensors.

        Args:
            sensors: List of sensor IDs to check

        Returns:
            SensorHealthReport with health assessment
        """
        statuses: Dict[str, HealthStatus] = {}
        details: Dict[str, Dict[str, Any]] = {}
        issues = []
        maintenance_needed = []

        healthy = 0
        degraded = 0
        failed = 0

        for sensor_id in sensors:
            # Check sensor registry for known status
            sensor_info = self._sensor_registry.get(sensor_id, {})

            # Determine health based on available info
            if sensor_info.get('last_reading_time'):
                last_reading = sensor_info['last_reading_time']
                age_s = (datetime.now(timezone.utc) - last_reading).total_seconds()

                if age_s > 300:
                    status = HealthStatus.CRITICAL
                    failed += 1
                    issues.append(f"Sensor {sensor_id}: No readings for {age_s:.0f}s")
                    maintenance_needed.append(sensor_id)
                elif age_s > 60:
                    status = HealthStatus.DEGRADED
                    degraded += 1
                    issues.append(f"Sensor {sensor_id}: Delayed readings ({age_s:.0f}s)")
                else:
                    status = HealthStatus.HEALTHY
                    healthy += 1
            else:
                # Unknown sensor - assume needs investigation
                status = HealthStatus.UNKNOWN
                issues.append(f"Sensor {sensor_id}: No health data available")

            statuses[sensor_id] = status
            details[sensor_id] = {
                'status': status.value,
                'last_reading': sensor_info.get('last_reading_time'),
                'reading_count_24h': sensor_info.get('reading_count_24h', 0),
            }

        # Determine overall status
        if failed > 0:
            overall = HealthStatus.CRITICAL
        elif degraded > 0:
            overall = HealthStatus.DEGRADED
        elif healthy == len(sensors):
            overall = HealthStatus.HEALTHY
        else:
            overall = HealthStatus.UNKNOWN

        report = SensorHealthReport(
            overall_status=overall,
            healthy_count=healthy,
            degraded_count=degraded,
            failed_count=failed,
            sensor_statuses=statuses,
            sensor_details=details,
            issues=issues,
            maintenance_required=maintenance_needed,
        )

        logger.info(
            f"Sensor health check: {healthy} healthy, "
            f"{degraded} degraded, {failed} failed"
        )

        return report

    def check_analyzer_calibration(self, analyzer: str) -> CalibrationStatus:
        """
        Check calibration status of an analyzer.

        Args:
            analyzer: Analyzer identifier

        Returns:
            CalibrationStatus with calibration details
        """
        # Get stored calibration record or create default
        if analyzer in self._calibration_records:
            status = self._calibration_records[analyzer]
        else:
            # Create default status for unknown analyzer
            status = CalibrationStatus(
                analyzer_id=analyzer,
                analyzer_type="UNKNOWN",
                state=CalibrationState.OVERDUE,
                requires_attention=True,
                recommended_action="Schedule initial calibration"
            )
            self._calibration_records[analyzer] = status
            logger.warning(f"Analyzer {analyzer} has no calibration record")

        # Update days until due
        if status.next_calibration_due:
            now = datetime.now(timezone.utc)
            delta = status.next_calibration_due - now
            status.days_until_due = delta.days

            # Update state based on days until due
            if status.days_until_due < 0:
                status.state = CalibrationState.OVERDUE
                status.requires_attention = True
                status.recommended_action = "Immediate calibration required"
            elif status.days_until_due <= 7:
                status.state = CalibrationState.DUE_SOON
                status.requires_attention = True
                status.recommended_action = "Schedule calibration within 7 days"

        logger.info(
            f"Calibration check for {analyzer}: state={status.state.value}, "
            f"days_until_due={status.days_until_due}"
        )

        return status

    def check_model_health(self, model_id: str) -> ModelHealthReport:
        """
        Check health status of an ML model.

        Args:
            model_id: Model identifier

        Returns:
            ModelHealthReport with health assessment
        """
        model_info = self._model_registry.get(model_id, {})
        issues = []
        recommendations = []

        # Determine health status
        status = HealthStatus.UNKNOWN

        if model_info:
            error_rate = model_info.get('error_rate', 0)
            accuracy = model_info.get('accuracy', 100)
            latency_p99 = model_info.get('latency_p99', 0)

            if error_rate > 10:
                status = HealthStatus.CRITICAL
                issues.append(f"High error rate: {error_rate}%")
                recommendations.append("Investigate model errors and retrain if needed")
            elif error_rate > 5 or accuracy < 90:
                status = HealthStatus.DEGRADED
                issues.append(f"Degraded accuracy: {accuracy}%")
                recommendations.append("Monitor model performance closely")
            elif latency_p99 > 1000:
                status = HealthStatus.DEGRADED
                issues.append(f"High latency: {latency_p99}ms p99")
                recommendations.append("Optimize model inference performance")
            else:
                status = HealthStatus.HEALTHY
        else:
            issues.append(f"Model {model_id} not found in registry")
            recommendations.append("Register model for health monitoring")

        report = ModelHealthReport(
            model_id=model_id,
            model_type=model_info.get('type', 'UNKNOWN'),
            model_version=model_info.get('version', '0.0.0'),
            status=status,
            last_prediction_time=model_info.get('last_prediction'),
            predictions_24h=model_info.get('predictions_24h', 0),
            accuracy_score=model_info.get('accuracy', 0),
            latency_p50_ms=model_info.get('latency_p50', 0),
            latency_p99_ms=model_info.get('latency_p99', 0),
            error_rate_percent=model_info.get('error_rate', 0),
            data_drift_detected=model_info.get('data_drift', False),
            model_drift_detected=model_info.get('model_drift', False),
            drift_score=model_info.get('drift_score', 0),
            memory_usage_mb=model_info.get('memory_mb', 0),
            inference_queue_depth=model_info.get('queue_depth', 0),
            issues=issues,
            recommendations=recommendations,
        )

        logger.info(f"Model health check for {model_id}: status={status.value}")

        return report

    def check_control_loop_health(self, loop: str) -> LoopHealthReport:
        """
        Check health status of a control loop.

        Args:
            loop: Control loop identifier

        Returns:
            LoopHealthReport with health assessment
        """
        loop_info = self._loop_registry.get(loop, {})
        issues = []

        # Determine health
        status = HealthStatus.UNKNOWN
        oscillation = False
        stability = 1.0

        if loop_info:
            error = abs(loop_info.get('error', 0))
            output = loop_info.get('output', 50)
            mode = loop_info.get('mode', 'AUTO')

            # Check for oscillation
            recent_errors = loop_info.get('recent_errors', [])
            if len(recent_errors) >= 10:
                oscillation = self._detect_oscillation(recent_errors)
                if oscillation:
                    issues.append("Oscillation detected in control loop")
                    stability = 0.3

            # Determine status
            if mode == 'MANUAL':
                status = HealthStatus.DEGRADED
                issues.append("Loop in manual mode")
            elif oscillation:
                status = HealthStatus.DEGRADED
            elif error > loop_info.get('error_threshold', 5):
                status = HealthStatus.DEGRADED
                issues.append(f"Error exceeds threshold: {error}")
            elif output > 95 or output < 5:
                status = HealthStatus.DEGRADED
                issues.append(f"Output near limits: {output}%")
            else:
                status = HealthStatus.HEALTHY
                stability = 0.9
        else:
            issues.append(f"Loop {loop} not found in registry")

        report = LoopHealthReport(
            loop_id=loop,
            loop_type=loop_info.get('type', 'PID'),
            status=status,
            mode=loop_info.get('mode', 'AUTO'),
            setpoint=loop_info.get('setpoint', 0),
            process_value=loop_info.get('pv', 0),
            output_percent=loop_info.get('output', 0),
            error=loop_info.get('error', 0),
            oscillation_detected=oscillation,
            stability_score=stability,
            kp=loop_info.get('kp', 1.0),
            ki=loop_info.get('ki', 0.0),
            kd=loop_info.get('kd', 0.0),
            tuning_quality=loop_info.get('tuning', 'UNKNOWN'),
            issues=issues,
        )

        logger.info(f"Control loop health check for {loop}: status={status.value}")

        return report

    def _detect_oscillation(self, errors: List[float]) -> bool:
        """Detect oscillation in error history using sign changes."""
        if len(errors) < 5:
            return False

        sign_changes = 0
        for i in range(1, len(errors)):
            if errors[i] * errors[i-1] < 0:  # Different signs
                sign_changes += 1

        # High number of sign changes indicates oscillation
        return sign_changes > len(errors) * 0.6

    def generate_system_health_dashboard(self) -> HealthDashboard:
        """
        Generate comprehensive system health dashboard.

        Returns:
            HealthDashboard with overall system health
        """
        critical_issues = []
        warnings = []

        # Collect component health
        data_quality_reports = []
        sensor_reports = []
        calibrations = list(self._calibration_records.values())
        model_reports = []
        loop_reports = []

        # Check all registered sensors
        if self._sensor_registry:
            sensor_report = self.check_sensor_health(list(self._sensor_registry.keys()))
            sensor_reports.append(sensor_report)

            if sensor_report.failed_count > 0:
                critical_issues.append(f"{sensor_report.failed_count} sensors failed")

        # Check all registered models
        for model_id in self._model_registry:
            model_report = self.check_model_health(model_id)
            model_reports.append(model_report)

            if model_report.status == HealthStatus.CRITICAL:
                critical_issues.append(f"Model {model_id} critical")
            elif model_report.status == HealthStatus.DEGRADED:
                warnings.append(f"Model {model_id} degraded")

        # Check all registered loops
        for loop_id in self._loop_registry:
            loop_report = self.check_control_loop_health(loop_id)
            loop_reports.append(loop_report)

            if loop_report.oscillation_detected:
                warnings.append(f"Loop {loop_id} oscillating")

        # Check calibrations
        for cal in calibrations:
            if cal.state == CalibrationState.OVERDUE:
                critical_issues.append(f"Analyzer {cal.analyzer_id} calibration overdue")
            elif cal.state == CalibrationState.DUE_SOON:
                warnings.append(f"Analyzer {cal.analyzer_id} calibration due soon")

        # Calculate counts
        total_sensors = len(self._sensor_registry)
        healthy_sensors = sum(
            1 for s in sensor_reports
            if s.overall_status == HealthStatus.HEALTHY
        ) if sensor_reports else 0

        total_models = len(self._model_registry)
        healthy_models = sum(
            1 for m in model_reports
            if m.status == HealthStatus.HEALTHY
        )

        total_loops = len(self._loop_registry)
        healthy_loops = sum(
            1 for l in loop_reports
            if l.status == HealthStatus.HEALTHY
        )

        # Determine overall status
        if critical_issues:
            overall_status = HealthStatus.CRITICAL
        elif warnings:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        # Calculate health score
        scores = []
        if total_sensors > 0:
            scores.append(healthy_sensors / total_sensors * 100)
        if total_models > 0:
            scores.append(healthy_models / total_models * 100)
        if total_loops > 0:
            scores.append(healthy_loops / total_loops * 100)

        health_score = statistics.mean(scores) if scores else 100.0

        # Determine component statuses
        data_quality_status = (
            data_quality_reports[0].overall_quality.value
            if data_quality_reports
            else HealthStatus.UNKNOWN
        )
        sensor_status = sensor_reports[0].overall_status if sensor_reports else HealthStatus.UNKNOWN
        model_status = (
            HealthStatus.HEALTHY
            if healthy_models == total_models and total_models > 0
            else HealthStatus.DEGRADED if healthy_models > 0
            else HealthStatus.UNKNOWN
        )
        loop_status = (
            HealthStatus.HEALTHY
            if healthy_loops == total_loops and total_loops > 0
            else HealthStatus.DEGRADED if healthy_loops > 0
            else HealthStatus.UNKNOWN
        )

        dashboard = HealthDashboard(
            overall_status=overall_status,
            health_score=health_score,
            data_quality_status=HealthStatus(data_quality_status) if isinstance(data_quality_status, str) else HealthStatus.UNKNOWN,
            sensor_health_status=sensor_status,
            model_health_status=model_status,
            control_loop_status=loop_status,
            total_sensors=total_sensors,
            healthy_sensors=healthy_sensors,
            total_models=total_models,
            healthy_models=healthy_models,
            total_loops=total_loops,
            healthy_loops=healthy_loops,
            critical_issues=critical_issues,
            warnings=warnings,
            data_quality_reports=data_quality_reports,
            sensor_health_reports=sensor_reports,
            calibration_statuses=calibrations,
            model_health_reports=model_reports,
            loop_health_reports=loop_reports,
        )

        # Compute provenance
        dashboard.provenance_hash = self._compute_provenance(dashboard)

        # Store in history
        self._health_history.append(dashboard)

        logger.info(
            f"Health dashboard generated: status={overall_status.value}, "
            f"score={health_score:.1f}"
        )

        return dashboard

    def register_sensor(
        self,
        sensor_id: str,
        sensor_type: str,
        unit_id: str
    ) -> None:
        """Register a sensor for health monitoring."""
        self._sensor_registry[sensor_id] = {
            'type': sensor_type,
            'unit_id': unit_id,
            'registered_at': datetime.now(timezone.utc),
        }
        logger.info(f"Registered sensor: {sensor_id} ({sensor_type})")

    def register_model(
        self,
        model_id: str,
        model_type: str,
        version: str
    ) -> None:
        """Register an ML model for health monitoring."""
        self._model_registry[model_id] = {
            'type': model_type,
            'version': version,
            'registered_at': datetime.now(timezone.utc),
        }
        logger.info(f"Registered model: {model_id} ({model_type} v{version})")

    def register_control_loop(
        self,
        loop_id: str,
        loop_type: str,
        setpoint: float
    ) -> None:
        """Register a control loop for health monitoring."""
        self._loop_registry[loop_id] = {
            'type': loop_type,
            'setpoint': setpoint,
            'mode': 'AUTO',
            'registered_at': datetime.now(timezone.utc),
        }
        logger.info(f"Registered control loop: {loop_id} ({loop_type})")

    def update_calibration(
        self,
        analyzer_id: str,
        analyzer_type: str,
        last_calibration: datetime,
        calibration_interval_days: int = 90
    ) -> None:
        """Update calibration record for an analyzer."""
        next_due = last_calibration + timedelta(days=calibration_interval_days)

        self._calibration_records[analyzer_id] = CalibrationStatus(
            analyzer_id=analyzer_id,
            analyzer_type=analyzer_type,
            state=CalibrationState.CALIBRATED,
            last_calibration=last_calibration,
            next_calibration_due=next_due,
        )
        logger.info(f"Updated calibration for {analyzer_id}, next due: {next_due}")

    def _compute_provenance(self, obj: BaseModel) -> str:
        """Compute SHA-256 provenance hash for audit."""
        content = obj.json(exclude={'provenance_hash'})
        return hashlib.sha256(content.encode()).hexdigest()
