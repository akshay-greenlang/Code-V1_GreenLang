"""
Evidence Packager for GL-012 SteamQual SteamQualityController

This module implements evidence package generation for steam quality events,
enabling comprehensive post-mortem analysis of quality excursions,
control actions, and system behavior.

Key Features:
    - Quality event evidence packaging
    - Control action evidence with before/after states
    - Sensor data evidence with calibration info
    - Timeline event reconstruction
    - Contributing factor analysis
    - Plot data for visualization
    - Evidence bundle generation for post-mortem

Use Cases:
    - Quality excursion investigation
    - Control performance analysis
    - Root cause analysis
    - Regulatory compliance documentation
    - Continuous improvement initiatives

Example:
    >>> packager = EvidencePackager(storage_path="/audit/evidence")
    >>> event_evidence = packager.create_quality_event_evidence(
    ...     event_id="EVT-001",
    ...     event_type="DRYNESS_LOW",
    ...     steam_header="HP",
    ...     sensor_data=sensor_readings,
    ...     calculation_traces=traces
    ... )
    >>> pack = packager.bundle_evidence_pack(
    ...     event_evidence,
    ...     include_plots=True,
    ...     include_timeline=True
    ... )

Author: GreenLang Steam Quality Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class EventSeverity(str, Enum):
    """Severity levels for quality events."""

    INFO = "INFO"
    WARNING = "WARNING"
    ALARM = "ALARM"
    CRITICAL = "CRITICAL"


class EventCategory(str, Enum):
    """Categories of quality events."""

    QUALITY_EXCURSION = "QUALITY_EXCURSION"
    CONTROL_DEVIATION = "CONTROL_DEVIATION"
    SENSOR_ANOMALY = "SENSOR_ANOMALY"
    EQUIPMENT_FAULT = "EQUIPMENT_FAULT"
    PROCESS_UPSET = "PROCESS_UPSET"
    OPERATOR_ACTION = "OPERATOR_ACTION"


class ContributingFactorType(str, Enum):
    """Types of contributing factors to quality events."""

    SENSOR_FAILURE = "SENSOR_FAILURE"
    SENSOR_DRIFT = "SENSOR_DRIFT"
    CONTROL_TUNING = "CONTROL_TUNING"
    EQUIPMENT_MALFUNCTION = "EQUIPMENT_MALFUNCTION"
    PROCESS_DISTURBANCE = "PROCESS_DISTURBANCE"
    OPERATOR_ERROR = "OPERATOR_ERROR"
    FEEDWATER_QUALITY = "FEEDWATER_QUALITY"
    LOAD_CHANGE = "LOAD_CHANGE"
    FUEL_QUALITY = "FUEL_QUALITY"
    AMBIENT_CONDITIONS = "AMBIENT_CONDITIONS"
    UNKNOWN = "UNKNOWN"


class TimelineEvent(BaseModel):
    """
    A single event in a timeline for post-mortem analysis.

    Represents a significant occurrence during a quality event
    with timestamp, description, and context.
    """

    event_id: UUID = Field(default_factory=uuid4, description="Unique event ID")
    timestamp: datetime = Field(..., description="Event timestamp")
    event_type: str = Field(..., description="Type of timeline event")
    description: str = Field(..., description="Event description")

    # Severity
    severity: EventSeverity = Field(
        default=EventSeverity.INFO, description="Event severity"
    )

    # Context
    source: str = Field(..., description="Event source (SENSOR, CONTROL, OPERATOR, SYSTEM)")
    tag_id: Optional[str] = Field(None, description="Associated tag ID if applicable")
    value: Optional[float] = Field(None, description="Associated value if applicable")
    unit: Optional[str] = Field(None, description="Value unit if applicable")

    # State change
    previous_state: Optional[str] = Field(None, description="Previous state if applicable")
    new_state: Optional[str] = Field(None, description="New state if applicable")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    class Config:
        frozen = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }


class ContributingFactor(BaseModel):
    """
    A contributing factor to a quality event.

    Documents factors that may have caused or contributed to
    a quality excursion or control deviation.
    """

    factor_id: UUID = Field(default_factory=uuid4, description="Unique factor ID")
    factor_type: ContributingFactorType = Field(..., description="Type of factor")
    factor_name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Detailed description")

    # Confidence
    confidence_pct: float = Field(
        50.0, ge=0, le=100, description="Confidence this is a contributing factor (%)"
    )
    evidence_strength: str = Field(
        default="MODERATE", description="Evidence strength (WEAK, MODERATE, STRONG)"
    )

    # Timing
    onset_time: Optional[datetime] = Field(
        None, description="When factor became relevant"
    )
    duration_minutes: Optional[float] = Field(
        None, ge=0, description="How long factor was present"
    )

    # Impact
    impact_severity: EventSeverity = Field(
        default=EventSeverity.WARNING, description="Impact severity"
    )
    affected_variables: List[str] = Field(
        default_factory=list, description="Variables affected by this factor"
    )

    # Evidence
    supporting_data: Dict[str, Any] = Field(
        default_factory=dict, description="Supporting data for this factor"
    )
    evidence_tags: List[str] = Field(
        default_factory=list, description="Tags containing supporting evidence"
    )

    # Remediation
    recommended_action: Optional[str] = Field(
        None, description="Recommended action to address"
    )
    prevention_measure: Optional[str] = Field(
        None, description="Measure to prevent recurrence"
    )

    class Config:
        frozen = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            UUID: lambda v: str(v),
        }


class PlotData(BaseModel):
    """
    Data for generating plots in evidence packages.

    Contains time series data and metadata for visualization
    of quality events and trends.
    """

    plot_id: str = Field(..., description="Unique plot identifier")
    plot_title: str = Field(..., description="Plot title")
    plot_type: str = Field(
        default="TIME_SERIES", description="Plot type (TIME_SERIES, SCATTER, BAR, HISTOGRAM)"
    )

    # Axes
    x_label: str = Field(default="Time", description="X-axis label")
    y_label: str = Field(..., description="Y-axis label")
    x_unit: Optional[str] = Field(None, description="X-axis unit")
    y_unit: Optional[str] = Field(None, description="Y-axis unit")

    # Data series
    series: List[Dict[str, Any]] = Field(
        default_factory=list, description="Data series for plotting"
    )
    # Each series: {"name": str, "x": [...], "y": [...], "color": str, "style": str}

    # Annotations
    annotations: List[Dict[str, Any]] = Field(
        default_factory=list, description="Plot annotations"
    )
    # Each annotation: {"x": value, "y": value, "text": str, "type": "POINT|VLINE|HLINE|REGION"}

    # Limits
    y_limits: Optional[Tuple[float, float]] = Field(None, description="Y-axis limits")
    reference_lines: List[Dict[str, Any]] = Field(
        default_factory=list, description="Reference lines (setpoints, limits)"
    )
    # Each line: {"value": float, "label": str, "style": "SOLID|DASHED", "color": str}

    # Metadata
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Plot generation timestamp"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SensorDataEvidence(BaseModel):
    """
    Evidence package for sensor data during a quality event.

    Contains raw sensor readings, calibration info, and quality flags.
    """

    evidence_id: UUID = Field(default_factory=uuid4, description="Unique evidence ID")
    evidence_type: str = Field(default="SENSOR_DATA", description="Evidence type")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )

    # Time range
    start_time: datetime = Field(..., description="Data start time")
    end_time: datetime = Field(..., description="Data end time")
    duration_minutes: float = Field(..., ge=0, description="Duration in minutes")

    # Sensors included
    sensor_tags: List[str] = Field(..., description="Sensor tags included")
    sensor_count: int = Field(0, ge=0, description="Number of sensors")

    # Data summary
    data_points_per_sensor: Dict[str, int] = Field(
        default_factory=dict, description="Data points per sensor"
    )
    total_data_points: int = Field(0, ge=0, description="Total data points")
    sample_rate_seconds: float = Field(
        1.0, gt=0, description="Sample rate in seconds"
    )

    # Data quality
    data_quality_flags: Dict[str, str] = Field(
        default_factory=dict, description="Quality flags per sensor"
    )
    missing_data_pct: float = Field(
        0.0, ge=0, le=100, description="Missing data percentage"
    )
    questionable_data_pct: float = Field(
        0.0, ge=0, le=100, description="Questionable data percentage"
    )

    # Calibration info
    calibration_status: Dict[str, str] = Field(
        default_factory=dict, description="Calibration status per sensor"
    )
    last_calibration_dates: Dict[str, datetime] = Field(
        default_factory=dict, description="Last calibration dates"
    )
    accuracy_specs: Dict[str, float] = Field(
        default_factory=dict, description="Accuracy specs per sensor (%)"
    )

    # Statistics per sensor
    sensor_statistics: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Statistics per sensor (min, max, avg, std)"
    )

    # Raw data reference
    raw_data_file: Optional[str] = Field(
        None, description="Path to raw data file"
    )
    raw_data_hash: Optional[str] = Field(
        None, description="SHA-256 hash of raw data"
    )

    # Hash for integrity
    evidence_hash: Optional[str] = Field(None, description="SHA-256 hash of evidence")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of evidence content."""
        data = self.dict(exclude={"evidence_hash"})
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class ControlActionEvidence(BaseModel):
    """
    Evidence package for control actions during a quality event.

    Documents control actions taken, their triggers, and results.
    """

    evidence_id: UUID = Field(default_factory=uuid4, description="Unique evidence ID")
    evidence_type: str = Field(default="CONTROL_ACTION", description="Evidence type")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )

    # Time range
    start_time: datetime = Field(..., description="Period start time")
    end_time: datetime = Field(..., description="Period end time")

    # Control loops
    control_loops: List[str] = Field(..., description="Control loops involved")

    # Actions taken
    actions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Control actions taken"
    )
    # Each action: {"timestamp": datetime, "loop_id": str, "action_type": str,
    #               "before_value": float, "after_value": float, "trigger": str}

    # Setpoint changes
    setpoint_changes: List[Dict[str, Any]] = Field(
        default_factory=list, description="Setpoint changes during period"
    )

    # Mode changes
    mode_changes: List[Dict[str, Any]] = Field(
        default_factory=list, description="Control mode changes (AUTO/MANUAL)"
    )

    # Operator actions
    operator_actions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Operator interventions"
    )

    # Performance metrics during period
    loop_performance: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Loop performance metrics"
    )

    # Alarms and alerts
    alarms_triggered: List[Dict[str, Any]] = Field(
        default_factory=list, description="Alarms triggered during period"
    )
    alerts_generated: List[Dict[str, Any]] = Field(
        default_factory=list, description="Alerts generated"
    )

    # Hash for integrity
    evidence_hash: Optional[str] = Field(None, description="SHA-256 hash of evidence")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of evidence content."""
        data = self.dict(exclude={"evidence_hash"})
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class QualityEventEvidence(BaseModel):
    """
    Comprehensive evidence package for a steam quality event.

    Contains all relevant data for post-mortem analysis of
    quality excursions, control deviations, and system issues.
    """

    evidence_id: UUID = Field(default_factory=uuid4, description="Unique evidence ID")
    evidence_type: str = Field(default="QUALITY_EVENT", description="Evidence type")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )

    # Event identification
    event_id: str = Field(..., description="Original event identifier")
    event_category: EventCategory = Field(..., description="Event category")
    event_type: str = Field(..., description="Specific event type (DRYNESS_LOW, etc.)")
    event_severity: EventSeverity = Field(..., description="Event severity")
    event_description: str = Field(..., description="Event description")

    # Location
    facility_id: str = Field(..., description="Facility identifier")
    facility_name: str = Field(..., description="Facility name")
    steam_header: str = Field(..., description="Affected steam header")
    affected_equipment: List[str] = Field(
        default_factory=list, description="Affected equipment list"
    )

    # Time window
    event_start: datetime = Field(..., description="Event start time")
    event_end: Optional[datetime] = Field(None, description="Event end time")
    event_duration_minutes: Optional[float] = Field(
        None, ge=0, description="Event duration in minutes"
    )

    # Pre-event window (for context)
    pre_event_start: datetime = Field(..., description="Pre-event context start")
    post_event_end: Optional[datetime] = Field(
        None, description="Post-event context end"
    )

    # Event metrics
    peak_deviation: float = Field(..., description="Peak deviation from normal")
    peak_deviation_time: datetime = Field(..., description="Time of peak deviation")
    average_deviation: float = Field(..., description="Average deviation during event")

    # Thresholds
    threshold_type: str = Field(..., description="Type of threshold exceeded")
    threshold_value: float = Field(..., description="Threshold value")
    peak_value: float = Field(..., description="Peak value during event")

    # Impact assessment
    estimated_production_impact: Optional[str] = Field(
        None, description="Estimated production impact"
    )
    estimated_energy_impact_mmbtu: Optional[float] = Field(
        None, ge=0, description="Estimated energy impact (MMBtu)"
    )
    affected_downstream_processes: List[str] = Field(
        default_factory=list, description="Affected downstream processes"
    )

    # Timeline
    timeline: List[TimelineEvent] = Field(
        default_factory=list, description="Event timeline"
    )

    # Contributing factors
    contributing_factors: List[ContributingFactor] = Field(
        default_factory=list, description="Identified contributing factors"
    )

    # Sub-evidence packages
    sensor_evidence: Optional[SensorDataEvidence] = Field(
        None, description="Sensor data evidence"
    )
    control_evidence: Optional[ControlActionEvidence] = Field(
        None, description="Control action evidence"
    )

    # Calculation traces
    calculation_trace_ids: List[str] = Field(
        default_factory=list, description="Related calculation trace IDs"
    )

    # Plots
    plots: List[PlotData] = Field(
        default_factory=list, description="Visualization plot data"
    )

    # Resolution
    is_resolved: bool = Field(False, description="Whether event is resolved")
    resolution_time: Optional[datetime] = Field(
        None, description="Time of resolution"
    )
    resolution_action: Optional[str] = Field(
        None, description="Action taken to resolve"
    )
    resolved_by: Optional[str] = Field(None, description="Who resolved the event")

    # Root cause
    root_cause_identified: bool = Field(
        False, description="Whether root cause was identified"
    )
    root_cause: Optional[str] = Field(None, description="Root cause if identified")
    root_cause_category: Optional[ContributingFactorType] = Field(
        None, description="Root cause category"
    )

    # Lessons learned
    lessons_learned: List[str] = Field(
        default_factory=list, description="Lessons learned"
    )
    corrective_actions: List[str] = Field(
        default_factory=list, description="Corrective actions taken"
    )
    preventive_actions: List[str] = Field(
        default_factory=list, description="Preventive actions planned"
    )

    # Prepared by
    prepared_by: str = Field(..., description="Evidence preparer")
    reviewed_by: Optional[str] = Field(None, description="Evidence reviewer")

    # Hash for integrity
    evidence_hash: Optional[str] = Field(None, description="SHA-256 hash of evidence")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            UUID: lambda v: str(v),
        }

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of evidence content."""
        data = self.dict(exclude={"evidence_hash"})
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class EvidencePack(BaseModel):
    """
    Complete evidence pack bundling all evidence for an event.

    Ready for export as ZIP archive for distribution and archival.
    """

    pack_id: UUID = Field(default_factory=uuid4, description="Unique pack ID")
    pack_type: str = Field(default="EVIDENCE_PACK", description="Pack type")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Pack creation timestamp"
    )

    # Source event
    source_event_id: str = Field(..., description="Source event ID")
    source_event_type: str = Field(..., description="Source event type")

    # Contents
    quality_event_evidence: QualityEventEvidence = Field(
        ..., description="Main quality event evidence"
    )
    additional_evidence: List[Dict[str, Any]] = Field(
        default_factory=list, description="Additional evidence items"
    )

    # Files included
    included_files: List[str] = Field(
        default_factory=list, description="Files included in pack"
    )
    total_file_count: int = Field(0, ge=0, description="Total file count")
    total_size_bytes: int = Field(0, ge=0, description="Total size in bytes")

    # Export info
    export_format: str = Field(default="ZIP", description="Export format")
    export_path: Optional[str] = Field(None, description="Export file path")
    export_hash: Optional[str] = Field(None, description="SHA-256 hash of export")

    # Metadata
    pack_version: str = Field(default="1.0.0", description="Pack format version")
    prepared_by: str = Field(..., description="Pack preparer")
    purpose: str = Field(
        default="POST_MORTEM", description="Pack purpose (POST_MORTEM, COMPLIANCE, ARCHIVE)"
    )

    # Hash for integrity
    pack_hash: Optional[str] = Field(None, description="SHA-256 hash of pack")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of pack content."""
        data = self.dict(exclude={"pack_hash", "export_hash"})
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


class EvidencePackager:
    """
    Evidence packager for steam quality events.

    Creates comprehensive evidence packages for post-mortem analysis,
    regulatory compliance, and continuous improvement.

    Attributes:
        storage_path: Path for storing evidence packages

    Example:
        >>> packager = EvidencePackager(storage_path="/audit/evidence")
        >>> evidence = packager.create_quality_event_evidence(...)
        >>> pack = packager.bundle_evidence_pack(evidence, ...)
        >>> packager.export_evidence_pack(pack, output_path)
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
    ):
        """
        Initialize evidence packager.

        Args:
            storage_path: Path for storing evidence packages
        """
        self.storage_path = Path(storage_path) if storage_path else None

        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)

        logger.info(
            "EvidencePackager initialized",
            extra={"storage_path": str(self.storage_path)}
        )

    def create_sensor_data_evidence(
        self,
        start_time: datetime,
        end_time: datetime,
        sensor_tags: List[str],
        sensor_data: Dict[str, List[Dict[str, Any]]],
        calibration_info: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> SensorDataEvidence:
        """
        Create sensor data evidence package.

        Args:
            start_time: Data start time
            end_time: Data end time
            sensor_tags: List of sensor tags
            sensor_data: Sensor data by tag, each containing list of
                         {"timestamp": datetime, "value": float, "quality": str}
            calibration_info: Optional calibration info per sensor

        Returns:
            SensorDataEvidence
        """
        duration_minutes = (end_time - start_time).total_seconds() / 60

        # Calculate data points per sensor
        data_points = {tag: len(sensor_data.get(tag, [])) for tag in sensor_tags}
        total_points = sum(data_points.values())

        # Calculate statistics per sensor
        sensor_statistics = {}
        for tag in sensor_tags:
            values = [d.get("value", 0) for d in sensor_data.get(tag, [])]
            if values:
                sensor_statistics[tag] = {
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values),
                    "std": self._calculate_std(values),
                }

        # Extract quality flags
        quality_flags = {}
        missing_count = 0
        questionable_count = 0
        for tag in sensor_tags:
            tag_data = sensor_data.get(tag, [])
            qualities = [d.get("quality", "GOOD") for d in tag_data]
            if qualities:
                # Most common quality
                quality_flags[tag] = max(set(qualities), key=qualities.count)
                missing_count += sum(1 for q in qualities if q == "BAD")
                questionable_count += sum(1 for q in qualities if q == "UNCERTAIN")

        missing_pct = (missing_count / total_points * 100) if total_points > 0 else 0
        questionable_pct = (questionable_count / total_points * 100) if total_points > 0 else 0

        # Extract calibration info
        calibration_status = {}
        last_calibration_dates = {}
        accuracy_specs = {}
        if calibration_info:
            for tag in sensor_tags:
                info = calibration_info.get(tag, {})
                calibration_status[tag] = info.get("status", "UNKNOWN")
                if "last_calibration" in info:
                    last_calibration_dates[tag] = info["last_calibration"]
                if "accuracy_pct" in info:
                    accuracy_specs[tag] = info["accuracy_pct"]

        # Calculate sample rate
        sample_rate = 1.0
        if total_points > 0 and duration_minutes > 0:
            sample_rate = (duration_minutes * 60) / (total_points / len(sensor_tags))

        evidence = SensorDataEvidence(
            start_time=start_time,
            end_time=end_time,
            duration_minutes=duration_minutes,
            sensor_tags=sensor_tags,
            sensor_count=len(sensor_tags),
            data_points_per_sensor=data_points,
            total_data_points=total_points,
            sample_rate_seconds=sample_rate,
            data_quality_flags=quality_flags,
            missing_data_pct=missing_pct,
            questionable_data_pct=questionable_pct,
            calibration_status=calibration_status,
            last_calibration_dates=last_calibration_dates,
            accuracy_specs=accuracy_specs,
            sensor_statistics=sensor_statistics,
        )

        # Calculate and set hash
        evidence_dict = evidence.dict()
        evidence_dict["evidence_hash"] = evidence.calculate_hash()
        evidence = SensorDataEvidence(**evidence_dict)

        logger.info(
            f"Sensor data evidence created: {evidence.evidence_id}",
            extra={"sensors": len(sensor_tags), "data_points": total_points}
        )

        return evidence

    def create_control_action_evidence(
        self,
        start_time: datetime,
        end_time: datetime,
        control_loops: List[str],
        actions: List[Dict[str, Any]],
        setpoint_changes: Optional[List[Dict[str, Any]]] = None,
        mode_changes: Optional[List[Dict[str, Any]]] = None,
        operator_actions: Optional[List[Dict[str, Any]]] = None,
        alarms: Optional[List[Dict[str, Any]]] = None,
        loop_performance: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> ControlActionEvidence:
        """
        Create control action evidence package.

        Args:
            start_time: Period start time
            end_time: Period end time
            control_loops: Control loops involved
            actions: List of control actions taken
            setpoint_changes: Optional setpoint changes
            mode_changes: Optional mode changes
            operator_actions: Optional operator interventions
            alarms: Optional alarms triggered
            loop_performance: Optional loop performance metrics

        Returns:
            ControlActionEvidence
        """
        evidence = ControlActionEvidence(
            start_time=start_time,
            end_time=end_time,
            control_loops=control_loops,
            actions=actions,
            setpoint_changes=setpoint_changes or [],
            mode_changes=mode_changes or [],
            operator_actions=operator_actions or [],
            alarms_triggered=alarms or [],
            loop_performance=loop_performance or {},
        )

        # Calculate and set hash
        evidence_dict = evidence.dict()
        evidence_dict["evidence_hash"] = evidence.calculate_hash()
        evidence = ControlActionEvidence(**evidence_dict)

        logger.info(
            f"Control action evidence created: {evidence.evidence_id}",
            extra={"loops": len(control_loops), "actions": len(actions)}
        )

        return evidence

    def create_quality_event_evidence(
        self,
        event_id: str,
        event_category: EventCategory,
        event_type: str,
        event_severity: EventSeverity,
        event_description: str,
        facility_id: str,
        facility_name: str,
        steam_header: str,
        event_start: datetime,
        peak_deviation: float,
        peak_deviation_time: datetime,
        average_deviation: float,
        threshold_type: str,
        threshold_value: float,
        peak_value: float,
        prepared_by: str,
        event_end: Optional[datetime] = None,
        pre_event_window_minutes: float = 30.0,
        post_event_window_minutes: float = 30.0,
        affected_equipment: Optional[List[str]] = None,
        sensor_evidence: Optional[SensorDataEvidence] = None,
        control_evidence: Optional[ControlActionEvidence] = None,
        timeline: Optional[List[TimelineEvent]] = None,
        contributing_factors: Optional[List[ContributingFactor]] = None,
        calculation_trace_ids: Optional[List[str]] = None,
        estimated_production_impact: Optional[str] = None,
        estimated_energy_impact_mmbtu: Optional[float] = None,
    ) -> QualityEventEvidence:
        """
        Create comprehensive quality event evidence.

        Args:
            event_id: Original event identifier
            event_category: Event category
            event_type: Specific event type
            event_severity: Event severity
            event_description: Event description
            facility_id: Facility identifier
            facility_name: Facility name
            steam_header: Affected steam header
            event_start: Event start time
            peak_deviation: Peak deviation from normal
            peak_deviation_time: Time of peak deviation
            average_deviation: Average deviation
            threshold_type: Type of threshold exceeded
            threshold_value: Threshold value
            peak_value: Peak value during event
            prepared_by: Evidence preparer
            event_end: Optional event end time
            pre_event_window_minutes: Pre-event context window
            post_event_window_minutes: Post-event context window
            affected_equipment: Optional affected equipment
            sensor_evidence: Optional sensor data evidence
            control_evidence: Optional control action evidence
            timeline: Optional event timeline
            contributing_factors: Optional contributing factors
            calculation_trace_ids: Optional calculation trace IDs
            estimated_production_impact: Optional production impact
            estimated_energy_impact_mmbtu: Optional energy impact

        Returns:
            QualityEventEvidence
        """
        # Calculate duration if event_end provided
        event_duration = None
        if event_end:
            event_duration = (event_end - event_start).total_seconds() / 60

        # Calculate context windows
        pre_event_start = event_start - timedelta(minutes=pre_event_window_minutes)
        post_event_end = None
        if event_end:
            post_event_end = event_end + timedelta(minutes=post_event_window_minutes)

        evidence = QualityEventEvidence(
            event_id=event_id,
            event_category=event_category,
            event_type=event_type,
            event_severity=event_severity,
            event_description=event_description,
            facility_id=facility_id,
            facility_name=facility_name,
            steam_header=steam_header,
            affected_equipment=affected_equipment or [],
            event_start=event_start,
            event_end=event_end,
            event_duration_minutes=event_duration,
            pre_event_start=pre_event_start,
            post_event_end=post_event_end,
            peak_deviation=peak_deviation,
            peak_deviation_time=peak_deviation_time,
            average_deviation=average_deviation,
            threshold_type=threshold_type,
            threshold_value=threshold_value,
            peak_value=peak_value,
            estimated_production_impact=estimated_production_impact,
            estimated_energy_impact_mmbtu=estimated_energy_impact_mmbtu,
            timeline=timeline or [],
            contributing_factors=contributing_factors or [],
            sensor_evidence=sensor_evidence,
            control_evidence=control_evidence,
            calculation_trace_ids=calculation_trace_ids or [],
            prepared_by=prepared_by,
        )

        # Calculate and set hash
        evidence_dict = evidence.dict()
        evidence_dict["evidence_hash"] = evidence.calculate_hash()
        evidence = QualityEventEvidence(**evidence_dict)

        logger.info(
            f"Quality event evidence created: {evidence.evidence_id}",
            extra={
                "event_id": event_id,
                "event_type": event_type,
                "severity": event_severity.value,
            }
        )

        return evidence

    def create_timeline_event(
        self,
        timestamp: datetime,
        event_type: str,
        description: str,
        source: str,
        severity: EventSeverity = EventSeverity.INFO,
        tag_id: Optional[str] = None,
        value: Optional[float] = None,
        unit: Optional[str] = None,
        previous_state: Optional[str] = None,
        new_state: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TimelineEvent:
        """
        Create a timeline event.

        Args:
            timestamp: Event timestamp
            event_type: Type of event
            description: Event description
            source: Event source
            severity: Event severity
            tag_id: Optional associated tag
            value: Optional associated value
            unit: Optional value unit
            previous_state: Optional previous state
            new_state: Optional new state
            metadata: Optional additional metadata

        Returns:
            TimelineEvent
        """
        return TimelineEvent(
            timestamp=timestamp,
            event_type=event_type,
            description=description,
            source=source,
            severity=severity,
            tag_id=tag_id,
            value=value,
            unit=unit,
            previous_state=previous_state,
            new_state=new_state,
            metadata=metadata or {},
        )

    def create_contributing_factor(
        self,
        factor_type: ContributingFactorType,
        factor_name: str,
        description: str,
        confidence_pct: float = 50.0,
        evidence_strength: str = "MODERATE",
        onset_time: Optional[datetime] = None,
        duration_minutes: Optional[float] = None,
        affected_variables: Optional[List[str]] = None,
        supporting_data: Optional[Dict[str, Any]] = None,
        recommended_action: Optional[str] = None,
        prevention_measure: Optional[str] = None,
    ) -> ContributingFactor:
        """
        Create a contributing factor.

        Args:
            factor_type: Type of factor
            factor_name: Human-readable name
            description: Detailed description
            confidence_pct: Confidence percentage
            evidence_strength: Strength of evidence
            onset_time: When factor became relevant
            duration_minutes: How long factor was present
            affected_variables: Variables affected
            supporting_data: Supporting data
            recommended_action: Recommended action
            prevention_measure: Prevention measure

        Returns:
            ContributingFactor
        """
        return ContributingFactor(
            factor_type=factor_type,
            factor_name=factor_name,
            description=description,
            confidence_pct=confidence_pct,
            evidence_strength=evidence_strength,
            onset_time=onset_time,
            duration_minutes=duration_minutes,
            affected_variables=affected_variables or [],
            supporting_data=supporting_data or {},
            recommended_action=recommended_action,
            prevention_measure=prevention_measure,
        )

    def create_plot_data(
        self,
        plot_id: str,
        plot_title: str,
        y_label: str,
        series: List[Dict[str, Any]],
        plot_type: str = "TIME_SERIES",
        x_label: str = "Time",
        y_unit: Optional[str] = None,
        annotations: Optional[List[Dict[str, Any]]] = None,
        reference_lines: Optional[List[Dict[str, Any]]] = None,
        y_limits: Optional[Tuple[float, float]] = None,
    ) -> PlotData:
        """
        Create plot data for visualization.

        Args:
            plot_id: Unique plot identifier
            plot_title: Plot title
            y_label: Y-axis label
            series: Data series for plotting
            plot_type: Type of plot
            x_label: X-axis label
            y_unit: Y-axis unit
            annotations: Plot annotations
            reference_lines: Reference lines
            y_limits: Y-axis limits

        Returns:
            PlotData
        """
        return PlotData(
            plot_id=plot_id,
            plot_title=plot_title,
            plot_type=plot_type,
            x_label=x_label,
            y_label=y_label,
            y_unit=y_unit,
            series=series,
            annotations=annotations or [],
            reference_lines=reference_lines or [],
            y_limits=y_limits,
        )

    def bundle_evidence_pack(
        self,
        quality_event_evidence: QualityEventEvidence,
        prepared_by: str,
        purpose: str = "POST_MORTEM",
        additional_evidence: Optional[List[Dict[str, Any]]] = None,
        include_plots: bool = True,
    ) -> EvidencePack:
        """
        Bundle all evidence into an evidence pack.

        Args:
            quality_event_evidence: Main quality event evidence
            prepared_by: Pack preparer
            purpose: Pack purpose
            additional_evidence: Optional additional evidence
            include_plots: Whether to include plots

        Returns:
            EvidencePack
        """
        # List files that would be included
        included_files = ["event_evidence.json"]

        if quality_event_evidence.sensor_evidence:
            included_files.append("sensor_data_evidence.json")

        if quality_event_evidence.control_evidence:
            included_files.append("control_action_evidence.json")

        if include_plots and quality_event_evidence.plots:
            for i, plot in enumerate(quality_event_evidence.plots):
                included_files.append(f"plots/{plot.plot_id}.json")

        if quality_event_evidence.timeline:
            included_files.append("timeline.json")

        # Estimate total size
        evidence_json = json.dumps(quality_event_evidence.dict(), default=str)
        total_size = len(evidence_json.encode("utf-8"))

        pack = EvidencePack(
            source_event_id=quality_event_evidence.event_id,
            source_event_type=quality_event_evidence.event_type,
            quality_event_evidence=quality_event_evidence,
            additional_evidence=additional_evidence or [],
            included_files=included_files,
            total_file_count=len(included_files),
            total_size_bytes=total_size,
            prepared_by=prepared_by,
            purpose=purpose,
        )

        # Calculate and set hash
        pack_dict = pack.dict()
        pack_dict["pack_hash"] = pack.calculate_hash()
        pack = EvidencePack(**pack_dict)

        logger.info(
            f"Evidence pack bundled: {pack.pack_id}",
            extra={
                "event_id": quality_event_evidence.event_id,
                "files": len(included_files),
                "size_bytes": total_size,
            }
        )

        return pack

    def export_evidence_pack(
        self,
        pack: EvidencePack,
        output_path: str,
        format: str = "ZIP",
    ) -> str:
        """
        Export evidence pack to file.

        Args:
            pack: Evidence pack to export
            output_path: Output file path
            format: Export format (ZIP or JSON)

        Returns:
            Path to exported file
        """
        output = Path(output_path)

        if format == "ZIP":
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
                # Main evidence
                evidence_json = json.dumps(
                    pack.quality_event_evidence.dict(),
                    indent=2,
                    default=str
                )
                zf.writestr("event_evidence.json", evidence_json)

                # Sensor evidence
                if pack.quality_event_evidence.sensor_evidence:
                    sensor_json = json.dumps(
                        pack.quality_event_evidence.sensor_evidence.dict(),
                        indent=2,
                        default=str
                    )
                    zf.writestr("sensor_data_evidence.json", sensor_json)

                # Control evidence
                if pack.quality_event_evidence.control_evidence:
                    control_json = json.dumps(
                        pack.quality_event_evidence.control_evidence.dict(),
                        indent=2,
                        default=str
                    )
                    zf.writestr("control_action_evidence.json", control_json)

                # Timeline
                if pack.quality_event_evidence.timeline:
                    timeline_json = json.dumps(
                        [t.dict() for t in pack.quality_event_evidence.timeline],
                        indent=2,
                        default=str
                    )
                    zf.writestr("timeline.json", timeline_json)

                # Plots
                for plot in pack.quality_event_evidence.plots:
                    plot_json = json.dumps(plot.dict(), indent=2, default=str)
                    zf.writestr(f"plots/{plot.plot_id}.json", plot_json)

                # Pack manifest
                manifest = {
                    "pack_id": str(pack.pack_id),
                    "created_at": pack.created_at.isoformat(),
                    "source_event_id": pack.source_event_id,
                    "pack_hash": pack.pack_hash,
                    "files": pack.included_files,
                }
                zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        else:  # JSON
            pack_json = json.dumps(pack.dict(), indent=2, default=str)
            output.write_text(pack_json)

        # Calculate export hash
        with open(output, "rb") as f:
            export_hash = hashlib.sha256(f.read()).hexdigest()

        logger.info(
            f"Evidence pack exported: {output}",
            extra={"format": format, "hash": export_hash[:16] + "..."}
        )

        return str(output)

    def store_evidence(
        self,
        evidence: Union[QualityEventEvidence, SensorDataEvidence, ControlActionEvidence],
    ) -> str:
        """
        Store evidence to configured storage path.

        Args:
            evidence: Evidence to store

        Returns:
            Storage path

        Raises:
            ValueError: If storage path not configured
        """
        if not self.storage_path:
            raise ValueError("Storage path not configured")

        # Create date-based directory
        date_path = evidence.created_at.strftime("%Y/%m/%d")
        full_path = self.storage_path / date_path
        full_path.mkdir(parents=True, exist_ok=True)

        filename = f"{evidence.evidence_type.lower()}_{evidence.evidence_id}.json"
        file_path = full_path / filename

        evidence_json = json.dumps(evidence.dict(), indent=2, default=str)
        file_path.write_text(evidence_json)

        logger.info(f"Evidence stored: {file_path}")
        return str(file_path)

    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
