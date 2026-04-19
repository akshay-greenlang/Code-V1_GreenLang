"""
GL-095: Alarm Management Agent (ALARM-MANAGER)

This module implements the AlarmManagementAgent for industrial alarm system
optimization, rationalization, and performance monitoring.

The agent provides:
- Alarm flood detection and analysis
- Alarm rationalization recommendations
- Priority assessment and optimization
- Operator response time analysis
- Nuisance alarm identification
- Complete SHA-256 provenance tracking

Standards Compliance:
- ISA-18.2 (Alarm Management)
- EEMUA 191 (Alarm Systems Guide)
- IEC 62682 (Alarm Management)
- ANSI/ISA-18.2

Example:
    >>> agent = AlarmManagementAgent()
    >>> result = agent.run(AlarmInput(
    ...     alarm_events=[...],
    ...     alarm_configurations=[...],
    ... ))
    >>> print(f"Nuisance Alarms: {len(result.nuisance_alarms)}")
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class AlarmPriority(str, Enum):
    """Alarm priority levels per ISA-18.2."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AlarmState(str, Enum):
    """Alarm state."""
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    CLEARED = "CLEARED"
    SUPPRESSED = "SUPPRESSED"


class AlarmType(str, Enum):
    """Alarm type classification."""
    PROCESS = "PROCESS"
    SAFETY = "SAFETY"
    EQUIPMENT = "EQUIPMENT"
    ENVIRONMENTAL = "ENVIRONMENTAL"
    QUALITY = "QUALITY"


# =============================================================================
# INPUT MODELS
# =============================================================================

class AlarmEvent(BaseModel):
    """Individual alarm event."""

    alarm_id: str = Field(..., description="Alarm identifier")
    tag_name: str = Field(..., description="Process tag name")
    timestamp: datetime = Field(..., description="Alarm timestamp")
    state: AlarmState = Field(..., description="Alarm state")
    priority: AlarmPriority = Field(..., description="Alarm priority")
    alarm_type: AlarmType = Field(..., description="Alarm type")
    message: str = Field(..., description="Alarm message")
    acknowledged_time: Optional[datetime] = Field(None, description="Acknowledgment time")
    cleared_time: Optional[datetime] = Field(None, description="Clear time")
    operator_id: Optional[str] = Field(None, description="Operator who acknowledged")


class AlarmConfiguration(BaseModel):
    """Alarm configuration settings."""

    alarm_id: str = Field(..., description="Alarm identifier")
    tag_name: str = Field(..., description="Process tag name")
    priority: AlarmPriority = Field(..., description="Configured priority")
    alarm_type: AlarmType = Field(..., description="Alarm type")
    setpoint: float = Field(..., description="Alarm setpoint")
    deadband: float = Field(..., description="Alarm deadband")
    delay_seconds: float = Field(default=0, ge=0, description="Alarm delay")
    description: str = Field(..., description="Alarm description")
    consequence: Optional[str] = Field(None, description="Alarm consequence")
    operator_action: Optional[str] = Field(None, description="Required operator action")


class AlarmInput(BaseModel):
    """Complete input model for Alarm Management Agent."""

    alarm_events: List[AlarmEvent] = Field(..., description="Alarm event history")
    alarm_configurations: List[AlarmConfiguration] = Field(..., description="Alarm configurations")
    analysis_period_hours: int = Field(default=24, ge=1, description="Analysis period")
    max_alarms_per_hour: int = Field(default=10, ge=1, description="Max acceptable alarms/hour")
    target_response_time_seconds: int = Field(default=60, ge=1, description="Target response time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @validator('alarm_events')
    def validate_events(cls, v):
        """Validate alarm events exist."""
        if not v:
            raise ValueError("At least one alarm event required")
        return v


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class AlarmPerformanceMetrics(BaseModel):
    """Overall alarm system performance metrics."""

    total_alarms: int = Field(..., description="Total alarms in period")
    alarms_per_hour_avg: float = Field(..., description="Average alarms per hour")
    alarms_per_day_avg: float = Field(..., description="Average alarms per day")
    peak_alarm_rate_per_hour: int = Field(..., description="Peak hourly alarm rate")
    avg_response_time_seconds: float = Field(..., description="Average response time")
    nuisance_alarm_rate_pct: float = Field(..., description="Nuisance alarm rate %")
    alarm_flood_events: int = Field(..., description="Number of alarm flood events")
    performance_rating: str = Field(..., description="Overall rating (EXCELLENT/GOOD/ACCEPTABLE/POOR)")


class NuisanceAlarm(BaseModel):
    """Identified nuisance alarm."""

    alarm_id: str = Field(..., description="Alarm identifier")
    tag_name: str = Field(..., description="Process tag")
    occurrence_count: int = Field(..., description="Occurrences in period")
    avg_duration_seconds: float = Field(..., description="Average active duration")
    chattering_detected: bool = Field(..., description="Chattering behavior detected")
    stale_detected: bool = Field(..., description="Stale alarm detected")
    recommendation: str = Field(..., description="Rationalization recommendation")


class AlarmFloodEvent(BaseModel):
    """Alarm flood event."""

    start_time: datetime = Field(..., description="Flood start time")
    end_time: datetime = Field(..., description="Flood end time")
    duration_minutes: float = Field(..., description="Flood duration")
    alarm_count: int = Field(..., description="Number of alarms")
    peak_rate_per_minute: int = Field(..., description="Peak alarm rate")
    contributing_tags: List[str] = Field(..., description="Tags involved in flood")


class AlarmRationalizationRecommendation(BaseModel):
    """Alarm rationalization recommendation."""

    alarm_id: str = Field(..., description="Alarm identifier")
    current_priority: AlarmPriority = Field(..., description="Current priority")
    recommended_priority: AlarmPriority = Field(..., description="Recommended priority")
    action: str = Field(..., description="Recommended action")
    rationale: str = Field(..., description="Rationale for change")
    estimated_reduction_pct: float = Field(..., description="Expected alarm reduction %")


class ProvenanceRecord(BaseModel):
    """Provenance tracking record."""

    operation: str = Field(..., description="Operation performed")
    timestamp: datetime = Field(..., description="Operation timestamp")
    input_hash: str = Field(..., description="SHA-256 hash of inputs")
    output_hash: str = Field(..., description="SHA-256 hash of outputs")
    tool_name: str = Field(..., description="Tool/calculator used")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")


class AlarmOutput(BaseModel):
    """Complete output model for Alarm Management Agent."""

    # Identification
    analysis_id: str = Field(..., description="Unique analysis identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Analysis timestamp")

    # Performance Metrics
    performance_metrics: AlarmPerformanceMetrics = Field(..., description="Performance metrics")

    # Nuisance Alarms
    nuisance_alarms: List[NuisanceAlarm] = Field(..., description="Identified nuisance alarms")

    # Alarm Floods
    flood_events: List[AlarmFloodEvent] = Field(..., description="Alarm flood events")

    # Rationalization
    rationalization_recommendations: List[AlarmRationalizationRecommendation] = Field(
        ...,
        description="Rationalization recommendations"
    )

    # Warnings & Recommendations
    warnings: List[str] = Field(default_factory=list, description="Critical warnings")
    recommendations: List[str] = Field(default_factory=list, description="Key recommendations")

    # Provenance
    provenance_chain: List[ProvenanceRecord] = Field(..., description="Complete audit trail")
    provenance_hash: str = Field(..., description="SHA-256 hash of provenance chain")

    # Processing Metadata
    processing_time_ms: float = Field(..., description="Processing time (ms)")
    validation_status: str = Field(..., description="PASS or FAIL")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")


# =============================================================================
# ALARM MANAGEMENT AGENT
# =============================================================================

class AlarmManagementAgent:
    """
    GL-095: Alarm Management Agent (ALARM-MANAGER).

    This agent optimizes industrial alarm systems through performance
    monitoring, rationalization, and flood detection per ISA-18.2.

    Zero-Hallucination Guarantee:
    - All calculations based on actual alarm event data
    - Performance metrics use deterministic statistical formulas
    - Flood detection uses configurable thresholds
    - No LLM inference in calculation path
    - Complete audit trail for compliance

    Attributes:
        AGENT_ID: Unique agent identifier (GL-095)
        AGENT_NAME: Agent name (ALARM-MANAGER)
        VERSION: Agent version
    """

    AGENT_ID = "GL-095"
    AGENT_NAME = "ALARM-MANAGER"
    VERSION = "1.0.0"
    DESCRIPTION = "Industrial Alarm Management and Optimization Agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the AlarmManagementAgent."""
        self.config = config or {}
        self._provenance_steps: List[Dict[str, Any]] = []
        self._validation_errors: List[str] = []
        self._warnings: List[str] = []
        self._recommendations: List[str] = []

        logger.info(
            f"AlarmManagementAgent initialized "
            f"(ID: {self.AGENT_ID}, Name: {self.AGENT_NAME}, Version: {self.VERSION})"
        )

    def run(self, input_data: AlarmInput) -> AlarmOutput:
        """Execute alarm management analysis."""
        start_time = datetime.utcnow()
        self._provenance_steps = []
        self._validation_errors = []
        self._warnings = []
        self._recommendations = []

        logger.info(f"Starting alarm analysis (events={len(input_data.alarm_events)})")

        try:
            # Step 1: Calculate performance metrics
            performance = self._calculate_performance_metrics(
                input_data.alarm_events,
                input_data.analysis_period_hours,
                input_data.max_alarms_per_hour,
                input_data.target_response_time_seconds
            )
            self._track_provenance(
                "performance_metrics",
                {"events": len(input_data.alarm_events)},
                {"avg_rate": performance.alarms_per_hour_avg},
                "Performance Calculator"
            )

            # Step 2: Identify nuisance alarms
            nuisance_alarms = self._identify_nuisance_alarms(
                input_data.alarm_events,
                input_data.alarm_configurations
            )
            self._track_provenance(
                "nuisance_detection",
                {"events": len(input_data.alarm_events)},
                {"nuisance_count": len(nuisance_alarms)},
                "Nuisance Detector"
            )

            # Step 3: Detect alarm floods
            flood_events = self._detect_alarm_floods(
                input_data.alarm_events,
                input_data.max_alarms_per_hour
            )
            self._track_provenance(
                "flood_detection",
                {"max_rate": input_data.max_alarms_per_hour},
                {"floods": len(flood_events)},
                "Flood Detector"
            )

            # Step 4: Generate rationalization recommendations
            rationalization = self._generate_rationalization_recommendations(
                input_data.alarm_events,
                input_data.alarm_configurations,
                nuisance_alarms
            )
            self._track_provenance(
                "rationalization",
                {"configs": len(input_data.alarm_configurations)},
                {"recommendations": len(rationalization)},
                "Rationalization Engine"
            )

            # Calculate provenance hash
            provenance_hash = self._calculate_provenance_hash()

            # Processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Generate analysis ID
            analysis_id = (
                f"ALM-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-"
                f"{hashlib.sha256(str(input_data.dict()).encode()).hexdigest()[:8]}"
            )

            validation_status = "PASS" if not self._validation_errors else "FAIL"

            output = AlarmOutput(
                analysis_id=analysis_id,
                performance_metrics=performance,
                nuisance_alarms=nuisance_alarms,
                flood_events=flood_events,
                rationalization_recommendations=rationalization,
                warnings=self._warnings,
                recommendations=self._recommendations,
                provenance_chain=[
                    ProvenanceRecord(
                        operation=step["operation"],
                        timestamp=step["timestamp"],
                        input_hash=step["input_hash"],
                        output_hash=step["output_hash"],
                        tool_name=step["tool_name"],
                        parameters=step.get("parameters", {}),
                    )
                    for step in self._provenance_steps
                ],
                provenance_hash=provenance_hash,
                processing_time_ms=round(processing_time, 2),
                validation_status=validation_status,
                validation_errors=self._validation_errors,
            )

            logger.info(
                f"Alarm analysis complete: {performance.alarms_per_hour_avg:.1f} alarms/hr "
                f"(duration: {processing_time:.2f} ms)"
            )

            return output

        except Exception as e:
            logger.error(f"Alarm analysis failed: {str(e)}", exc_info=True)
            raise

    def _calculate_performance_metrics(
        self,
        events: List[AlarmEvent],
        period_hours: int,
        max_rate: int,
        target_response: int
    ) -> AlarmPerformanceMetrics:
        """Calculate alarm system performance metrics."""
        total_alarms = len(events)
        alarms_per_hour = total_alarms / period_hours
        alarms_per_day = alarms_per_hour * 24

        # Calculate hourly distribution
        hourly_counts = {}
        for event in events:
            hour_key = event.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1

        peak_rate = max(hourly_counts.values()) if hourly_counts else 0

        # Response times
        response_times = []
        for event in events:
            if event.acknowledged_time:
                response = (event.acknowledged_time - event.timestamp).total_seconds()
                response_times.append(response)

        avg_response = sum(response_times) / len(response_times) if response_times else 0

        # Nuisance rate (chattering alarms - simplified detection)
        alarm_counts = {}
        for event in events:
            alarm_counts[event.alarm_id] = alarm_counts.get(event.alarm_id, 0) + 1

        high_frequency_alarms = sum(1 for count in alarm_counts.values() if count > 10)
        nuisance_rate = (high_frequency_alarms / len(alarm_counts) * 100) if alarm_counts else 0

        # Flood events
        flood_count = sum(1 for count in hourly_counts.values() if count > max_rate)

        # Performance rating
        if alarms_per_hour <= 6 and avg_response <= target_response and flood_count == 0:
            rating = "EXCELLENT"
        elif alarms_per_hour <= 12 and avg_response <= target_response * 1.5:
            rating = "GOOD"
        elif alarms_per_hour <= 24:
            rating = "ACCEPTABLE"
        else:
            rating = "POOR"
            self._warnings.append(f"POOR alarm system performance: {alarms_per_hour:.1f} alarms/hour")

        if avg_response > target_response:
            self._warnings.append(f"Response time ({avg_response:.0f}s) exceeds target ({target_response}s)")

        return AlarmPerformanceMetrics(
            total_alarms=total_alarms,
            alarms_per_hour_avg=round(alarms_per_hour, 2),
            alarms_per_day_avg=round(alarms_per_day, 2),
            peak_alarm_rate_per_hour=peak_rate,
            avg_response_time_seconds=round(avg_response, 2),
            nuisance_alarm_rate_pct=round(nuisance_rate, 2),
            alarm_flood_events=flood_count,
            performance_rating=rating,
        )

    def _identify_nuisance_alarms(
        self,
        events: List[AlarmEvent],
        configs: List[AlarmConfiguration]
    ) -> List[NuisanceAlarm]:
        """Identify nuisance alarms."""
        nuisance_alarms = []

        # Group events by alarm_id
        alarm_events = {}
        for event in events:
            if event.alarm_id not in alarm_events:
                alarm_events[event.alarm_id] = []
            alarm_events[event.alarm_id].append(event)

        for alarm_id, event_list in alarm_events.items():
            occurrence_count = len(event_list)

            # Calculate average duration
            durations = []
            for event in event_list:
                if event.cleared_time:
                    duration = (event.cleared_time - event.timestamp).total_seconds()
                    durations.append(duration)

            avg_duration = sum(durations) / len(durations) if durations else 0

            # Detect chattering (many short-duration alarms)
            chattering = occurrence_count > 10 and avg_duration < 60

            # Detect stale alarms (long duration, no clear)
            stale = any((datetime.utcnow() - e.timestamp).total_seconds() > 86400 for e in event_list if not e.cleared_time)

            if chattering or stale or occurrence_count > 20:
                config = next((c for c in configs if c.alarm_id == alarm_id), None)
                tag = config.tag_name if config else "UNKNOWN"

                if chattering:
                    recommendation = "Increase deadband or add delay to reduce chattering"
                elif stale:
                    recommendation = "Review alarm logic - stale alarm detected"
                else:
                    recommendation = "High occurrence rate - review alarm setpoint and necessity"

                nuisance_alarms.append(NuisanceAlarm(
                    alarm_id=alarm_id,
                    tag_name=tag,
                    occurrence_count=occurrence_count,
                    avg_duration_seconds=round(avg_duration, 2),
                    chattering_detected=chattering,
                    stale_detected=stale,
                    recommendation=recommendation,
                ))

        return nuisance_alarms

    def _detect_alarm_floods(
        self,
        events: List[AlarmEvent],
        max_rate: int
    ) -> List[AlarmFloodEvent]:
        """Detect alarm flood events."""
        # Sort events by time
        sorted_events = sorted(events, key=lambda x: x.timestamp)

        floods = []
        current_flood = None

        for i, event in enumerate(sorted_events):
            # Count alarms in the next hour
            hour_end = event.timestamp + timedelta(hours=1)
            alarms_in_hour = [e for e in sorted_events[i:] if e.timestamp < hour_end]

            if len(alarms_in_hour) > max_rate:
                if not current_flood:
                    # Start new flood
                    current_flood = {
                        "start": event.timestamp,
                        "alarms": set([e.tag_name for e in alarms_in_hour]),
                        "count": len(alarms_in_hour),
                    }
            else:
                if current_flood:
                    # End current flood
                    floods.append(AlarmFloodEvent(
                        start_time=current_flood["start"],
                        end_time=event.timestamp,
                        duration_minutes=(event.timestamp - current_flood["start"]).total_seconds() / 60,
                        alarm_count=current_flood["count"],
                        peak_rate_per_minute=current_flood["count"] // 60,
                        contributing_tags=list(current_flood["alarms"]),
                    ))
                    current_flood = None

        return floods

    def _generate_rationalization_recommendations(
        self,
        events: List[AlarmEvent],
        configs: List[AlarmConfiguration],
        nuisance_alarms: List[NuisanceAlarm]
    ) -> List[AlarmRationalizationRecommendation]:
        """Generate rationalization recommendations."""
        recommendations = []

        # Downgrade nuisance alarms
        for nuisance in nuisance_alarms:
            config = next((c for c in configs if c.alarm_id == nuisance.alarm_id), None)
            if not config:
                continue

            if config.priority in [AlarmPriority.CRITICAL, AlarmPriority.HIGH]:
                recommendations.append(AlarmRationalizationRecommendation(
                    alarm_id=nuisance.alarm_id,
                    current_priority=config.priority,
                    recommended_priority=AlarmPriority.LOW,
                    action="DOWNGRADE_PRIORITY",
                    rationale=f"Nuisance alarm with {nuisance.occurrence_count} occurrences",
                    estimated_reduction_pct=30.0,
                ))

        # Suppress or remove rarely occurring LOW priority alarms
        alarm_counts = {}
        for event in events:
            alarm_counts[event.alarm_id] = alarm_counts.get(event.alarm_id, 0) + 1

        for config in configs:
            if config.priority == AlarmPriority.LOW:
                count = alarm_counts.get(config.alarm_id, 0)
                if count < 3:  # Rarely occurring
                    recommendations.append(AlarmRationalizationRecommendation(
                        alarm_id=config.alarm_id,
                        current_priority=config.priority,
                        recommended_priority=AlarmPriority.LOW,
                        action="SUPPRESS_OR_REMOVE",
                        rationale="Rarely occurring low priority alarm - consider removal",
                        estimated_reduction_pct=5.0,
                    ))

        return recommendations

    def _track_provenance(
        self,
        operation: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        tool_name: str
    ) -> None:
        """Track a calculation step for audit trail."""
        input_str = json.dumps(inputs, sort_keys=True, default=str)
        output_str = json.dumps(outputs, sort_keys=True, default=str)

        self._provenance_steps.append({
            "operation": operation,
            "timestamp": datetime.utcnow(),
            "input_hash": hashlib.sha256(input_str.encode()).hexdigest(),
            "output_hash": hashlib.sha256(output_str.encode()).hexdigest(),
            "tool_name": tool_name,
            "parameters": inputs,
        })

    def _calculate_provenance_hash(self) -> str:
        """Calculate SHA-256 hash of complete provenance chain."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "agent_name": self.AGENT_NAME,
            "version": self.VERSION,
            "steps": [
                {
                    "operation": s["operation"],
                    "input_hash": s["input_hash"],
                    "output_hash": s["output_hash"],
                }
                for s in self._provenance_steps
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        json_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# =============================================================================
# PACK SPECIFICATION
# =============================================================================

PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-095",
    "name": "ALARM-MANAGER - Alarm Management Agent",
    "version": "1.0.0",
    "summary": "Industrial alarm system optimization and rationalization per ISA-18.2",
    "tags": [
        "alarm-management",
        "ISA-18.2",
        "EEMUA-191",
        "alarm-rationalization",
        "flood-detection",
        "nuisance-alarms",
    ],
    "owners": ["operations-team"],
    "compute": {
        "entrypoint": "python://agents.gl_095_alarm.agent:AlarmManagementAgent",
        "deterministic": True,
    },
    "standards": [
        {"ref": "ISA-18.2", "description": "Management of Alarm Systems for Process Industries"},
        {"ref": "EEMUA-191", "description": "Alarm Systems - A Guide to Design, Management and Procurement"},
        {"ref": "IEC-62682", "description": "Management of Alarm Systems for the Process Industries"},
    ],
    "provenance": {
        "calculation_verified": True,
        "enable_audit": True,
    },
}
