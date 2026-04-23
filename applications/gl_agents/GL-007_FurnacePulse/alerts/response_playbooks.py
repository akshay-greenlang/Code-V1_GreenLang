"""
ResponsePlaybookManager - Guided Response Procedures for FurnacePulse Alerts

This module implements the ResponsePlaybookManager for providing structured
response guidance for each alert type. It includes playbook templates,
step-by-step recommended actions, evidence requirements, and integration
with CMMS (Computerized Maintenance Management System) for work order creation.

Example:
    >>> manager = ResponsePlaybookManager(config)
    >>> playbook = manager.get_playbook(AlertCode.HOTSPOT_WARNING)
    >>> for step in playbook.steps:
    ...     print(f"{step.sequence}: {step.title}")
    >>> work_order = manager.create_work_order(alert, playbook)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class AlertCode(str, Enum):
    """Alert taxonomy codes (mirrored from alert_orchestrator for independence)."""

    HOTSPOT_ADVISORY = "A-001"
    HOTSPOT_WARNING = "A-002"
    HOTSPOT_URGENT = "A-003"
    EFFICIENCY_DEGRADATION = "A-010"
    DRAFT_INSTABILITY = "A-020"
    SENSOR_DRIFT_STUCK = "A-030"


class StepType(str, Enum):
    """Type of playbook step."""

    VERIFICATION = "VERIFICATION"
    INVESTIGATION = "INVESTIGATION"
    MITIGATION = "MITIGATION"
    ESCALATION = "ESCALATION"
    DOCUMENTATION = "DOCUMENTATION"
    NOTIFICATION = "NOTIFICATION"


class StepPriority(str, Enum):
    """Priority level for a playbook step."""

    REQUIRED = "REQUIRED"
    RECOMMENDED = "RECOMMENDED"
    OPTIONAL = "OPTIONAL"
    CONDITIONAL = "CONDITIONAL"


class EvidenceType(str, Enum):
    """Types of evidence that can be collected."""

    SCREENSHOT = "SCREENSHOT"
    DATA_EXPORT = "DATA_EXPORT"
    PHOTO = "PHOTO"
    VIDEO = "VIDEO"
    LOG_FILE = "LOG_FILE"
    MEASUREMENT = "MEASUREMENT"
    OPERATOR_NOTE = "OPERATOR_NOTE"
    SIGNATURE = "SIGNATURE"
    CHECKLIST = "CHECKLIST"


class WorkOrderPriority(str, Enum):
    """Priority levels for work orders."""

    EMERGENCY = "EMERGENCY"
    URGENT = "URGENT"
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"
    SCHEDULED = "SCHEDULED"


class WorkOrderStatus(str, Enum):
    """Status of a work order."""

    DRAFT = "DRAFT"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    ON_HOLD = "ON_HOLD"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class EvidenceRequirement(BaseModel):
    """Requirement for evidence collection."""

    evidence_id: str = Field(default_factory=lambda: str(uuid4()))
    evidence_type: EvidenceType = Field(..., description="Type of evidence required")
    description: str = Field(..., description="What evidence to collect")
    is_mandatory: bool = Field(default=True, description="Whether evidence is required")
    validation_rules: List[str] = Field(
        default_factory=list, description="Rules for validating evidence"
    )
    retention_days: int = Field(default=365, description="How long to retain evidence")


class PlaybookStep(BaseModel):
    """A single step in a response playbook."""

    step_id: str = Field(default_factory=lambda: str(uuid4()))
    sequence: int = Field(..., ge=1, description="Step sequence number")
    title: str = Field(..., description="Short step title")
    description: str = Field(..., description="Detailed step instructions")
    step_type: StepType = Field(..., description="Type of step")
    priority: StepPriority = Field(default=StepPriority.REQUIRED)
    estimated_minutes: int = Field(default=5, ge=1, description="Estimated time to complete")
    responsible_role: str = Field(..., description="Role responsible for this step")
    preconditions: List[str] = Field(
        default_factory=list, description="Conditions that must be met before this step"
    )
    evidence_requirements: List[EvidenceRequirement] = Field(
        default_factory=list, description="Evidence to collect for this step"
    )
    success_criteria: List[str] = Field(
        default_factory=list, description="How to verify step completion"
    )
    failure_actions: List[str] = Field(
        default_factory=list, description="What to do if step fails"
    )
    automation_available: bool = Field(
        default=False, description="Whether this step can be automated"
    )
    automation_command: Optional[str] = Field(
        None, description="Command or script for automation"
    )
    safety_warnings: List[str] = Field(
        default_factory=list, description="Safety considerations"
    )
    reference_documents: List[str] = Field(
        default_factory=list, description="Related SOPs or documentation"
    )


class Playbook(BaseModel):
    """Complete response playbook for an alert type."""

    playbook_id: str = Field(default_factory=lambda: str(uuid4()))
    alert_code: AlertCode = Field(..., description="Alert code this playbook addresses")
    name: str = Field(..., description="Playbook name")
    description: str = Field(..., description="Overview of the response procedure")
    version: str = Field(default="1.0.0", description="Playbook version")
    effective_date: datetime = Field(default_factory=datetime.utcnow)
    review_date: Optional[datetime] = Field(None, description="Next review date")
    owner: str = Field(..., description="Playbook owner/maintainer")
    steps: List[PlaybookStep] = Field(..., description="Ordered list of response steps")
    total_estimated_minutes: int = Field(default=0, description="Total estimated time")
    escalation_thresholds: Dict[str, int] = Field(
        default_factory=dict, description="Thresholds for automatic escalation"
    )
    requires_supervisor_approval: bool = Field(
        default=False, description="Whether supervisor approval is needed"
    )
    tags: List[str] = Field(default_factory=list, description="Categorization tags")
    provenance_hash: str = Field(default="", description="SHA-256 hash of playbook content")

    @validator("total_estimated_minutes", pre=True, always=True)
    def calculate_total_time(cls, v, values):
        """Calculate total estimated time from steps."""
        if "steps" in values and values["steps"]:
            return sum(step.estimated_minutes for step in values["steps"])
        return v or 0

    class Config:
        use_enum_values = True


class StepExecution(BaseModel):
    """Record of a playbook step execution."""

    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    step_id: str = Field(..., description="Playbook step ID")
    alert_id: str = Field(..., description="Associated alert ID")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)
    executed_by: str = Field(..., description="User who executed the step")
    status: str = Field(default="IN_PROGRESS")
    outcome: Optional[str] = Field(None, description="Step outcome description")
    evidence_collected: List[Dict[str, Any]] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    duration_minutes: Optional[float] = Field(None)


class PlaybookExecution(BaseModel):
    """Record of a complete playbook execution."""

    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    playbook_id: str = Field(..., description="Playbook being executed")
    alert_id: str = Field(..., description="Alert being addressed")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)
    status: str = Field(default="IN_PROGRESS")
    current_step: int = Field(default=1, ge=1)
    step_executions: List[StepExecution] = Field(default_factory=list)
    overall_outcome: Optional[str] = Field(None)
    lessons_learned: Optional[str] = Field(None)
    work_orders_created: List[str] = Field(default_factory=list)


class WorkOrder(BaseModel):
    """Work order for CMMS integration."""

    work_order_id: str = Field(default_factory=lambda: str(uuid4()))
    external_id: Optional[str] = Field(None, description="CMMS external reference")
    alert_id: str = Field(..., description="Originating alert ID")
    playbook_id: Optional[str] = Field(None, description="Associated playbook")
    title: str = Field(..., description="Work order title")
    description: str = Field(..., description="Detailed work description")
    priority: WorkOrderPriority = Field(default=WorkOrderPriority.NORMAL)
    status: WorkOrderStatus = Field(default=WorkOrderStatus.DRAFT)
    asset_id: Optional[str] = Field(None, description="Asset/equipment identifier")
    asset_name: Optional[str] = Field(None, description="Asset/equipment name")
    location: Optional[str] = Field(None, description="Physical location")
    work_type: str = Field(default="CORRECTIVE", description="Type of work")
    assigned_to: Optional[str] = Field(None, description="Assigned technician/team")
    requested_by: str = Field(..., description="Person requesting the work")
    due_date: Optional[datetime] = Field(None, description="Target completion date")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    estimated_hours: float = Field(default=1.0, ge=0)
    actual_hours: Optional[float] = Field(None)
    parts_required: List[Dict[str, Any]] = Field(default_factory=list)
    safety_requirements: List[str] = Field(default_factory=list)
    completion_notes: Optional[str] = Field(None)
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")


class CMMSConfig(BaseModel):
    """Configuration for CMMS integration."""

    enabled: bool = Field(default=False, description="Whether CMMS integration is enabled")
    system_type: str = Field(default="SAP_PM", description="CMMS system type")
    api_endpoint: Optional[str] = Field(None, description="CMMS API endpoint")
    api_key: Optional[str] = Field(None, description="API authentication key")
    default_plant: str = Field(default="PLANT01", description="Default plant code")
    work_center: str = Field(default="FURNACE", description="Default work center")
    auto_create_orders: bool = Field(
        default=False, description="Automatically create work orders"
    )
    require_approval: bool = Field(
        default=True, description="Require approval before submission"
    )
    notification_type: str = Field(
        default="M2", description="CMMS notification type code"
    )


class ResponsePlaybookManagerConfig(BaseModel):
    """Configuration for ResponsePlaybookManager."""

    cmms_config: CMMSConfig = Field(default_factory=CMMSConfig)
    playbook_directory: str = Field(
        default="playbooks/", description="Directory for playbook files"
    )
    enable_automation: bool = Field(
        default=False, description="Enable automated step execution"
    )
    track_execution_history: bool = Field(
        default=True, description="Track playbook execution history"
    )
    max_execution_history: int = Field(
        default=1000, ge=100, description="Maximum history entries to retain"
    )


# Pre-defined playbooks for each alert type
def _create_hotspot_advisory_playbook() -> Playbook:
    """Create playbook for A-001 Hotspot Advisory."""
    return Playbook(
        alert_code=AlertCode.HOTSPOT_ADVISORY,
        name="Hotspot Advisory Response",
        description="Response procedure for early-warning hotspot detection when TMT is approaching threshold",
        owner="Process Engineering",
        steps=[
            PlaybookStep(
                sequence=1,
                title="Verify Alert Validity",
                description="Confirm the TMT reading is accurate and not a sensor artifact. "
                           "Check for recent calibration issues or maintenance activities.",
                step_type=StepType.VERIFICATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=5,
                responsible_role="Operator",
                evidence_requirements=[
                    EvidenceRequirement(
                        evidence_type=EvidenceType.SCREENSHOT,
                        description="Screenshot of TMT trend showing the approaching threshold",
                        is_mandatory=True,
                    ),
                ],
                success_criteria=[
                    "TMT reading confirmed on multiple sources",
                    "No active sensor maintenance tickets",
                ],
            ),
            PlaybookStep(
                sequence=2,
                title="Check Adjacent Sensors",
                description="Review readings from adjacent tube metal thermocouples to identify "
                           "if the temperature rise is localized or widespread.",
                step_type=StepType.INVESTIGATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=10,
                responsible_role="Operator",
                evidence_requirements=[
                    EvidenceRequirement(
                        evidence_type=EvidenceType.DATA_EXPORT,
                        description="Export of adjacent TMT readings for the past 2 hours",
                        is_mandatory=True,
                    ),
                ],
                success_criteria=[
                    "Adjacent sensors reviewed",
                    "Localization pattern identified",
                ],
            ),
            PlaybookStep(
                sequence=3,
                title="Review Operating Conditions",
                description="Check current firing rate, feed rate, and any recent process changes "
                           "that could explain the temperature rise.",
                step_type=StepType.INVESTIGATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=10,
                responsible_role="Operator",
                evidence_requirements=[
                    EvidenceRequirement(
                        evidence_type=EvidenceType.OPERATOR_NOTE,
                        description="Notes on current operating conditions and any deviations",
                        is_mandatory=True,
                    ),
                ],
                success_criteria=[
                    "Operating parameters reviewed",
                    "Recent changes documented",
                ],
            ),
            PlaybookStep(
                sequence=4,
                title="Adjust Operating Parameters (if needed)",
                description="If temperature rise is due to controllable factors, make appropriate "
                           "adjustments to firing pattern or feed rate.",
                step_type=StepType.MITIGATION,
                priority=StepPriority.CONDITIONAL,
                estimated_minutes=15,
                responsible_role="Operator",
                preconditions=["Root cause identified", "Adjustment is within operating limits"],
                safety_warnings=[
                    "Do not exceed maximum rate of change for firing adjustments",
                    "Coordinate with control room before making changes",
                ],
            ),
            PlaybookStep(
                sequence=5,
                title="Document and Close",
                description="Document findings and actions taken. Close the advisory if condition "
                           "stabilizes, or escalate if condition worsens.",
                step_type=StepType.DOCUMENTATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=5,
                responsible_role="Operator",
                evidence_requirements=[
                    EvidenceRequirement(
                        evidence_type=EvidenceType.OPERATOR_NOTE,
                        description="Summary of investigation and actions taken",
                        is_mandatory=True,
                    ),
                ],
                success_criteria=[
                    "Documentation complete",
                    "Alert status updated appropriately",
                ],
            ),
        ],
        escalation_thresholds={
            "temperature_increase_degC": 10,
            "time_at_threshold_minutes": 30,
        },
        tags=["hotspot", "TMT", "advisory", "early-warning"],
    )


def _create_hotspot_warning_playbook() -> Playbook:
    """Create playbook for A-002 Hotspot Warning."""
    return Playbook(
        alert_code=AlertCode.HOTSPOT_WARNING,
        name="Hotspot Warning Response",
        description="Response procedure for sustained TMT exceedance or accelerating rate-of-rise",
        owner="Process Engineering",
        requires_supervisor_approval=True,
        steps=[
            PlaybookStep(
                sequence=1,
                title="Immediate Verification",
                description="Immediately verify the TMT exceedance on the DCS and confirm trend data.",
                step_type=StepType.VERIFICATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=3,
                responsible_role="Operator",
                evidence_requirements=[
                    EvidenceRequirement(
                        evidence_type=EvidenceType.SCREENSHOT,
                        description="Screenshot of TMT reading exceeding threshold",
                        is_mandatory=True,
                    ),
                ],
            ),
            PlaybookStep(
                sequence=2,
                title="Notify Shift Supervisor",
                description="Immediately notify the shift supervisor of the warning condition.",
                step_type=StepType.NOTIFICATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=2,
                responsible_role="Operator",
                success_criteria=["Supervisor acknowledged notification"],
            ),
            PlaybookStep(
                sequence=3,
                title="Assess Rate of Change",
                description="Calculate the rate of temperature rise over the past 30 minutes. "
                           "Determine if the condition is accelerating.",
                step_type=StepType.INVESTIGATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=10,
                responsible_role="Process Engineer",
                evidence_requirements=[
                    EvidenceRequirement(
                        evidence_type=EvidenceType.DATA_EXPORT,
                        description="TMT trend data export for rate-of-rise calculation",
                        is_mandatory=True,
                    ),
                    EvidenceRequirement(
                        evidence_type=EvidenceType.MEASUREMENT,
                        description="Calculated rate of rise (degC/hour)",
                        is_mandatory=True,
                    ),
                ],
            ),
            PlaybookStep(
                sequence=4,
                title="Evaluate Tube Remaining Life",
                description="Reference the tube life model to assess remaining life at current temperature.",
                step_type=StepType.INVESTIGATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=15,
                responsible_role="Reliability",
                reference_documents=["SOP-TUBE-LIFE-001", "Tube Remaining Life Calculator"],
            ),
            PlaybookStep(
                sequence=5,
                title="Implement Mitigation",
                description="Reduce firing rate or adjust burner pattern to reduce tube temperature. "
                           "Consider temporary load reduction if necessary.",
                step_type=StepType.MITIGATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=20,
                responsible_role="Operator",
                preconditions=["Supervisor approval obtained"],
                safety_warnings=[
                    "Follow rate-of-change limits for firing adjustments",
                    "Monitor downstream process impacts",
                    "Do not bypass any safety interlocks",
                ],
            ),
            PlaybookStep(
                sequence=6,
                title="Create Inspection Work Order",
                description="Create a work order for visual or infrared inspection of the affected tube.",
                step_type=StepType.DOCUMENTATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=10,
                responsible_role="Reliability",
            ),
            PlaybookStep(
                sequence=7,
                title="Monitor Recovery",
                description="Monitor TMT for next 2 hours to confirm temperature is decreasing.",
                step_type=StepType.VERIFICATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=120,
                responsible_role="Operator",
                success_criteria=[
                    "TMT trending downward",
                    "No new warnings triggered",
                ],
            ),
        ],
        escalation_thresholds={
            "temperature_increase_degC": 20,
            "time_at_threshold_minutes": 15,
            "rate_of_rise_degC_per_hour": 15,
        },
        tags=["hotspot", "TMT", "warning", "exceedance"],
    )


def _create_hotspot_urgent_playbook() -> Playbook:
    """Create playbook for A-003 Hotspot Urgent."""
    return Playbook(
        alert_code=AlertCode.HOTSPOT_URGENT,
        name="Hotspot Urgent Response",
        description="Emergency response for high-confidence hotspot with risk to tube integrity",
        owner="Safety",
        requires_supervisor_approval=True,
        steps=[
            PlaybookStep(
                sequence=1,
                title="IMMEDIATE: Reduce Firing",
                description="IMMEDIATELY reduce firing rate to minimum safe level for the affected zone.",
                step_type=StepType.MITIGATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=2,
                responsible_role="Operator",
                safety_warnings=[
                    "This is a time-critical action",
                    "Follow emergency procedures",
                    "Do not wait for supervisor approval for initial response",
                ],
            ),
            PlaybookStep(
                sequence=2,
                title="Alert Emergency Response",
                description="Activate emergency response protocol. Notify shift supervisor, safety, "
                           "and on-call management.",
                step_type=StepType.NOTIFICATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=5,
                responsible_role="Operator",
                success_criteria=[
                    "Shift supervisor notified",
                    "Safety team notified",
                    "Management notified",
                ],
            ),
            PlaybookStep(
                sequence=3,
                title="Assess Shutdown Necessity",
                description="Evaluate whether emergency shutdown is required based on temperature "
                           "trend and tube condition assessment.",
                step_type=StepType.INVESTIGATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=10,
                responsible_role="Process Engineer",
                safety_warnings=[
                    "If tube rupture appears imminent, initiate emergency shutdown immediately",
                ],
            ),
            PlaybookStep(
                sequence=4,
                title="Clear Personnel from Area",
                description="If shutdown is not immediate, ensure all non-essential personnel "
                           "are clear of the furnace area as a precaution.",
                step_type=StepType.MITIGATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=15,
                responsible_role="Safety",
                evidence_requirements=[
                    EvidenceRequirement(
                        evidence_type=EvidenceType.CHECKLIST,
                        description="Personnel accountability checklist",
                        is_mandatory=True,
                    ),
                ],
            ),
            PlaybookStep(
                sequence=5,
                title="Continuous Monitoring",
                description="Maintain continuous monitoring of affected tube and adjacent tubes. "
                           "Report any further temperature increases immediately.",
                step_type=StepType.VERIFICATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=60,
                responsible_role="Operator",
            ),
            PlaybookStep(
                sequence=6,
                title="Document Incident",
                description="Complete incident documentation including timeline, actions taken, "
                           "and personnel involved.",
                step_type=StepType.DOCUMENTATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=30,
                responsible_role="Shift Supervisor",
                evidence_requirements=[
                    EvidenceRequirement(
                        evidence_type=EvidenceType.DATA_EXPORT,
                        description="Complete data export for incident timeline",
                        is_mandatory=True,
                    ),
                    EvidenceRequirement(
                        evidence_type=EvidenceType.OPERATOR_NOTE,
                        description="Detailed incident narrative",
                        is_mandatory=True,
                    ),
                ],
            ),
        ],
        escalation_thresholds={
            "immediate_shutdown_temp_degC": 1050,
        },
        tags=["hotspot", "TMT", "urgent", "emergency", "tube-integrity"],
    )


def _create_efficiency_degradation_playbook() -> Playbook:
    """Create playbook for A-010 Efficiency Degradation."""
    return Playbook(
        alert_code=AlertCode.EFFICIENCY_DEGRADATION,
        name="Efficiency Degradation Response",
        description="Response procedure for efficiency or specific fuel consumption deviation from baseline",
        owner="Process Engineering",
        steps=[
            PlaybookStep(
                sequence=1,
                title="Verify Efficiency Calculation",
                description="Confirm the efficiency calculation inputs are accurate. Check for any "
                           "instrument errors or calibration issues.",
                step_type=StepType.VERIFICATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=15,
                responsible_role="Process Engineer",
                evidence_requirements=[
                    EvidenceRequirement(
                        evidence_type=EvidenceType.DATA_EXPORT,
                        description="Efficiency calculation input parameters",
                        is_mandatory=True,
                    ),
                ],
            ),
            PlaybookStep(
                sequence=2,
                title="Check for Process Changes",
                description="Review any recent feed quality changes, throughput adjustments, or "
                           "operating mode changes that could explain efficiency variation.",
                step_type=StepType.INVESTIGATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=20,
                responsible_role="Process Engineer",
            ),
            PlaybookStep(
                sequence=3,
                title="Analyze Heat Loss Indicators",
                description="Review flue gas temperature, excess air, and skin temperatures for "
                           "indications of heat loss issues.",
                step_type=StepType.INVESTIGATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=30,
                responsible_role="Process Engineer",
                evidence_requirements=[
                    EvidenceRequirement(
                        evidence_type=EvidenceType.DATA_EXPORT,
                        description="Heat loss indicator trends for past 7 days",
                        is_mandatory=True,
                    ),
                ],
            ),
            PlaybookStep(
                sequence=4,
                title="Check Fouling Indicators",
                description="Review tube-side and fireside fouling indicators. Compare current "
                           "U-values to clean baseline.",
                step_type=StepType.INVESTIGATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=20,
                responsible_role="Process Engineer",
            ),
            PlaybookStep(
                sequence=5,
                title="Develop Action Plan",
                description="Based on root cause, develop a plan to restore efficiency. This may "
                           "include tuning, cleaning, or maintenance activities.",
                step_type=StepType.MITIGATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=30,
                responsible_role="Process Engineer",
            ),
            PlaybookStep(
                sequence=6,
                title="Create Work Orders",
                description="Create any necessary work orders for maintenance or cleaning activities.",
                step_type=StepType.DOCUMENTATION,
                priority=StepPriority.CONDITIONAL,
                estimated_minutes=15,
                responsible_role="Reliability",
            ),
        ],
        tags=["efficiency", "SFC", "heat-rate", "performance"],
    )


def _create_draft_instability_playbook() -> Playbook:
    """Create playbook for A-020 Draft Instability."""
    return Playbook(
        alert_code=AlertCode.DRAFT_INSTABILITY,
        name="Draft Instability Response",
        description="Response procedure for draft variance indicating control issues",
        owner="OT/Controls",
        steps=[
            PlaybookStep(
                sequence=1,
                title="Verify Draft Readings",
                description="Confirm draft readings are accurate. Check for instrument errors or "
                           "transmitter issues.",
                step_type=StepType.VERIFICATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=10,
                responsible_role="OT/Controls",
            ),
            PlaybookStep(
                sequence=2,
                title="Check Damper Positions",
                description="Verify all damper positions and confirm they are responding correctly "
                           "to control signals.",
                step_type=StepType.INVESTIGATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=15,
                responsible_role="Operator",
            ),
            PlaybookStep(
                sequence=3,
                title="Review Control Loop Performance",
                description="Analyze the draft control loop for oscillations, saturation, or other "
                           "abnormal behavior.",
                step_type=StepType.INVESTIGATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=20,
                responsible_role="OT/Controls",
                evidence_requirements=[
                    EvidenceRequirement(
                        evidence_type=EvidenceType.DATA_EXPORT,
                        description="Control loop historian data for past 24 hours",
                        is_mandatory=True,
                    ),
                ],
            ),
            PlaybookStep(
                sequence=4,
                title="Check External Factors",
                description="Evaluate external factors such as wind conditions, stack effects, or "
                           "interactions with adjacent equipment.",
                step_type=StepType.INVESTIGATION,
                priority=StepPriority.RECOMMENDED,
                estimated_minutes=15,
                responsible_role="Process Engineer",
            ),
            PlaybookStep(
                sequence=5,
                title="Tune or Repair",
                description="Based on findings, either retune the control loop or create a work "
                           "order for mechanical repairs.",
                step_type=StepType.MITIGATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=30,
                responsible_role="OT/Controls",
            ),
        ],
        tags=["draft", "control", "damper", "instability"],
    )


def _create_sensor_drift_stuck_playbook() -> Playbook:
    """Create playbook for A-030 Sensor Drift/Stuck."""
    return Playbook(
        alert_code=AlertCode.SENSOR_DRIFT_STUCK,
        name="Sensor Drift/Stuck Response",
        description="Response procedure for sensors that fail drift or stuck-value tests",
        owner="OT/Controls",
        steps=[
            PlaybookStep(
                sequence=1,
                title="Identify Affected Sensor",
                description="Confirm the specific sensor(s) affected and their role in process control.",
                step_type=StepType.VERIFICATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=5,
                responsible_role="Operator",
            ),
            PlaybookStep(
                sequence=2,
                title="Assess Impact on Control",
                description="Evaluate whether the sensor is used in closed-loop control. If so, "
                           "determine appropriate compensating actions.",
                step_type=StepType.INVESTIGATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=10,
                responsible_role="OT/Controls",
            ),
            PlaybookStep(
                sequence=3,
                title="Compare to Redundant Sensors",
                description="Compare the suspect sensor to any redundant or adjacent sensors to "
                           "validate the fault.",
                step_type=StepType.VERIFICATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=10,
                responsible_role="Operator",
                evidence_requirements=[
                    EvidenceRequirement(
                        evidence_type=EvidenceType.DATA_EXPORT,
                        description="Comparison trend of suspect and reference sensors",
                        is_mandatory=True,
                    ),
                ],
            ),
            PlaybookStep(
                sequence=4,
                title="Implement Backup Strategy",
                description="If sensor is confirmed faulty and impacts control, switch to backup "
                           "sensor or manual control as appropriate.",
                step_type=StepType.MITIGATION,
                priority=StepPriority.CONDITIONAL,
                estimated_minutes=15,
                responsible_role="Operator",
                safety_warnings=[
                    "Follow procedure for switching to manual control",
                    "Notify control room of control mode change",
                ],
            ),
            PlaybookStep(
                sequence=5,
                title="Create Calibration/Repair Work Order",
                description="Create a work order for sensor calibration verification or replacement.",
                step_type=StepType.DOCUMENTATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=10,
                responsible_role="OT/Controls",
            ),
            PlaybookStep(
                sequence=6,
                title="Update Data Quality Flags",
                description="Ensure the data quality system is flagging data from the suspect sensor "
                           "appropriately.",
                step_type=StepType.DOCUMENTATION,
                priority=StepPriority.REQUIRED,
                estimated_minutes=5,
                responsible_role="OT/Controls",
            ),
        ],
        tags=["sensor", "drift", "stuck", "data-quality", "calibration"],
    )


class ResponsePlaybookManager:
    """
    Manager for response playbooks and CMMS integration.

    This class provides access to predefined playbooks for each alert type,
    tracks playbook execution, and integrates with CMMS for work order creation.

    Attributes:
        config: Manager configuration
        playbooks: Dictionary of available playbooks by alert code
        executions: Active and historical playbook executions
        work_orders: Created work orders

    Example:
        >>> config = ResponsePlaybookManagerConfig()
        >>> manager = ResponsePlaybookManager(config)
        >>> playbook = manager.get_playbook(AlertCode.HOTSPOT_WARNING)
        >>> execution = manager.start_execution(playbook, alert_id="alert-123")
    """

    def __init__(self, config: ResponsePlaybookManagerConfig):
        """
        Initialize ResponsePlaybookManager.

        Args:
            config: Manager configuration
        """
        self.config = config
        self.playbooks: Dict[AlertCode, Playbook] = {}
        self.executions: Dict[str, PlaybookExecution] = {}
        self.work_orders: Dict[str, WorkOrder] = {}

        # Initialize built-in playbooks
        self._initialize_playbooks()

        logger.info(
            "ResponsePlaybookManager initialized with %d playbooks",
            len(self.playbooks)
        )

    def _initialize_playbooks(self) -> None:
        """Initialize the built-in playbook library."""
        playbook_creators = [
            _create_hotspot_advisory_playbook,
            _create_hotspot_warning_playbook,
            _create_hotspot_urgent_playbook,
            _create_efficiency_degradation_playbook,
            _create_draft_instability_playbook,
            _create_sensor_drift_stuck_playbook,
        ]

        for creator in playbook_creators:
            playbook = creator()
            playbook.provenance_hash = self._calculate_playbook_hash(playbook)
            self.playbooks[AlertCode(playbook.alert_code)] = playbook
            logger.debug("Loaded playbook: %s", playbook.name)

    def _calculate_playbook_hash(self, playbook: Playbook) -> str:
        """Calculate SHA-256 hash of playbook content for version tracking."""
        content = f"{playbook.alert_code}|{playbook.name}|{playbook.version}|"
        content += "|".join(f"{s.sequence}:{s.title}" for s in playbook.steps)
        return hashlib.sha256(content.encode()).hexdigest()

    def get_playbook(self, alert_code: AlertCode) -> Optional[Playbook]:
        """
        Get the playbook for an alert code.

        Args:
            alert_code: Alert taxonomy code

        Returns:
            Playbook if found, None otherwise
        """
        return self.playbooks.get(alert_code)

    def get_all_playbooks(self) -> List[Playbook]:
        """Get all available playbooks."""
        return list(self.playbooks.values())

    def start_execution(
        self,
        playbook: Playbook,
        alert_id: str,
        executor_id: str,
    ) -> PlaybookExecution:
        """
        Start executing a playbook for an alert.

        Args:
            playbook: Playbook to execute
            alert_id: Alert being addressed
            executor_id: User starting the execution

        Returns:
            PlaybookExecution tracking object
        """
        execution = PlaybookExecution(
            playbook_id=playbook.playbook_id,
            alert_id=alert_id,
        )

        self.executions[execution.execution_id] = execution

        logger.info(
            "Started playbook execution %s for alert %s",
            execution.execution_id, alert_id
        )

        return execution

    def complete_step(
        self,
        execution_id: str,
        step_id: str,
        executor_id: str,
        outcome: str,
        evidence: Optional[List[Dict[str, Any]]] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Record completion of a playbook step.

        Args:
            execution_id: Playbook execution ID
            step_id: Step ID being completed
            executor_id: User completing the step
            outcome: Outcome description
            evidence: Collected evidence (optional)
            notes: Additional notes (optional)

        Returns:
            True if step recorded successfully
        """
        if execution_id not in self.executions:
            logger.warning("Unknown execution ID: %s", execution_id)
            return False

        execution = self.executions[execution_id]

        # Create step execution record
        step_execution = StepExecution(
            step_id=step_id,
            alert_id=execution.alert_id,
            executed_by=executor_id,
            status="COMPLETED",
            outcome=outcome,
            completed_at=datetime.utcnow(),
        )

        if evidence:
            step_execution.evidence_collected = evidence
        if notes:
            step_execution.notes.append(notes)

        # Calculate duration
        step_execution.duration_minutes = (
            step_execution.completed_at - step_execution.started_at
        ).total_seconds() / 60

        execution.step_executions.append(step_execution)
        execution.current_step += 1

        logger.info(
            "Completed step %s in execution %s",
            step_id, execution_id
        )

        return True

    def complete_execution(
        self,
        execution_id: str,
        overall_outcome: str,
        lessons_learned: Optional[str] = None,
    ) -> bool:
        """
        Complete a playbook execution.

        Args:
            execution_id: Playbook execution ID
            overall_outcome: Overall outcome description
            lessons_learned: Any lessons learned (optional)

        Returns:
            True if execution completed successfully
        """
        if execution_id not in self.executions:
            logger.warning("Unknown execution ID: %s", execution_id)
            return False

        execution = self.executions[execution_id]
        execution.status = "COMPLETED"
        execution.completed_at = datetime.utcnow()
        execution.overall_outcome = overall_outcome
        execution.lessons_learned = lessons_learned

        logger.info(
            "Completed playbook execution %s with outcome: %s",
            execution_id, overall_outcome
        )

        return True

    def create_work_order(
        self,
        alert_id: str,
        title: str,
        description: str,
        priority: WorkOrderPriority,
        requested_by: str,
        playbook_id: Optional[str] = None,
        asset_id: Optional[str] = None,
        asset_name: Optional[str] = None,
        location: Optional[str] = None,
        due_date: Optional[datetime] = None,
        estimated_hours: float = 1.0,
        safety_requirements: Optional[List[str]] = None,
    ) -> WorkOrder:
        """
        Create a work order for CMMS integration.

        Args:
            alert_id: Originating alert ID
            title: Work order title
            description: Detailed work description
            priority: Work order priority
            requested_by: Person requesting the work
            playbook_id: Associated playbook (optional)
            asset_id: Asset/equipment identifier (optional)
            asset_name: Asset/equipment name (optional)
            location: Physical location (optional)
            due_date: Target completion date (optional)
            estimated_hours: Estimated work hours
            safety_requirements: Safety requirements (optional)

        Returns:
            Created WorkOrder object
        """
        work_order = WorkOrder(
            alert_id=alert_id,
            playbook_id=playbook_id,
            title=title,
            description=description,
            priority=priority,
            requested_by=requested_by,
            asset_id=asset_id,
            asset_name=asset_name,
            location=location,
            due_date=due_date,
            estimated_hours=estimated_hours,
            safety_requirements=safety_requirements or [],
        )

        # Calculate provenance hash
        work_order.provenance_hash = self._calculate_work_order_hash(work_order)

        self.work_orders[work_order.work_order_id] = work_order

        # Add to execution if playbook is active
        if playbook_id:
            for execution in self.executions.values():
                if execution.playbook_id == playbook_id and execution.status == "IN_PROGRESS":
                    execution.work_orders_created.append(work_order.work_order_id)
                    break

        logger.info(
            "Created work order %s: %s (priority=%s)",
            work_order.work_order_id, title, priority.value
        )

        return work_order

    def _calculate_work_order_hash(self, work_order: WorkOrder) -> str:
        """Calculate SHA-256 hash for work order audit trail."""
        content = (
            f"{work_order.work_order_id}|{work_order.alert_id}|"
            f"{work_order.title}|{work_order.created_at.isoformat()}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    async def submit_to_cmms(self, work_order_id: str) -> Tuple[bool, Optional[str]]:
        """
        Submit a work order to the CMMS system.

        Args:
            work_order_id: Work order to submit

        Returns:
            Tuple of (success, external_id or error message)
        """
        if not self.config.cmms_config.enabled:
            logger.warning("CMMS integration is not enabled")
            return False, "CMMS integration not enabled"

        if work_order_id not in self.work_orders:
            logger.warning("Unknown work order ID: %s", work_order_id)
            return False, "Unknown work order ID"

        work_order = self.work_orders[work_order_id]

        if self.config.cmms_config.require_approval:
            if work_order.status != WorkOrderStatus.APPROVED:
                logger.warning("Work order requires approval before submission")
                return False, "Work order requires approval"

        # In production, this would make an API call to the CMMS
        # For now, we simulate the submission
        try:
            logger.info("Submitting work order %s to CMMS", work_order_id)

            # Simulate API call
            import asyncio
            await asyncio.sleep(0.1)

            # Generate mock external ID
            external_id = f"{self.config.cmms_config.notification_type}-{work_order_id[:8].upper()}"
            work_order.external_id = external_id
            work_order.status = WorkOrderStatus.PENDING_APPROVAL
            work_order.updated_at = datetime.utcnow()

            logger.info(
                "Work order submitted successfully: %s -> %s",
                work_order_id, external_id
            )

            return True, external_id

        except Exception as e:
            logger.error("CMMS submission failed: %s", str(e))
            return False, str(e)

    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get playbook execution statistics.

        Returns:
            Dictionary with execution statistics
        """
        total_executions = len(self.executions)
        completed = sum(
            1 for e in self.executions.values() if e.status == "COMPLETED"
        )
        in_progress = sum(
            1 for e in self.executions.values() if e.status == "IN_PROGRESS"
        )

        avg_duration = 0.0
        completed_executions = [
            e for e in self.executions.values()
            if e.status == "COMPLETED" and e.completed_at
        ]
        if completed_executions:
            durations = [
                (e.completed_at - e.started_at).total_seconds() / 60
                for e in completed_executions
            ]
            avg_duration = sum(durations) / len(durations)

        return {
            "total_executions": total_executions,
            "completed": completed,
            "in_progress": in_progress,
            "average_duration_minutes": round(avg_duration, 1),
            "total_work_orders": len(self.work_orders),
            "work_orders_by_status": {
                status.value: sum(
                    1 for wo in self.work_orders.values()
                    if wo.status == status
                )
                for status in WorkOrderStatus
            },
        }
