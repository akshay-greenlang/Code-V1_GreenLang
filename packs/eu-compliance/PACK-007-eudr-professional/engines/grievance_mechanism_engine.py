"""
GrievanceMechanismEngine - Stakeholder complaint handling and grievance resolution for EUDR

This module implements grievance mechanism for PACK-007 EUDR Professional Pack.
Provides stakeholder complaint registration, triage, investigation, FPIC verification,
and resolution tracking per EU Regulation 2023/1115 and UN Guiding Principles on Business
and Human Rights.

Example:
    >>> config = GrievanceConfig(enabled=True, anonymous=True)
    >>> engine = GrievanceMechanismEngine(config)
    >>> complaint = engine.register_complaint({...})
    >>> investigation = engine.start_investigation(complaint.complaint_id)
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field, validator
import hashlib
import logging
from datetime import datetime, timedelta
from enum import Enum
import json
import secrets

logger = logging.getLogger(__name__)


class ComplaintType(str, Enum):
    """Types of grievances/complaints."""
    DEFORESTATION = "DEFORESTATION"
    LAND_RIGHTS = "LAND_RIGHTS"
    INDIGENOUS_RIGHTS = "INDIGENOUS_RIGHTS"
    LABOR_RIGHTS = "LABOR_RIGHTS"
    ENVIRONMENTAL_DAMAGE = "ENVIRONMENTAL_DAMAGE"
    FPIC_VIOLATION = "FPIC_VIOLATION"
    CORRUPTION = "CORRUPTION"
    DATA_INACCURACY = "DATA_INACCURACY"
    SAFETY_VIOLATION = "SAFETY_VIOLATION"
    OTHER = "OTHER"


class ComplaintSeverity(str, Enum):
    """Complaint severity levels."""
    CRITICAL = "CRITICAL"  # Immediate investigation required
    HIGH = "HIGH"  # Investigation within 24 hours
    MEDIUM = "MEDIUM"  # Investigation within 5 days
    LOW = "LOW"  # Investigation within 14 days


class ComplaintStatus(str, Enum):
    """Complaint status values."""
    SUBMITTED = "SUBMITTED"
    TRIAGED = "TRIAGED"
    UNDER_INVESTIGATION = "UNDER_INVESTIGATION"
    PENDING_RESOLUTION = "PENDING_RESOLUTION"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"
    ESCALATED = "ESCALATED"


class ComplaintSource(str, Enum):
    """Source of complaint."""
    INTERNAL_EMPLOYEE = "INTERNAL_EMPLOYEE"
    SUPPLIER = "SUPPLIER"
    LOCAL_COMMUNITY = "LOCAL_COMMUNITY"
    NGO = "NGO"
    GOVERNMENT_AGENCY = "GOVERNMENT_AGENCY"
    ANONYMOUS = "ANONYMOUS"
    WHISTLEBLOWER = "WHISTLEBLOWER"


class GrievanceConfig(BaseModel):
    """Configuration for grievance mechanism engine."""

    enabled: bool = Field(True, description="Enable grievance mechanism")
    anonymous: bool = Field(True, description="Allow anonymous submissions")
    response_sla_days: int = Field(5, ge=1, le=30, description="Days to acknowledge complaint")
    resolution_sla_days: int = Field(30, ge=1, le=180, description="Days to resolve complaint")
    auto_triage: bool = Field(True, description="Automatic triage based on severity")
    fpic_verification: bool = Field(True, description="Enable FPIC verification")
    whistleblower_protection: bool = Field(True, description="Enable whistleblower protection")
    escalation_threshold_days: int = Field(45, ge=7, le=90, description="Days before auto-escalation")


class ComplaintTimeline(BaseModel):
    """Timeline entry for complaint."""

    timestamp: datetime = Field(..., description="Event timestamp")
    event_type: str = Field(..., description="Event type")
    actor: str = Field(..., description="Actor who performed event")
    description: str = Field(..., description="Event description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class Complaint(BaseModel):
    """Grievance complaint record."""

    complaint_id: str = Field(..., description="Unique complaint identifier")
    source: ComplaintSource = Field(..., description="Complaint source")
    type: ComplaintType = Field(..., description="Complaint type")
    severity: ComplaintSeverity = Field(..., description="Severity level")
    description: str = Field(..., description="Detailed complaint description")
    linked_plots: List[str] = Field(default_factory=list, description="Linked plot IDs")
    linked_suppliers: List[str] = Field(default_factory=list, description="Linked supplier IDs")
    linked_dds: List[str] = Field(default_factory=list, description="Linked DDS IDs")
    status: ComplaintStatus = Field(..., description="Current status")
    timeline: List[ComplaintTimeline] = Field(..., description="Status timeline")
    submitted_at: datetime = Field(..., description="Submission timestamp")
    acknowledged_at: Optional[datetime] = Field(None, description="Acknowledgment timestamp")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    submitter_contact: Optional[str] = Field(None, description="Submitter contact (if not anonymous)")
    is_anonymous: bool = Field(False, description="Whether complaint is anonymous")
    tracking_code: str = Field(..., description="Public tracking code for status checks")
    confidential: bool = Field(False, description="Whether complaint contains sensitive data")


class TriageResult(BaseModel):
    """Complaint triage result."""

    triage_id: str = Field(..., description="Triage identifier")
    complaint_id: str = Field(..., description="Complaint identifier")
    assigned_severity: ComplaintSeverity = Field(..., description="Assigned severity")
    assigned_investigator: str = Field(..., description="Assigned investigator")
    investigation_deadline: datetime = Field(..., description="Investigation deadline")
    triage_notes: str = Field(..., description="Triage notes")
    recommended_actions: List[str] = Field(..., description="Recommended immediate actions")
    requires_escalation: bool = Field(False, description="Whether immediate escalation needed")


class EvidenceItem(BaseModel):
    """Evidence item for investigation."""

    evidence_id: str = Field(..., description="Evidence identifier")
    evidence_type: str = Field(..., description="Evidence type (DOCUMENT, PHOTO, TESTIMONY, DATA)")
    description: str = Field(..., description="Evidence description")
    source: str = Field(..., description="Evidence source")
    collected_at: datetime = Field(..., description="Collection timestamp")
    file_hash: Optional[str] = Field(None, description="SHA-256 hash if file evidence")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    chain_of_custody: List[str] = Field(..., description="Chain of custody log")


class Investigation(BaseModel):
    """Complaint investigation record."""

    investigation_id: str = Field(..., description="Investigation identifier")
    complaint_id: str = Field(..., description="Related complaint ID")
    investigator: str = Field(..., description="Lead investigator")
    team_members: List[str] = Field(default_factory=list, description="Investigation team")
    started_at: datetime = Field(..., description="Investigation start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Investigation completion timestamp")
    findings: str = Field(..., description="Investigation findings")
    evidence: List[EvidenceItem] = Field(..., description="Collected evidence")
    verified: bool = Field(False, description="Whether complaint was verified")
    verification_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Verification confidence")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")
    status: str = Field("IN_PROGRESS", description="Investigation status")


class EvidenceLink(BaseModel):
    """Link between investigation and evidence."""

    link_id: str = Field(..., description="Link identifier")
    investigation_id: str = Field(..., description="Investigation ID")
    evidence_id: str = Field(..., description="Evidence ID")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    linked_at: datetime = Field(..., description="Link timestamp")
    linked_by: str = Field(..., description="User who created link")


class Resolution(BaseModel):
    """Complaint resolution record."""

    resolution_id: str = Field(..., description="Resolution identifier")
    complaint_id: str = Field(..., description="Complaint ID")
    investigation_id: str = Field(..., description="Investigation ID")
    resolution_type: str = Field(..., description="Resolution type (SUBSTANTIATED, UNSUBSTANTIATED, PARTIAL)")
    corrective_actions: List[str] = Field(..., description="Corrective actions taken")
    preventive_actions: List[str] = Field(..., description="Preventive actions implemented")
    compensation_provided: bool = Field(False, description="Whether compensation was provided")
    resolved_at: datetime = Field(..., description="Resolution timestamp")
    resolver: str = Field(..., description="Person who resolved complaint")
    submitter_notified: bool = Field(False, description="Whether submitter was notified")
    appeal_deadline: Optional[datetime] = Field(None, description="Deadline for appeal")


class SLAStatus(BaseModel):
    """SLA tracking for complaint."""

    complaint_id: str = Field(..., description="Complaint identifier")
    response_sla_met: bool = Field(..., description="Whether response SLA was met")
    response_due: datetime = Field(..., description="Response due date")
    response_actual: Optional[datetime] = Field(None, description="Actual response date")
    resolution_sla_met: Optional[bool] = Field(None, description="Whether resolution SLA was met")
    resolution_due: datetime = Field(..., description="Resolution due date")
    resolution_actual: Optional[datetime] = Field(None, description="Actual resolution date")
    days_overdue: int = Field(0, description="Days overdue (if any)")
    escalation_triggered: bool = Field(False, description="Whether escalation was triggered")


class GrievanceStatistics(BaseModel):
    """Grievance mechanism statistics."""

    total_complaints: int = Field(..., ge=0, description="Total complaints received")
    complaints_by_type: Dict[str, int] = Field(..., description="Complaints grouped by type")
    complaints_by_severity: Dict[str, int] = Field(..., description="Complaints grouped by severity")
    complaints_by_status: Dict[str, int] = Field(..., description="Complaints grouped by status")
    avg_resolution_days: float = Field(..., ge=0.0, description="Average days to resolution")
    sla_compliance_rate: float = Field(..., ge=0.0, le=1.0, description="SLA compliance rate")
    substantiated_rate: float = Field(..., ge=0.0, le=1.0, description="Rate of substantiated complaints")
    anonymous_complaints: int = Field(..., ge=0, description="Number of anonymous complaints")
    period_start: datetime = Field(..., description="Statistics period start")
    period_end: datetime = Field(..., description="Statistics period end")


class FPICResult(BaseModel):
    """Free Prior Informed Consent verification result."""

    fpic_id: str = Field(..., description="FPIC verification identifier")
    plot_id: str = Field(..., description="Plot identifier")
    indigenous_community: str = Field(..., description="Indigenous community name")
    consent_obtained: bool = Field(..., description="Whether FPIC was obtained")
    consent_date: Optional[datetime] = Field(None, description="Date consent was obtained")
    consultation_process: str = Field(..., description="Description of consultation process")
    documentation: List[str] = Field(..., description="Supporting documentation IDs")
    verification_status: str = Field(..., description="Verification status")
    verified_by: Optional[str] = Field(None, description="Third-party verifier (if applicable)")
    issues_identified: List[str] = Field(default_factory=list, description="Issues identified")


class AnonymousReceipt(BaseModel):
    """Receipt for anonymous submission."""

    receipt_id: str = Field(..., description="Receipt identifier")
    tracking_code: str = Field(..., description="Tracking code for status checks")
    submission_timestamp: datetime = Field(..., description="Submission timestamp")
    estimated_response_date: datetime = Field(..., description="Estimated response date")
    instructions: str = Field(..., description="Instructions for using tracking code")


class SecureReport(BaseModel):
    """Secure whistleblower report."""

    report_id: str = Field(..., description="Report identifier")
    encrypted_content: str = Field(..., description="Encrypted report content")
    tracking_code: str = Field(..., description="Secure tracking code")
    submission_timestamp: datetime = Field(..., description="Submission timestamp")
    protection_level: str = Field("MAXIMUM", description="Protection level")
    secure_channel: str = Field(..., description="Secure communication channel")


class GrievanceMechanismEngine:
    """
    Grievance mechanism engine for EUDR compliance.

    Implements stakeholder complaint handling, investigation, FPIC verification,
    and resolution tracking per EUDR requirements and UN Guiding Principles.

    Attributes:
        config: Engine configuration
        complaints: Complaint registry
        investigations: Investigation registry
        resolutions: Resolution registry

    Example:
        >>> config = GrievanceConfig(enabled=True)
        >>> engine = GrievanceMechanismEngine(config)
        >>> complaint_data = {...}
        >>> complaint = engine.register_complaint(complaint_data)
        >>> print(f"Complaint registered: {complaint.complaint_id}")
    """

    def __init__(self, config: GrievanceConfig):
        """Initialize grievance mechanism engine."""
        self.config = config
        self.complaints: Dict[str, Complaint] = {}
        self.investigations: Dict[str, Investigation] = {}
        self.resolutions: Dict[str, Resolution] = {}
        logger.info(f"GrievanceMechanismEngine initialized with response_sla={config.response_sla_days} days, "
                   f"resolution_sla={config.resolution_sla_days} days")

    def register_complaint(self, complaint_data: Dict[str, Any]) -> Complaint:
        """
        Register new grievance complaint.

        Args:
            complaint_data: Complaint data including type, severity, description, etc.

        Returns:
            Registered complaint with tracking code

        Raises:
            ValueError: If required fields are missing or mechanism is disabled
        """
        try:
            if not self.config.enabled:
                raise ValueError("Grievance mechanism is disabled")

            # Generate IDs
            complaint_id = f"COMPLAINT-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"
            tracking_code = f"GRV-{secrets.token_hex(6).upper()}"

            # Extract data
            source = ComplaintSource(complaint_data.get('source', ComplaintSource.ANONYMOUS))
            complaint_type = ComplaintType(complaint_data['type'])
            severity = ComplaintSeverity(complaint_data.get('severity', ComplaintSeverity.MEDIUM))
            description = complaint_data['description']
            is_anonymous = source == ComplaintSource.ANONYMOUS or complaint_data.get('anonymous', False)

            # Create initial timeline
            submitted_at = datetime.utcnow()
            timeline = [ComplaintTimeline(
                timestamp=submitted_at,
                event_type="SUBMITTED",
                actor="SYSTEM",
                description="Complaint submitted and registered"
            )]

            # Create complaint
            complaint = Complaint(
                complaint_id=complaint_id,
                source=source,
                type=complaint_type,
                severity=severity,
                description=description,
                linked_plots=complaint_data.get('linked_plots', []),
                linked_suppliers=complaint_data.get('linked_suppliers', []),
                linked_dds=complaint_data.get('linked_dds', []),
                status=ComplaintStatus.SUBMITTED,
                timeline=timeline,
                submitted_at=submitted_at,
                submitter_contact=complaint_data.get('contact') if not is_anonymous else None,
                is_anonymous=is_anonymous,
                tracking_code=tracking_code,
                confidential=complaint_data.get('confidential', False)
            )

            # Store complaint
            self.complaints[complaint_id] = complaint

            # Auto-triage if enabled
            if self.config.auto_triage:
                self.triage_complaint(complaint_id)

            logger.info(f"Registered complaint {complaint_id} (type={complaint_type.value}, "
                       f"severity={severity.value}, anonymous={is_anonymous})")
            return complaint

        except Exception as e:
            logger.error(f"Failed to register complaint: {str(e)}", exc_info=True)
            raise

    def triage_complaint(self, complaint_id: str) -> TriageResult:
        """
        Triage complaint to assign severity and investigator.

        Args:
            complaint_id: Complaint identifier

        Returns:
            Triage result with assignments

        Raises:
            ValueError: If complaint not found
        """
        try:
            complaint = self.complaints.get(complaint_id)
            if not complaint:
                raise ValueError(f"Complaint {complaint_id} not found")

            # Assign investigator based on complaint type
            investigator_map = {
                ComplaintType.DEFORESTATION: "Environmental Compliance Team",
                ComplaintType.LAND_RIGHTS: "Human Rights Officer",
                ComplaintType.INDIGENOUS_RIGHTS: "Human Rights Officer",
                ComplaintType.LABOR_RIGHTS: "Labor Compliance Officer",
                ComplaintType.FPIC_VIOLATION: "Human Rights Officer",
                ComplaintType.CORRUPTION: "Ethics & Compliance Officer",
                ComplaintType.DATA_INACCURACY: "Data Quality Manager",
            }
            assigned_investigator = investigator_map.get(complaint.type, "General Compliance Officer")

            # Calculate investigation deadline based on severity
            deadline_map = {
                ComplaintSeverity.CRITICAL: timedelta(hours=24),
                ComplaintSeverity.HIGH: timedelta(days=3),
                ComplaintSeverity.MEDIUM: timedelta(days=7),
                ComplaintSeverity.LOW: timedelta(days=14),
            }
            investigation_deadline = complaint.submitted_at + deadline_map[complaint.severity]

            # Determine recommended actions
            recommended_actions = []
            if complaint.severity == ComplaintSeverity.CRITICAL:
                recommended_actions.append("Immediate site inspection required")
                recommended_actions.append("Notify senior management within 2 hours")
            if complaint.type == ComplaintType.DEFORESTATION:
                recommended_actions.append("Verify plot geolocation data")
                recommended_actions.append("Review satellite imagery for affected area")
            if complaint.type == ComplaintType.FPIC_VIOLATION:
                recommended_actions.append("Engage with indigenous community representatives")
                recommended_actions.append("Suspend operations pending FPIC verification")

            # Check if escalation needed
            requires_escalation = (
                complaint.severity == ComplaintSeverity.CRITICAL or
                complaint.type in [ComplaintType.FPIC_VIOLATION, ComplaintType.CORRUPTION]
            )

            triage_result = TriageResult(
                triage_id=f"TRIAGE-{complaint_id}",
                complaint_id=complaint_id,
                assigned_severity=complaint.severity,
                assigned_investigator=assigned_investigator,
                investigation_deadline=investigation_deadline,
                triage_notes=f"Auto-triaged based on type={complaint.type.value}, severity={complaint.severity.value}",
                recommended_actions=recommended_actions,
                requires_escalation=requires_escalation
            )

            # Update complaint status
            complaint.status = ComplaintStatus.TRIAGED
            complaint.timeline.append(ComplaintTimeline(
                timestamp=datetime.utcnow(),
                event_type="TRIAGED",
                actor="AUTO_TRIAGE_SYSTEM",
                description=f"Assigned to {assigned_investigator}, deadline {investigation_deadline.isoformat()}",
                metadata={"investigator": assigned_investigator, "deadline": investigation_deadline.isoformat()}
            ))

            logger.info(f"Triaged complaint {complaint_id}: assigned to {assigned_investigator}, "
                       f"deadline {investigation_deadline.isoformat()}")
            return triage_result

        except Exception as e:
            logger.error(f"Failed to triage complaint: {str(e)}", exc_info=True)
            raise

    def start_investigation(self, complaint_id: str) -> Investigation:
        """
        Start investigation for complaint.

        Args:
            complaint_id: Complaint identifier

        Returns:
            Investigation record

        Raises:
            ValueError: If complaint not found or already under investigation
        """
        try:
            complaint = self.complaints.get(complaint_id)
            if not complaint:
                raise ValueError(f"Complaint {complaint_id} not found")

            if complaint.status == ComplaintStatus.UNDER_INVESTIGATION:
                raise ValueError(f"Complaint {complaint_id} already under investigation")

            # Create investigation
            investigation_id = f"INV-{complaint_id}"
            investigation = Investigation(
                investigation_id=investigation_id,
                complaint_id=complaint_id,
                investigator="Compliance Investigator",
                team_members=["Field Officer", "Data Analyst"],
                started_at=datetime.utcnow(),
                findings="Investigation in progress",
                evidence=[],
                status="IN_PROGRESS"
            )

            # Store investigation
            self.investigations[investigation_id] = investigation

            # Update complaint
            complaint.status = ComplaintStatus.UNDER_INVESTIGATION
            complaint.timeline.append(ComplaintTimeline(
                timestamp=datetime.utcnow(),
                event_type="INVESTIGATION_STARTED",
                actor=investigation.investigator,
                description=f"Investigation started by {investigation.investigator}"
            ))

            logger.info(f"Started investigation {investigation_id} for complaint {complaint_id}")
            return investigation

        except Exception as e:
            logger.error(f"Failed to start investigation: {str(e)}", exc_info=True)
            raise

    def link_evidence(self, investigation_id: str, evidence: Dict[str, Any]) -> EvidenceLink:
        """
        Link evidence to investigation.

        Args:
            investigation_id: Investigation identifier
            evidence: Evidence data

        Returns:
            Evidence link record

        Raises:
            ValueError: If investigation not found
        """
        try:
            investigation = self.investigations.get(investigation_id)
            if not investigation:
                raise ValueError(f"Investigation {investigation_id} not found")

            # Create evidence item
            evidence_id = f"EVD-{investigation_id}-{len(investigation.evidence) + 1:03d}"
            evidence_item = EvidenceItem(
                evidence_id=evidence_id,
                evidence_type=evidence['type'],
                description=evidence['description'],
                source=evidence.get('source', 'UNKNOWN'),
                collected_at=datetime.utcnow(),
                file_hash=evidence.get('file_hash'),
                metadata=evidence.get('metadata', {}),
                chain_of_custody=[f"Collected by {investigation.investigator} at {datetime.utcnow().isoformat()}"]
            )

            # Add to investigation
            investigation.evidence.append(evidence_item)

            # Create link
            link = EvidenceLink(
                link_id=f"LINK-{evidence_id}",
                investigation_id=investigation_id,
                evidence_id=evidence_id,
                relevance_score=evidence.get('relevance_score', 1.0),
                linked_at=datetime.utcnow(),
                linked_by=investigation.investigator
            )

            logger.info(f"Linked evidence {evidence_id} to investigation {investigation_id}")
            return link

        except Exception as e:
            logger.error(f"Failed to link evidence: {str(e)}", exc_info=True)
            raise

    def resolve_complaint(self, complaint_id: str, resolution_data: Dict[str, Any]) -> Resolution:
        """
        Resolve complaint with outcome and actions.

        Args:
            complaint_id: Complaint identifier
            resolution_data: Resolution data including type, actions, etc.

        Returns:
            Resolution record

        Raises:
            ValueError: If complaint or investigation not found
        """
        try:
            complaint = self.complaints.get(complaint_id)
            if not complaint:
                raise ValueError(f"Complaint {complaint_id} not found")

            investigation_id = f"INV-{complaint_id}"
            investigation = self.investigations.get(investigation_id)
            if not investigation:
                raise ValueError(f"Investigation {investigation_id} not found")

            # Create resolution
            resolution_id = f"RES-{complaint_id}"
            resolved_at = datetime.utcnow()

            resolution = Resolution(
                resolution_id=resolution_id,
                complaint_id=complaint_id,
                investigation_id=investigation_id,
                resolution_type=resolution_data['type'],
                corrective_actions=resolution_data.get('corrective_actions', []),
                preventive_actions=resolution_data.get('preventive_actions', []),
                compensation_provided=resolution_data.get('compensation_provided', False),
                resolved_at=resolved_at,
                resolver=resolution_data.get('resolver', 'Compliance Manager'),
                submitter_notified=False,
                appeal_deadline=resolved_at + timedelta(days=30)
            )

            # Store resolution
            self.resolutions[resolution_id] = resolution

            # Update complaint
            complaint.status = ComplaintStatus.RESOLVED
            complaint.resolved_at = resolved_at
            complaint.timeline.append(ComplaintTimeline(
                timestamp=resolved_at,
                event_type="RESOLVED",
                actor=resolution.resolver,
                description=f"Complaint resolved as {resolution.resolution_type}",
                metadata={"resolution_type": resolution.resolution_type}
            ))

            # Update investigation
            investigation.status = "COMPLETED"
            investigation.completed_at = resolved_at
            investigation.verified = resolution.resolution_type == "SUBSTANTIATED"

            logger.info(f"Resolved complaint {complaint_id} as {resolution.resolution_type}")
            return resolution

        except Exception as e:
            logger.error(f"Failed to resolve complaint: {str(e)}", exc_info=True)
            raise

    def track_sla(self, complaint_id: str) -> SLAStatus:
        """
        Track SLA compliance for complaint.

        Args:
            complaint_id: Complaint identifier

        Returns:
            SLA status

        Raises:
            ValueError: If complaint not found
        """
        try:
            complaint = self.complaints.get(complaint_id)
            if not complaint:
                raise ValueError(f"Complaint {complaint_id} not found")

            # Calculate response SLA
            response_due = complaint.submitted_at + timedelta(days=self.config.response_sla_days)
            response_actual = complaint.acknowledged_at
            response_sla_met = response_actual is not None and response_actual <= response_due

            # Calculate resolution SLA
            resolution_due = complaint.submitted_at + timedelta(days=self.config.resolution_sla_days)
            resolution_actual = complaint.resolved_at
            resolution_sla_met = None
            if resolution_actual:
                resolution_sla_met = resolution_actual <= resolution_due

            # Calculate days overdue
            now = datetime.utcnow()
            days_overdue = 0
            if not response_actual and now > response_due:
                days_overdue = (now - response_due).days
            elif not resolution_actual and now > resolution_due:
                days_overdue = (now - resolution_due).days

            # Check escalation
            escalation_triggered = days_overdue >= self.config.escalation_threshold_days

            sla_status = SLAStatus(
                complaint_id=complaint_id,
                response_sla_met=response_sla_met,
                response_due=response_due,
                response_actual=response_actual,
                resolution_sla_met=resolution_sla_met,
                resolution_due=resolution_due,
                resolution_actual=resolution_actual,
                days_overdue=days_overdue,
                escalation_triggered=escalation_triggered
            )

            logger.info(f"SLA status for {complaint_id}: response_met={response_sla_met}, "
                       f"resolution_met={resolution_sla_met}, overdue={days_overdue} days")
            return sla_status

        except Exception as e:
            logger.error(f"Failed to track SLA: {str(e)}", exc_info=True)
            raise

    def generate_statistics(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> GrievanceStatistics:
        """
        Generate grievance mechanism statistics.

        Args:
            start_date: Period start (defaults to 30 days ago)
            end_date: Period end (defaults to now)

        Returns:
            Comprehensive statistics
        """
        try:
            if not start_date:
                start_date = datetime.utcnow() - timedelta(days=30)
            if not end_date:
                end_date = datetime.utcnow()

            # Filter complaints in period
            period_complaints = [
                c for c in self.complaints.values()
                if start_date <= c.submitted_at <= end_date
            ]

            # Count by type
            complaints_by_type = {}
            for complaint_type in ComplaintType:
                count = sum(1 for c in period_complaints if c.type == complaint_type)
                complaints_by_type[complaint_type.value] = count

            # Count by severity
            complaints_by_severity = {}
            for severity in ComplaintSeverity:
                count = sum(1 for c in period_complaints if c.severity == severity)
                complaints_by_severity[severity.value] = count

            # Count by status
            complaints_by_status = {}
            for status in ComplaintStatus:
                count = sum(1 for c in period_complaints if c.status == status)
                complaints_by_status[status.value] = count

            # Calculate average resolution days
            resolved_complaints = [c for c in period_complaints if c.resolved_at]
            if resolved_complaints:
                total_days = sum((c.resolved_at - c.submitted_at).days for c in resolved_complaints)
                avg_resolution_days = total_days / len(resolved_complaints)
            else:
                avg_resolution_days = 0.0

            # Calculate SLA compliance rate
            sla_met_count = 0
            for complaint in period_complaints:
                sla = self.track_sla(complaint.complaint_id)
                if sla.response_sla_met and (sla.resolution_sla_met is None or sla.resolution_sla_met):
                    sla_met_count += 1
            sla_compliance_rate = sla_met_count / len(period_complaints) if period_complaints else 0.0

            # Calculate substantiated rate
            substantiated_count = sum(
                1 for res in self.resolutions.values()
                if res.resolution_type == "SUBSTANTIATED" and res.complaint_id in [c.complaint_id for c in period_complaints]
            )
            substantiated_rate = substantiated_count / len(resolved_complaints) if resolved_complaints else 0.0

            # Count anonymous complaints
            anonymous_complaints = sum(1 for c in period_complaints if c.is_anonymous)

            statistics = GrievanceStatistics(
                total_complaints=len(period_complaints),
                complaints_by_type=complaints_by_type,
                complaints_by_severity=complaints_by_severity,
                complaints_by_status=complaints_by_status,
                avg_resolution_days=avg_resolution_days,
                sla_compliance_rate=sla_compliance_rate,
                substantiated_rate=substantiated_rate,
                anonymous_complaints=anonymous_complaints,
                period_start=start_date,
                period_end=end_date
            )

            logger.info(f"Generated statistics: {len(period_complaints)} complaints, "
                       f"avg resolution {avg_resolution_days:.1f} days, SLA compliance {sla_compliance_rate:.1%}")
            return statistics

        except Exception as e:
            logger.error(f"Failed to generate statistics: {str(e)}", exc_info=True)
            raise

    def check_fpic(self, plot_data: Dict[str, Any]) -> FPICResult:
        """
        Check Free Prior Informed Consent for plot.

        Args:
            plot_data: Plot data including location and community information

        Returns:
            FPIC verification result

        Raises:
            ValueError: If FPIC verification is disabled
        """
        try:
            if not self.config.fpic_verification:
                raise ValueError("FPIC verification is disabled")

            plot_id = plot_data['plot_id']
            indigenous_community = plot_data.get('indigenous_community', 'UNKNOWN')

            # Mock FPIC verification (in production, check actual documentation)
            consent_obtained = plot_data.get('fpic_obtained', False)
            consent_date = plot_data.get('fpic_date')
            consultation_process = plot_data.get('consultation_process', 'No consultation process documented')
            documentation = plot_data.get('fpic_documentation', [])

            # Identify issues
            issues_identified = []
            if not consent_obtained:
                issues_identified.append("No FPIC documentation found")
            if not documentation:
                issues_identified.append("No supporting documentation provided")
            if consent_date and (datetime.utcnow() - consent_date).days > 365:
                issues_identified.append("FPIC consent is older than 12 months - renewal recommended")

            verification_status = "VERIFIED" if consent_obtained and not issues_identified else "UNVERIFIED"

            result = FPICResult(
                fpic_id=f"FPIC-{plot_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                plot_id=plot_id,
                indigenous_community=indigenous_community,
                consent_obtained=consent_obtained,
                consent_date=consent_date,
                consultation_process=consultation_process,
                documentation=documentation,
                verification_status=verification_status,
                verified_by=plot_data.get('verified_by'),
                issues_identified=issues_identified
            )

            logger.info(f"FPIC check for plot {plot_id}: consent={consent_obtained}, "
                       f"status={verification_status}, issues={len(issues_identified)}")
            return result

        except Exception as e:
            logger.error(f"FPIC check failed: {str(e)}", exc_info=True)
            raise

    def anonymous_submission(self, data: Dict[str, Any]) -> AnonymousReceipt:
        """
        Handle anonymous complaint submission.

        Args:
            data: Complaint data

        Returns:
            Anonymous receipt with tracking code

        Raises:
            ValueError: If anonymous submissions are disabled
        """
        try:
            if not self.config.anonymous:
                raise ValueError("Anonymous submissions are disabled")

            # Force anonymous
            data['source'] = ComplaintSource.ANONYMOUS
            data['anonymous'] = True

            # Register complaint
            complaint = self.register_complaint(data)

            # Generate receipt
            estimated_response = complaint.submitted_at + timedelta(days=self.config.response_sla_days)

            receipt = AnonymousReceipt(
                receipt_id=f"RCPT-{complaint.complaint_id}",
                tracking_code=complaint.tracking_code,
                submission_timestamp=complaint.submitted_at,
                estimated_response_date=estimated_response,
                instructions=f"Use tracking code '{complaint.tracking_code}' to check status. "
                            f"Keep this code secure. Expected response by {estimated_response.strftime('%Y-%m-%d')}."
            )

            logger.info(f"Anonymous submission registered with tracking code {complaint.tracking_code}")
            return receipt

        except Exception as e:
            logger.error(f"Anonymous submission failed: {str(e)}", exc_info=True)
            raise

    def whistleblower_report(self, data: Dict[str, Any]) -> SecureReport:
        """
        Handle secure whistleblower report submission.

        Args:
            data: Report data

        Returns:
            Secure report with encrypted tracking

        Raises:
            ValueError: If whistleblower protection is disabled
        """
        try:
            if not self.config.whistleblower_protection:
                raise ValueError("Whistleblower protection is disabled")

            # Force whistleblower source
            data['source'] = ComplaintSource.WHISTLEBLOWER
            data['anonymous'] = True
            data['confidential'] = True

            # Register complaint
            complaint = self.register_complaint(data)

            # Create secure report (mock encryption - use real crypto in production)
            encrypted_content = hashlib.sha256(json.dumps(data).encode()).hexdigest()
            secure_tracking_code = f"WB-{secrets.token_hex(12).upper()}"

            report = SecureReport(
                report_id=f"WB-{complaint.complaint_id}",
                encrypted_content=encrypted_content,
                tracking_code=secure_tracking_code,
                submission_timestamp=complaint.submitted_at,
                protection_level="MAXIMUM",
                secure_channel="ENCRYPTED_PORTAL"
            )

            logger.info(f"Whistleblower report registered with maximum protection")
            return report

        except Exception as e:
            logger.error(f"Whistleblower report failed: {str(e)}", exc_info=True)
            raise
