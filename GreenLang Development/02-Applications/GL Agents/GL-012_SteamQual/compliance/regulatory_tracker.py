"""
GL-012 SteamQual - Regulatory Compliance Tracker

Comprehensive tracking of regulatory compliance for steam systems including
steam system regulations, environmental permits, and energy efficiency standards.

Regulatory References:
- ASME Boiler and Pressure Vessel Code
- EPA 40 CFR Part 60, 63, 75, 98
- OSHA 29 CFR 1910
- DOE 10 CFR 430, 431 (Energy Efficiency)
- State and Local Air Quality Permits

This module provides:
1. Regulatory requirement tracking and management
2. Compliance status monitoring and alerting
3. Permit tracking with renewal dates
4. Energy efficiency standard compliance
5. Audit trail for regulatory inspections

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# REGULATORY TRACKER ENUMERATIONS
# =============================================================================

class RegulatoryAgency(Enum):
    """Regulatory agencies with jurisdiction over steam systems."""

    # Federal
    EPA = "US Environmental Protection Agency"
    OSHA = "Occupational Safety and Health Administration"
    DOE = "US Department of Energy"

    # Standards Organizations
    ASME = "American Society of Mechanical Engineers"
    NFPA = "National Fire Protection Association"
    API = "American Petroleum Institute"

    # State/Local
    STATE_EPA = "State Environmental Agency"
    LOCAL_AQD = "Local Air Quality District"
    STATE_BOILER = "State Boiler Inspector"


class RegulationType(Enum):
    """Types of regulatory requirements."""

    EMISSION_LIMIT = "emission_limit"
    MONITORING_REQUIREMENT = "monitoring_requirement"
    REPORTING_REQUIREMENT = "reporting_requirement"
    SAFETY_STANDARD = "safety_standard"
    EFFICIENCY_STANDARD = "efficiency_standard"
    PERMIT_CONDITION = "permit_condition"
    INSPECTION_REQUIREMENT = "inspection_requirement"
    RECORDKEEPING = "recordkeeping"


class ComplianceStatus(Enum):
    """Compliance status for regulatory requirements."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PENDING_VERIFICATION = "pending_verification"
    WAIVER_GRANTED = "waiver_granted"
    UNDER_REVIEW = "under_review"
    NOT_APPLICABLE = "not_applicable"
    EXPIRED = "expired"


class PermitType(Enum):
    """Types of environmental and operating permits."""

    TITLE_V = "title_v_operating_permit"
    MINOR_SOURCE = "minor_source_permit"
    NSR_PSD = "new_source_review_psd"
    NSR_NNSR = "new_source_review_nnsr"
    BOILER_OPERATING = "boiler_operating_permit"
    PRESSURE_VESSEL = "pressure_vessel_permit"
    WASTEWATER = "wastewater_discharge_permit"
    STORMWATER = "stormwater_permit"


class AlertPriority(Enum):
    """Priority levels for compliance alerts."""

    CRITICAL = "critical"      # Immediate action required
    HIGH = "high"              # Action within 24 hours
    MEDIUM = "medium"          # Action within 7 days
    LOW = "low"                # Action within 30 days
    INFORMATIONAL = "informational"


# =============================================================================
# REGULATORY REQUIREMENTS
# =============================================================================

@dataclass
class RegulatoryRequirement:
    """
    Single regulatory requirement with compliance tracking.

    Defines a specific regulatory requirement applicable
    to steam systems with full citation.
    """

    requirement_id: str
    title: str
    description: str
    agency: RegulatoryAgency
    regulation_type: RegulationType
    cfr_citation: str
    effective_date: datetime
    applicability_criteria: str
    compliance_frequency: str  # "continuous", "daily", "monthly", "annual"
    compliance_method: str
    documentation_required: List[str]
    penalty_per_violation: Optional[Decimal] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "requirement_id": self.requirement_id,
            "title": self.title,
            "description": self.description,
            "agency": self.agency.value,
            "regulation_type": self.regulation_type.value,
            "cfr_citation": self.cfr_citation,
            "effective_date": self.effective_date.isoformat(),
            "applicability_criteria": self.applicability_criteria,
            "compliance_frequency": self.compliance_frequency,
            "compliance_method": self.compliance_method,
            "documentation_required": self.documentation_required,
            "penalty_per_violation": str(self.penalty_per_violation) if self.penalty_per_violation else None,
            "notes": self.notes,
        }


# Standard steam system regulatory requirements
STEAM_SYSTEM_REQUIREMENTS: List[RegulatoryRequirement] = [
    # EPA GHG Reporting
    RegulatoryRequirement(
        requirement_id="EPA-GHG-001",
        title="GHG Emissions Reporting",
        description="Annual reporting of greenhouse gas emissions from stationary combustion sources",
        agency=RegulatoryAgency.EPA,
        regulation_type=RegulationType.REPORTING_REQUIREMENT,
        cfr_citation="40 CFR Part 98 Subpart C",
        effective_date=datetime(2010, 1, 1, tzinfo=timezone.utc),
        applicability_criteria="Facilities emitting >= 25,000 MT CO2e annually",
        compliance_frequency="annual",
        compliance_method="Calculate and report via EPA e-GGRT",
        documentation_required=[
            "Fuel consumption records",
            "Emission calculations",
            "Quality assurance records",
            "Calibration records",
        ],
        penalty_per_violation=Decimal("37500"),
        notes="Report due March 31 each year",
    ),

    # EPA NSPS for Boilers
    RegulatoryRequirement(
        requirement_id="EPA-NSPS-001",
        title="NSPS Subpart Db Emission Limits",
        description="Emission limits for SO2, NOx, and PM from industrial boilers",
        agency=RegulatoryAgency.EPA,
        regulation_type=RegulationType.EMISSION_LIMIT,
        cfr_citation="40 CFR Part 60 Subpart Db",
        effective_date=datetime(1987, 6, 19, tzinfo=timezone.utc),
        applicability_criteria="Steam generating units > 100 MMBtu/hr constructed after June 19, 1984",
        compliance_frequency="continuous",
        compliance_method="CEMS or periodic stack testing",
        documentation_required=[
            "CEMS data records",
            "Stack test reports",
            "Operating logs",
            "Fuel analysis records",
        ],
        penalty_per_violation=Decimal("51744"),
    ),

    # EPA MACT for Boilers
    RegulatoryRequirement(
        requirement_id="EPA-MACT-001",
        title="Boiler MACT HAP Limits",
        description="Maximum achievable control technology limits for HAP emissions",
        agency=RegulatoryAgency.EPA,
        regulation_type=RegulationType.EMISSION_LIMIT,
        cfr_citation="40 CFR Part 63 Subpart DDDDD",
        effective_date=datetime(2013, 1, 31, tzinfo=timezone.utc),
        applicability_criteria="Major source industrial boilers",
        compliance_frequency="continuous",
        compliance_method="Tune-up, energy assessment, stack testing",
        documentation_required=[
            "Tune-up records",
            "Energy assessment report",
            "Stack test reports",
            "CMS records",
        ],
    ),

    # ASME Boiler Code
    RegulatoryRequirement(
        requirement_id="ASME-BPV-001",
        title="Boiler and Pressure Vessel Code Compliance",
        description="Design, construction, and operation per ASME BPVC",
        agency=RegulatoryAgency.ASME,
        regulation_type=RegulationType.SAFETY_STANDARD,
        cfr_citation="ASME BPVC Section I, IV",
        effective_date=datetime(2000, 1, 1, tzinfo=timezone.utc),
        applicability_criteria="All steam boilers and pressure vessels",
        compliance_frequency="continuous",
        compliance_method="Periodic inspection by authorized inspector",
        documentation_required=[
            "ASME data reports",
            "Inspection certificates",
            "Repair records",
            "Safety valve certifications",
        ],
    ),

    # OSHA Process Safety
    RegulatoryRequirement(
        requirement_id="OSHA-PSM-001",
        title="Process Safety Management",
        description="PSM requirements for facilities with highly hazardous chemicals",
        agency=RegulatoryAgency.OSHA,
        regulation_type=RegulationType.SAFETY_STANDARD,
        cfr_citation="29 CFR 1910.119",
        effective_date=datetime(1992, 5, 26, tzinfo=timezone.utc),
        applicability_criteria="Facilities with > 10,000 lb flammable materials",
        compliance_frequency="continuous",
        compliance_method="PSM program implementation and audit",
        documentation_required=[
            "Process safety information",
            "Process hazard analysis",
            "Operating procedures",
            "Training records",
            "Mechanical integrity records",
            "Incident investigation reports",
        ],
        penalty_per_violation=Decimal("15625"),
    ),

    # DOE Energy Efficiency
    RegulatoryRequirement(
        requirement_id="DOE-EFF-001",
        title="Commercial Boiler Efficiency Standards",
        description="Minimum efficiency standards for commercial boilers",
        agency=RegulatoryAgency.DOE,
        regulation_type=RegulationType.EFFICIENCY_STANDARD,
        cfr_citation="10 CFR 431.86",
        effective_date=datetime(2012, 3, 2, tzinfo=timezone.utc),
        applicability_criteria="Commercial packaged boilers",
        compliance_frequency="at_installation",
        compliance_method="Equipment certification",
        documentation_required=[
            "Equipment certification",
            "Efficiency test reports",
            "Installation records",
        ],
    ),

    # Continuous Monitoring
    RegulatoryRequirement(
        requirement_id="EPA-CEMS-001",
        title="Continuous Emission Monitoring",
        description="CEMS requirements for regulated emission sources",
        agency=RegulatoryAgency.EPA,
        regulation_type=RegulationType.MONITORING_REQUIREMENT,
        cfr_citation="40 CFR Part 75",
        effective_date=datetime(1995, 1, 1, tzinfo=timezone.utc),
        applicability_criteria="Acid Rain Program affected units",
        compliance_frequency="continuous",
        compliance_method="Operate and maintain certified CEMS",
        documentation_required=[
            "CEMS hourly data",
            "Quality assurance test records",
            "Calibration records",
            "Maintenance logs",
        ],
    ),
]


# =============================================================================
# PERMIT TRACKING
# =============================================================================

@dataclass
class EnvironmentalPermit:
    """
    Environmental or operating permit with tracking.

    Tracks permit status, conditions, and renewal dates.
    """

    permit_id: str
    permit_number: str
    permit_type: PermitType
    issuing_agency: RegulatoryAgency
    facility_id: str
    facility_name: str
    issue_date: datetime
    expiration_date: datetime
    renewal_application_due: datetime

    # Permit conditions
    conditions: List[Dict[str, Any]]
    emission_limits: List[Dict[str, Any]]
    monitoring_requirements: List[str]
    reporting_requirements: List[str]

    # Status
    status: ComplianceStatus
    last_inspection_date: Optional[datetime] = None
    next_inspection_date: Optional[datetime] = None
    modification_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "permit_id": self.permit_id,
            "permit_number": self.permit_number,
            "permit_type": self.permit_type.value,
            "issuing_agency": self.issuing_agency.value,
            "facility_id": self.facility_id,
            "facility_name": self.facility_name,
            "issue_date": self.issue_date.isoformat(),
            "expiration_date": self.expiration_date.isoformat(),
            "renewal_application_due": self.renewal_application_due.isoformat(),
            "conditions": self.conditions,
            "emission_limits": self.emission_limits,
            "monitoring_requirements": self.monitoring_requirements,
            "reporting_requirements": self.reporting_requirements,
            "status": self.status.value,
            "last_inspection_date": self.last_inspection_date.isoformat() if self.last_inspection_date else None,
            "next_inspection_date": self.next_inspection_date.isoformat() if self.next_inspection_date else None,
            "modification_history": self.modification_history,
        }

    def days_until_expiration(self) -> int:
        """Calculate days until permit expires."""
        now = datetime.now(timezone.utc)
        delta = self.expiration_date - now
        return delta.days

    def days_until_renewal_due(self) -> int:
        """Calculate days until renewal application is due."""
        now = datetime.now(timezone.utc)
        delta = self.renewal_application_due - now
        return delta.days

    def is_renewal_required(self, threshold_days: int = 180) -> bool:
        """Check if renewal application should be initiated."""
        return self.days_until_renewal_due() <= threshold_days


# =============================================================================
# COMPLIANCE TRACKING
# =============================================================================

@dataclass
class ComplianceRecord:
    """
    Record of compliance status for a specific requirement.

    Provides complete audit trail for compliance verification.
    """

    record_id: str
    requirement_id: str
    evaluation_date: datetime
    evaluation_period_start: datetime
    evaluation_period_end: datetime
    status: ComplianceStatus
    measured_value: Optional[Decimal]
    limit_value: Optional[Decimal]
    unit: Optional[str]
    evaluator_id: str
    evidence_documents: List[str]
    findings: str
    corrective_actions: Optional[List[Dict[str, Any]]]
    provenance_hash: str
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "record_id": self.record_id,
            "requirement_id": self.requirement_id,
            "evaluation_date": self.evaluation_date.isoformat(),
            "evaluation_period_start": self.evaluation_period_start.isoformat(),
            "evaluation_period_end": self.evaluation_period_end.isoformat(),
            "status": self.status.value,
            "measured_value": str(self.measured_value) if self.measured_value else None,
            "limit_value": str(self.limit_value) if self.limit_value else None,
            "unit": self.unit,
            "evaluator_id": self.evaluator_id,
            "evidence_documents": self.evidence_documents,
            "findings": self.findings,
            "corrective_actions": self.corrective_actions,
            "provenance_hash": self.provenance_hash,
            "notes": self.notes,
        }


@dataclass
class ComplianceAlert:
    """
    Compliance alert for upcoming deadlines or violations.

    Provides notification for compliance-related events.
    """

    alert_id: str
    timestamp: datetime
    priority: AlertPriority
    alert_type: str  # "deadline", "violation", "renewal", "inspection"
    title: str
    description: str
    related_requirement_id: Optional[str]
    related_permit_id: Optional[str]
    due_date: Optional[datetime]
    action_required: str
    responsible_party: Optional[str]
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "alert_type": self.alert_type,
            "title": self.title,
            "description": self.description,
            "related_requirement_id": self.related_requirement_id,
            "related_permit_id": self.related_permit_id,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "action_required": self.action_required,
            "responsible_party": self.responsible_party,
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
        }


# =============================================================================
# REGULATORY COMPLIANCE TRACKER
# =============================================================================

class RegulatoryComplianceTracker:
    """
    Central tracker for regulatory compliance.

    Manages requirements, permits, compliance records, and alerts
    for steam system regulatory compliance.

    Example:
        >>> tracker = RegulatoryComplianceTracker()
        >>> tracker.add_permit(permit)
        >>> record = tracker.evaluate_compliance(requirement_id, measured_value)
        >>> alerts = tracker.check_upcoming_deadlines()
    """

    VERSION = "1.0.0"

    def __init__(self) -> None:
        """Initialize regulatory compliance tracker."""
        self._requirements: Dict[str, RegulatoryRequirement] = {}
        self._permits: Dict[str, EnvironmentalPermit] = {}
        self._compliance_records: Dict[str, List[ComplianceRecord]] = {}
        self._alerts: List[ComplianceAlert] = []

        # Load standard requirements
        for req in STEAM_SYSTEM_REQUIREMENTS:
            self._requirements[req.requirement_id] = req

        logger.info(
            f"RegulatoryComplianceTracker initialized with "
            f"{len(self._requirements)} requirements"
        )

    def add_requirement(self, requirement: RegulatoryRequirement) -> None:
        """
        Add a regulatory requirement to tracking.

        Args:
            requirement: Regulatory requirement to add
        """
        self._requirements[requirement.requirement_id] = requirement
        self._compliance_records[requirement.requirement_id] = []
        logger.info(f"Added requirement: {requirement.requirement_id}")

    def get_requirement(self, requirement_id: str) -> Optional[RegulatoryRequirement]:
        """Get requirement by ID."""
        return self._requirements.get(requirement_id)

    def get_requirements_by_agency(
        self,
        agency: RegulatoryAgency,
    ) -> List[RegulatoryRequirement]:
        """Get all requirements from a specific agency."""
        return [
            req for req in self._requirements.values()
            if req.agency == agency
        ]

    def get_requirements_by_type(
        self,
        regulation_type: RegulationType,
    ) -> List[RegulatoryRequirement]:
        """Get all requirements of a specific type."""
        return [
            req for req in self._requirements.values()
            if req.regulation_type == regulation_type
        ]

    def add_permit(self, permit: EnvironmentalPermit) -> None:
        """
        Add an environmental permit to tracking.

        Args:
            permit: Permit to add
        """
        self._permits[permit.permit_id] = permit
        logger.info(f"Added permit: {permit.permit_id} ({permit.permit_number})")

    def get_permit(self, permit_id: str) -> Optional[EnvironmentalPermit]:
        """Get permit by ID."""
        return self._permits.get(permit_id)

    def get_permits_by_type(
        self,
        permit_type: PermitType,
    ) -> List[EnvironmentalPermit]:
        """Get all permits of a specific type."""
        return [
            p for p in self._permits.values()
            if p.permit_type == permit_type
        ]

    def evaluate_compliance(
        self,
        requirement_id: str,
        evaluation_period_start: datetime,
        evaluation_period_end: datetime,
        measured_value: Optional[Union[Decimal, float]] = None,
        limit_value: Optional[Union[Decimal, float]] = None,
        unit: Optional[str] = None,
        evaluator_id: str = "system",
        evidence_documents: Optional[List[str]] = None,
        findings: str = "",
    ) -> ComplianceRecord:
        """
        Evaluate and record compliance status.

        Uses deterministic comparison against limits.

        Args:
            requirement_id: Requirement being evaluated
            evaluation_period_start: Start of evaluation period
            evaluation_period_end: End of evaluation period
            measured_value: Measured/calculated value
            limit_value: Applicable limit
            unit: Unit of measurement
            evaluator_id: ID of evaluator
            evidence_documents: Supporting documents
            findings: Findings summary

        Returns:
            ComplianceRecord with status
        """
        timestamp = datetime.now(timezone.utc)

        requirement = self._requirements.get(requirement_id)
        if requirement is None:
            raise ValueError(f"Unknown requirement: {requirement_id}")

        # Determine compliance status
        status = ComplianceStatus.COMPLIANT
        if measured_value is not None and limit_value is not None:
            measured_dec = Decimal(str(measured_value))
            limit_dec = Decimal(str(limit_value))

            # For emission limits, measured should be <= limit
            if measured_dec > limit_dec:
                status = ComplianceStatus.NON_COMPLIANT

        # Provenance hash
        provenance_data = {
            "requirement_id": requirement_id,
            "evaluation_date": timestamp.isoformat(),
            "measured_value": str(measured_value) if measured_value else None,
            "status": status.value,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        record = ComplianceRecord(
            record_id=f"CR-{timestamp.strftime('%Y%m%d%H%M%S')}-{requirement_id[:8]}",
            requirement_id=requirement_id,
            evaluation_date=timestamp,
            evaluation_period_start=evaluation_period_start,
            evaluation_period_end=evaluation_period_end,
            status=status,
            measured_value=Decimal(str(measured_value)) if measured_value else None,
            limit_value=Decimal(str(limit_value)) if limit_value else None,
            unit=unit,
            evaluator_id=evaluator_id,
            evidence_documents=evidence_documents or [],
            findings=findings,
            corrective_actions=None if status == ComplianceStatus.COMPLIANT else [],
            provenance_hash=provenance_hash,
        )

        # Store record
        if requirement_id not in self._compliance_records:
            self._compliance_records[requirement_id] = []
        self._compliance_records[requirement_id].append(record)

        # Generate alert if non-compliant
        if status == ComplianceStatus.NON_COMPLIANT:
            self._create_alert(
                priority=AlertPriority.HIGH,
                alert_type="violation",
                title=f"Non-compliance: {requirement.title}",
                description=f"Measured value {measured_value} exceeds limit {limit_value}",
                related_requirement_id=requirement_id,
                action_required="Investigate and implement corrective action",
            )

        logger.info(f"Compliance evaluation: {requirement_id} = {status.value}")
        return record

    def get_compliance_history(
        self,
        requirement_id: str,
    ) -> List[ComplianceRecord]:
        """Get compliance history for a requirement."""
        return self._compliance_records.get(requirement_id, [])

    def check_upcoming_deadlines(
        self,
        days_ahead: int = 30,
    ) -> List[ComplianceAlert]:
        """
        Check for upcoming compliance deadlines.

        Args:
            days_ahead: Days to look ahead

        Returns:
            List of alerts for upcoming deadlines
        """
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(days=days_ahead)
        alerts = []

        # Check permit renewals
        for permit in self._permits.values():
            if permit.renewal_application_due <= cutoff:
                days_left = permit.days_until_renewal_due()
                priority = AlertPriority.CRITICAL if days_left <= 7 else \
                           AlertPriority.HIGH if days_left <= 30 else \
                           AlertPriority.MEDIUM

                alert = self._create_alert(
                    priority=priority,
                    alert_type="renewal",
                    title=f"Permit Renewal Due: {permit.permit_number}",
                    description=f"Renewal application due in {days_left} days",
                    related_permit_id=permit.permit_id,
                    due_date=permit.renewal_application_due,
                    action_required="Submit renewal application",
                )
                alerts.append(alert)

            if permit.expiration_date <= cutoff:
                days_left = permit.days_until_expiration()
                priority = AlertPriority.CRITICAL

                alert = self._create_alert(
                    priority=priority,
                    alert_type="deadline",
                    title=f"Permit Expiring: {permit.permit_number}",
                    description=f"Permit expires in {days_left} days",
                    related_permit_id=permit.permit_id,
                    due_date=permit.expiration_date,
                    action_required="Ensure permit renewal before expiration",
                )
                alerts.append(alert)

        return alerts

    def check_reporting_deadlines(
        self,
        days_ahead: int = 30,
    ) -> List[ComplianceAlert]:
        """
        Check for upcoming regulatory reporting deadlines.

        Args:
            days_ahead: Days to look ahead

        Returns:
            List of alerts for reporting deadlines
        """
        now = datetime.now(timezone.utc)
        alerts = []

        # Common reporting deadlines
        reporting_deadlines = [
            {
                "requirement_id": "EPA-GHG-001",
                "name": "Annual GHG Report",
                "due_month": 3,
                "due_day": 31,
            },
            {
                "requirement_id": "EPA-CEMS-001",
                "name": "Quarterly CEMS Report",
                "due_months": [1, 4, 7, 10],
                "due_day": 30,
            },
        ]

        for deadline in reporting_deadlines:
            if deadline.get("due_month"):
                due_date = datetime(
                    now.year if now.month <= deadline["due_month"] else now.year + 1,
                    deadline["due_month"],
                    deadline["due_day"],
                    tzinfo=timezone.utc
                )
                if due_date <= now + timedelta(days=days_ahead):
                    days_left = (due_date - now).days
                    priority = AlertPriority.HIGH if days_left <= 14 else AlertPriority.MEDIUM

                    alert = self._create_alert(
                        priority=priority,
                        alert_type="deadline",
                        title=f"Reporting Deadline: {deadline['name']}",
                        description=f"Report due in {days_left} days",
                        related_requirement_id=deadline["requirement_id"],
                        due_date=due_date,
                        action_required="Complete and submit required report",
                    )
                    alerts.append(alert)

        return alerts

    def _create_alert(
        self,
        priority: AlertPriority,
        alert_type: str,
        title: str,
        description: str,
        related_requirement_id: Optional[str] = None,
        related_permit_id: Optional[str] = None,
        due_date: Optional[datetime] = None,
        action_required: str = "",
        responsible_party: Optional[str] = None,
    ) -> ComplianceAlert:
        """Create and store a compliance alert."""
        timestamp = datetime.now(timezone.utc)

        alert = ComplianceAlert(
            alert_id=f"CA-{timestamp.strftime('%Y%m%d%H%M%S')}-{len(self._alerts):04d}",
            timestamp=timestamp,
            priority=priority,
            alert_type=alert_type,
            title=title,
            description=description,
            related_requirement_id=related_requirement_id,
            related_permit_id=related_permit_id,
            due_date=due_date,
            action_required=action_required,
            responsible_party=responsible_party,
        )

        self._alerts.append(alert)
        logger.info(f"Created alert: {alert.alert_id} - {title}")
        return alert

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> bool:
        """
        Acknowledge a compliance alert.

        Args:
            alert_id: Alert to acknowledge
            acknowledged_by: User acknowledging

        Returns:
            True if acknowledged, False if not found
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = datetime.now(timezone.utc)
                logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True
        return False

    def get_active_alerts(
        self,
        min_priority: Optional[AlertPriority] = None,
    ) -> List[ComplianceAlert]:
        """
        Get active (unacknowledged) alerts.

        Args:
            min_priority: Minimum priority to include

        Returns:
            List of active alerts
        """
        priority_order = [
            AlertPriority.CRITICAL,
            AlertPriority.HIGH,
            AlertPriority.MEDIUM,
            AlertPriority.LOW,
            AlertPriority.INFORMATIONAL,
        ]

        alerts = [a for a in self._alerts if not a.acknowledged]

        if min_priority:
            min_idx = priority_order.index(min_priority)
            alerts = [
                a for a in alerts
                if priority_order.index(a.priority) <= min_idx
            ]

        # Sort by priority
        return sorted(alerts, key=lambda a: priority_order.index(a.priority))

    def get_compliance_summary(
        self,
        facility_id: str,
    ) -> Dict[str, Any]:
        """
        Generate compliance summary for a facility.

        Args:
            facility_id: Facility identifier

        Returns:
            Comprehensive compliance summary
        """
        timestamp = datetime.now(timezone.utc)

        # Count compliance status
        status_counts: Dict[str, int] = {s.value: 0 for s in ComplianceStatus}
        for records in self._compliance_records.values():
            if records:
                latest = records[-1]
                status_counts[latest.status.value] += 1

        # Summarize permits
        permit_summary = {
            "total": len(self._permits),
            "expiring_30_days": sum(1 for p in self._permits.values() if p.days_until_expiration() <= 30),
            "renewal_due_180_days": sum(1 for p in self._permits.values() if p.is_renewal_required()),
        }

        # Summarize alerts
        active_alerts = self.get_active_alerts()
        alert_summary = {
            "total_active": len(active_alerts),
            "critical": sum(1 for a in active_alerts if a.priority == AlertPriority.CRITICAL),
            "high": sum(1 for a in active_alerts if a.priority == AlertPriority.HIGH),
        }

        # Calculate overall compliance rate
        total_evaluated = sum(status_counts.values())
        compliant_count = status_counts.get(ComplianceStatus.COMPLIANT.value, 0)
        compliance_rate = (
            f"{(compliant_count / total_evaluated * 100):.1f}%"
            if total_evaluated > 0 else "N/A"
        )

        # Report hash
        report_data = {
            "facility_id": facility_id,
            "timestamp": timestamp.isoformat(),
            "status_counts": status_counts,
            "permit_count": len(self._permits),
        }
        report_hash = hashlib.sha256(
            json.dumps(report_data, sort_keys=True).encode()
        ).hexdigest()

        return {
            "report_metadata": {
                "report_id": f"RCS-{timestamp.strftime('%Y%m%d%H%M%S')}",
                "facility_id": facility_id,
                "generated_at": timestamp.isoformat(),
                "report_hash": report_hash,
                "generator_version": self.VERSION,
            },
            "compliance_overview": {
                "total_requirements": len(self._requirements),
                "requirements_evaluated": total_evaluated,
                "compliance_rate": compliance_rate,
                "status_breakdown": status_counts,
            },
            "permit_status": permit_summary,
            "alert_status": alert_summary,
            "critical_items": [a.to_dict() for a in active_alerts if a.priority == AlertPriority.CRITICAL],
        }

    def generate_regulatory_report(
        self,
        facility_id: str,
        report_period_start: datetime,
        report_period_end: datetime,
        include_all_records: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive regulatory compliance report.

        Args:
            facility_id: Facility identifier
            report_period_start: Report period start
            report_period_end: Report period end
            include_all_records: Include all compliance records

        Returns:
            Complete regulatory report for audit
        """
        timestamp = datetime.now(timezone.utc)

        # Filter records to period
        period_records: Dict[str, List[Dict]] = {}
        for req_id, records in self._compliance_records.items():
            period_records[req_id] = [
                r.to_dict() for r in records
                if report_period_start <= r.evaluation_date <= report_period_end
            ] if include_all_records else (
                [records[-1].to_dict()] if records else []
            )

        # Get permits
        permits = [p.to_dict() for p in self._permits.values()]

        # Get alerts in period
        period_alerts = [
            a.to_dict() for a in self._alerts
            if report_period_start <= a.timestamp <= report_period_end
        ]

        # Report hash
        report_data = {
            "facility_id": facility_id,
            "period_start": report_period_start.isoformat(),
            "period_end": report_period_end.isoformat(),
            "record_count": sum(len(r) for r in period_records.values()),
        }
        report_hash = hashlib.sha256(
            json.dumps(report_data, sort_keys=True).encode()
        ).hexdigest()

        return {
            "report_metadata": {
                "report_id": f"RCR-{timestamp.strftime('%Y%m%d%H%M%S')}",
                "report_type": "Regulatory Compliance Report",
                "facility_id": facility_id,
                "report_period_start": report_period_start.isoformat(),
                "report_period_end": report_period_end.isoformat(),
                "generated_at": timestamp.isoformat(),
                "report_hash": report_hash,
                "generator_version": self.VERSION,
            },
            "requirements": [r.to_dict() for r in self._requirements.values()],
            "permits": permits,
            "compliance_records": period_records,
            "alerts": period_alerts,
            "summary": self.get_compliance_summary(facility_id),
        }


# =============================================================================
# ENERGY EFFICIENCY STANDARDS TRACKER
# =============================================================================

@dataclass
class EfficiencyStandard:
    """
    Energy efficiency standard specification.

    Defines minimum efficiency requirements for equipment.
    """

    standard_id: str
    title: str
    equipment_type: str
    min_efficiency: Decimal
    efficiency_metric: str  # "thermal_efficiency", "AFUE", "combustion_efficiency"
    cfr_reference: str
    effective_date: datetime
    applicability: str
    test_method: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "standard_id": self.standard_id,
            "title": self.title,
            "equipment_type": self.equipment_type,
            "min_efficiency": str(self.min_efficiency),
            "efficiency_metric": self.efficiency_metric,
            "cfr_reference": self.cfr_reference,
            "effective_date": self.effective_date.isoformat(),
            "applicability": self.applicability,
            "test_method": self.test_method,
        }


EFFICIENCY_STANDARDS: List[EfficiencyStandard] = [
    EfficiencyStandard(
        standard_id="DOE-BOILER-001",
        title="Commercial Gas-Fired Boiler Efficiency",
        equipment_type="gas_fired_boiler",
        min_efficiency=Decimal("82"),
        efficiency_metric="thermal_efficiency",
        cfr_reference="10 CFR 431.86",
        effective_date=datetime(2012, 3, 2, tzinfo=timezone.utc),
        applicability="Gas-fired hot water commercial packaged boilers >= 300,000 Btu/hr",
        test_method="ASHRAE 90.1",
    ),
    EfficiencyStandard(
        standard_id="DOE-BOILER-002",
        title="Commercial Oil-Fired Boiler Efficiency",
        equipment_type="oil_fired_boiler",
        min_efficiency=Decimal("84"),
        efficiency_metric="thermal_efficiency",
        cfr_reference="10 CFR 431.86",
        effective_date=datetime(2012, 3, 2, tzinfo=timezone.utc),
        applicability="Oil-fired hot water commercial packaged boilers >= 300,000 Btu/hr",
        test_method="ASHRAE 90.1",
    ),
    EfficiencyStandard(
        standard_id="DOE-BOILER-003",
        title="Steam Boiler Efficiency",
        equipment_type="steam_boiler",
        min_efficiency=Decimal("80"),
        efficiency_metric="thermal_efficiency",
        cfr_reference="10 CFR 431.86",
        effective_date=datetime(2012, 3, 2, tzinfo=timezone.utc),
        applicability="Steam commercial packaged boilers >= 300,000 Btu/hr",
        test_method="ASHRAE 90.1",
    ),
]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_regulatory_tracker() -> RegulatoryComplianceTracker:
    """Factory function to create regulatory compliance tracker."""
    return RegulatoryComplianceTracker()


def get_requirements_by_agency(
    agency: RegulatoryAgency,
) -> List[RegulatoryRequirement]:
    """
    Get standard requirements for a regulatory agency.

    Args:
        agency: Regulatory agency

    Returns:
        List of requirements
    """
    return [
        req for req in STEAM_SYSTEM_REQUIREMENTS
        if req.agency == agency
    ]


def get_efficiency_standards(
    equipment_type: Optional[str] = None,
) -> List[EfficiencyStandard]:
    """
    Get efficiency standards, optionally filtered by equipment type.

    Args:
        equipment_type: Optional equipment type filter

    Returns:
        List of efficiency standards
    """
    if equipment_type:
        return [s for s in EFFICIENCY_STANDARDS if s.equipment_type == equipment_type]
    return EFFICIENCY_STANDARDS.copy()


def get_all_requirements() -> List[RegulatoryRequirement]:
    """Get all standard steam system requirements."""
    return STEAM_SYSTEM_REQUIREMENTS.copy()
