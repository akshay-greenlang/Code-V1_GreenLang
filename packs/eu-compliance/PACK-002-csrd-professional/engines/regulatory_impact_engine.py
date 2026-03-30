# -*- coding: utf-8 -*-
"""
RegulatoryImpactEngine - PACK-002 CSRD Professional Engine 6

Regulatory change impact analysis engine tracking EFRAG, EU Commission,
ESMA, ISSB, and national authority changes. Detects compliance gaps,
generates impact assessments, and maintains a regulatory deadline calendar
with 20+ pre-built deadlines for ESRS Set 2, national transpositions,
and ISSB convergence.

Features:
    - Register and track regulatory changes by source and severity
    - Assess impact on data points, calculations, and disclosures
    - Detect compliance gaps between current state and new requirements
    - Maintain a regulatory calendar with jurisdiction-specific deadlines
    - Version-controlled change history with SHA-256 provenance
    - Pre-built deadlines for key CSRD/ESRS milestones

Zero-Hallucination:
    - All impact assessments use deterministic matching rules
    - Gap detection compares explicit requirement lists
    - Deadline calculations use calendar arithmetic only
    - No LLM involvement in gap or impact determination

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-002 CSRD Professional
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RegulationSource(str, Enum):
    """Source of regulatory change."""

    EFRAG = "efrag"
    EU_COMMISSION = "eu_commission"
    ESMA = "esma"
    ISSB = "issb"
    NATIONAL_AUTHORITY = "national_authority"

class ChangeSeverity(str, Enum):
    """Severity of a regulatory change."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"

class DeadlineStatus(str, Enum):
    """Status of a regulatory deadline."""

    UPCOMING = "upcoming"
    IMMINENT = "imminent"
    OVERDUE = "overdue"
    COMPLETED = "completed"
    NOT_APPLICABLE = "not_applicable"

class GapStatus(str, Enum):
    """Current compliance status for a gap."""

    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class RegulatoryChange(BaseModel):
    """A registered regulatory change."""

    change_id: str = Field(default_factory=_new_uuid, description="Change ID")
    regulation: str = Field(..., description="Regulation name (e.g. 'ESRS E1')")
    source: RegulationSource = Field(..., description="Issuing body")
    description: str = Field(..., description="Description of the change")
    effective_date: date = Field(..., description="When the change takes effect")
    severity: ChangeSeverity = Field(..., description="Impact severity")
    affected_standards: List[str] = Field(
        default_factory=list,
        description="ESRS standards affected (e.g. ['E1', 'E2', 'S1'])",
    )
    source_url: Optional[str] = Field(
        None, description="URL to official source document"
    )
    registered_at: datetime = Field(
        default_factory=utcnow, description="Registration timestamp"
    )
    provenance_hash: str = Field("", description="SHA-256 hash")

class ImpactAssessment(BaseModel):
    """Impact assessment for a regulatory change."""

    assessment_id: str = Field(default_factory=_new_uuid, description="Assessment ID")
    change_id: str = Field(..., description="Regulatory change assessed")
    affected_data_points: List[str] = Field(
        default_factory=list, description="ESRS data points affected"
    )
    affected_calculations: List[str] = Field(
        default_factory=list, description="Calculations requiring update"
    )
    new_disclosure_requirements: List[str] = Field(
        default_factory=list, description="New disclosures required"
    )
    deprecated_requirements: List[str] = Field(
        default_factory=list, description="Requirements being deprecated"
    )
    remediation_effort_hours: float = Field(
        0.0, ge=0.0, description="Estimated remediation effort"
    )
    priority: str = Field("medium", description="Remediation priority")
    assessed_at: datetime = Field(
        default_factory=utcnow, description="Assessment timestamp"
    )
    provenance_hash: str = Field("", description="SHA-256 hash")

class ComplianceGap(BaseModel):
    """A detected gap between current state and requirements."""

    gap_id: str = Field(default_factory=_new_uuid, description="Gap ID")
    standard: str = Field(..., description="ESRS standard (e.g. 'ESRS E1')")
    requirement: str = Field(..., description="Specific requirement")
    current_status: GapStatus = Field(
        GapStatus.NOT_ASSESSED, description="Current compliance status"
    )
    gap_description: str = Field("", description="Description of the gap")
    priority: str = Field("medium", description="Gap priority")
    remediation_plan: str = Field("", description="Planned remediation approach")
    provenance_hash: str = Field("", description="SHA-256 hash")

class RegulatoryDeadline(BaseModel):
    """A regulatory deadline with status tracking."""

    deadline_id: str = Field(default_factory=_new_uuid, description="Deadline ID")
    regulation: str = Field(..., description="Regulation name")
    jurisdiction: str = Field("EU", description="Applicable jurisdiction")
    deadline: date = Field(..., description="Deadline date")
    description: str = Field(..., description="What must be done by this date")
    status: DeadlineStatus = Field(
        DeadlineStatus.UPCOMING, description="Current status"
    )
    days_remaining: int = Field(0, description="Days until deadline")

class RegulatoryCalendar(BaseModel):
    """A calendar of regulatory deadlines."""

    calendar_id: str = Field(default_factory=_new_uuid, description="Calendar ID")
    deadlines: List[RegulatoryDeadline] = Field(
        default_factory=list, description="All deadlines sorted by date"
    )
    jurisdictions: List[str] = Field(
        default_factory=list, description="Jurisdictions included"
    )
    total_upcoming: int = Field(0, description="Upcoming deadlines count")
    total_overdue: int = Field(0, description="Overdue deadlines count")
    generated_at: datetime = Field(default_factory=utcnow, description="Generated at")
    provenance_hash: str = Field("", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class RegulatoryConfig(BaseModel):
    """Configuration for the regulatory impact engine."""

    imminent_threshold_days: int = Field(
        30, ge=1, description="Days before deadline is considered imminent"
    )
    auto_assess_new_changes: bool = Field(
        False, description="Automatically assess impact when change is registered"
    )
    default_jurisdiction: str = Field(
        "EU", description="Default jurisdiction for calendar"
    )
    track_issb_convergence: bool = Field(
        True, description="Include ISSB convergence deadlines"
    )

# ---------------------------------------------------------------------------
# Pre-Built Deadlines
# ---------------------------------------------------------------------------

_DEFAULT_DEADLINES: List[Dict[str, Any]] = [
    # CSRD Phase 1: Large PIEs (>500 employees)
    {"regulation": "CSRD Phase 1", "jurisdiction": "EU", "deadline": "2025-01-01",
     "description": "CSRD reporting begins for large PIEs (>500 employees, FY2024)"},
    {"regulation": "CSRD Phase 1 Filing", "jurisdiction": "EU", "deadline": "2025-06-30",
     "description": "First CSRD reports due in annual management reports"},
    # CSRD Phase 2: Other large companies
    {"regulation": "CSRD Phase 2", "jurisdiction": "EU", "deadline": "2026-01-01",
     "description": "CSRD reporting begins for other large companies (FY2025)"},
    {"regulation": "CSRD Phase 2 Filing", "jurisdiction": "EU", "deadline": "2026-06-30",
     "description": "Phase 2 CSRD reports due in annual management reports"},
    # CSRD Phase 3: Listed SMEs
    {"regulation": "CSRD Phase 3", "jurisdiction": "EU", "deadline": "2027-01-01",
     "description": "CSRD reporting begins for listed SMEs (FY2026)"},
    # ESRS Set 2
    {"regulation": "ESRS Set 2 Exposure Draft", "jurisdiction": "EU", "deadline": "2026-06-30",
     "description": "EFRAG expected to publish ESRS Set 2 exposure drafts"},
    {"regulation": "ESRS Set 2 Adoption", "jurisdiction": "EU", "deadline": "2027-06-30",
     "description": "EU Commission expected to adopt ESRS Set 2 delegated acts"},
    {"regulation": "ESRS Sector Standards Draft", "jurisdiction": "EU", "deadline": "2026-12-31",
     "description": "EFRAG sector-specific ESRS standards exposure drafts"},
    # National Transpositions
    {"regulation": "CSRD Transposition - Germany", "jurisdiction": "DE", "deadline": "2025-07-06",
     "description": "Germany CSRD transposition into national law (CSRIG)"},
    {"regulation": "CSRD Transposition - France", "jurisdiction": "FR", "deadline": "2025-07-06",
     "description": "France CSRD transposition into national law"},
    {"regulation": "CSRD Transposition - Netherlands", "jurisdiction": "NL", "deadline": "2025-07-06",
     "description": "Netherlands CSRD transposition into national law"},
    {"regulation": "CSRD Transposition - Italy", "jurisdiction": "IT", "deadline": "2025-07-06",
     "description": "Italy CSRD transposition into national law"},
    {"regulation": "CSRD Transposition - Spain", "jurisdiction": "ES", "deadline": "2025-07-06",
     "description": "Spain CSRD transposition into national law"},
    # ISSB Convergence
    {"regulation": "ISSB S1/S2 Effective", "jurisdiction": "Global", "deadline": "2025-01-01",
     "description": "ISSB IFRS S1 and S2 standards effective date"},
    {"regulation": "ISSB-ESRS Interoperability Guide", "jurisdiction": "Global", "deadline": "2026-03-31",
     "description": "EFRAG/ISSB interoperability guidance publication"},
    {"regulation": "ISSB S3 Biodiversity Draft", "jurisdiction": "Global", "deadline": "2026-12-31",
     "description": "ISSB expected exposure draft on biodiversity standard"},
    # XBRL / Digital Tagging
    {"regulation": "ESRS XBRL Taxonomy v1.0", "jurisdiction": "EU", "deadline": "2025-12-31",
     "description": "EFRAG ESRS XBRL taxonomy v1.0 finalization"},
    {"regulation": "ESEF XBRL Digital Tagging", "jurisdiction": "EU", "deadline": "2026-01-01",
     "description": "Digital tagging of CSRD reports using ESRS XBRL taxonomy"},
    # Assurance
    {"regulation": "Limited Assurance Standard", "jurisdiction": "EU", "deadline": "2026-10-01",
     "description": "EU limited assurance standard for sustainability reporting adoption"},
    {"regulation": "Reasonable Assurance Roadmap", "jurisdiction": "EU", "deadline": "2028-10-01",
     "description": "EU roadmap for transition to reasonable assurance"},
    # Third Country
    {"regulation": "CSRD Third Country Regime", "jurisdiction": "EU", "deadline": "2028-01-01",
     "description": "Non-EU companies with EU turnover >150M begin reporting"},
    # Taxonomy
    {"regulation": "EU Taxonomy Environmental DA", "jurisdiction": "EU", "deadline": "2026-01-01",
     "description": "EU Taxonomy environmental delegated acts for remaining objectives"},
]

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RegulatoryImpactEngine:
    """Regulatory change impact analysis engine.

    Tracks regulatory changes, assesses their impact on CSRD reporting,
    detects compliance gaps, and maintains a regulatory deadline calendar.

    Attributes:
        config: Engine configuration.
        changes: Registered regulatory changes keyed by change_id.
        assessments: Impact assessments keyed by assessment_id.
        gaps: Detected compliance gaps.
        custom_deadlines: User-registered deadlines.

    Example:
        >>> engine = RegulatoryImpactEngine()
        >>> engine.register_change(RegulatoryChange(
        ...     regulation="ESRS E1 Amendment",
        ...     source=RegulationSource.EFRAG,
        ...     description="New GHG intensity metric required",
        ...     effective_date=date(2026, 1, 1),
        ...     severity=ChangeSeverity.HIGH,
        ... ))
        >>> calendar = await engine.generate_calendar(["EU", "DE"])
    """

    def __init__(self, config: Optional[RegulatoryConfig] = None) -> None:
        """Initialize RegulatoryImpactEngine.

        Args:
            config: Engine configuration. Uses defaults if not provided.
        """
        self.config = config or RegulatoryConfig()
        self.changes: Dict[str, RegulatoryChange] = {}
        self.assessments: Dict[str, ImpactAssessment] = {}
        self.gaps: List[ComplianceGap] = []
        self.custom_deadlines: List[RegulatoryDeadline] = []
        logger.info(
            "RegulatoryImpactEngine initialized (version=%s)", _MODULE_VERSION
        )

    # -- Change Registration ------------------------------------------------

    def register_change(self, change: RegulatoryChange) -> str:
        """Register a regulatory change for tracking.

        Args:
            change: Regulatory change definition.

        Returns:
            Change ID.
        """
        change.provenance_hash = _compute_hash(change)
        self.changes[change.change_id] = change

        logger.info(
            "Regulatory change registered: %s (source=%s, severity=%s, effective=%s)",
            change.regulation,
            change.source.value,
            change.severity.value,
            change.effective_date.isoformat(),
        )
        return change.change_id

    # -- Impact Assessment --------------------------------------------------

    async def assess_impact(
        self,
        change_id: str,
        current_report: Dict[str, Any],
    ) -> ImpactAssessment:
        """Analyze the impact of a regulatory change on current reporting.

        Compares the change requirements against the current report state
        to identify affected data points, calculations, and disclosures.

        Args:
            change_id: Regulatory change to assess.
            current_report: Current report state with data points and metadata.

        Returns:
            ImpactAssessment with detailed impact analysis.

        Raises:
            ValueError: If change_id not found.
        """
        change = self.changes.get(change_id)
        if change is None:
            raise ValueError(f"Regulatory change '{change_id}' not found")

        logger.info("Assessing impact for change: %s", change.regulation)

        # Determine affected data points
        affected_data_points = self._find_affected_data_points(
            change, current_report
        )

        # Determine affected calculations
        affected_calcs = self._find_affected_calculations(
            change, current_report
        )

        # Identify new disclosures required
        new_disclosures = self._find_new_disclosures(change, current_report)

        # Identify deprecated requirements
        deprecated = self._find_deprecated_requirements(change, current_report)

        # Estimate effort
        effort = self._estimate_remediation_effort(
            affected_data_points, affected_calcs, new_disclosures
        )

        # Determine priority based on severity and time to effective date
        priority = self._determine_priority(change)

        assessment = ImpactAssessment(
            change_id=change_id,
            affected_data_points=affected_data_points,
            affected_calculations=affected_calcs,
            new_disclosure_requirements=new_disclosures,
            deprecated_requirements=deprecated,
            remediation_effort_hours=effort,
            priority=priority,
        )
        assessment.provenance_hash = _compute_hash(assessment)

        self.assessments[assessment.assessment_id] = assessment

        logger.info(
            "Impact assessment complete: %d data points, %d calculations, "
            "%d new disclosures, effort=%.1f hours",
            len(affected_data_points),
            len(affected_calcs),
            len(new_disclosures),
            effort,
        )
        return assessment

    # -- Gap Detection ------------------------------------------------------

    async def detect_gaps(
        self,
        current_state: Dict[str, Any],
        new_requirements: List[Dict[str, Any]],
    ) -> List[ComplianceGap]:
        """Detect compliance gaps between current state and new requirements.

        Args:
            current_state: Current compliance state with covered requirements.
            new_requirements: List of new requirement definitions, each with
                'standard', 'requirement', and 'priority' keys.

        Returns:
            List of detected ComplianceGap objects.
        """
        covered = set(current_state.get("covered_requirements", []))
        detected_gaps: List[ComplianceGap] = []

        for req in new_requirements:
            req_id = req.get("requirement", "")
            standard = req.get("standard", "Unknown")
            priority = req.get("priority", "medium")

            if req_id in covered:
                continue

            # Check for partial coverage
            partially_covered = any(
                req_id.startswith(c) or c.startswith(req_id) for c in covered
            )

            gap = ComplianceGap(
                standard=standard,
                requirement=req_id,
                current_status=(
                    GapStatus.PARTIAL if partially_covered else GapStatus.NON_COMPLIANT
                ),
                gap_description=req.get(
                    "description",
                    f"Requirement '{req_id}' from {standard} is not met",
                ),
                priority=priority,
                remediation_plan=req.get("remediation_hint", ""),
            )
            gap.provenance_hash = _compute_hash(gap)
            detected_gaps.append(gap)

        self.gaps.extend(detected_gaps)

        logger.info(
            "Gap detection: %d gaps found from %d requirements",
            len(detected_gaps),
            len(new_requirements),
        )
        return detected_gaps

    # -- Calendar -----------------------------------------------------------

    async def generate_calendar(
        self,
        jurisdictions: Optional[List[str]] = None,
    ) -> RegulatoryCalendar:
        """Generate a regulatory deadline calendar.

        Includes pre-built CSRD/ESRS deadlines plus any custom deadlines,
        filtered by jurisdiction. Deadlines are sorted by date and annotated
        with days remaining and status.

        Args:
            jurisdictions: Jurisdictions to include. None includes all.

        Returns:
            RegulatoryCalendar with sorted deadlines.
        """
        target_jurisdictions = jurisdictions or [self.config.default_jurisdiction]

        deadlines: List[RegulatoryDeadline] = []
        today = date.today()

        # Add pre-built deadlines
        for dl_def in _DEFAULT_DEADLINES:
            dl_jurisdiction = dl_def["jurisdiction"]

            # Include "EU" deadlines for any EU member state, "Global" always
            include = (
                dl_jurisdiction in target_jurisdictions
                or dl_jurisdiction == "Global"
                or (dl_jurisdiction == "EU" and any(
                    j in target_jurisdictions
                    for j in ["EU", "DE", "FR", "NL", "IT", "ES", "AT", "BE",
                              "IE", "FI", "SE", "DK", "PT", "PL", "CZ"]
                ))
            )

            if not include:
                continue

            dl_date = date.fromisoformat(dl_def["deadline"])
            days_remaining = (dl_date - today).days

            if days_remaining < 0:
                status = DeadlineStatus.OVERDUE
            elif days_remaining <= self.config.imminent_threshold_days:
                status = DeadlineStatus.IMMINENT
            else:
                status = DeadlineStatus.UPCOMING

            deadlines.append(
                RegulatoryDeadline(
                    regulation=dl_def["regulation"],
                    jurisdiction=dl_jurisdiction,
                    deadline=dl_date,
                    description=dl_def["description"],
                    status=status,
                    days_remaining=days_remaining,
                )
            )

        # Add custom deadlines
        for custom_dl in self.custom_deadlines:
            if jurisdictions and custom_dl.jurisdiction not in target_jurisdictions:
                continue
            days_remaining = (custom_dl.deadline - today).days
            custom_dl.days_remaining = days_remaining
            if days_remaining < 0:
                custom_dl.status = DeadlineStatus.OVERDUE
            elif days_remaining <= self.config.imminent_threshold_days:
                custom_dl.status = DeadlineStatus.IMMINENT
            else:
                custom_dl.status = DeadlineStatus.UPCOMING
            deadlines.append(custom_dl)

        # Sort by date
        deadlines.sort(key=lambda d: d.deadline)

        # Counts
        total_upcoming = sum(
            1 for d in deadlines if d.status in (DeadlineStatus.UPCOMING, DeadlineStatus.IMMINENT)
        )
        total_overdue = sum(
            1 for d in deadlines if d.status == DeadlineStatus.OVERDUE
        )

        calendar = RegulatoryCalendar(
            deadlines=deadlines,
            jurisdictions=target_jurisdictions,
            total_upcoming=total_upcoming,
            total_overdue=total_overdue,
        )
        calendar.provenance_hash = _compute_hash(calendar)

        logger.info(
            "Regulatory calendar generated: %d deadlines (%d upcoming, %d overdue)",
            len(deadlines),
            total_upcoming,
            total_overdue,
        )
        return calendar

    # -- Query Methods ------------------------------------------------------

    def get_change_history(self) -> List[RegulatoryChange]:
        """Get version-controlled log of all registered changes.

        Returns:
            List of RegulatoryChange objects sorted by registration time.
        """
        return sorted(
            self.changes.values(),
            key=lambda c: c.registered_at,
        )

    async def get_upcoming_deadlines(
        self, days_ahead: int = 90
    ) -> List[RegulatoryDeadline]:
        """Get deadlines within the specified number of days.

        Args:
            days_ahead: Number of days to look ahead.

        Returns:
            List of upcoming RegulatoryDeadline objects.
        """
        calendar = await self.generate_calendar()
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)

        return [
            d
            for d in calendar.deadlines
            if today <= d.deadline <= cutoff
        ]

    def add_custom_deadline(self, deadline: RegulatoryDeadline) -> str:
        """Add a custom regulatory deadline.

        Args:
            deadline: Deadline definition.

        Returns:
            Deadline ID.
        """
        self.custom_deadlines.append(deadline)
        logger.info(
            "Custom deadline added: %s (%s, %s)",
            deadline.regulation,
            deadline.jurisdiction,
            deadline.deadline.isoformat(),
        )
        return deadline.deadline_id

    # -- Internal Helpers ---------------------------------------------------

    def _find_affected_data_points(
        self,
        change: RegulatoryChange,
        current_report: Dict[str, Any],
    ) -> List[str]:
        """Find data points affected by a regulatory change.

        Matches change's affected standards against report data points.

        Args:
            change: Regulatory change.
            current_report: Current report state.

        Returns:
            List of affected data point identifiers.
        """
        affected: List[str] = []
        report_data_points = current_report.get("data_points", {})

        for dp_id in report_data_points:
            for standard in change.affected_standards:
                if dp_id.upper().startswith(standard.upper()):
                    affected.append(dp_id)
                    break

        return affected

    def _find_affected_calculations(
        self,
        change: RegulatoryChange,
        current_report: Dict[str, Any],
    ) -> List[str]:
        """Find calculations affected by a regulatory change.

        Args:
            change: Regulatory change.
            current_report: Current report state.

        Returns:
            List of affected calculation identifiers.
        """
        affected: List[str] = []
        calculations = current_report.get("calculations", {})

        for calc_id, calc_meta in calculations.items():
            calc_standards = calc_meta.get("standards", []) if isinstance(calc_meta, dict) else []
            if any(
                std in change.affected_standards for std in calc_standards
            ):
                affected.append(calc_id)

        return affected

    def _find_new_disclosures(
        self,
        change: RegulatoryChange,
        current_report: Dict[str, Any],
    ) -> List[str]:
        """Find new disclosure requirements introduced by a change.

        Args:
            change: Regulatory change with description.
            current_report: Current report state.

        Returns:
            List of new disclosure requirement descriptions.
        """
        current_disclosures = set(current_report.get("disclosures", []))
        new_disclosures: List[str] = []

        # The change may reference new standards that are not in current disclosures
        for standard in change.affected_standards:
            standard_key = f"ESRS_{standard}"
            if standard_key not in current_disclosures:
                new_disclosures.append(
                    f"New disclosure requirement for {standard} per {change.regulation}"
                )

        return new_disclosures

    def _find_deprecated_requirements(
        self,
        change: RegulatoryChange,
        current_report: Dict[str, Any],
    ) -> List[str]:
        """Find requirements deprecated by a change.

        Args:
            change: Regulatory change.
            current_report: Current report state.

        Returns:
            List of deprecated requirement identifiers.
        """
        deprecated_list = current_report.get("deprecated_by_change", {})
        return deprecated_list.get(change.change_id, [])

    def _estimate_remediation_effort(
        self,
        data_points: List[str],
        calculations: List[str],
        new_disclosures: List[str],
    ) -> float:
        """Estimate remediation effort in hours.

        Uses a simple heuristic:
        - 2 hours per affected data point
        - 4 hours per affected calculation
        - 8 hours per new disclosure

        Args:
            data_points: Affected data points.
            calculations: Affected calculations.
            new_disclosures: New disclosure requirements.

        Returns:
            Estimated effort in hours.
        """
        effort = (
            len(data_points) * 2.0
            + len(calculations) * 4.0
            + len(new_disclosures) * 8.0
        )
        return round(effort, 1)

    def _determine_priority(self, change: RegulatoryChange) -> str:
        """Determine remediation priority based on severity and timeline.

        Args:
            change: Regulatory change.

        Returns:
            Priority string (critical/high/medium/low).
        """
        days_until = (change.effective_date - date.today()).days

        if change.severity == ChangeSeverity.CRITICAL:
            return "critical"
        if change.severity == ChangeSeverity.HIGH and days_until < 180:
            return "critical"
        if change.severity == ChangeSeverity.HIGH:
            return "high"
        if change.severity == ChangeSeverity.MEDIUM and days_until < 90:
            return "high"
        if change.severity == ChangeSeverity.MEDIUM:
            return "medium"
        return "low"

    # -- Reset --------------------------------------------------------------

    def reset(self) -> None:
        """Reset engine state, clearing all changes, assessments, and gaps."""
        self.changes.clear()
        self.assessments.clear()
        self.gaps.clear()
        self.custom_deadlines.clear()
        logger.info("RegulatoryImpactEngine reset")
