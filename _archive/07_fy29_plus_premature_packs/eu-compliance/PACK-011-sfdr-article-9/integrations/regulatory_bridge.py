# -*- coding: utf-8 -*-
"""
RegulatoryBridge - SFDR/Taxonomy/Benchmark Regulatory Updates for Article 9
=============================================================================

This module connects PACK-011 (SFDR Article 9) with regulatory update
tracking services to monitor SFDR 2.0 developments, Taxonomy Regulation
amendments, Benchmark Regulation updates, and ESMA/ESA guidance that
affects Article 9 product compliance. It provides structured regulatory
event tracking, compliance deadline management, and impact assessment
for upcoming regulatory changes.

Architecture:
    PACK-011 SFDR Art 9 --> RegulatoryBridge --> Regulatory Sources
                                  |
                                  v
    SFDR 2.0 Tracking, Taxonomy Updates, BMR Amendments, Deadline Alerts

Regulatory Context:
    The SFDR regulatory landscape is evolving rapidly with SFDR 2.0
    consultation (expected 2025-2026), Taxonomy Regulation amendments
    (environmental delegated acts), Benchmark Regulation updates for
    CTB/PAB, and ongoing ESMA Q&As. Article 9 products face the highest
    regulatory scrutiny and must track all relevant changes.

Example:
    >>> config = RegulatoryBridgeConfig()
    >>> bridge = RegulatoryBridge(config)
    >>> result = bridge.check_for_updates()
    >>> print(f"Updates: {result.total_updates}, critical: {result.critical_count}")

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-011 SFDR Article 9
Version: 1.0.0
Status: Production Ready
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# =============================================================================
# Utility Helpers
# =============================================================================

def _hash_data(data: Any) -> str:
    """Compute a SHA-256 hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()

# =============================================================================
# Enums
# =============================================================================

class RegulatoryFramework(str, Enum):
    """Regulatory frameworks relevant to Article 9."""
    SFDR = "sfdr"
    SFDR_RTS = "sfdr_rts"
    SFDR_2_0 = "sfdr_2_0"
    TAXONOMY_REGULATION = "taxonomy_regulation"
    TAXONOMY_CDA = "taxonomy_cda"
    BENCHMARK_REGULATION = "benchmark_regulation"
    MIFID_II = "mifid_ii"
    CSRD = "csrd"
    ESMA_QA = "esma_qa"
    EBA_GUIDANCE = "eba_guidance"
    EIOPA_GUIDANCE = "eiopa_guidance"

class UpdateSeverity(str, Enum):
    """Severity of a regulatory update."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"

class EventType(str, Enum):
    """Type of regulatory event."""
    NEW_REGULATION = "new_regulation"
    AMENDMENT = "amendment"
    DELEGATED_ACT = "delegated_act"
    TECHNICAL_STANDARD = "technical_standard"
    GUIDANCE = "guidance"
    QA_UPDATE = "qa_update"
    CONSULTATION = "consultation"
    FINAL_REPORT = "final_report"
    EFFECTIVE_DATE = "effective_date"
    TRANSITION_PERIOD_END = "transition_period_end"

class ComplianceStatus(str, Enum):
    """Compliance status for a regulatory requirement."""
    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_YET_APPLICABLE = "not_yet_applicable"
    UNDER_REVIEW = "under_review"

class ImpactArea(str, Enum):
    """Area impacted by a regulatory change."""
    CLASSIFICATION = "classification"
    DISCLOSURE_PRE_CONTRACTUAL = "disclosure_pre_contractual"
    DISCLOSURE_PERIODIC = "disclosure_periodic"
    DISCLOSURE_WEBSITE = "disclosure_website"
    PAI_REPORTING = "pai_reporting"
    TAXONOMY_ALIGNMENT = "taxonomy_alignment"
    BENCHMARK_DESIGNATION = "benchmark_designation"
    DNSH_ASSESSMENT = "dnsh_assessment"
    GOOD_GOVERNANCE = "good_governance"
    SUSTAINABLE_INVESTMENT_DEFINITION = "sustainable_investment_definition"
    DATA_REQUIREMENTS = "data_requirements"
    NAMING_CONVENTIONS = "naming_conventions"
    TRANSITION_PLANS = "transition_plans"

# =============================================================================
# Data Models
# =============================================================================

class RegulatoryBridgeConfig(BaseModel):
    """Configuration for the Regulatory Bridge."""
    monitored_frameworks: List[str] = Field(
        default_factory=lambda: [f.value for f in RegulatoryFramework],
        description="Regulatory frameworks to monitor",
    )
    alert_severity_threshold: UpdateSeverity = Field(
        default=UpdateSeverity.MEDIUM,
        description="Minimum severity to trigger alerts",
    )
    check_interval_hours: int = Field(
        default=24, ge=1, le=168,
        description="How often to check for updates (hours)",
    )
    include_consultations: bool = Field(
        default=True,
        description="Include draft/consultation documents",
    )
    include_sfdr_2_0_tracking: bool = Field(
        default=True,
        description="Track SFDR 2.0 developments",
    )
    deadline_alert_days: int = Field(
        default=90, ge=1,
        description="Days before deadline to raise alert",
    )
    enable_provenance: bool = Field(
        default=True, description="Enable provenance hash tracking"
    )

class RegulatoryEvent(BaseModel):
    """A single regulatory event or update."""
    event_id: str = Field(default="", description="Event identifier")
    framework: str = Field(default="sfdr", description="Regulatory framework")
    event_type: str = Field(default="guidance", description="Event type")
    title: str = Field(default="", description="Event title")
    description: str = Field(default="", description="Event description")
    severity: str = Field(default="medium", description="Severity level")
    publication_date: str = Field(
        default="", description="Publication date"
    )
    effective_date: str = Field(
        default="", description="Effective date (if applicable)"
    )
    transition_end_date: str = Field(
        default="", description="Transition period end date"
    )
    impact_areas: List[str] = Field(
        default_factory=list, description="Areas impacted"
    )
    article_9_specific: bool = Field(
        default=False,
        description="Whether this specifically affects Article 9",
    )
    action_required: bool = Field(
        default=False, description="Whether action is required"
    )
    recommended_actions: List[str] = Field(
        default_factory=list, description="Recommended actions"
    )
    source_url: str = Field(default="", description="Source URL")
    source_document: str = Field(
        default="", description="Source document reference"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class ComplianceDeadline(BaseModel):
    """A compliance deadline for Article 9."""
    deadline_id: str = Field(default="", description="Deadline identifier")
    framework: str = Field(default="sfdr", description="Regulatory framework")
    requirement: str = Field(default="", description="Requirement description")
    deadline_date: str = Field(default="", description="Deadline date (ISO)")
    days_remaining: int = Field(
        default=0, description="Days remaining to deadline"
    )
    status: str = Field(
        default="not_yet_applicable", description="Compliance status"
    )
    impact_areas: List[str] = Field(
        default_factory=list, description="Areas impacted"
    )
    action_items: List[str] = Field(
        default_factory=list, description="Action items for compliance"
    )
    severity: str = Field(
        default="medium", description="Deadline severity"
    )
    is_overdue: bool = Field(
        default=False, description="Whether deadline is overdue"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class SFDR2TrackingStatus(BaseModel):
    """SFDR 2.0 development tracking status."""
    current_phase: str = Field(
        default="consultation",
        description="Current SFDR 2.0 phase",
    )
    expected_final_text: str = Field(
        default="", description="Expected date for final legislative text"
    )
    expected_application_date: str = Field(
        default="", description="Expected application date"
    )
    key_proposals: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Key SFDR 2.0 proposals and their status",
    )
    article_9_impact: List[str] = Field(
        default_factory=list,
        description="Expected impact on Article 9 products",
    )
    preparation_actions: List[str] = Field(
        default_factory=list,
        description="Actions to prepare for SFDR 2.0",
    )
    last_updated: str = Field(
        default="", description="Last tracking update"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

class UpdateCheckResult(BaseModel):
    """Result of checking for regulatory updates."""
    check_id: str = Field(default="", description="Check identifier")
    checked_at: str = Field(default="", description="Check timestamp")
    frameworks_checked: List[str] = Field(
        default_factory=list, description="Frameworks checked"
    )
    total_updates: int = Field(default=0, description="Total updates found")
    critical_count: int = Field(
        default=0, description="Critical severity updates"
    )
    high_count: int = Field(default=0, description="High severity updates")
    medium_count: int = Field(default=0, description="Medium severity updates")
    events: List[RegulatoryEvent] = Field(
        default_factory=list, description="Regulatory events found"
    )
    upcoming_deadlines: List[ComplianceDeadline] = Field(
        default_factory=list, description="Upcoming compliance deadlines"
    )
    sfdr_2_status: Optional[SFDR2TrackingStatus] = Field(
        default=None, description="SFDR 2.0 tracking status"
    )
    action_required: bool = Field(
        default=False,
        description="Whether any action is required",
    )
    summary: str = Field(default="", description="Summary of findings")
    errors: List[str] = Field(default_factory=list, description="Errors")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    execution_time_ms: float = Field(default=0.0, description="Execution time")

# =============================================================================
# Known Regulatory Events Database
# =============================================================================

KNOWN_REGULATORY_EVENTS: List[Dict[str, Any]] = [
    {
        "event_id": "REG-SFDR-001",
        "framework": "sfdr",
        "event_type": "technical_standard",
        "title": "SFDR RTS Annex III/V Templates (Art 9 specific)",
        "description": "Pre-contractual (Annex III) and periodic (Annex V) disclosure templates for Art 9 products",
        "severity": "critical",
        "publication_date": "2023-04-12",
        "effective_date": "2023-01-01",
        "impact_areas": ["disclosure_pre_contractual", "disclosure_periodic"],
        "article_9_specific": True,
        "action_required": False,
    },
    {
        "event_id": "REG-SFDR-002",
        "framework": "sfdr_2_0",
        "event_type": "consultation",
        "title": "SFDR 2.0 Comprehensive Review - Product Categorization",
        "description": "Commission consultation on replacing Art 8/9 with sustainability categories",
        "severity": "high",
        "publication_date": "2025-09-17",
        "impact_areas": ["classification", "sustainable_investment_definition"],
        "article_9_specific": True,
        "action_required": True,
        "recommended_actions": [
            "Review consultation document for Art 9 implications",
            "Assess impact of potential category system on current Art 9 products",
            "Prepare transition plan for SFDR 2.0 migration",
        ],
    },
    {
        "event_id": "REG-TAX-001",
        "framework": "taxonomy_regulation",
        "event_type": "delegated_act",
        "title": "Taxonomy Environmental Delegated Act - 4 remaining objectives",
        "description": "Delegated acts for water, circular economy, pollution, biodiversity",
        "severity": "high",
        "publication_date": "2024-06-27",
        "effective_date": "2025-01-01",
        "impact_areas": ["taxonomy_alignment", "disclosure_pre_contractual"],
        "article_9_specific": True,
        "action_required": True,
        "recommended_actions": [
            "Update taxonomy alignment calculations for all 6 objectives",
            "Extend DNSH assessment to cover new delegated act criteria",
        ],
    },
    {
        "event_id": "REG-BMR-001",
        "framework": "benchmark_regulation",
        "event_type": "amendment",
        "title": "CTB/PAB Minimum Standards Update",
        "description": "Updated minimum standards for Climate Transition and Paris-Aligned Benchmarks",
        "severity": "medium",
        "publication_date": "2024-01-15",
        "impact_areas": ["benchmark_designation"],
        "article_9_specific": False,
        "action_required": False,
    },
    {
        "event_id": "REG-ESMA-001",
        "framework": "esma_qa",
        "event_type": "qa_update",
        "title": "ESMA Q&A on SFDR - Fund Naming Guidelines",
        "description": "ESMA guidelines on ESG and sustainability-related fund names",
        "severity": "high",
        "publication_date": "2024-08-14",
        "effective_date": "2024-11-21",
        "impact_areas": ["naming_conventions", "classification"],
        "article_9_specific": True,
        "action_required": True,
        "recommended_actions": [
            "Review product name against ESMA naming guidelines",
            "Ensure 80% sustainable investment threshold for sustainability naming",
            "Verify no exclusion conflicts with fund name claims",
        ],
    },
    {
        "event_id": "REG-SFDR-003",
        "framework": "sfdr",
        "event_type": "guidance",
        "title": "ESA Joint Supervisory Statement on SFDR PAI Disclosures",
        "description": "Guidance on calculation and disclosure of all 18 mandatory PAI indicators",
        "severity": "medium",
        "publication_date": "2025-03-15",
        "impact_areas": ["pai_reporting", "data_requirements"],
        "article_9_specific": True,
        "action_required": False,
    },
    {
        "event_id": "REG-CSRD-001",
        "framework": "csrd",
        "event_type": "effective_date",
        "title": "CSRD Phase 1 Application - Large Listed Companies",
        "description": "First wave of CSRD reporting improves investee data availability for PAI",
        "severity": "medium",
        "publication_date": "2024-01-01",
        "effective_date": "2025-01-01",
        "impact_areas": ["data_requirements", "pai_reporting"],
        "article_9_specific": False,
        "action_required": False,
    },
]

KNOWN_DEADLINES: List[Dict[str, Any]] = [
    {
        "deadline_id": "DL-SFDR-001",
        "framework": "sfdr",
        "requirement": "Annual PAI Statement publication (entity-level)",
        "deadline_date": "2026-06-30",
        "impact_areas": ["pai_reporting"],
        "action_items": [
            "Calculate all 18 mandatory PAI indicators",
            "Include at least 2 optional PAI indicators per table",
            "Publish PAI statement on website",
        ],
        "severity": "critical",
    },
    {
        "deadline_id": "DL-SFDR-002",
        "framework": "sfdr",
        "requirement": "Annex V periodic disclosure publication",
        "deadline_date": "2026-04-30",
        "impact_areas": ["disclosure_periodic"],
        "action_items": [
            "Populate Annex V template with FY2025 data",
            "Include taxonomy alignment ratios",
            "Include impact measurement results",
        ],
        "severity": "critical",
    },
    {
        "deadline_id": "DL-TAX-001",
        "framework": "taxonomy_regulation",
        "requirement": "Full 6-objective taxonomy alignment disclosure",
        "deadline_date": "2026-01-01",
        "impact_areas": ["taxonomy_alignment"],
        "action_items": [
            "Extend alignment calculation to all 6 objectives",
            "Include CDA (gas/nuclear) disclosures",
        ],
        "severity": "high",
    },
    {
        "deadline_id": "DL-ESMA-001",
        "framework": "esma_qa",
        "requirement": "ESMA Fund Naming Guidelines Compliance",
        "deadline_date": "2025-11-21",
        "impact_areas": ["naming_conventions"],
        "action_items": [
            "Ensure product name aligns with ESMA ESG naming guidelines",
            "Verify 80% sustainability threshold for naming",
        ],
        "severity": "high",
    },
]

# =============================================================================
# Regulatory Bridge
# =============================================================================

class RegulatoryBridge:
    """Bridge for regulatory update tracking and compliance monitoring.

    Monitors SFDR, Taxonomy, Benchmark, and related regulatory changes
    that affect Article 9 product compliance. Provides structured event
    tracking, deadline management, and SFDR 2.0 preparation support.

    Attributes:
        config: Bridge configuration.
        _events_db: Known regulatory events database.
        _deadlines_db: Known compliance deadlines.

    Example:
        >>> bridge = RegulatoryBridge(RegulatoryBridgeConfig())
        >>> result = bridge.check_for_updates()
        >>> for event in result.events:
        ...     print(f"{event.severity}: {event.title}")
    """

    def __init__(self, config: Optional[RegulatoryBridgeConfig] = None) -> None:
        """Initialize the Regulatory Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or RegulatoryBridgeConfig()
        self.logger = logger
        self._events_db = list(KNOWN_REGULATORY_EVENTS)
        self._deadlines_db = list(KNOWN_DEADLINES)

        self.logger.info(
            "RegulatoryBridge initialized: frameworks=%d, severity_threshold=%s, "
            "sfdr_2_tracking=%s, known_events=%d, known_deadlines=%d",
            len(self.config.monitored_frameworks),
            self.config.alert_severity_threshold.value,
            self.config.include_sfdr_2_0_tracking,
            len(self._events_db),
            len(self._deadlines_db),
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def check_for_updates(
        self,
        since_date: Optional[str] = None,
    ) -> UpdateCheckResult:
        """Check for regulatory updates affecting Article 9 products.

        Scans all monitored regulatory frameworks for new events,
        evaluates upcoming deadlines, and tracks SFDR 2.0 status.

        Args:
            since_date: Only include events after this date (ISO format).

        Returns:
            UpdateCheckResult with events, deadlines, and SFDR 2.0 status.
        """
        start_time = time.time()
        errors: List[str] = []
        warnings: List[str] = []

        # Get relevant events
        events = self._get_filtered_events(since_date)

        # Get upcoming deadlines
        deadlines = self._get_upcoming_deadlines()

        # Get SFDR 2.0 status
        sfdr_2_status = None
        if self.config.include_sfdr_2_0_tracking:
            sfdr_2_status = self._get_sfdr_2_tracking()

        # Count by severity
        severity_counts: Dict[str, int] = {
            "critical": 0, "high": 0, "medium": 0, "low": 0, "informational": 0,
        }
        for event in events:
            sev = event.severity
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        # Determine if action required
        action_required = any(e.action_required for e in events)
        overdue_deadlines = [d for d in deadlines if d.is_overdue]
        if overdue_deadlines:
            action_required = True
            warnings.append(
                f"{len(overdue_deadlines)} compliance deadline(s) are overdue"
            )

        # Build summary
        summary = self._build_summary(events, deadlines, sfdr_2_status)

        elapsed_ms = (time.time() - start_time) * 1000

        result = UpdateCheckResult(
            check_id=f"CHK-{utcnow().strftime('%Y%m%d%H%M%S')}",
            checked_at=utcnow().isoformat(),
            frameworks_checked=self.config.monitored_frameworks,
            total_updates=len(events),
            critical_count=severity_counts.get("critical", 0),
            high_count=severity_counts.get("high", 0),
            medium_count=severity_counts.get("medium", 0),
            events=events,
            upcoming_deadlines=deadlines,
            sfdr_2_status=sfdr_2_status,
            action_required=action_required,
            summary=summary,
            errors=errors,
            warnings=warnings,
            execution_time_ms=elapsed_ms,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _hash_data(
                result.model_dump(
                    exclude={"provenance_hash", "events", "upcoming_deadlines"}
                )
            )

        self.logger.info(
            "RegulatoryBridge: %d updates found (critical=%d, high=%d), "
            "%d deadlines, action_required=%s, elapsed=%.1fms",
            len(events), severity_counts.get("critical", 0),
            severity_counts.get("high", 0),
            len(deadlines), action_required, elapsed_ms,
        )
        return result

    def get_compliance_deadlines(self) -> List[ComplianceDeadline]:
        """Get all known compliance deadlines.

        Returns:
            List of compliance deadlines sorted by date.
        """
        return self._get_upcoming_deadlines()

    def get_sfdr_2_tracking(self) -> SFDR2TrackingStatus:
        """Get SFDR 2.0 development tracking status.

        Returns:
            SFDR2TrackingStatus with current development phase.
        """
        return self._get_sfdr_2_tracking()

    def get_article_9_events(self) -> List[RegulatoryEvent]:
        """Get events specifically affecting Article 9 products.

        Returns:
            List of Article 9 specific regulatory events.
        """
        events = self._parse_events(self._events_db)
        return [e for e in events if e.article_9_specific]

    def assess_compliance_impact(
        self,
        current_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess how regulatory changes impact current compliance posture.

        Args:
            current_config: Current product configuration.

        Returns:
            Impact assessment with per-area compliance status.
        """
        events = self._parse_events(self._events_db)
        art_9_events = [e for e in events if e.article_9_specific]

        impact_areas: Dict[str, Dict[str, Any]] = {}
        for area in ImpactArea:
            relevant_events = [
                e for e in art_9_events if area.value in e.impact_areas
            ]
            status = "compliant"
            if any(e.action_required for e in relevant_events):
                status = "under_review"

            impact_areas[area.value] = {
                "status": status,
                "event_count": len(relevant_events),
                "latest_event": relevant_events[-1].title if relevant_events else "",
                "actions_required": sum(1 for e in relevant_events if e.action_required),
            }

        return {
            "overall_status": "under_review" if any(
                a["status"] != "compliant" for a in impact_areas.values()
            ) else "compliant",
            "impact_areas": impact_areas,
            "total_areas": len(impact_areas),
            "areas_requiring_action": sum(
                1 for a in impact_areas.values() if a["actions_required"] > 0
            ),
            "assessed_at": utcnow().isoformat(),
        }

    def add_custom_event(
        self,
        event_data: Dict[str, Any],
    ) -> RegulatoryEvent:
        """Add a custom regulatory event to the tracking database.

        Args:
            event_data: Event data dict.

        Returns:
            Parsed RegulatoryEvent.
        """
        self._events_db.append(event_data)
        events = self._parse_events([event_data])
        return events[0] if events else RegulatoryEvent()

    def add_custom_deadline(
        self,
        deadline_data: Dict[str, Any],
    ) -> ComplianceDeadline:
        """Add a custom compliance deadline.

        Args:
            deadline_data: Deadline data dict.

        Returns:
            Parsed ComplianceDeadline.
        """
        self._deadlines_db.append(deadline_data)
        deadlines = self._parse_deadlines([deadline_data])
        return deadlines[0] if deadlines else ComplianceDeadline()

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _get_filtered_events(
        self,
        since_date: Optional[str],
    ) -> List[RegulatoryEvent]:
        """Get events filtered by date and framework."""
        events = self._parse_events(self._events_db)

        # Filter by monitored frameworks
        events = [
            e for e in events
            if e.framework in self.config.monitored_frameworks
        ]

        # Filter by date
        if since_date:
            events = [
                e for e in events
                if e.publication_date >= since_date
            ]

        # Filter consultations if disabled
        if not self.config.include_consultations:
            events = [
                e for e in events
                if e.event_type != "consultation"
            ]

        # Sort by severity then date
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "informational": 4}
        events.sort(key=lambda e: (severity_order.get(e.severity, 4), e.publication_date))

        return events

    def _get_upcoming_deadlines(self) -> List[ComplianceDeadline]:
        """Get upcoming compliance deadlines."""
        now = utcnow()
        deadlines = self._parse_deadlines(self._deadlines_db)

        for deadline in deadlines:
            try:
                dl_date = datetime.fromisoformat(deadline.deadline_date)
                if dl_date.tzinfo is None:
                    dl_date = dl_date.replace(tzinfo=timezone.utc)
                delta = (dl_date - now).days
                deadline.days_remaining = max(0, delta)
                deadline.is_overdue = delta < 0

                # Set severity based on urgency
                if deadline.is_overdue:
                    deadline.severity = "critical"
                elif delta <= 30:
                    deadline.severity = "critical"
                elif delta <= self.config.deadline_alert_days:
                    deadline.severity = "high"
            except (ValueError, TypeError):
                deadline.days_remaining = -1

        deadlines.sort(key=lambda d: d.days_remaining if d.days_remaining >= 0 else 9999)
        return deadlines

    def _get_sfdr_2_tracking(self) -> SFDR2TrackingStatus:
        """Get SFDR 2.0 development tracking status."""
        status = SFDR2TrackingStatus(
            current_phase="consultation",
            expected_final_text="2026-Q4",
            expected_application_date="2028-H1",
            key_proposals=[
                {
                    "proposal": "Replace Art 8/9 with product categories",
                    "status": "under_consultation",
                    "impact": "Art 9 may become 'Sustainable' or 'Transition' category",
                },
                {
                    "proposal": "Harmonised sustainable investment definition",
                    "status": "under_consultation",
                    "impact": "May change SI qualification criteria for Art 9",
                },
                {
                    "proposal": "Enhanced PAI framework with quantitative thresholds",
                    "status": "proposed",
                    "impact": "May add minimum PAI thresholds for highest category",
                },
                {
                    "proposal": "Mandatory transition plan disclosure",
                    "status": "proposed",
                    "impact": "Art 9 products may need investee transition plans",
                },
            ],
            article_9_impact=[
                "Art 9 classification may be replaced with stricter sustainability category",
                "SI definition standardization may raise qualification bar",
                "Enhanced PAI thresholds expected for top-tier products",
                "Transition plan requirements may add new disclosure obligations",
                "Naming guidelines may be codified into regulation",
            ],
            preparation_actions=[
                "Track SFDR 2.0 consultation responses and outcomes",
                "Map current Art 9 features to proposed new categories",
                "Begin transition planning for potential reclassification",
                "Prepare data infrastructure for enhanced PAI thresholds",
                "Review product naming against expected new guidelines",
            ],
            last_updated=utcnow().isoformat(),
        )

        if self.config.enable_provenance:
            status.provenance_hash = _hash_data(
                status.model_dump(exclude={"provenance_hash"})
            )

        return status

    def _parse_events(
        self,
        raw_events: List[Dict[str, Any]],
    ) -> List[RegulatoryEvent]:
        """Parse raw event dicts into RegulatoryEvent models."""
        events: List[RegulatoryEvent] = []
        for raw in raw_events:
            event = RegulatoryEvent(
                event_id=str(raw.get("event_id", "")),
                framework=str(raw.get("framework", "sfdr")),
                event_type=str(raw.get("event_type", "guidance")),
                title=str(raw.get("title", "")),
                description=str(raw.get("description", "")),
                severity=str(raw.get("severity", "medium")),
                publication_date=str(raw.get("publication_date", "")),
                effective_date=str(raw.get("effective_date", "")),
                transition_end_date=str(raw.get("transition_end_date", "")),
                impact_areas=raw.get("impact_areas", []),
                article_9_specific=bool(raw.get("article_9_specific", False)),
                action_required=bool(raw.get("action_required", False)),
                recommended_actions=raw.get("recommended_actions", []),
                source_url=str(raw.get("source_url", "")),
                source_document=str(raw.get("source_document", "")),
            )
            if self.config.enable_provenance:
                event.provenance_hash = _hash_data(
                    event.model_dump(exclude={"provenance_hash"})
                )
            events.append(event)
        return events

    def _parse_deadlines(
        self,
        raw_deadlines: List[Dict[str, Any]],
    ) -> List[ComplianceDeadline]:
        """Parse raw deadline dicts into ComplianceDeadline models."""
        deadlines: List[ComplianceDeadline] = []
        for raw in raw_deadlines:
            deadline = ComplianceDeadline(
                deadline_id=str(raw.get("deadline_id", "")),
                framework=str(raw.get("framework", "sfdr")),
                requirement=str(raw.get("requirement", "")),
                deadline_date=str(raw.get("deadline_date", "")),
                impact_areas=raw.get("impact_areas", []),
                action_items=raw.get("action_items", []),
                severity=str(raw.get("severity", "medium")),
            )
            if self.config.enable_provenance:
                deadline.provenance_hash = _hash_data(
                    deadline.model_dump(exclude={"provenance_hash"})
                )
            deadlines.append(deadline)
        return deadlines

    def _build_summary(
        self,
        events: List[RegulatoryEvent],
        deadlines: List[ComplianceDeadline],
        sfdr_2: Optional[SFDR2TrackingStatus],
    ) -> str:
        """Build a human-readable summary of regulatory findings."""
        parts: List[str] = []

        if events:
            critical = sum(1 for e in events if e.severity == "critical")
            art_9 = sum(1 for e in events if e.article_9_specific)
            parts.append(
                f"{len(events)} regulatory updates ({critical} critical, "
                f"{art_9} Art 9 specific)"
            )

        overdue = [d for d in deadlines if d.is_overdue]
        upcoming = [d for d in deadlines if not d.is_overdue and d.days_remaining <= 90]
        if overdue:
            parts.append(f"{len(overdue)} overdue deadline(s)")
        if upcoming:
            parts.append(f"{len(upcoming)} deadline(s) within 90 days")

        if sfdr_2:
            parts.append(f"SFDR 2.0 phase: {sfdr_2.current_phase}")

        return "; ".join(parts) if parts else "No notable regulatory updates"
