# -*- coding: utf-8 -*-
"""
RegulatoryTrackingBridge - SFDR Regulatory Updates Monitoring
==============================================================

This module monitors SFDR regulatory updates from ESMA, the European
Commission, EBA, and National Competent Authorities. It tracks SFDR
Level 1, SFDR RTS, Taxonomy Regulation updates, and SFDR 2.0 proposals,
providing impact assessment and deadline tracking.

Architecture:
    Regulatory Sources --> RegulatoryTrackingBridge --> Impact Assessment
                                 |
                                 v
    ESMA/EC/EBA/NCA --> Timeline + Deadlines + Change Impact

Example:
    >>> config = RegulatoryTrackingConfig()
    >>> bridge = RegulatoryTrackingBridge(config)
    >>> updates = bridge.check_updates()
    >>> deadlines = bridge.get_upcoming_deadlines()

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-010 SFDR Article 8
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Helpers
# =============================================================================


def _utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _hash_data(data: Any) -> str:
    """Compute a SHA-256 hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


# =============================================================================
# Enums
# =============================================================================


class RegulatorySource(str, Enum):
    """Source of regulatory updates."""
    ESMA = "esma"
    EUROPEAN_COMMISSION = "european_commission"
    EBA = "eba"
    NCA = "national_competent_authority"
    EFRAG = "efrag"
    PLATFORM_SUSTAINABLE_FINANCE = "platform_sustainable_finance"


class ImpactLevel(str, Enum):
    """Impact level of a regulatory change."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


class RegulationStatus(str, Enum):
    """Status of a regulation or update."""
    PROPOSED = "proposed"
    CONSULTATION = "consultation"
    ADOPTED = "adopted"
    IN_FORCE = "in_force"
    AMENDED = "amended"
    REPEALED = "repealed"


class AlertChannel(str, Enum):
    """Notification channel for regulatory alerts."""
    EMAIL = "email"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"
    LOG = "log"


# =============================================================================
# Data Models
# =============================================================================


class RegulatoryTrackingConfig(BaseModel):
    """Configuration for the Regulatory Tracking Bridge."""
    monitoring_scope: List[str] = Field(
        default_factory=lambda: [
            "sfdr_level_1", "sfdr_rts", "taxonomy_regulation",
            "sfdr_2_proposals", "mifid_sustainability",
        ],
        description="Regulations to monitor",
    )
    alert_channels: List[str] = Field(
        default_factory=lambda: ["dashboard", "log"],
        description="Alert notification channels",
    )
    impact_threshold: ImpactLevel = Field(
        default=ImpactLevel.MEDIUM,
        description="Minimum impact level for alerts",
    )
    check_interval_hours: int = Field(
        default=24, ge=1, le=168,
        description="Check interval in hours",
    )
    track_ncas: List[str] = Field(
        default_factory=lambda: ["AMF", "BaFin", "CSSF", "AFM", "CNMV"],
        description="National Competent Authorities to track",
    )


class RegulatoryEvent(BaseModel):
    """A single regulatory event or update."""
    event_id: str = Field(default="", description="Event identifier")
    title: str = Field(default="", description="Event title")
    description: str = Field(default="", description="Event description")
    source: RegulatorySource = Field(
        default=RegulatorySource.ESMA, description="Source authority"
    )
    regulation: str = Field(default="", description="Affected regulation")
    status: RegulationStatus = Field(
        default=RegulationStatus.IN_FORCE, description="Regulation status"
    )
    impact_level: ImpactLevel = Field(
        default=ImpactLevel.MEDIUM, description="Impact level"
    )
    effective_date: str = Field(default="", description="Effective date")
    deadline_date: str = Field(default="", description="Compliance deadline")
    published_date: str = Field(default="", description="Publication date")
    article_8_impact: bool = Field(
        default=False, description="Specific impact on Article 8"
    )
    affected_areas: List[str] = Field(
        default_factory=list, description="Affected disclosure areas"
    )
    action_required: str = Field(default="", description="Required action")
    reference_url: str = Field(default="", description="Reference URL")


class UpdateCheckResult(BaseModel):
    """Result of checking for regulatory updates."""
    total_events: int = Field(default=0, description="Total events found")
    critical_events: int = Field(default=0, description="Critical events")
    high_events: int = Field(default=0, description="High impact events")
    events: List[RegulatoryEvent] = Field(
        default_factory=list, description="Regulatory events"
    )
    last_checked: str = Field(default="", description="Last check timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class DeadlineEntry(BaseModel):
    """An upcoming regulatory deadline."""
    regulation: str = Field(default="", description="Regulation name")
    deadline_date: str = Field(default="", description="Deadline date")
    days_remaining: int = Field(default=0, description="Days until deadline")
    impact_level: ImpactLevel = Field(
        default=ImpactLevel.MEDIUM, description="Impact level"
    )
    description: str = Field(default="", description="Deadline description")
    action_required: str = Field(default="", description="Required action")


# =============================================================================
# Tracked Regulations and Events
# =============================================================================


REGULATORY_SOURCES: Dict[str, Dict[str, str]] = {
    "esma": {
        "name": "European Securities and Markets Authority",
        "url": "https://www.esma.europa.eu",
        "scope": "SFDR supervisory guidance, Q&A, enforcement",
    },
    "european_commission": {
        "name": "European Commission",
        "url": "https://ec.europa.eu",
        "scope": "SFDR Level 1, delegated acts, SFDR 2.0",
    },
    "eba": {
        "name": "European Banking Authority",
        "url": "https://www.eba.europa.eu",
        "scope": "ESG risk management, Pillar 3 disclosures",
    },
    "nca": {
        "name": "National Competent Authorities",
        "url": "various",
        "scope": "Local supervisory guidance and enforcement",
    },
    "efrag": {
        "name": "European Financial Reporting Advisory Group",
        "url": "https://www.efrag.org",
        "scope": "Sustainability reporting standards alignment",
    },
}

TRACKED_REGULATIONS: List[Dict[str, Any]] = [
    {
        "id": "SFDR_L1",
        "name": "SFDR Level 1 (Regulation 2019/2088)",
        "status": "in_force",
        "effective_date": "2021-03-10",
        "scope": "Core SFDR obligations and product classification",
    },
    {
        "id": "SFDR_RTS",
        "name": "SFDR Regulatory Technical Standards (Delegated Regulation 2022/1288)",
        "status": "in_force",
        "effective_date": "2023-01-01",
        "scope": "PAI indicators, pre-contractual/periodic templates",
    },
    {
        "id": "TAXONOMY_REG",
        "name": "EU Taxonomy Regulation (2020/852)",
        "status": "in_force",
        "effective_date": "2020-07-12",
        "scope": "Taxonomy alignment for SFDR disclosures",
    },
    {
        "id": "SFDR_20",
        "name": "SFDR 2.0 Proposal",
        "status": "proposed",
        "effective_date": "2027-01-01",
        "scope": "Comprehensive SFDR revision: categories, PAI expansion",
    },
    {
        "id": "TAX_CDA",
        "name": "Taxonomy Complementary Delegated Act (Gas/Nuclear)",
        "status": "in_force",
        "effective_date": "2023-01-01",
        "scope": "Gas and nuclear inclusion criteria",
    },
    {
        "id": "TAX_ENV4",
        "name": "Taxonomy Environmental Delegated Act (Objectives 3-6)",
        "status": "in_force",
        "effective_date": "2024-01-01",
        "scope": "Water, circular economy, pollution, biodiversity criteria",
    },
    {
        "id": "MIFID_SUS",
        "name": "MiFID II Sustainability Preferences",
        "status": "in_force",
        "effective_date": "2022-08-02",
        "scope": "Client sustainability preferences in suitability",
    },
    {
        "id": "CSRD_ALIGN",
        "name": "CSRD Alignment with SFDR",
        "status": "in_force",
        "effective_date": "2024-01-01",
        "scope": "CSRD data availability for SFDR PAI indicators",
    },
]

REGULATORY_EVENTS: List[Dict[str, Any]] = [
    {
        "event_id": "SFDR-2026-001",
        "title": "SFDR 2.0 Commission proposal published",
        "description": "European Commission publishes proposal for comprehensive SFDR revision",
        "source": "european_commission",
        "regulation": "SFDR_20",
        "status": "proposed",
        "impact_level": "critical",
        "published_date": "2025-06-15",
        "effective_date": "2027-01-01",
        "article_8_impact": True,
        "affected_areas": ["classification", "disclosure_templates", "pai_indicators"],
        "action_required": "Review proposed changes and assess product reclassification impact",
    },
    {
        "event_id": "SFDR-2026-002",
        "title": "ESMA updated Q&A on Article 8 disclosures",
        "description": "Clarification on taxonomy alignment disclosure in pre-contractual documents",
        "source": "esma",
        "regulation": "SFDR_RTS",
        "status": "in_force",
        "impact_level": "high",
        "published_date": "2025-09-01",
        "article_8_impact": True,
        "affected_areas": ["taxonomy_disclosure", "annex_ii"],
        "action_required": "Update pre-contractual disclosures per new Q&A guidance",
    },
    {
        "event_id": "SFDR-2026-003",
        "title": "PAI indicator methodology update",
        "description": "Updated calculation methodology for PAI indicators 1-3",
        "source": "esma",
        "regulation": "SFDR_RTS",
        "status": "in_force",
        "impact_level": "high",
        "published_date": "2025-12-15",
        "effective_date": "2026-01-01",
        "article_8_impact": True,
        "affected_areas": ["pai_calculation", "periodic_disclosure"],
        "action_required": "Update PAI calculation engine with revised methodology",
    },
    {
        "event_id": "SFDR-2026-004",
        "title": "Taxonomy Regulation amendment: social taxonomy proposal",
        "description": "Proposal for social taxonomy framework to complement environmental taxonomy",
        "source": "european_commission",
        "regulation": "TAXONOMY_REG",
        "status": "proposed",
        "impact_level": "medium",
        "published_date": "2026-01-15",
        "article_8_impact": True,
        "affected_areas": ["social_characteristics", "taxonomy_alignment"],
        "action_required": "Monitor development; prepare for social taxonomy integration",
    },
    {
        "event_id": "SFDR-2026-005",
        "title": "EET v2.0 specification update",
        "description": "Updated European ESG Template with expanded SFDR fields",
        "source": "european_commission",
        "regulation": "SFDR_RTS",
        "status": "adopted",
        "impact_level": "medium",
        "published_date": "2025-11-01",
        "effective_date": "2026-03-01",
        "article_8_impact": True,
        "affected_areas": ["eet_export", "data_exchange"],
        "action_required": "Update EET bridge to support v2.0 field mappings",
    },
    {
        "event_id": "SFDR-2026-006",
        "title": "ESMA supervisory briefing on greenwashing",
        "description": "Guidance on supervisory expectations for Article 8 product naming",
        "source": "esma",
        "regulation": "SFDR_L1",
        "status": "in_force",
        "impact_level": "high",
        "published_date": "2025-10-01",
        "article_8_impact": True,
        "affected_areas": ["product_naming", "marketing_materials"],
        "action_required": "Review product naming against ESMA fund naming guidelines",
    },
    {
        "event_id": "SFDR-2026-007",
        "title": "Annual PAI reporting deadline",
        "description": "Annual PAI statement publication deadline for financial market participants",
        "source": "esma",
        "regulation": "SFDR_RTS",
        "status": "in_force",
        "impact_level": "critical",
        "published_date": "2026-01-01",
        "deadline_date": "2026-06-30",
        "article_8_impact": True,
        "affected_areas": ["pai_reporting", "website_disclosure"],
        "action_required": "Publish annual PAI statement on website by June 30",
    },
    {
        "event_id": "SFDR-2026-008",
        "title": "Periodic reporting period end",
        "description": "End of annual reporting period for periodic Annex IV disclosures",
        "source": "esma",
        "regulation": "SFDR_RTS",
        "status": "in_force",
        "impact_level": "high",
        "published_date": "2025-12-31",
        "deadline_date": "2026-12-31",
        "article_8_impact": True,
        "affected_areas": ["annex_iv", "periodic_disclosure"],
        "action_required": "Prepare periodic disclosure for FY2026",
    },
    {
        "event_id": "SFDR-2026-009",
        "title": "CSRD reporting data availability",
        "description": "First CSRD reporting cycle data available for SFDR PAI indicators",
        "source": "efrag",
        "regulation": "CSRD_ALIGN",
        "status": "in_force",
        "impact_level": "medium",
        "published_date": "2026-01-01",
        "article_8_impact": True,
        "affected_areas": ["pai_data_quality", "data_coverage"],
        "action_required": "Integrate CSRD-reported data to improve PAI data quality",
    },
    {
        "event_id": "SFDR-2026-010",
        "title": "MiFID sustainability preferences update",
        "description": "Updated guidance on integrating sustainability preferences with SFDR",
        "source": "esma",
        "regulation": "MIFID_SUS",
        "status": "in_force",
        "impact_level": "medium",
        "published_date": "2025-07-01",
        "article_8_impact": True,
        "affected_areas": ["target_market", "eet_export"],
        "action_required": "Update EET export and target market assessments",
    },
]


# =============================================================================
# Regulatory Tracking Bridge
# =============================================================================


class RegulatoryTrackingBridge:
    """Monitor SFDR regulatory updates and compliance deadlines.

    Tracks regulatory developments from ESMA, the European Commission,
    EBA, and National Competent Authorities. Provides impact assessment
    and deadline tracking for SFDR Article 8 products.

    Attributes:
        config: Bridge configuration.

    Example:
        >>> bridge = RegulatoryTrackingBridge(RegulatoryTrackingConfig())
        >>> updates = bridge.check_updates()
        >>> print(f"Found {updates.total_events} events")
    """

    def __init__(
        self, config: Optional[RegulatoryTrackingConfig] = None
    ) -> None:
        """Initialize the Regulatory Tracking Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or RegulatoryTrackingConfig()
        self.logger = logger

        self.logger.info(
            "RegulatoryTrackingBridge initialized: scope=%s, channels=%s",
            self.config.monitoring_scope,
            self.config.alert_channels,
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def check_updates(
        self,
        since_date: Optional[str] = None,
    ) -> UpdateCheckResult:
        """Check for regulatory updates since a given date.

        Args:
            since_date: ISO date string to filter events after (optional).

        Returns:
            UpdateCheckResult with relevant regulatory events.
        """
        events: List[RegulatoryEvent] = []
        critical = 0
        high = 0

        for event_data in REGULATORY_EVENTS:
            # Filter by monitoring scope
            regulation = event_data.get("regulation", "")
            in_scope = any(
                scope.upper() in regulation.upper()
                for scope in self.config.monitoring_scope
            ) or not self.config.monitoring_scope

            if not in_scope and regulation:
                continue

            # Filter by date
            if since_date:
                pub_date = event_data.get("published_date", "")
                if pub_date and pub_date < since_date:
                    continue

            event = RegulatoryEvent(
                event_id=event_data.get("event_id", ""),
                title=event_data.get("title", ""),
                description=event_data.get("description", ""),
                source=RegulatorySource(event_data.get("source", "esma")),
                regulation=regulation,
                status=RegulationStatus(event_data.get("status", "in_force")),
                impact_level=ImpactLevel(event_data.get("impact_level", "medium")),
                effective_date=event_data.get("effective_date", ""),
                deadline_date=event_data.get("deadline_date", ""),
                published_date=event_data.get("published_date", ""),
                article_8_impact=event_data.get("article_8_impact", False),
                affected_areas=event_data.get("affected_areas", []),
                action_required=event_data.get("action_required", ""),
            )
            events.append(event)

            if event.impact_level == ImpactLevel.CRITICAL:
                critical += 1
            elif event.impact_level == ImpactLevel.HIGH:
                high += 1

        result = UpdateCheckResult(
            total_events=len(events),
            critical_events=critical,
            high_events=high,
            events=events,
            last_checked=_utcnow().isoformat(),
        )
        result.provenance_hash = _hash_data({
            "total": len(events), "critical": critical,
            "checked_at": result.last_checked,
        })

        self.logger.info(
            "Regulatory update check: %d events (%d critical, %d high)",
            len(events), critical, high,
        )
        return result

    def get_timeline(self) -> List[Dict[str, Any]]:
        """Get a chronological timeline of tracked regulations.

        Returns:
            Sorted list of regulatory milestones.
        """
        timeline: List[Dict[str, Any]] = []

        for reg in TRACKED_REGULATIONS:
            timeline.append({
                "date": reg["effective_date"],
                "regulation": reg["name"],
                "id": reg["id"],
                "status": reg["status"],
                "type": "effective_date",
                "scope": reg["scope"],
            })

        for event_data in REGULATORY_EVENTS:
            if event_data.get("deadline_date"):
                timeline.append({
                    "date": event_data["deadline_date"],
                    "regulation": event_data.get("title", ""),
                    "id": event_data.get("event_id", ""),
                    "status": "deadline",
                    "type": "deadline",
                    "action": event_data.get("action_required", ""),
                })

            if event_data.get("effective_date"):
                timeline.append({
                    "date": event_data["effective_date"],
                    "regulation": event_data.get("title", ""),
                    "id": event_data.get("event_id", ""),
                    "status": event_data.get("status", ""),
                    "type": "effective_date",
                    "action": event_data.get("action_required", ""),
                })

        timeline.sort(key=lambda x: x.get("date", ""))
        return timeline

    def assess_impact(
        self,
        event_id: str,
    ) -> Dict[str, Any]:
        """Assess the impact of a specific regulatory event.

        Args:
            event_id: Regulatory event identifier.

        Returns:
            Impact assessment with affected areas and recommended actions.
        """
        for event_data in REGULATORY_EVENTS:
            if event_data.get("event_id") == event_id:
                impact = ImpactLevel(event_data.get("impact_level", "medium"))
                return {
                    "event_id": event_id,
                    "title": event_data.get("title", ""),
                    "impact_level": impact.value,
                    "article_8_impact": event_data.get("article_8_impact", False),
                    "affected_areas": event_data.get("affected_areas", []),
                    "action_required": event_data.get("action_required", ""),
                    "compliance_changes_needed": self._estimate_changes(
                        event_data
                    ),
                    "estimated_effort_days": self._estimate_effort(impact),
                    "provenance_hash": _hash_data({
                        "event": event_id, "impact": impact.value,
                    }),
                }

        return {"event_id": event_id, "error": "Event not found"}

    def get_upcoming_deadlines(
        self,
        days_ahead: int = 180,
    ) -> List[DeadlineEntry]:
        """Get upcoming regulatory deadlines.

        Args:
            days_ahead: Number of days to look ahead.

        Returns:
            List of upcoming deadlines sorted by date.
        """
        now = _utcnow()
        deadlines: List[DeadlineEntry] = []

        for event_data in REGULATORY_EVENTS:
            deadline_str = event_data.get("deadline_date", "")
            if not deadline_str:
                continue

            try:
                deadline_dt = datetime.fromisoformat(deadline_str).replace(
                    tzinfo=timezone.utc
                )
            except (ValueError, TypeError):
                continue

            days_remaining = (deadline_dt - now).days
            if 0 <= days_remaining <= days_ahead:
                deadlines.append(DeadlineEntry(
                    regulation=event_data.get("title", ""),
                    deadline_date=deadline_str,
                    days_remaining=days_remaining,
                    impact_level=ImpactLevel(
                        event_data.get("impact_level", "medium")
                    ),
                    description=event_data.get("description", ""),
                    action_required=event_data.get("action_required", ""),
                ))

        deadlines.sort(key=lambda d: d.days_remaining)

        self.logger.info(
            "Found %d upcoming deadlines within %d days",
            len(deadlines), days_ahead,
        )
        return deadlines

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _estimate_changes(
        self, event_data: Dict[str, Any]
    ) -> List[str]:
        """Estimate compliance changes needed for an event.

        Args:
            event_data: Regulatory event data.

        Returns:
            List of estimated changes.
        """
        changes: List[str] = []
        areas = event_data.get("affected_areas", [])

        area_to_change = {
            "classification": "Review and potentially update product classification",
            "disclosure_templates": "Update Annex II/III/IV disclosure templates",
            "pai_indicators": "Update PAI indicator definitions and calculations",
            "pai_calculation": "Revise PAI calculation methodology",
            "taxonomy_disclosure": "Update taxonomy alignment disclosures",
            "annex_ii": "Update pre-contractual disclosure (Annex II)",
            "annex_iv": "Update periodic disclosure (Annex IV)",
            "periodic_disclosure": "Prepare periodic disclosure report",
            "website_disclosure": "Update website sustainability disclosure",
            "pai_reporting": "Prepare and publish annual PAI statement",
            "eet_export": "Update EET data export format and fields",
            "product_naming": "Review product name compliance with naming guidelines",
            "data_exchange": "Update data exchange templates",
            "social_characteristics": "Review social characteristics disclosures",
            "taxonomy_alignment": "Update taxonomy alignment calculations",
            "target_market": "Update target market sustainability preferences",
            "pai_data_quality": "Improve PAI data quality with new data sources",
            "data_coverage": "Expand data coverage for PAI indicators",
            "marketing_materials": "Review marketing materials for compliance",
        }

        for area in areas:
            change = area_to_change.get(area, f"Review {area} for compliance")
            changes.append(change)

        return changes

    @staticmethod
    def _estimate_effort(impact: ImpactLevel) -> int:
        """Estimate implementation effort in days based on impact level.

        Args:
            impact: Impact level.

        Returns:
            Estimated effort in business days.
        """
        effort_map = {
            ImpactLevel.CRITICAL: 30,
            ImpactLevel.HIGH: 15,
            ImpactLevel.MEDIUM: 7,
            ImpactLevel.LOW: 3,
            ImpactLevel.INFORMATIONAL: 1,
        }
        return effort_map.get(impact, 5)
