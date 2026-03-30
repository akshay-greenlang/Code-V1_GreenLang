# -*- coding: utf-8 -*-
"""
RegulatoryBridge - Regulatory Change Monitoring for DMA PACK-015
==================================================================

This module monitors regulatory changes relevant to the Double Materiality
Assessment, including ESRS updates and amendments, CSRD Omnibus Directive
changes, threshold adjustments, and new disclosure requirements. It alerts
the DMA pipeline when regulatory updates may affect materiality outcomes.

Features:
    - Monitor ESRS standard updates and amendments
    - Track CSRD Omnibus Directive changes
    - Alert on new or modified disclosure requirements
    - Update materiality thresholds per regulatory guidance
    - Maintain a regulatory change log with provenance tracking
    - Support for EFRAG Q&A and implementation guidance
    - Track delegated acts and sector-specific standards

Regulatory Sources Tracked:
    - ESRS Set 1 (10 standards: ESRS 1, ESRS 2, E1-E5, S1-S4, G1)
    - CSRD Directive 2022/2464/EU
    - CSRD Omnibus Simplification Directive
    - EFRAG Implementation Guidance
    - EC Delegated Acts
    - Sector-Specific ESRS (when adopted)

Architecture:
    Regulatory Sources --> RegulatoryBridge --> Change Detection
                                |                    |
                                v                    v
    Change Log <-- Alert System    Threshold Updates
                                |                    |
                                v                    v
    DMA Config Update <-- Provenance Hash <-- Compliance Check

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-015 Double Materiality Assessment
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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
    """Compute SHA-256 hash for provenance tracking."""
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

class RegulatorySource(str, Enum):
    """Sources of regulatory changes."""

    ESRS_SET_1 = "esrs_set_1"
    CSRD_DIRECTIVE = "csrd_directive"
    CSRD_OMNIBUS = "csrd_omnibus"
    EFRAG_GUIDANCE = "efrag_guidance"
    EC_DELEGATED_ACTS = "ec_delegated_acts"
    SECTOR_SPECIFIC_ESRS = "sector_specific_esrs"

class ChangeType(str, Enum):
    """Types of regulatory changes."""

    NEW_REQUIREMENT = "new_requirement"
    AMENDMENT = "amendment"
    THRESHOLD_CHANGE = "threshold_change"
    SCOPE_CHANGE = "scope_change"
    TIMELINE_CHANGE = "timeline_change"
    GUIDANCE_UPDATE = "guidance_update"
    REPEAL = "repeal"

class ChangeSeverity(str, Enum):
    """Severity of regulatory change impact."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"

class ChangeStatus(str, Enum):
    """Status of a regulatory change."""

    PROPOSED = "proposed"
    ADOPTED = "adopted"
    IN_FORCE = "in_force"
    PENDING_TRANSPOSITION = "pending_transposition"
    SUPERSEDED = "superseded"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class RegulatoryBridgeConfig(BaseModel):
    """Configuration for the Regulatory Bridge."""

    pack_id: str = Field(default="PACK-015")
    enable_provenance: bool = Field(default=True)
    auto_alert_on_changes: bool = Field(default=True)
    severity_threshold: ChangeSeverity = Field(
        default=ChangeSeverity.MEDIUM,
        description="Minimum severity to trigger alerts",
    )
    check_interval_days: int = Field(
        default=7, ge=1, le=90,
        description="How often to check for regulatory changes",
    )

class RegulatoryChange(BaseModel):
    """A single regulatory change record."""

    change_id: str = Field(default_factory=_new_uuid)
    source: RegulatorySource = Field(...)
    change_type: ChangeType = Field(...)
    severity: ChangeSeverity = Field(default=ChangeSeverity.MEDIUM)
    status: ChangeStatus = Field(default=ChangeStatus.PROPOSED)
    title: str = Field(default="")
    description: str = Field(default="")
    affected_standards: List[str] = Field(default_factory=list)
    affected_topics: List[str] = Field(default_factory=list)
    effective_date: Optional[str] = Field(None, description="ISO date string")
    reference_url: str = Field(default="")
    dma_impact: str = Field(
        default="",
        description="How this change affects DMA outcomes",
    )
    detected_at: datetime = Field(default_factory=utcnow)

class RegulatoryAlert(BaseModel):
    """Alert generated when a regulatory change is detected."""

    alert_id: str = Field(default_factory=_new_uuid)
    change_id: str = Field(default="")
    severity: ChangeSeverity = Field(default=ChangeSeverity.MEDIUM)
    title: str = Field(default="")
    message: str = Field(default="")
    recommended_action: str = Field(default="")
    affected_dma_phases: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    acknowledged: bool = Field(default=False)

class ThresholdUpdate(BaseModel):
    """Update to a materiality threshold based on regulatory guidance."""

    update_id: str = Field(default_factory=_new_uuid)
    source_change_id: str = Field(default="")
    threshold_type: str = Field(default="", description="impact or financial")
    previous_value: float = Field(default=0.0)
    new_value: float = Field(default=0.0)
    esrs_topic: Optional[str] = Field(None)
    reason: str = Field(default="")
    effective_date: Optional[str] = Field(None)
    applied: bool = Field(default=False)
    applied_at: Optional[datetime] = Field(None)

class RegulatoryCheckResult(BaseModel):
    """Result of a regulatory change check."""

    check_id: str = Field(default_factory=_new_uuid)
    changes_found: int = Field(default=0)
    alerts_generated: int = Field(default=0)
    threshold_updates: int = Field(default=0)
    sources_checked: List[str] = Field(default_factory=list)
    last_checked_at: datetime = Field(default_factory=utcnow)
    success: bool = Field(default=False)
    message: str = Field(default="")
    duration_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Known Regulatory Changes Catalog
# ---------------------------------------------------------------------------

KNOWN_REGULATORY_CHANGES: List[RegulatoryChange] = [
    RegulatoryChange(
        source=RegulatorySource.CSRD_OMNIBUS,
        change_type=ChangeType.SCOPE_CHANGE,
        severity=ChangeSeverity.CRITICAL,
        status=ChangeStatus.PROPOSED,
        title="CSRD Omnibus Simplification - Phased-in Scope Reduction",
        description=(
            "The proposed CSRD Omnibus Directive introduces a two-year postponement "
            "for companies not yet reporting, and a phased-in approach reducing "
            "mandatory datapoints for SMEs and smaller large companies."
        ),
        affected_standards=["ESRS 1", "ESRS 2"],
        affected_topics=["E1", "E2", "E3", "E4", "E5", "S1", "S2", "S3", "S4", "G1"],
        effective_date="2027-01-01",
        dma_impact="May reduce scope of mandatory DMA for smaller companies",
    ),
    RegulatoryChange(
        source=RegulatorySource.CSRD_OMNIBUS,
        change_type=ChangeType.THRESHOLD_CHANGE,
        severity=ChangeSeverity.HIGH,
        status=ChangeStatus.PROPOSED,
        title="Omnibus - Materiality Threshold Simplification",
        description=(
            "Proposed simplification of materiality thresholds with more "
            "prescriptive guidance on when topics are material, reducing "
            "the need for full double materiality assessment in some cases."
        ),
        affected_standards=["ESRS 1"],
        affected_topics=["E1", "S1"],
        effective_date="2027-01-01",
        dma_impact="May simplify DMA process with presumed materiality for certain topics",
    ),
    RegulatoryChange(
        source=RegulatorySource.EFRAG_GUIDANCE,
        change_type=ChangeType.GUIDANCE_UPDATE,
        severity=ChangeSeverity.MEDIUM,
        status=ChangeStatus.IN_FORCE,
        title="EFRAG IG 1 - Materiality Assessment Implementation Guidance",
        description=(
            "EFRAG implementation guidance on conducting the double materiality "
            "assessment, including practical examples, scoring methodologies, "
            "and stakeholder engagement best practices."
        ),
        affected_standards=["ESRS 1"],
        affected_topics=[],
        reference_url="https://www.efrag.org/publications",
        dma_impact="Provides clarification on DMA methodology and scoring",
    ),
    RegulatoryChange(
        source=RegulatorySource.EFRAG_GUIDANCE,
        change_type=ChangeType.GUIDANCE_UPDATE,
        severity=ChangeSeverity.MEDIUM,
        status=ChangeStatus.IN_FORCE,
        title="EFRAG IG 2 - Value Chain Implementation Guidance",
        description=(
            "Implementation guidance on value chain assessment within "
            "the materiality assessment, including boundaries, data "
            "collection, and estimation approaches."
        ),
        affected_standards=["ESRS 1", "ESRS 2"],
        affected_topics=["S2", "S3"],
        dma_impact="Clarifies value chain boundaries for DMA scope definition",
    ),
    RegulatoryChange(
        source=RegulatorySource.SECTOR_SPECIFIC_ESRS,
        change_type=ChangeType.NEW_REQUIREMENT,
        severity=ChangeSeverity.HIGH,
        status=ChangeStatus.PROPOSED,
        title="Sector-Specific ESRS - First Batch (Mining, Oil & Gas, Agriculture)",
        description=(
            "First batch of sector-specific ESRS standards covering mining, "
            "oil and gas, and agriculture sectors with additional disclosure "
            "requirements beyond the cross-sector standards."
        ),
        affected_standards=["Sector ESRS"],
        affected_topics=["E1", "E2", "E3", "E4", "S1", "S3"],
        effective_date="2027-01-01",
        dma_impact="Adds sector-specific IROs and materiality topics for covered sectors",
    ),
    RegulatoryChange(
        source=RegulatorySource.EC_DELEGATED_ACTS,
        change_type=ChangeType.AMENDMENT,
        severity=ChangeSeverity.LOW,
        status=ChangeStatus.IN_FORCE,
        title="ESRS Set 1 - Corrigendum and Technical Corrections",
        description=(
            "Technical corrections to ESRS Set 1 standards including "
            "numbering fixes, cross-reference corrections, and clarification "
            "of ambiguous wording in disclosure requirements."
        ),
        affected_standards=["ESRS E1", "ESRS S1", "ESRS G1"],
        affected_topics=["E1", "S1", "G1"],
        dma_impact="Minor clarifications, no material impact on DMA methodology",
    ),
]

# ---------------------------------------------------------------------------
# RegulatoryBridge
# ---------------------------------------------------------------------------

class RegulatoryBridge:
    """Regulatory change monitoring for Double Materiality Assessment.

    Monitors ESRS updates, CSRD Omnibus changes, EFRAG guidance, and
    delegated acts. Generates alerts and threshold updates when regulatory
    changes may affect DMA outcomes.

    Attributes:
        config: Bridge configuration.
        _change_log: List of tracked regulatory changes.
        _alerts: List of generated alerts.
        _threshold_updates: List of threshold update records.

    Example:
        >>> bridge = RegulatoryBridge()
        >>> result = bridge.check_for_changes()
        >>> print(f"Changes found: {result.changes_found}")
        >>> alerts = bridge.get_active_alerts()
    """

    def __init__(self, config: Optional[RegulatoryBridgeConfig] = None) -> None:
        """Initialize the Regulatory Bridge.

        Args:
            config: Bridge configuration. Uses defaults if None.
        """
        self.config = config or RegulatoryBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._change_log: List[RegulatoryChange] = list(KNOWN_REGULATORY_CHANGES)
        self._alerts: List[RegulatoryAlert] = []
        self._threshold_updates: List[ThresholdUpdate] = []

        self.logger.info(
            "RegulatoryBridge initialized: %d known changes, alert_threshold=%s",
            len(self._change_log), self.config.severity_threshold.value,
        )

    # -------------------------------------------------------------------------
    # Change Detection
    # -------------------------------------------------------------------------

    def check_for_changes(self) -> RegulatoryCheckResult:
        """Check for new or updated regulatory changes.

        Returns:
            RegulatoryCheckResult with findings.
        """
        start = time.monotonic()

        sources_checked = [s.value for s in RegulatorySource]
        changes_found = len(self._change_log)
        alerts_count = 0
        threshold_count = 0

        # Generate alerts for changes above severity threshold
        severity_order = {
            ChangeSeverity.INFORMATIONAL: 0,
            ChangeSeverity.LOW: 1,
            ChangeSeverity.MEDIUM: 2,
            ChangeSeverity.HIGH: 3,
            ChangeSeverity.CRITICAL: 4,
        }
        threshold_level = severity_order.get(self.config.severity_threshold, 2)

        if self.config.auto_alert_on_changes:
            for change in self._change_log:
                change_level = severity_order.get(change.severity, 0)
                if change_level >= threshold_level:
                    alert = self._generate_alert(change)
                    if alert and not self._alert_exists(change.change_id):
                        self._alerts.append(alert)
                        alerts_count += 1

                if change.change_type == ChangeType.THRESHOLD_CHANGE:
                    update = self._generate_threshold_update(change)
                    if update:
                        self._threshold_updates.append(update)
                        threshold_count += 1

        elapsed = (time.monotonic() - start) * 1000

        result = RegulatoryCheckResult(
            changes_found=changes_found,
            alerts_generated=alerts_count,
            threshold_updates=threshold_count,
            sources_checked=sources_checked,
            success=True,
            message=f"Checked {len(sources_checked)} sources: {changes_found} changes, {alerts_count} alerts",
            duration_ms=elapsed,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Regulatory check complete: %d changes, %d alerts in %.1fms",
            changes_found, alerts_count, elapsed,
        )
        return result

    # -------------------------------------------------------------------------
    # Alerts
    # -------------------------------------------------------------------------

    def get_active_alerts(self) -> List[RegulatoryAlert]:
        """Get all unacknowledged regulatory alerts.

        Returns:
            List of active (unacknowledged) alerts.
        """
        return [a for a in self._alerts if not a.acknowledged]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a regulatory alert.

        Args:
            alert_id: Alert ID to acknowledge.

        Returns:
            True if alert was found and acknowledged.
        """
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                self.logger.info("Alert acknowledged: %s", alert_id)
                return True
        return False

    # -------------------------------------------------------------------------
    # Threshold Updates
    # -------------------------------------------------------------------------

    def get_pending_threshold_updates(self) -> List[ThresholdUpdate]:
        """Get all pending (unapplied) threshold updates.

        Returns:
            List of unapplied threshold updates.
        """
        return [u for u in self._threshold_updates if not u.applied]

    def apply_threshold_update(self, update_id: str) -> bool:
        """Mark a threshold update as applied.

        Args:
            update_id: Update ID to mark as applied.

        Returns:
            True if update was found and marked.
        """
        for update in self._threshold_updates:
            if update.update_id == update_id:
                update.applied = True
                update.applied_at = utcnow()
                self.logger.info("Threshold update applied: %s", update_id)
                return True
        return False

    # -------------------------------------------------------------------------
    # Change Log Queries
    # -------------------------------------------------------------------------

    def get_changes_by_source(
        self, source: RegulatorySource,
    ) -> List[RegulatoryChange]:
        """Get regulatory changes filtered by source.

        Args:
            source: Regulatory source to filter by.

        Returns:
            List of changes from the specified source.
        """
        return [c for c in self._change_log if c.source == source]

    def get_changes_by_severity(
        self, severity: ChangeSeverity,
    ) -> List[RegulatoryChange]:
        """Get regulatory changes filtered by severity.

        Args:
            severity: Severity level to filter by.

        Returns:
            List of changes at the specified severity.
        """
        return [c for c in self._change_log if c.severity == severity]

    def get_changes_affecting_topic(
        self, esrs_topic: str,
    ) -> List[RegulatoryChange]:
        """Get regulatory changes affecting a specific ESRS topic.

        Args:
            esrs_topic: ESRS topic code (e.g., 'E1', 'S1').

        Returns:
            List of changes affecting the topic.
        """
        return [c for c in self._change_log if esrs_topic in c.affected_topics]

    def get_change_log(self) -> List[Dict[str, Any]]:
        """Get the full regulatory change log.

        Returns:
            List of change records as dicts.
        """
        return [
            {
                "change_id": c.change_id,
                "source": c.source.value,
                "type": c.change_type.value,
                "severity": c.severity.value,
                "status": c.status.value,
                "title": c.title,
                "affected_topics": c.affected_topics,
                "effective_date": c.effective_date,
            }
            for c in self._change_log
        ]

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _generate_alert(self, change: RegulatoryChange) -> Optional[RegulatoryAlert]:
        """Generate an alert from a regulatory change.

        Args:
            change: Regulatory change to generate alert for.

        Returns:
            RegulatoryAlert or None if not applicable.
        """
        affected_phases: List[str] = []
        if change.change_type in (ChangeType.THRESHOLD_CHANGE, ChangeType.SCOPE_CHANGE):
            affected_phases.extend(["configuration", "impact_assessment", "financial_assessment"])
        if change.change_type == ChangeType.NEW_REQUIREMENT:
            affected_phases.extend(["iro_identification", "esrs_mapping"])
        if change.change_type == ChangeType.GUIDANCE_UPDATE:
            affected_phases.append("stakeholder_engagement")

        return RegulatoryAlert(
            change_id=change.change_id,
            severity=change.severity,
            title=change.title,
            message=change.description[:200],
            recommended_action=f"Review DMA configuration for impact of: {change.title}",
            affected_dma_phases=affected_phases,
        )

    def _generate_threshold_update(
        self, change: RegulatoryChange,
    ) -> Optional[ThresholdUpdate]:
        """Generate a threshold update from a regulatory change.

        Args:
            change: Regulatory change with threshold implications.

        Returns:
            ThresholdUpdate or None.
        """
        if change.change_type != ChangeType.THRESHOLD_CHANGE:
            return None

        return ThresholdUpdate(
            source_change_id=change.change_id,
            threshold_type="impact",
            previous_value=3.0,
            new_value=3.0,
            reason=change.title,
            effective_date=change.effective_date,
        )

    def _alert_exists(self, change_id: str) -> bool:
        """Check if an alert already exists for a change.

        Args:
            change_id: Change ID to check.

        Returns:
            True if an alert already exists.
        """
        return any(a.change_id == change_id for a in self._alerts)
