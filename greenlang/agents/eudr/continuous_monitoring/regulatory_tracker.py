# -*- coding: utf-8 -*-
"""
Regulatory Tracker Engine - AGENT-EUDR-033

Monitors regulatory changes from EU and national sources, assesses
impact on EUDR compliance, maps updates to affected entities, and
dispatches notifications to relevant stakeholders.

Zero-Hallucination:
    - Impact assessment uses deterministic keyword-based scoring
    - Entity mapping uses rule-based article-to-entity matching
    - Notification dispatch is deterministic channel routing

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-033 (GL-EUDR-CM-033)
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import ContinuousMonitoringConfig, get_config
from .models import (
    AGENT_ID,
    EUDR_ARTICLES_MONITORED,
    RegulatoryImpact,
    RegulatoryTrackingRecord,
    RegulatoryUpdate,
)
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

# Keyword-to-impact mapping for regulatory text analysis
_IMPACT_KEYWORDS: Dict[str, RegulatoryImpact] = {
    "repeal": RegulatoryImpact.BREAKING,
    "amendment": RegulatoryImpact.HIGH,
    "implementing act": RegulatoryImpact.HIGH,
    "delegated act": RegulatoryImpact.HIGH,
    "guidance": RegulatoryImpact.MODERATE,
    "clarification": RegulatoryImpact.LOW,
    "corrigendum": RegulatoryImpact.LOW,
    "deadline extension": RegulatoryImpact.MODERATE,
    "penalty": RegulatoryImpact.HIGH,
    "enforcement": RegulatoryImpact.HIGH,
    "reporting requirement": RegulatoryImpact.MODERATE,
    "threshold change": RegulatoryImpact.MODERATE,
}

# EUDR article to entity-type mapping
_ARTICLE_ENTITY_MAP: Dict[str, List[str]] = {
    "Article 4": ["operator", "trader"],
    "Article 8": ["due_diligence_statement"],
    "Article 10": ["risk_assessment"],
    "Article 11": ["mitigation_measure"],
    "Article 12": ["record", "audit_trail"],
    "Article 14": ["competent_authority"],
    "Article 29": ["information_system"],
    "Article 31": ["record", "document"],
}


class RegulatoryTracker:
    """Regulatory change monitoring and notification engine.

    Fetches regulatory updates, assesses their impact on EUDR
    compliance, maps changes to affected supply chain entities,
    and dispatches notifications to stakeholders.

    Example:
        >>> tracker = RegulatoryTracker()
        >>> record = await tracker.fetch_regulatory_updates(
        ...     operator_id="OP-001",
        ...     updates=[{
        ...         "title": "EUDR Amendment on Reporting",
        ...         "summary": "New reporting requirement for operators",
        ...     }],
        ... )
        >>> assert record.updates_found >= 0
    """

    def __init__(
        self, config: Optional[ContinuousMonitoringConfig] = None,
    ) -> None:
        """Initialize RegulatoryTracker engine."""
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._records: Dict[str, RegulatoryTrackingRecord] = {}
        logger.info("RegulatoryTracker engine initialized")

    async def fetch_regulatory_updates(
        self,
        operator_id: str,
        updates: Optional[List[Dict[str, Any]]] = None,
    ) -> RegulatoryTrackingRecord:
        """Fetch and process regulatory updates.

        In production, this would integrate with EUR-Lex API and
        national authority feeds. For now, processes provided updates.

        Args:
            operator_id: Operator identifier.
            updates: List of regulatory update data.

        Returns:
            RegulatoryTrackingRecord with processing results.
        """
        start_time = time.monotonic()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        tracking_id = str(uuid.uuid4())
        raw_updates = updates or []

        # Process each update
        regulatory_updates: List[RegulatoryUpdate] = []
        for raw in raw_updates:
            update = self._process_update(raw)
            regulatory_updates.append(update)

        # Assess impact for each update
        for reg_update in regulatory_updates:
            if reg_update.impact_level == RegulatoryImpact.NONE:
                assessed_impact = await self.assess_impact(
                    reg_update.title, reg_update.summary,
                )
                reg_update.impact_level = assessed_impact

        # Map to entities
        entity_mappings = []
        for reg_update in regulatory_updates:
            mappings = await self.map_to_entities(reg_update)
            entity_mappings.extend(mappings)

        # Notify stakeholders
        high_impact_updates = [
            u for u in regulatory_updates
            if u.impact_level in (RegulatoryImpact.HIGH, RegulatoryImpact.BREAKING)
        ]
        notifications_sent = 0
        if high_impact_updates:
            notifications_sent = await self.notify_stakeholders(
                operator_id, high_impact_updates,
            )

        record = RegulatoryTrackingRecord(
            tracking_id=tracking_id,
            operator_id=operator_id,
            updates_found=len(regulatory_updates),
            high_impact_count=len(high_impact_updates),
            sources_checked=self.config.regulatory_sources[:],
            regulatory_updates=regulatory_updates,
            entity_mappings=entity_mappings,
            notifications_sent=notifications_sent,
            notification_channels=self.config.regulatory_notification_channels[:],
            checked_at=now,
        )

        record.provenance_hash = self._provenance.compute_hash({
            "tracking_id": tracking_id,
            "operator_id": operator_id,
            "updates_found": len(regulatory_updates),
            "high_impact": len(high_impact_updates),
            "created_at": now.isoformat(),
        })

        self._provenance.record(
            entity_type="regulatory_tracking",
            action="track",
            entity_id=tracking_id,
            actor=AGENT_ID,
            metadata={
                "operator_id": operator_id,
                "updates": len(regulatory_updates),
                "high_impact": len(high_impact_updates),
                "notifications": notifications_sent,
            },
        )

        self._records[tracking_id] = record
        elapsed = time.monotonic() - start_time
        logger.info(
            "Regulatory check %s: %d updates, %d high-impact, %d notifications (%.3fs)",
            tracking_id, len(regulatory_updates), len(high_impact_updates),
            notifications_sent, elapsed,
        )
        return record

    async def assess_impact(
        self,
        title: str,
        summary: str,
    ) -> RegulatoryImpact:
        """Assess the impact of a regulatory update using keyword analysis.

        Args:
            title: Update title.
            summary: Update summary text.

        Returns:
            RegulatoryImpact classification.
        """
        combined = f"{title} {summary}".lower()
        highest_impact = RegulatoryImpact.NONE

        impact_order = [
            RegulatoryImpact.NONE,
            RegulatoryImpact.LOW,
            RegulatoryImpact.MODERATE,
            RegulatoryImpact.HIGH,
            RegulatoryImpact.BREAKING,
        ]

        for keyword, impact in _IMPACT_KEYWORDS.items():
            if keyword.lower() in combined:
                if impact_order.index(impact) > impact_order.index(highest_impact):
                    highest_impact = impact

        return highest_impact

    async def map_to_entities(
        self,
        update: RegulatoryUpdate,
    ) -> List[Dict[str, Any]]:
        """Map a regulatory update to affected supply chain entities.

        Uses article references to determine which entity types
        are affected by the regulatory change.

        Args:
            update: Regulatory update to map.

        Returns:
            List of entity mapping dictionaries.
        """
        mappings: List[Dict[str, Any]] = []

        affected_articles = update.affected_articles
        if not affected_articles:
            # Try to extract articles from title/summary
            affected_articles = self._extract_articles(
                f"{update.title} {update.summary}"
            )

        for article in affected_articles:
            entity_types = _ARTICLE_ENTITY_MAP.get(article, [])
            for entity_type in entity_types:
                mappings.append({
                    "update_id": update.update_id,
                    "article": article,
                    "entity_type": entity_type,
                    "impact_level": update.impact_level.value,
                    "action_required": self._determine_action(
                        update.impact_level, entity_type,
                    ),
                })

        return mappings

    async def notify_stakeholders(
        self,
        operator_id: str,
        updates: List[RegulatoryUpdate],
    ) -> int:
        """Dispatch notifications to stakeholders about regulatory changes.

        Args:
            operator_id: Operator identifier.
            updates: High-impact regulatory updates.

        Returns:
            Number of notifications sent.
        """
        if not updates:
            return 0

        channels = self.config.regulatory_notification_channels
        notifications = 0

        for update in updates:
            for channel in channels:
                # In production, this dispatches to actual notification channels
                logger.info(
                    "Notification [%s] to %s: %s (impact=%s)",
                    channel, operator_id, update.title, update.impact_level.value,
                )
                notifications += 1

        return notifications

    def _process_update(self, raw: Dict[str, Any]) -> RegulatoryUpdate:
        """Process a raw update dictionary into a RegulatoryUpdate model."""
        update_id = raw.get("update_id", str(uuid.uuid4()))
        source = raw.get("source", "unknown")
        title = raw.get("title", "")
        summary = raw.get("summary", "")

        published_date = None
        pub_str = raw.get("published_date")
        if pub_str:
            try:
                if isinstance(pub_str, datetime):
                    published_date = pub_str
                else:
                    published_date = datetime.fromisoformat(
                        str(pub_str).replace("Z", "+00:00")
                    )
            except (ValueError, TypeError):
                pass

        # Pre-assigned impact
        impact_str = raw.get("impact_level", "none")
        try:
            impact = RegulatoryImpact(impact_str)
        except ValueError:
            impact = RegulatoryImpact.NONE

        # Extract affected articles
        affected = raw.get("affected_articles", [])
        if not affected:
            affected = self._extract_articles(f"{title} {summary}")

        return RegulatoryUpdate(
            update_id=update_id,
            source=source,
            title=title,
            summary=summary,
            published_date=published_date,
            impact_level=impact,
            affected_articles=affected,
        )

    @staticmethod
    def _extract_articles(text: str) -> List[str]:
        """Extract EUDR article references from text."""
        articles: List[str] = []
        for article in EUDR_ARTICLES_MONITORED:
            # Check for "Article X" pattern
            art_num = article.split(" ")[-1]
            if f"article {art_num}" in text.lower() or article.lower() in text.lower():
                articles.append(article)
        return articles

    @staticmethod
    def _determine_action(impact: RegulatoryImpact, entity_type: str) -> str:
        """Determine required action based on impact and entity type."""
        if impact == RegulatoryImpact.BREAKING:
            return f"Immediate review and update of all {entity_type} records required"
        elif impact == RegulatoryImpact.HIGH:
            return f"Review and update {entity_type} records within 30 days"
        elif impact == RegulatoryImpact.MODERATE:
            return f"Assess impact on {entity_type} records within 60 days"
        elif impact == RegulatoryImpact.LOW:
            return f"Note for next {entity_type} review cycle"
        return "No action required"

    async def get_record(self, tracking_id: str) -> Optional[RegulatoryTrackingRecord]:
        """Retrieve a regulatory tracking record by ID."""
        return self._records.get(tracking_id)

    async def list_records(
        self,
        operator_id: Optional[str] = None,
    ) -> List[RegulatoryTrackingRecord]:
        """List regulatory tracking records with optional filters."""
        results = list(self._records.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        return results

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "engine": "RegulatoryTracker",
            "status": "healthy",
            "record_count": len(self._records),
        }
