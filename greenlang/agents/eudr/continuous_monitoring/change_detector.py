# -*- coding: utf-8 -*-
"""
Change Detector Engine - AGENT-EUDR-033

Detects and classifies entity changes across the supply chain including
supplier status, certifications, geolocations, risk scores, compliance
posture, and regulatory environment changes. Computes weighted impact
scores and recommends corrective actions.

Zero-Hallucination:
    - All impact scoring uses Decimal-weighted arithmetic
    - Change detection uses deterministic diff comparisons
    - Action recommendations are rule-based, not LLM-generated

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
    ActionRecommendation,
    ChangeDetectionRecord,
    ChangeImpact,
    ChangeType,
)
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)

# Maps change types to their default base severity for impact calculation
_CHANGE_BASE_SEVERITY: Dict[ChangeType, Decimal] = {
    ChangeType.SUPPLIER_STATUS: Decimal("60"),
    ChangeType.CERTIFICATION: Decimal("70"),
    ChangeType.GEOLOCATION: Decimal("65"),
    ChangeType.RISK_SCORE: Decimal("55"),
    ChangeType.COMPLIANCE_STATUS: Decimal("80"),
    ChangeType.REGULATORY: Decimal("75"),
    ChangeType.DEFORESTATION: Decimal("90"),
    ChangeType.OWNERSHIP: Decimal("50"),
}


class ChangeDetector:
    """Entity change detection and impact assessment engine.

    Compares entity states to detect changes, calculates weighted
    impact scores, categorizes change types, and recommends actions.

    Example:
        >>> detector = ChangeDetector()
        >>> changes = await detector.detect_changes(
        ...     operator_id="OP-001",
        ...     entity_snapshots=[{
        ...         "entity_id": "S-001",
        ...         "old_state": {"status": "active"},
        ...         "new_state": {"status": "suspended"},
        ...     }],
        ... )
        >>> assert len(changes) > 0
    """

    def __init__(
        self, config: Optional[ContinuousMonitoringConfig] = None,
    ) -> None:
        """Initialize ChangeDetector engine."""
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._records: Dict[str, ChangeDetectionRecord] = {}
        logger.info("ChangeDetector engine initialized")

    async def detect_changes(
        self,
        operator_id: str,
        entity_snapshots: List[Dict[str, Any]],
    ) -> List[ChangeDetectionRecord]:
        """Detect changes across entity snapshots.

        Args:
            operator_id: Operator identifier.
            entity_snapshots: List of entity snapshot pairs (old/new state).

        Returns:
            List of ChangeDetectionRecord for each detected change.
        """
        start_time = time.monotonic()
        records: List[ChangeDetectionRecord] = []

        for snapshot in entity_snapshots:
            entity_id = snapshot.get("entity_id", "")
            entity_type = snapshot.get("entity_type", "supplier")
            old_state = snapshot.get("old_state", {})
            new_state = snapshot.get("new_state", {})

            # Find changed fields
            changed_fields = self._diff_states(old_state, new_state)
            if not changed_fields:
                continue

            # Categorize the change type
            change_type = self.categorize_change_type(changed_fields, entity_type)

            # Calculate impact
            impact_score = self.calculate_change_impact(
                change_type, old_state, new_state, changed_fields,
            )

            # Determine impact level
            impact_level = self._score_to_impact(impact_score)

            # Recommend actions
            actions = self.recommend_actions(
                change_type, impact_level, entity_id, changed_fields,
            )

            description = self._build_description(
                entity_id, entity_type, changed_fields,
            )

            now = datetime.now(timezone.utc).replace(microsecond=0)
            detection_id = str(uuid.uuid4())

            record = ChangeDetectionRecord(
                detection_id=detection_id,
                operator_id=operator_id,
                entity_id=entity_id,
                entity_type=entity_type,
                change_type=change_type,
                change_impact=impact_level,
                impact_score=impact_score,
                description=description,
                old_state=old_state,
                new_state=new_state,
                recommended_actions=actions,
                detected_at=now,
            )

            record.provenance_hash = self._provenance.compute_hash({
                "detection_id": detection_id,
                "entity_id": entity_id,
                "change_type": change_type.value,
                "impact_score": str(impact_score),
                "created_at": now.isoformat(),
            })

            self._provenance.record(
                entity_type="change_detection",
                action="detect",
                entity_id=detection_id,
                actor=AGENT_ID,
                metadata={
                    "operator_id": operator_id,
                    "entity_id": entity_id,
                    "change_type": change_type.value,
                    "impact": impact_level.value,
                },
            )

            self._records[detection_id] = record
            records.append(record)

        elapsed = time.monotonic() - start_time
        logger.info(
            "Change detection for %s: %d snapshots -> %d changes (%.3fs)",
            operator_id, len(entity_snapshots), len(records), elapsed,
        )
        return records

    def calculate_change_impact(
        self,
        change_type: ChangeType,
        old_state: Dict[str, Any],
        new_state: Dict[str, Any],
        changed_fields: List[str],
    ) -> Decimal:
        """Calculate weighted impact score for a change.

        Uses configured weights for compliance, risk, supply chain,
        and regulatory dimensions.

        Args:
            change_type: Type of change detected.
            old_state: Previous entity state.
            new_state: Current entity state.
            changed_fields: List of changed field names.

        Returns:
            Impact score 0-100 as Decimal.
        """
        weights = self.config.get_change_impact_weights()
        base_severity = _CHANGE_BASE_SEVERITY.get(change_type, Decimal("50"))

        # Compliance dimension
        compliance_factor = Decimal("1.0")
        if change_type in (ChangeType.COMPLIANCE_STATUS, ChangeType.CERTIFICATION):
            compliance_factor = Decimal("1.5")
        compliance_component = base_severity * weights["compliance"] * compliance_factor

        # Risk dimension
        risk_factor = Decimal("1.0")
        if change_type in (ChangeType.RISK_SCORE, ChangeType.DEFORESTATION):
            risk_factor = Decimal("1.4")
        risk_component = base_severity * weights["risk"] * risk_factor

        # Supply chain dimension
        sc_factor = Decimal("1.0")
        if change_type in (ChangeType.SUPPLIER_STATUS, ChangeType.OWNERSHIP):
            sc_factor = Decimal("1.3")
        sc_component = base_severity * weights["supply_chain"] * sc_factor

        # Regulatory dimension
        reg_factor = Decimal("1.0")
        if change_type == ChangeType.REGULATORY:
            reg_factor = Decimal("1.6")
        reg_component = base_severity * weights["regulatory"] * reg_factor

        # Field count multiplier (more fields changed = higher impact)
        field_multiplier = min(Decimal("1") + Decimal(str(len(changed_fields))) * Decimal("0.05"), Decimal("1.5"))

        raw_score = (
            compliance_component + risk_component + sc_component + reg_component
        ) * field_multiplier

        # Normalize to 0-100
        score = min(Decimal("100"), raw_score.quantize(Decimal("0.01")))
        return score

    def categorize_change_type(
        self,
        changed_fields: List[str],
        entity_type: str,
    ) -> ChangeType:
        """Categorize the type of change based on affected fields.

        Args:
            changed_fields: List of changed field names.
            entity_type: Entity type string.

        Returns:
            ChangeType classification.
        """
        field_lower = [f.lower() for f in changed_fields]

        if any("certification" in f or "cert" in f for f in field_lower):
            return ChangeType.CERTIFICATION
        if any("geolocation" in f or "lat" in f or "lon" in f or "coordinate" in f for f in field_lower):
            return ChangeType.GEOLOCATION
        if any("risk" in f for f in field_lower):
            return ChangeType.RISK_SCORE
        if any("compliance" in f for f in field_lower):
            return ChangeType.COMPLIANCE_STATUS
        if any("regulatory" in f or "regulation" in f for f in field_lower):
            return ChangeType.REGULATORY
        if any("deforestation" in f or "forest" in f for f in field_lower):
            return ChangeType.DEFORESTATION
        if any("owner" in f for f in field_lower):
            return ChangeType.OWNERSHIP
        return ChangeType.SUPPLIER_STATUS

    def recommend_actions(
        self,
        change_type: ChangeType,
        impact: ChangeImpact,
        entity_id: str,
        changed_fields: List[str],
    ) -> List[ActionRecommendation]:
        """Recommend corrective actions based on change analysis.

        Args:
            change_type: Type of change detected.
            impact: Impact level of the change.
            entity_id: Affected entity identifier.
            changed_fields: List of changed fields.

        Returns:
            List of recommended actions.
        """
        actions: List[ActionRecommendation] = []

        if impact in (ChangeImpact.CRITICAL, ChangeImpact.HIGH):
            actions.append(ActionRecommendation(
                action=f"Investigate {change_type.value} change for entity {entity_id}",
                priority="critical" if impact == ChangeImpact.CRITICAL else "high",
                deadline_days=3 if impact == ChangeImpact.CRITICAL else 7,
                category="investigation",
            ))

        if change_type == ChangeType.CERTIFICATION:
            actions.append(ActionRecommendation(
                action=f"Verify certification status for {entity_id}",
                priority="high",
                deadline_days=7,
                category="verification",
            ))
        elif change_type == ChangeType.COMPLIANCE_STATUS:
            actions.append(ActionRecommendation(
                action=f"Re-run compliance audit for {entity_id}",
                priority="high",
                deadline_days=14,
                category="compliance",
            ))
        elif change_type == ChangeType.DEFORESTATION:
            actions.append(ActionRecommendation(
                action=f"Suspend sourcing from {entity_id} pending investigation",
                priority="critical",
                deadline_days=1,
                category="supply_chain",
            ))
        elif change_type == ChangeType.OWNERSHIP:
            actions.append(ActionRecommendation(
                action=f"Re-assess due diligence for {entity_id} after ownership change",
                priority="high",
                deadline_days=30,
                category="due_diligence",
            ))

        if impact == ChangeImpact.MODERATE:
            actions.append(ActionRecommendation(
                action=f"Update monitoring parameters for {entity_id}",
                priority="medium",
                deadline_days=14,
                category="monitoring",
            ))

        return actions

    @staticmethod
    def _diff_states(old_state: Dict[str, Any], new_state: Dict[str, Any]) -> List[str]:
        """Find fields that differ between old and new state."""
        changed: List[str] = []
        all_keys = set(list(old_state.keys()) + list(new_state.keys()))
        for key in all_keys:
            old_val = old_state.get(key)
            new_val = new_state.get(key)
            if str(old_val) != str(new_val):
                changed.append(key)
        return changed

    def _score_to_impact(self, score: Decimal) -> ChangeImpact:
        """Convert a numeric impact score to ChangeImpact level."""
        if score >= Decimal("80"):
            return ChangeImpact.CRITICAL
        elif score >= Decimal("60"):
            return ChangeImpact.HIGH
        elif score >= Decimal("40"):
            return ChangeImpact.MODERATE
        elif score >= Decimal("20"):
            return ChangeImpact.LOW
        return ChangeImpact.NEGLIGIBLE

    @staticmethod
    def _build_description(
        entity_id: str,
        entity_type: str,
        changed_fields: List[str],
    ) -> str:
        """Build a human-readable change description."""
        field_list = ", ".join(changed_fields[:5])
        suffix = f" (+{len(changed_fields) - 5} more)" if len(changed_fields) > 5 else ""
        return f"Change detected in {entity_type} {entity_id}: fields [{field_list}{suffix}]"

    async def get_detection(self, detection_id: str) -> Optional[ChangeDetectionRecord]:
        """Retrieve a change detection record by ID."""
        return self._records.get(detection_id)

    async def list_detections(
        self,
        operator_id: Optional[str] = None,
        change_type: Optional[str] = None,
        impact: Optional[str] = None,
    ) -> List[ChangeDetectionRecord]:
        """List change detection records with optional filters."""
        results = list(self._records.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        if change_type:
            results = [r for r in results if r.change_type.value == change_type]
        if impact:
            results = [r for r in results if r.change_impact.value == impact]
        return results

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "engine": "ChangeDetector",
            "status": "healthy",
            "detection_count": len(self._records),
        }
