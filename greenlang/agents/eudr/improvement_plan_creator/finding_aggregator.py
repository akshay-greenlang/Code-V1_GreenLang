# -*- coding: utf-8 -*-
"""
Finding Aggregator Engine - AGENT-EUDR-035: Improvement Plan Creator

Consolidates findings from multiple upstream EUDR agents (EUDR-016 through
EUDR-034) into a unified, deduplicated view of compliance status. Performs
source identification, similarity-based deduplication, severity
classification, and confidence-weighted ranking.

Zero-Hallucination:
    - All deduplication uses deterministic string similarity (no LLM)
    - Severity counting is pure arithmetic
    - Confidence thresholds are config-driven

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-035 (GL-EUDR-IPC-035)
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import ImprovementPlanCreatorConfig, get_config
from .models import (
    AGENT_ID,
    AggregatedFindings,
    Finding,
    FindingSource,
    GapSeverity,
)
from .provenance import ProvenanceTracker
from . import metrics as m

logger = logging.getLogger(__name__)


class FindingAggregator:
    """Aggregates findings from upstream EUDR agents.

    Collects, deduplicates, and classifies findings from risk assessment,
    country risk, supplier risk, commodity risk, deforestation alerts,
    legal compliance, document authentication, satellite monitoring,
    mitigation measure, and audit manager agents.

    Example:
        >>> engine = FindingAggregator()
        >>> result = await engine.aggregate_findings("OP-001", findings)
        >>> assert result.duplicates_removed >= 0
    """

    def __init__(self, config: Optional[ImprovementPlanCreatorConfig] = None) -> None:
        """Initialize FindingAggregator.

        Args:
            config: Optional configuration override.
        """
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._store: Dict[str, AggregatedFindings] = {}
        logger.info("FindingAggregator initialized")

    async def aggregate_findings(
        self,
        operator_id: str,
        findings: List[Finding],
        plan_id: str = "",
    ) -> AggregatedFindings:
        """Aggregate and deduplicate findings from multiple sources.

        Args:
            operator_id: Operator under review.
            findings: Raw findings from upstream agents.
            plan_id: Optional parent plan identifier.

        Returns:
            AggregatedFindings with deduplication and severity counts.
        """
        start = time.monotonic()
        aggregation_id = f"AGG-{uuid.uuid4().hex[:12]}"

        # Filter stale findings
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self.config.finding_staleness_days
        )
        fresh = [f for f in findings if f.detected_at >= cutoff]

        # Filter by confidence threshold
        threshold = self.config.finding_confidence_threshold
        confident = [
            f for f in fresh
            if f.risk_score >= threshold * 100 or f.severity in (
                GapSeverity.CRITICAL, GapSeverity.HIGH
            )
        ]

        # Deduplicate
        deduped, removed = self._deduplicate(confident)

        # Enforce max limit
        max_findings = self.config.max_findings_per_aggregation
        deduped = deduped[:max_findings]

        # Count by severity
        critical = sum(1 for f in deduped if f.severity == GapSeverity.CRITICAL)
        high = sum(1 for f in deduped if f.severity == GapSeverity.HIGH)
        medium = sum(1 for f in deduped if f.severity == GapSeverity.MEDIUM)
        low = sum(1 for f in deduped if f.severity == GapSeverity.LOW)

        # Collect unique source agents
        source_agents = sorted(set(f.source_agent_id for f in deduped if f.source_agent_id))

        # Compute provenance
        provenance_data = {
            "aggregation_id": aggregation_id,
            "operator_id": operator_id,
            "total_input": len(findings),
            "total_output": len(deduped),
            "duplicates_removed": removed,
        }
        provenance_hash = self._provenance.compute_hash(provenance_data)

        result = AggregatedFindings(
            aggregation_id=aggregation_id,
            operator_id=operator_id,
            findings=deduped,
            total_findings=len(deduped),
            critical_count=critical,
            high_count=high,
            medium_count=medium,
            low_count=low,
            source_agents=source_agents,
            duplicates_removed=removed,
            provenance_hash=provenance_hash,
        )

        self._store[aggregation_id] = result

        # Metrics
        elapsed = time.monotonic() - start
        m.observe_finding_aggregation_duration(elapsed)
        for f in deduped:
            m.record_finding_aggregated(f.source.value)
        if removed > 0:
            m.record_duplicates_removed(removed)

        self._provenance.record(
            "aggregation", "create", aggregation_id, AGENT_ID,
            metadata={"operator_id": operator_id, "count": len(deduped)},
        )

        logger.info(
            "Aggregated %d findings (%d duplicates removed) for %s in %.1fms",
            len(deduped), removed, operator_id, elapsed * 1000,
        )
        return result

    def _deduplicate(
        self, findings: List[Finding]
    ) -> tuple[List[Finding], int]:
        """Remove duplicate findings using title similarity.

        Uses exact title matching for deduplication. Findings with the
        same title from the same source are considered duplicates; the
        one with the higher risk score is retained.

        Args:
            findings: Findings to deduplicate.

        Returns:
            Tuple of (deduplicated list, number removed).
        """
        seen: Dict[str, Finding] = {}
        removed = 0

        for finding in findings:
            key = f"{finding.source.value}:{finding.title.lower().strip()}"
            if key in seen:
                # Keep the one with higher risk score
                if finding.risk_score > seen[key].risk_score:
                    seen[key] = finding
                removed += 1
            else:
                seen[key] = finding

        deduped = list(seen.values())
        # Sort by severity score descending, then by detected_at
        severity_order = {
            GapSeverity.CRITICAL: 4,
            GapSeverity.HIGH: 3,
            GapSeverity.MEDIUM: 2,
            GapSeverity.LOW: 1,
            GapSeverity.INFORMATIONAL: 0,
        }
        deduped.sort(
            key=lambda f: (severity_order.get(f.severity, 0), float(f.risk_score)),
            reverse=True,
        )
        return deduped, removed

    async def get_aggregation(self, aggregation_id: str) -> Optional[AggregatedFindings]:
        """Retrieve a stored aggregation by ID.

        Args:
            aggregation_id: Aggregation identifier.

        Returns:
            AggregatedFindings or None if not found.
        """
        return self._store.get(aggregation_id)

    async def list_aggregations(
        self,
        operator_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[AggregatedFindings]:
        """List stored aggregations with optional filtering.

        Args:
            operator_id: Filter by operator.
            limit: Max results to return.
            offset: Number of results to skip.

        Returns:
            List of AggregatedFindings.
        """
        results = list(self._store.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        results.sort(key=lambda r: r.aggregated_at, reverse=True)
        return results[offset:offset + limit]

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "engine": "FindingAggregator",
            "status": "healthy",
            "aggregations_stored": len(self._store),
        }
