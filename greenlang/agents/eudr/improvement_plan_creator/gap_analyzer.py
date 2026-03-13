# -*- coding: utf-8 -*-
"""
Gap Analyzer Engine - AGENT-EUDR-035: Improvement Plan Creator

Performs compliance gap analysis by comparing aggregated findings against
EUDR regulatory requirements. Maps findings to specific articles,
calculates severity scores, identifies compliance deltas, and generates
structured gap records for action planning.

Zero-Hallucination:
    - All severity scores are deterministic Decimal arithmetic
    - Gap-to-article mapping uses static lookup tables
    - No LLM involvement in gap scoring or classification

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-035 (GL-EUDR-IPC-035)
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import ImprovementPlanCreatorConfig, get_config
from .models import (
    AGENT_ID,
    AggregatedFindings,
    ComplianceGap,
    Finding,
    GapSeverity,
    GAP_SEVERITY_THRESHOLDS,
)
from .provenance import ProvenanceTracker
from . import metrics as m

logger = logging.getLogger(__name__)

# EUDR article-to-requirement mapping (static, no hallucination)
_EUDR_REQUIREMENTS: Dict[str, Dict[str, str]] = {
    "Article 3": {
        "ref": "Art. 3 - Prohibition",
        "requirement": "Products placed on EU market must be deforestation-free",
    },
    "Article 4": {
        "ref": "Art. 4 - Due Diligence Obligations",
        "requirement": "Operators must exercise due diligence before placing or exporting products",
    },
    "Article 8": {
        "ref": "Art. 8 - Traceability",
        "requirement": "Supply chain geolocation and traceability data required",
    },
    "Article 9": {
        "ref": "Art. 9 - Information Requirements",
        "requirement": "Product-level geolocation data for all plots of land",
    },
    "Article 10": {
        "ref": "Art. 10 - Risk Assessment",
        "requirement": "Comprehensive risk assessment of deforestation and degradation risk",
    },
    "Article 11": {
        "ref": "Art. 11 - Risk Mitigation",
        "requirement": "Adequate risk mitigation measures when risk is not negligible",
    },
    "Article 12": {
        "ref": "Art. 12 - Monitoring & Review",
        "requirement": "Ongoing monitoring and periodic review of due diligence",
    },
    "Article 14": {
        "ref": "Art. 14 - Due Diligence Statement",
        "requirement": "Submit DDS before placing or exporting products",
    },
    "Article 29": {
        "ref": "Art. 29 - Competent Authorities",
        "requirement": "Cooperation with competent authority checks and audits",
    },
    "Article 31": {
        "ref": "Art. 31 - Record Keeping",
        "requirement": "Retain due diligence records for minimum 5 years",
    },
}

# Finding source to typical EUDR article mapping
_SOURCE_ARTICLE_MAP: Dict[str, str] = {
    "risk_assessment": "Article 10",
    "country_risk": "Article 10",
    "supplier_risk": "Article 10",
    "commodity_risk": "Article 10",
    "deforestation_alert": "Article 3",
    "legal_compliance": "Article 4",
    "document_authentication": "Article 14",
    "satellite_monitoring": "Article 12",
    "mitigation_measure": "Article 11",
    "audit_manager": "Article 29",
    "manual": "Article 4",
}


class GapAnalyzer:
    """Analyzes compliance gaps from aggregated findings.

    Compares the current compliance state (as evidenced by findings)
    against EUDR regulatory requirements to identify gaps, calculate
    severity scores, and produce structured gap records.

    Example:
        >>> engine = GapAnalyzer()
        >>> gaps = await engine.analyze_gaps(aggregation, "PLAN-001")
        >>> assert all(g.severity_score >= Decimal("0") for g in gaps)
    """

    def __init__(self, config: Optional[ImprovementPlanCreatorConfig] = None) -> None:
        """Initialize GapAnalyzer.

        Args:
            config: Optional configuration override.
        """
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._store: Dict[str, List[ComplianceGap]] = {}
        logger.info("GapAnalyzer initialized")

    async def analyze_gaps(
        self,
        aggregation: AggregatedFindings,
        plan_id: str = "",
    ) -> List[ComplianceGap]:
        """Analyze compliance gaps from aggregated findings.

        Args:
            aggregation: Aggregated findings to analyze.
            plan_id: Parent plan identifier.

        Returns:
            List of identified compliance gaps.
        """
        start = time.monotonic()
        gaps: List[ComplianceGap] = []

        for finding in aggregation.findings:
            gap = self._finding_to_gap(finding, plan_id)
            if gap is not None:
                gaps.append(gap)
                m.record_gap_identified(gap.severity.value)

        # Enforce max gaps
        gaps = gaps[:self.config.max_gaps_per_analysis]

        # Store for later retrieval
        self._store[plan_id] = gaps

        # Provenance
        provenance_data = {
            "plan_id": plan_id,
            "aggregation_id": aggregation.aggregation_id,
            "gaps_identified": len(gaps),
        }
        provenance_hash = self._provenance.compute_hash(provenance_data)

        for gap in gaps:
            gap.provenance_hash = provenance_hash

        self._provenance.record(
            "gap_analysis", "analyze", plan_id, AGENT_ID,
            metadata={"count": len(gaps), "aggregation_id": aggregation.aggregation_id},
        )

        elapsed = time.monotonic() - start
        m.observe_gap_analysis_duration(elapsed)

        logger.info(
            "Identified %d compliance gaps for plan %s in %.1fms",
            len(gaps), plan_id, elapsed * 1000,
        )
        return gaps

    def _finding_to_gap(
        self, finding: Finding, plan_id: str
    ) -> Optional[ComplianceGap]:
        """Convert a finding to a compliance gap.

        Args:
            finding: Source finding.
            plan_id: Parent plan identifier.

        Returns:
            ComplianceGap or None if finding does not indicate a gap.
        """
        # Determine article reference
        article_key = finding.eudr_article_ref or _SOURCE_ARTICLE_MAP.get(
            finding.source.value, "Article 4"
        )
        requirement_info = _EUDR_REQUIREMENTS.get(article_key, {})

        # Calculate severity score (normalized 0-1 from 0-100 risk score)
        severity_score = (finding.risk_score / Decimal("100")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        severity_score = min(max(severity_score, Decimal("0")), Decimal("1"))

        # Classify severity
        severity = self._classify_severity(severity_score)

        gap_id = f"GAP-{uuid.uuid4().hex[:12]}"

        return ComplianceGap(
            gap_id=gap_id,
            plan_id=plan_id,
            title=f"Gap: {finding.title}",
            description=(
                f"Compliance gap identified from {finding.source.value} "
                f"finding: {finding.description or finding.title}"
            ),
            severity=severity,
            severity_score=severity_score,
            eudr_article_ref=requirement_info.get("ref", article_key),
            eudr_requirement=requirement_info.get("requirement", ""),
            current_state=f"Finding detected: {finding.title}",
            required_state=requirement_info.get("requirement", "Full compliance"),
            finding_ids=[finding.finding_id],
            commodity=finding.commodity,
            risk_dimension=finding.source.value,
        )

    def _classify_severity(self, score: Decimal) -> GapSeverity:
        """Classify gap severity from normalized score.

        Args:
            score: Normalized severity score (0-1).

        Returns:
            GapSeverity enum value.
        """
        if score >= self.config.gap_severity_critical_threshold:
            return GapSeverity.CRITICAL
        elif score >= self.config.gap_severity_high_threshold:
            return GapSeverity.HIGH
        elif score >= self.config.gap_severity_medium_threshold:
            return GapSeverity.MEDIUM
        elif score >= self.config.gap_severity_low_threshold:
            return GapSeverity.LOW
        return GapSeverity.INFORMATIONAL

    async def get_gaps(self, plan_id: str) -> List[ComplianceGap]:
        """Retrieve stored gaps for a plan.

        Args:
            plan_id: Plan identifier.

        Returns:
            List of ComplianceGap for the plan.
        """
        return self._store.get(plan_id, [])

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "engine": "GapAnalyzer",
            "status": "healthy",
            "plans_analyzed": len(self._store),
        }
