# -*- coding: utf-8 -*-
"""
Root Cause Mapper Engine - AGENT-EUDR-035: Improvement Plan Creator

Performs root cause analysis using 5-Whys and Ishikawa (fishbone) diagram
methods. Maps compliance gaps to underlying systemic causes, identifies
cross-cutting patterns, and produces structured root cause records linked
to gaps and actions.

Zero-Hallucination:
    - 5-Whys depth is config-limited (default 5, max 10)
    - Fishbone categories are enum-constrained
    - Confidence scoring is deterministic rule-based
    - No LLM involvement in causal chain construction

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
    ComplianceGap,
    FishboneAnalysis,
    FishboneCategory,
    GapSeverity,
    RootCause,
)
from .provenance import ProvenanceTracker
from . import metrics as m

logger = logging.getLogger(__name__)

# Risk dimension to fishbone category mapping
_DIMENSION_CATEGORY_MAP: Dict[str, FishboneCategory] = {
    "risk_assessment": FishboneCategory.PROCESS,
    "country_risk": FishboneCategory.ENVIRONMENT,
    "supplier_risk": FishboneCategory.SUPPLIERS,
    "commodity_risk": FishboneCategory.DATA,
    "deforestation_alert": FishboneCategory.ENVIRONMENT,
    "legal_compliance": FishboneCategory.POLICY,
    "document_authentication": FishboneCategory.TECHNOLOGY,
    "satellite_monitoring": FishboneCategory.TECHNOLOGY,
    "mitigation_measure": FishboneCategory.MANAGEMENT,
    "audit_manager": FishboneCategory.PROCESS,
    "manual": FishboneCategory.PEOPLE,
}

# 5-Whys template chains by category
_WHYS_TEMPLATES: Dict[FishboneCategory, List[str]] = {
    FishboneCategory.PROCESS: [
        "Process not executed as designed",
        "Process design does not address the requirement",
        "Requirements not fully captured in process design",
        "Stakeholder requirements not communicated to process owners",
        "No formal requirements gathering mechanism exists",
    ],
    FishboneCategory.TECHNOLOGY: [
        "Technology system did not detect the issue",
        "Detection rules/algorithms are incomplete",
        "System configuration not updated for new requirements",
        "Change management process not triggered for regulatory updates",
        "No automated regulatory change tracking system deployed",
    ],
    FishboneCategory.PEOPLE: [
        "Staff did not follow the prescribed procedure",
        "Procedure not clearly communicated to staff",
        "Training program does not cover this scenario",
        "Training needs assessment is outdated",
        "No periodic competency review process exists",
    ],
    FishboneCategory.POLICY: [
        "Policy does not address the identified gap",
        "Policy has not been updated for current regulations",
        "Policy review cycle is too infrequent",
        "No trigger mechanism for policy updates on regulatory change",
        "Regulatory monitoring function is understaffed",
    ],
    FishboneCategory.DATA: [
        "Data quality insufficient for compliance verification",
        "Data collection process has gaps",
        "Data source integration is incomplete",
        "No data quality monitoring framework in place",
        "Data governance strategy does not prioritize compliance data",
    ],
    FishboneCategory.ENVIRONMENT: [
        "External conditions changed beyond current controls",
        "Monitoring of external conditions is insufficient",
        "Risk scanning scope does not cover emerging threats",
        "Environmental scanning frequency is inadequate",
        "No early warning system for external risk factors",
    ],
    FishboneCategory.SUPPLIERS: [
        "Supplier did not meet compliance requirements",
        "Supplier requirements not clearly communicated",
        "Supplier onboarding process lacks compliance checks",
        "Supplier management framework is incomplete",
        "No comprehensive supplier compliance program exists",
    ],
    FishboneCategory.MANAGEMENT: [
        "Management oversight did not identify the gap",
        "Reporting mechanisms do not surface compliance status",
        "Management reporting framework lacks compliance KPIs",
        "Compliance KPIs not aligned with regulatory requirements",
        "No integrated compliance management system deployed",
    ],
}


class RootCauseMapper:
    """Maps compliance gaps to root causes via 5-Whys and fishbone analysis.

    Performs structured root cause analysis for each compliance gap,
    building 5-Whys causal chains and organizing results in fishbone
    (Ishikawa) diagram format. Identifies systemic (cross-cutting) root
    causes that affect multiple gaps.

    Example:
        >>> engine = RootCauseMapper()
        >>> causes = await engine.analyze_root_causes(gaps, "PLAN-001")
        >>> assert all(rc.depth >= 1 for rc in causes)
    """

    def __init__(self, config: Optional[ImprovementPlanCreatorConfig] = None) -> None:
        """Initialize RootCauseMapper.

        Args:
            config: Optional configuration override.
        """
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._cause_store: Dict[str, List[RootCause]] = {}
        self._fishbone_store: Dict[str, FishboneAnalysis] = {}
        logger.info("RootCauseMapper initialized")

    async def analyze_root_causes(
        self,
        gaps: List[ComplianceGap],
        plan_id: str = "",
    ) -> List[RootCause]:
        """Perform root cause analysis for a set of compliance gaps.

        Args:
            gaps: Compliance gaps to analyze.
            plan_id: Parent plan identifier.

        Returns:
            List of identified root causes.
        """
        start = time.monotonic()
        all_causes: List[RootCause] = []

        for gap in gaps:
            causes = self._five_whys(gap)
            all_causes.extend(causes)

        # Identify systemic root causes (appearing in 2+ gaps)
        cause_descriptions: Dict[str, List[str]] = {}
        for cause in all_causes:
            desc_key = cause.description.lower().strip()
            cause_descriptions.setdefault(desc_key, []).append(cause.gap_id)

        for cause in all_causes:
            desc_key = cause.description.lower().strip()
            if len(cause_descriptions.get(desc_key, [])) >= 2:
                cause.systemic = True

        # Record metrics
        for cause in all_causes:
            m.record_root_cause_mapped(cause.category.value)

        systemic_count = sum(1 for c in all_causes if c.systemic)
        m.set_systemic_root_causes(systemic_count)

        # Store
        self._cause_store[plan_id] = all_causes

        # Provenance
        provenance_data = {
            "plan_id": plan_id,
            "root_causes": len(all_causes),
            "systemic": systemic_count,
        }
        provenance_hash = self._provenance.compute_hash(provenance_data)
        for cause in all_causes:
            cause.provenance_hash = provenance_hash

        self._provenance.record(
            "root_cause_analysis", "analyze", plan_id, AGENT_ID,
            metadata={"count": len(all_causes), "systemic": systemic_count},
        )

        elapsed = time.monotonic() - start
        m.observe_root_cause_mapping_duration(elapsed)

        logger.info(
            "Mapped %d root causes (%d systemic) for plan %s in %.1fms",
            len(all_causes), systemic_count, plan_id, elapsed * 1000,
        )
        return all_causes

    def _five_whys(self, gap: ComplianceGap) -> List[RootCause]:
        """Perform 5-Whys analysis for a single gap.

        Args:
            gap: Compliance gap to analyze.

        Returns:
            List of root causes at increasing depth.
        """
        start = time.monotonic()
        category = _DIMENSION_CATEGORY_MAP.get(
            gap.risk_dimension, FishboneCategory.PROCESS
        )
        template_chain = _WHYS_TEMPLATES.get(
            category, _WHYS_TEMPLATES[FishboneCategory.PROCESS]
        )

        max_depth = min(self.config.five_whys_max_depth, len(template_chain))
        causes: List[RootCause] = []
        chain: List[str] = []

        for depth in range(1, max_depth + 1):
            why_text = template_chain[depth - 1]
            chain.append(why_text)

            # Confidence increases with depth (deeper = more root)
            confidence = Decimal(str(0.3 + (depth * 0.12))).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            confidence = min(confidence, Decimal("1.0"))

            root_cause = RootCause(
                root_cause_id=f"RC-{uuid.uuid4().hex[:12]}",
                gap_id=gap.gap_id,
                category=category,
                description=why_text,
                analysis_chain=list(chain),
                depth=depth,
                confidence=confidence,
                contributing_factors=[gap.title],
            )
            causes.append(root_cause)

        elapsed = time.monotonic() - start
        m.observe_five_whys_duration(elapsed)
        return causes

    async def build_fishbone(
        self,
        gap: ComplianceGap,
        root_causes: Optional[List[RootCause]] = None,
    ) -> FishboneAnalysis:
        """Build a fishbone (Ishikawa) diagram analysis for a gap.

        Args:
            gap: Target compliance gap.
            root_causes: Optional pre-computed root causes.

        Returns:
            FishboneAnalysis with categorized root causes.
        """
        start = time.monotonic()

        if root_causes is None:
            root_causes = self._five_whys(gap)

        # Organize by category
        categories: Dict[str, List[RootCause]] = {}
        for cause in root_causes:
            cat_key = cause.category.value
            categories.setdefault(cat_key, [])
            # Enforce max causes per category
            if len(categories[cat_key]) < self.config.fishbone_max_causes_per_category:
                categories[cat_key].append(cause)

        # Determine primary root cause (deepest with highest confidence)
        primary_id: Optional[str] = None
        if root_causes:
            deepest = max(root_causes, key=lambda c: (c.depth, float(c.confidence)))
            primary_id = deepest.root_cause_id

        systemic_ids = [c.root_cause_id for c in root_causes if c.systemic]

        analysis_id = f"FB-{uuid.uuid4().hex[:12]}"

        provenance_data = {
            "analysis_id": analysis_id,
            "gap_id": gap.gap_id,
            "categories": len(categories),
            "causes": len(root_causes),
        }
        provenance_hash = self._provenance.compute_hash(provenance_data)

        result = FishboneAnalysis(
            analysis_id=analysis_id,
            gap_id=gap.gap_id,
            categories=categories,
            primary_root_cause_id=primary_id,
            systemic_causes=systemic_ids,
            provenance_hash=provenance_hash,
        )

        self._fishbone_store[analysis_id] = result

        elapsed = time.monotonic() - start
        m.observe_fishbone_analysis_duration(elapsed)

        logger.info(
            "Built fishbone for gap %s: %d categories, %d causes in %.1fms",
            gap.gap_id, len(categories), len(root_causes), elapsed * 1000,
        )
        return result

    async def get_root_causes(self, plan_id: str) -> List[RootCause]:
        """Retrieve stored root causes for a plan.

        Args:
            plan_id: Plan identifier.

        Returns:
            List of RootCause for the plan.
        """
        return self._cause_store.get(plan_id, [])

    async def get_fishbone(self, analysis_id: str) -> Optional[FishboneAnalysis]:
        """Retrieve a fishbone analysis by ID.

        Args:
            analysis_id: Analysis identifier.

        Returns:
            FishboneAnalysis or None.
        """
        return self._fishbone_store.get(analysis_id)

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "engine": "RootCauseMapper",
            "status": "healthy",
            "plans_analyzed": len(self._cause_store),
            "fishbones_built": len(self._fishbone_store),
        }
