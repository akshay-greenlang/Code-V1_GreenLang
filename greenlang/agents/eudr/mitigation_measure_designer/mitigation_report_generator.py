# -*- coding: utf-8 -*-
"""
Mitigation Report Generator Engine - AGENT-EUDR-029

Generates DDS-ready Article 11 compliance reports summarizing mitigation
strategy design, measure outcomes, effectiveness estimates, verification
results, regulatory references, and the complete provenance chain.

Reports are structured for inclusion in the EU Due Diligence Statement
(DDS) as required by EUDR Article 4(2) and Article 31 record-keeping
obligations.

Report Sections:
    1. Risk Trigger Summary   -- Source assessment, operator, commodity,
       composite score, risk level, dimension breakdown
    2. Strategy Overview      -- Strategy ID, target score, measure count,
       cumulative reduction, feasibility assessment
    3. Measures Detail        -- Per-measure: title, category, dimension,
       priority, status, expected/actual reduction, evidence
    4. Verification Results   -- Pre/post scores, absolute/percentage
       reduction, classification, recommendations
    5. Regulatory Mapping     -- EUDR article references for each measure
       category, compliance determination
    6. Provenance Chain       -- Complete SHA-256 hash chain for audit trail

Zero-Hallucination Guarantees:
    - All report data sourced from deterministic engine outputs
    - No LLM involvement in data aggregation or scoring
    - Hash chain verifiable by any third party
    - Regulatory mapping uses static EUDR article references

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-029 Mitigation Measure Designer (GL-EUDR-MMD-029)
Regulation: EU 2023/1115 (EUDR) Articles 4, 10, 11, 29, 31
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import MitigationMeasureDesignerConfig, get_config
from .models import (
    AGENT_ID,
    AGENT_VERSION,
    Article11Category,
    EUDRCommodity,
    MeasurePriority,
    MeasureStatus,
    MeasureSummary,
    MitigationMeasure,
    MitigationReport,
    MitigationStrategy,
    RiskDimension,
    RiskLevel,
    RiskTrigger,
    VerificationReport,
    VerificationResult,
)
from .provenance import GENESIS_HASH, ProvenanceTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EUDR Article references per measure category
# ---------------------------------------------------------------------------

_ARTICLE_REFERENCES: Dict[Article11Category, Dict[str, str]] = {
    Article11Category.ADDITIONAL_INFO: {
        "primary": "EUDR Article 10(1), Article 11(1)(a)",
        "description": (
            "Gathering additional information, including documents, "
            "data, or other evidence, regarding the products or their "
            "country of production."
        ),
        "record_keeping": "EUDR Article 31(1)(d)",
        "dds_requirement": "EUDR Article 4(2)",
    },
    Article11Category.INDEPENDENT_AUDIT: {
        "primary": "EUDR Article 10(1), Article 11(1)(b)",
        "description": (
            "Carrying out independent surveys and audits, including "
            "field audits, to verify compliance with EUDR requirements."
        ),
        "record_keeping": "EUDR Article 31(1)(e)",
        "dds_requirement": "EUDR Article 4(2)",
    },
    Article11Category.OTHER_MEASURES: {
        "primary": "EUDR Article 10(1), Article 11(1)(c)",
        "description": (
            "Any other risk mitigation measures that are adequate to "
            "reach the conclusion that the risk is negligible, including "
            "contractual clauses, capacity building, or supply chain "
            "restructuring."
        ),
        "record_keeping": "EUDR Article 31(1)(f)",
        "dds_requirement": "EUDR Article 4(2)",
    },
}

# ---------------------------------------------------------------------------
# Risk level display mapping
# ---------------------------------------------------------------------------

_RISK_LEVEL_LABELS: Dict[RiskLevel, str] = {
    RiskLevel.NEGLIGIBLE: "Negligible (no further action required)",
    RiskLevel.LOW: "Low (routine monitoring sufficient)",
    RiskLevel.STANDARD: "Standard (mitigation measures required)",
    RiskLevel.HIGH: "High (enhanced mitigation required)",
    RiskLevel.CRITICAL: "Critical (immediate action required)",
}

# ---------------------------------------------------------------------------
# Verification result display mapping
# ---------------------------------------------------------------------------

_VERIFICATION_LABELS: Dict[VerificationResult, str] = {
    VerificationResult.SUFFICIENT: (
        "SUFFICIENT - Risk reduced to acceptable level"
    ),
    VerificationResult.PARTIAL: (
        "PARTIAL - Risk reduced but remains above target"
    ),
    VerificationResult.INSUFFICIENT: (
        "INSUFFICIENT - No meaningful risk reduction achieved"
    ),
}


class MitigationReportGenerator:
    """Generates DDS-ready Article 11 compliance reports.

    Assembles structured reports from strategy design outputs,
    measure implementation status, effectiveness estimates,
    and verification results. Computes SHA-256 provenance hash
    over the entire report for regulatory audit trail.

    Attributes:
        _config: Agent configuration.
        _provenance: Provenance tracker for audit trail.

    Example:
        >>> generator = MitigationReportGenerator()
        >>> report = generator.generate_report(
        ...     strategy=strategy,
        ...     risk_trigger=trigger,
        ...     verification=verification,
        ... )
        >>> assert report.report_id.startswith("rpt-")
        >>> assert report.provenance_hash != ""
    """

    def __init__(
        self,
        config: Optional[MitigationMeasureDesignerConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize MitigationReportGenerator.

        Args:
            config: Agent configuration. Uses get_config() if None.
            provenance: Provenance tracker instance.
        """
        self._config = config or get_config()
        self._provenance = provenance or ProvenanceTracker()
        logger.info(
            "MitigationReportGenerator initialized: "
            "format=%s, include_evidence=%s, include_provenance=%s",
            self._config.report_format,
            self._config.include_evidence_summary,
            self._config.include_provenance,
        )

    def generate_report(
        self,
        strategy: MitigationStrategy,
        risk_trigger: RiskTrigger,
        verification: Optional[VerificationReport] = None,
    ) -> MitigationReport:
        """Generate a complete mitigation compliance report.

        Assembles all sections into a MitigationReport model and
        computes the overall provenance hash for audit integrity.

        Args:
            strategy: The mitigation strategy being reported on.
            risk_trigger: Original risk trigger from EUDR-028.
            verification: Optional verification report (None if
                verification has not been performed yet).

        Returns:
            MitigationReport with all sections populated.
        """
        start_time = time.monotonic()
        report_id = f"rpt-{uuid.uuid4().hex[:12]}"
        logger.info(
            "Generating report: id=%s, strategy=%s, operator=%s",
            report_id,
            strategy.strategy_id,
            risk_trigger.operator_id,
        )

        # Build measures summary
        measures_summary = self._build_measures_summary(strategy.measures)

        # Determine post-mitigation score
        post_score = self._determine_post_score(strategy, verification)

        # Determine verification result
        verification_result = (
            verification.result if verification else None
        )

        # Compute provenance hash over all report data
        provenance_hash = self._compute_report_hash(
            report_id=report_id,
            strategy=strategy,
            risk_trigger=risk_trigger,
            verification=verification,
            measures_summary=measures_summary,
        )

        report = MitigationReport(
            report_id=report_id,
            strategy_id=strategy.strategy_id,
            operator_id=risk_trigger.operator_id,
            commodity=risk_trigger.commodity,
            pre_score=risk_trigger.composite_score,
            post_score=post_score,
            measures_summary=measures_summary,
            verification_result=verification_result,
            provenance_hash=provenance_hash,
        )

        # Record provenance entry
        self._provenance.create_entry(
            step="generate_report",
            source="mitigation_report_generator",
            input_hash=self._provenance.compute_hash(
                {"strategy_id": strategy.strategy_id}
            ),
            output_hash=provenance_hash,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Report generated: id=%s, measures=%d, "
            "pre=%s, post=%s, verification=%s, elapsed=%.1fms",
            report_id,
            len(measures_summary),
            risk_trigger.composite_score,
            post_score,
            verification_result.value if verification_result else "pending",
            elapsed_ms,
        )

        return report

    def format_for_dds(
        self,
        strategy: MitigationStrategy,
        risk_trigger: RiskTrigger,
        verification: Optional[VerificationReport] = None,
    ) -> Dict[str, Any]:
        """Generate a DDS-ready structured report dictionary.

        Produces a comprehensive dictionary suitable for inclusion
        in the EU Due Diligence Statement, including all required
        sections: risk trigger, strategy, measures, verification,
        regulatory references, and provenance chain.

        Args:
            strategy: The mitigation strategy being reported on.
            risk_trigger: Original risk trigger from EUDR-028.
            verification: Optional verification report.

        Returns:
            Dictionary with all DDS sections.
        """
        start_time = time.monotonic()
        logger.info(
            "Formatting DDS report: strategy=%s, operator=%s",
            strategy.strategy_id,
            risk_trigger.operator_id,
        )

        # Build each section
        risk_trigger_section = self._build_risk_trigger_section(
            risk_trigger
        )
        strategy_section = self._build_strategy_section(strategy)
        measures_section = self._build_measures_section(strategy.measures)
        verification_section = self._build_verification_section(
            strategy=strategy,
            risk_trigger=risk_trigger,
            verification=verification,
        )
        regulatory_section = self._build_regulatory_section(
            strategy.measures
        )
        recommendations = self._generate_recommendations(
            strategy=strategy,
            risk_trigger=risk_trigger,
            verification=verification,
        )

        # Provenance chain
        provenance_section = self._build_provenance_section(
            strategy=strategy,
            risk_trigger=risk_trigger,
            verification=verification,
        )

        # Compute overall report hash
        report_hash = self._compute_report_hash(
            report_id=f"dds-{uuid.uuid4().hex[:12]}",
            strategy=strategy,
            risk_trigger=risk_trigger,
            verification=verification,
            measures_summary=self._build_measures_summary(
                strategy.measures
            ),
        )

        dds_report: Dict[str, Any] = {
            "report_metadata": {
                "agent_id": AGENT_ID,
                "agent_version": AGENT_VERSION,
                "report_type": "EUDR Article 11 Mitigation Report",
                "generated_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "format": self._config.report_format,
                "provenance_hash": report_hash,
            },
            "risk_trigger": risk_trigger_section,
            "strategy": strategy_section,
            "measures": measures_section,
            "verification": verification_section,
            "regulatory_compliance": regulatory_section,
            "recommendations": recommendations,
        }

        if self._config.include_provenance:
            dds_report["provenance_chain"] = provenance_section

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "DDS report formatted: strategy=%s, elapsed=%.1fms",
            strategy.strategy_id,
            elapsed_ms,
        )

        return dds_report

    def _build_risk_trigger_section(
        self, risk_trigger: RiskTrigger,
    ) -> Dict[str, Any]:
        """Build the risk trigger section of the report.

        Args:
            risk_trigger: Source risk trigger from EUDR-028.

        Returns:
            Dictionary with risk trigger details.
        """
        dimension_breakdown: Dict[str, str] = {}
        for dim, score in risk_trigger.risk_dimensions.items():
            dimension_breakdown[dim.value] = str(score)

        return {
            "assessment_id": risk_trigger.assessment_id,
            "operator_id": risk_trigger.operator_id,
            "commodity": risk_trigger.commodity.value,
            "composite_score": str(risk_trigger.composite_score),
            "risk_level": risk_trigger.risk_level.value,
            "risk_level_description": _RISK_LEVEL_LABELS.get(
                risk_trigger.risk_level,
                risk_trigger.risk_level.value,
            ),
            "triggered_at": risk_trigger.triggered_at.isoformat(),
            "dimension_breakdown": dimension_breakdown,
            "elevated_dimensions": [
                dim.value
                for dim, score in risk_trigger.risk_dimensions.items()
                if score > self._config.low_max
            ],
        }

    def _build_strategy_section(
        self, strategy: MitigationStrategy,
    ) -> Dict[str, Any]:
        """Build the strategy overview section of the report.

        Args:
            strategy: The mitigation strategy.

        Returns:
            Dictionary with strategy overview details.
        """
        # Count measures by status
        status_breakdown: Dict[str, int] = {}
        for measure in strategy.measures:
            status_name = measure.status.value
            status_breakdown[status_name] = (
                status_breakdown.get(status_name, 0) + 1
            )

        # Count measures by category
        category_breakdown: Dict[str, int] = {}
        for measure in strategy.measures:
            cat = measure.article11_category.value
            category_breakdown[cat] = (
                category_breakdown.get(cat, 0) + 1
            )

        # Calculate cumulative expected reduction
        cumulative_reduction = self._calculate_cumulative_reduction(
            strategy.measures
        )

        return {
            "strategy_id": strategy.strategy_id,
            "workflow_id": strategy.workflow_id,
            "pre_mitigation_score": str(strategy.pre_mitigation_score),
            "target_score": str(strategy.target_score),
            "post_mitigation_score": (
                str(strategy.post_mitigation_score)
                if strategy.post_mitigation_score is not None
                else "pending"
            ),
            "total_measures": len(strategy.measures),
            "cumulative_expected_reduction": str(cumulative_reduction),
            "status": strategy.status.value if hasattr(
                strategy.status, "value"
            ) else str(strategy.status),
            "designed_at": strategy.designed_at.isoformat(),
            "designed_by": strategy.designed_by,
            "status_breakdown": status_breakdown,
            "category_breakdown": category_breakdown,
            "provenance_hash": strategy.provenance_hash,
        }

    def _build_measures_section(
        self, measures: List[MitigationMeasure],
    ) -> List[Dict[str, Any]]:
        """Build the per-measure detail section of the report.

        Args:
            measures: List of mitigation measures.

        Returns:
            List of measure detail dictionaries.
        """
        result: List[Dict[str, Any]] = []
        for measure in measures:
            measure_dict: Dict[str, Any] = {
                "measure_id": measure.measure_id,
                "title": measure.title,
                "description": measure.description,
                "article11_category": measure.article11_category.value,
                "target_dimension": measure.target_dimension.value,
                "status": measure.status.value,
                "priority": measure.priority.value,
                "expected_risk_reduction": str(
                    measure.expected_risk_reduction
                ),
                "actual_risk_reduction": (
                    str(measure.actual_risk_reduction)
                    if measure.actual_risk_reduction is not None
                    else None
                ),
                "template_id": measure.template_id,
                "assigned_to": measure.assigned_to,
                "deadline": (
                    measure.deadline.isoformat()
                    if measure.deadline
                    else None
                ),
                "started_at": (
                    measure.started_at.isoformat()
                    if measure.started_at
                    else None
                ),
                "completed_at": (
                    measure.completed_at.isoformat()
                    if measure.completed_at
                    else None
                ),
                "regulatory_reference": _ARTICLE_REFERENCES.get(
                    measure.article11_category, {}
                ).get("primary", "EUDR Article 11"),
            }

            # Include evidence summary if configured
            if self._config.include_evidence_summary:
                measure_dict["evidence_count"] = len(
                    measure.evidence_ids
                )
                measure_dict["evidence_ids"] = measure.evidence_ids

            result.append(measure_dict)

        return result

    def _build_verification_section(
        self,
        strategy: MitigationStrategy,
        risk_trigger: RiskTrigger,
        verification: Optional[VerificationReport] = None,
    ) -> Dict[str, Any]:
        """Build the verification results section of the report.

        Args:
            strategy: The mitigation strategy.
            risk_trigger: Original risk trigger.
            verification: Optional verification report.

        Returns:
            Dictionary with verification details.
        """
        if verification is None:
            return {
                "status": "pending",
                "message": (
                    "Verification has not been performed yet. "
                    "Measures are still in implementation phase."
                ),
                "pre_score": str(risk_trigger.composite_score),
                "target_score": str(strategy.target_score),
            }

        pre = verification.pre_score
        post = verification.post_score
        absolute_reduction = pre - post
        percentage_reduction = (
            ((pre - post) / pre * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            if pre > Decimal("0")
            else Decimal("0")
        )

        return {
            "status": "completed",
            "verification_id": verification.verification_id,
            "pre_score": str(pre),
            "post_score": str(post),
            "target_score": str(strategy.target_score),
            "absolute_reduction": str(absolute_reduction),
            "percentage_reduction": str(percentage_reduction),
            "result": verification.result.value,
            "result_description": _VERIFICATION_LABELS.get(
                verification.result,
                verification.result.value,
            ),
            "verified_at": verification.verified_at.isoformat(),
            "verified_by": verification.verified_by,
            "target_achieved": (
                verification.result == VerificationResult.SUFFICIENT
            ),
            "provenance_hash": verification.provenance_hash,
        }

    def _build_regulatory_section(
        self, measures: List[MitigationMeasure],
    ) -> Dict[str, Any]:
        """Build the regulatory mapping section.

        Maps each measure category to the relevant EUDR article
        references and provides an overall compliance determination.

        Args:
            measures: List of mitigation measures.

        Returns:
            Dictionary with regulatory references and compliance status.
        """
        # Collect unique categories used
        categories_used: Dict[str, Dict[str, str]] = {}
        for measure in measures:
            cat = measure.article11_category
            if cat.value not in categories_used:
                ref = _ARTICLE_REFERENCES.get(cat, {})
                categories_used[cat.value] = {
                    "category": cat.value,
                    "primary_reference": ref.get(
                        "primary", "EUDR Article 11"
                    ),
                    "description": ref.get("description", ""),
                    "record_keeping": ref.get(
                        "record_keeping", "EUDR Article 31"
                    ),
                    "dds_requirement": ref.get(
                        "dds_requirement", "EUDR Article 4(2)"
                    ),
                }

        # Count measures per category
        category_counts: Dict[str, int] = {}
        for measure in measures:
            cat_val = measure.article11_category.value
            category_counts[cat_val] = (
                category_counts.get(cat_val, 0) + 1
            )

        # Compliance determination
        has_all_categories = len(categories_used) > 0
        all_completed = all(
            m.status in (
                MeasureStatus.COMPLETED,
                MeasureStatus.VERIFIED,
                MeasureStatus.CLOSED,
            )
            for m in measures
        ) if measures else False

        return {
            "regulation": "EU 2023/1115 (EUDR)",
            "primary_articles": [
                "Article 10 (Risk Assessment)",
                "Article 11 (Risk Mitigation)",
                "Article 4(2) (Due Diligence Statement)",
                "Article 31 (Record-keeping)",
            ],
            "article11_categories_applied": categories_used,
            "category_measure_counts": category_counts,
            "compliance_determination": {
                "measures_designed": len(measures) > 0,
                "all_categories_covered": has_all_categories,
                "all_measures_completed": all_completed,
                "ready_for_dds": (
                    has_all_categories and all_completed
                ),
            },
            "enforcement_dates": {
                "large_operators": "2025-12-30",
                "sme_operators": "2026-06-30",
            },
        }

    def _generate_recommendations(
        self,
        strategy: MitigationStrategy,
        risk_trigger: RiskTrigger,
        verification: Optional[VerificationReport] = None,
    ) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on report data.

        Provides prioritized recommendations based on the current
        state of the mitigation strategy and verification outcome.

        Args:
            strategy: The mitigation strategy.
            risk_trigger: Original risk trigger.
            verification: Optional verification report.

        Returns:
            List of recommendation dictionaries with priority
            and description.
        """
        recommendations: List[Dict[str, str]] = []

        # Recommendation 1: Based on verification result
        if verification is not None:
            if verification.result == VerificationResult.SUFFICIENT:
                recommendations.append({
                    "priority": "low",
                    "category": "monitoring",
                    "recommendation": (
                        "Risk reduced to acceptable level. Continue "
                        "routine monitoring per EUDR Article 8(3). "
                        "Update Due Diligence Statement to reflect "
                        "mitigation outcomes."
                    ),
                    "reference": "EUDR Article 8(3), Article 4(2)",
                })
            elif verification.result == VerificationResult.PARTIAL:
                gap = verification.post_score - strategy.target_score
                recommendations.append({
                    "priority": "high",
                    "category": "additional_mitigation",
                    "recommendation": (
                        f"Risk reduced but remains {gap} points "
                        f"above target ({strategy.target_score}). "
                        f"Design and implement additional mitigation "
                        f"measures per EUDR Article 11."
                    ),
                    "reference": "EUDR Article 10(2), Article 11",
                })
                recommendations.append({
                    "priority": "medium",
                    "category": "escalation",
                    "recommendation": (
                        "Consider escalating to enhanced due "
                        "diligence level. Engage independent "
                        "auditors for field verification."
                    ),
                    "reference": "EUDR Article 11(1)(b)",
                })
            elif verification.result == VerificationResult.INSUFFICIENT:
                recommendations.append({
                    "priority": "critical",
                    "category": "strategy_redesign",
                    "recommendation": (
                        "No meaningful risk reduction achieved. "
                        "Immediate strategy review and redesign "
                        "required. Consider sourcing suspension "
                        "from affected supplier."
                    ),
                    "reference": "EUDR Article 11(1), Article 10(2)",
                })
                recommendations.append({
                    "priority": "critical",
                    "category": "management_escalation",
                    "recommendation": (
                        "Escalate to management for decision on "
                        "continued sourcing relationship per EUDR "
                        "Article 11(1). Document decision rationale "
                        "for regulatory record-keeping."
                    ),
                    "reference": "EUDR Article 11(1), Article 31",
                })
        else:
            recommendations.append({
                "priority": "medium",
                "category": "verification",
                "recommendation": (
                    "Verification has not been performed. Schedule "
                    "post-implementation risk reassessment to confirm "
                    "mitigation effectiveness."
                ),
                "reference": "EUDR Article 10, Article 11",
            })

        # Recommendation 2: Check for incomplete measures
        incomplete = [
            m for m in strategy.measures
            if m.status in (
                MeasureStatus.PROPOSED,
                MeasureStatus.APPROVED,
                MeasureStatus.IN_PROGRESS,
            )
        ]
        if incomplete:
            recommendations.append({
                "priority": "high",
                "category": "implementation",
                "recommendation": (
                    f"{len(incomplete)} measure(s) remain incomplete. "
                    f"Ensure all measures are implemented and "
                    f"evidence is collected before finalizing the "
                    f"Due Diligence Statement."
                ),
                "reference": "EUDR Article 11, Article 4(2)",
            })

        # Recommendation 3: Check for measures without evidence
        if self._config.evidence_required:
            no_evidence = [
                m for m in strategy.measures
                if m.status in (
                    MeasureStatus.COMPLETED,
                    MeasureStatus.VERIFIED,
                    MeasureStatus.CLOSED,
                ) and not m.evidence_ids
            ]
            if no_evidence:
                recommendations.append({
                    "priority": "high",
                    "category": "evidence",
                    "recommendation": (
                        f"{len(no_evidence)} completed measure(s) "
                        f"lack supporting evidence. Collect and "
                        f"attach evidence per EUDR Article 31 "
                        f"record-keeping requirements."
                    ),
                    "reference": "EUDR Article 31(1)",
                })

        # Recommendation 4: Dimension-specific
        for dim, score in risk_trigger.risk_dimensions.items():
            if score >= self._config.high_max:
                recommendations.append({
                    "priority": "high",
                    "category": "dimension_alert",
                    "recommendation": (
                        f"Risk dimension '{dim.value}' remains at "
                        f"critical level ({score}). Prioritize "
                        f"targeted mitigation for this dimension."
                    ),
                    "reference": "EUDR Article 10(2), Article 11",
                })

        return recommendations

    def _build_provenance_section(
        self,
        strategy: MitigationStrategy,
        risk_trigger: RiskTrigger,
        verification: Optional[VerificationReport] = None,
    ) -> Dict[str, Any]:
        """Build the provenance chain section for audit trail.

        Constructs or retrieves the provenance hash chain covering
        the full lifecycle from risk trigger through report generation.

        Args:
            strategy: The mitigation strategy.
            risk_trigger: Original risk trigger.
            verification: Optional verification report.

        Returns:
            Dictionary with provenance chain details.
        """
        # Build chain from steps
        steps = [
            {
                "step": "risk_trigger",
                "source": "AGENT-EUDR-028",
                "data": {
                    "assessment_id": risk_trigger.assessment_id,
                    "composite_score": str(
                        risk_trigger.composite_score
                    ),
                },
            },
            {
                "step": "strategy_design",
                "source": "AGENT-EUDR-029",
                "data": {
                    "strategy_id": strategy.strategy_id,
                    "measure_count": len(strategy.measures),
                },
            },
        ]

        if verification is not None:
            steps.append({
                "step": "verification",
                "source": "AGENT-EUDR-029",
                "data": {
                    "verification_id": verification.verification_id,
                    "result": verification.result.value,
                },
            })

        steps.append({
            "step": "report_generation",
            "source": "AGENT-EUDR-029",
            "data": {
                "strategy_id": strategy.strategy_id,
                "operator_id": risk_trigger.operator_id,
            },
        })

        chain = self._provenance.build_chain(
            steps=steps,
            genesis_hash=GENESIS_HASH,
        )

        # Verify chain integrity
        is_valid = self._provenance.verify_chain(chain)

        return {
            "algorithm": "sha256",
            "chain_length": len(chain),
            "chain_valid": is_valid,
            "genesis_hash": GENESIS_HASH,
            "entries": chain,
            "strategy_hash": strategy.provenance_hash,
            "verification_hash": (
                verification.provenance_hash
                if verification
                else None
            ),
        }

    def _build_measures_summary(
        self, measures: List[MitigationMeasure],
    ) -> List[MeasureSummary]:
        """Build MeasureSummary list for the MitigationReport model.

        Args:
            measures: List of mitigation measures.

        Returns:
            List of MeasureSummary instances.
        """
        summaries: List[MeasureSummary] = []
        for m in measures:
            summary = MeasureSummary(
                measure_id=m.measure_id,
                title=m.title,
                article11_category=m.article11_category,
                target_dimension=m.target_dimension,
                status=m.status,
                priority=m.priority,
                expected_risk_reduction=m.expected_risk_reduction,
                actual_risk_reduction=m.actual_risk_reduction,
            )
            summaries.append(summary)
        return summaries

    def _determine_post_score(
        self,
        strategy: MitigationStrategy,
        verification: Optional[VerificationReport] = None,
    ) -> Decimal:
        """Determine the post-mitigation score for the report.

        Uses verification post_score if available, otherwise falls
        back to strategy.post_mitigation_score, and finally to an
        estimated score based on expected reductions.

        Args:
            strategy: The mitigation strategy.
            verification: Optional verification report.

        Returns:
            Post-mitigation risk score as Decimal.
        """
        if verification is not None:
            return verification.post_score

        if strategy.post_mitigation_score is not None:
            return strategy.post_mitigation_score

        # Estimate based on expected cumulative reduction
        cumulative = self._calculate_cumulative_reduction(
            strategy.measures
        )
        reduction_abs = (
            strategy.pre_mitigation_score
            * cumulative
            / Decimal("100")
        )
        estimated = strategy.pre_mitigation_score - reduction_abs
        estimated = max(estimated, Decimal("0"))
        return estimated.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    def _calculate_cumulative_reduction(
        self, measures: List[MitigationMeasure],
    ) -> Decimal:
        """Calculate cumulative expected reduction using diminishing returns.

        Uses the formula: Total = 1 - Product(1 - Ri) for each
        measure i, where Ri = expected_risk_reduction / 100.

        Args:
            measures: List of mitigation measures.

        Returns:
            Cumulative reduction percentage (0-100).
        """
        if not measures:
            return Decimal("0")

        product = Decimal("1")
        for m in measures:
            ri = m.expected_risk_reduction / Decimal("100")
            ri = min(ri, Decimal("1"))
            product *= (Decimal("1") - ri)

        total = (Decimal("1") - product) * Decimal("100")
        cap = self._config.max_effectiveness_cap
        total = min(total, cap)
        return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _compute_report_hash(
        self,
        report_id: str,
        strategy: MitigationStrategy,
        risk_trigger: RiskTrigger,
        verification: Optional[VerificationReport],
        measures_summary: List[MeasureSummary],
    ) -> str:
        """Compute SHA-256 hash over all report data for provenance.

        Args:
            report_id: Report identifier.
            strategy: The mitigation strategy.
            risk_trigger: Original risk trigger.
            verification: Optional verification report.
            measures_summary: List of measure summaries.

        Returns:
            64-character hex SHA-256 hash string.
        """
        hash_data: Dict[str, Any] = {
            "report_id": report_id,
            "strategy_id": strategy.strategy_id,
            "operator_id": risk_trigger.operator_id,
            "commodity": risk_trigger.commodity.value,
            "pre_score": str(risk_trigger.composite_score),
            "target_score": str(strategy.target_score),
            "measure_count": len(measures_summary),
            "measure_ids": [ms.measure_id for ms in measures_summary],
            "strategy_hash": strategy.provenance_hash,
        }

        if verification is not None:
            hash_data["verification_id"] = verification.verification_id
            hash_data["verification_result"] = verification.result.value
            hash_data["post_score"] = str(verification.post_score)
            hash_data["verification_hash"] = (
                verification.provenance_hash
            )

        return self._provenance.compute_hash(hash_data)

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status.

        Returns:
            Dictionary with engine status and configuration.
        """
        return {
            "engine": "MitigationReportGenerator",
            "status": "available",
            "config": {
                "report_format": self._config.report_format,
                "include_evidence_summary": (
                    self._config.include_evidence_summary
                ),
                "include_provenance": (
                    self._config.include_provenance
                ),
            },
        }
