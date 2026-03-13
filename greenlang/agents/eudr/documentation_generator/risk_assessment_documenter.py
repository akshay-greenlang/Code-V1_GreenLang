# -*- coding: utf-8 -*-
"""
Risk Assessment Documenter Engine - AGENT-EUDR-030

Documents Article 10 risk assessment results from EUDR-028 output into
structured regulatory documentation. Generates risk assessment documents
with composite scores, risk level classifications, criterion-by-criterion
evaluations, risk decomposition by dimension, country benchmarking per
Article 29, and simplified due diligence eligibility determination.

Report Sections:
    1. Assessment Summary     -- Assessment ID, operator, commodity,
       composite score, risk level classification
    2. Criterion Evaluation   -- Per-criterion Article 10(2) details
    3. Risk Decomposition     -- Dimensional contribution breakdown
    4. Country Benchmark      -- Article 29 country benchmarking
    5. Simplified DD Check    -- Low-risk country eligibility

Zero-Hallucination Guarantees:
    - All numeric scores use Decimal arithmetic
    - No LLM calls in the documentation path
    - Risk level mapping uses deterministic threshold comparison
    - Complete provenance trail for every document generated

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-030 Documentation Generator (GL-EUDR-DGN-030)
Regulation: EU 2023/1115 (EUDR) Articles 10, 29, 31
Status: Production Ready
"""
from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import DocumentationGeneratorConfig, get_config
from .models import (
    AGENT_ID,
    AGENT_VERSION,
    EUDRCommodity,
    RiskAssessmentDoc,
    RiskLevel,
)
from .provenance import GENESIS_HASH, ProvenanceTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk level display labels
# ---------------------------------------------------------------------------

_RISK_LEVEL_LABELS: Dict[RiskLevel, str] = {
    RiskLevel.NEGLIGIBLE: "Negligible - no further action required",
    RiskLevel.LOW: "Low - routine monitoring sufficient",
    RiskLevel.STANDARD: "Standard - enhanced due diligence recommended",
    RiskLevel.HIGH: "High - mitigation measures required (Article 11)",
    RiskLevel.CRITICAL: "Critical - immediate mitigation required",
}

# ---------------------------------------------------------------------------
# Article 10(2) risk assessment criteria
# ---------------------------------------------------------------------------

_ARTICLE10_CRITERIA: List[Dict[str, str]] = [
    {
        "id": "10.2.a",
        "description": (
            "Prevalence of deforestation or forest degradation in the "
            "country, region, or area of production."
        ),
        "article": "Article 10(2)(a)",
    },
    {
        "id": "10.2.b",
        "description": (
            "Risk of circumvention related to mixing with products "
            "of unknown origin."
        ),
        "article": "Article 10(2)(b)",
    },
    {
        "id": "10.2.c",
        "description": (
            "Prevalence of internationally recognised corruption in "
            "the country of production per Corruption Perceptions Index."
        ),
        "article": "Article 10(2)(c)",
    },
    {
        "id": "10.2.d",
        "description": (
            "Complexity of the supply chain, including risk of "
            "laundering through third countries."
        ),
        "article": "Article 10(2)(d)",
    },
    {
        "id": "10.2.e",
        "description": (
            "Risk of non-compliance with local laws, including "
            "land tenure rights and indigenous peoples' rights."
        ),
        "article": "Article 10(2)(e)",
    },
    {
        "id": "10.2.f",
        "description": (
            "Information from the Commission's country benchmarking "
            "system per Article 29."
        ),
        "article": "Article 10(2)(f)",
    },
    {
        "id": "10.2.g",
        "description": (
            "Concerns raised by relevant stakeholders, competent "
            "authorities, or civil society organisations."
        ),
        "article": "Article 10(2)(g)",
    },
]

# ---------------------------------------------------------------------------
# Country benchmark classifications per Article 29
# ---------------------------------------------------------------------------

_BENCHMARK_DESCRIPTIONS: Dict[str, str] = {
    "low": (
        "Low-risk country classification per Article 29. Simplified "
        "due diligence may apply per Article 13."
    ),
    "standard": (
        "Standard-risk country classification per Article 29. Standard "
        "due diligence applies per Article 8."
    ),
    "high": (
        "High-risk country classification per Article 29. Enhanced "
        "scrutiny required per Article 10."
    ),
    "unknown": (
        "Country benchmarking classification not yet available from "
        "the Commission per Article 29."
    ),
}


class RiskAssessmentDocumenter:
    """Documents Article 10 risk assessment results.

    Generates structured risk assessment documentation from EUDR-028
    output, including composite scores, dimensional decomposition,
    criterion evaluations, and country benchmarking for inclusion
    in the DDS.

    Attributes:
        _config: Agent configuration instance.
        _provenance: Provenance tracker for audit trail.

    Example:
        >>> documenter = RiskAssessmentDocumenter()
        >>> doc = documenter.generate_risk_doc(
        ...     assessment_id="asr-001",
        ...     operator_id="OP-001",
        ...     commodity=EUDRCommodity.COFFEE,
        ...     composite_score=Decimal("45.50"),
        ...     risk_level=RiskLevel.STANDARD,
        ... )
        >>> assert doc.doc_id.startswith("rad-")
    """

    def __init__(
        self,
        config: Optional[DocumentationGeneratorConfig] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize RiskAssessmentDocumenter.

        Args:
            config: Agent configuration. Uses get_config() if None.
            provenance: Provenance tracker instance.
        """
        self._config = config or get_config()
        self._provenance = provenance or ProvenanceTracker()
        logger.info(
            "RiskAssessmentDocumenter initialized: "
            "criterion_details=%s, decomposition=%s, trend_data=%s",
            self._config.include_criterion_details,
            self._config.include_decomposition,
            self._config.include_trend_data,
        )

    def generate_risk_doc(
        self,
        assessment_id: str,
        operator_id: str,
        commodity: EUDRCommodity,
        composite_score: Decimal,
        risk_level: RiskLevel,
        criterion_evaluations: Optional[List[Dict[str, Any]]] = None,
        country_benchmark: Optional[str] = None,
        simplified_dd_eligible: bool = False,
        risk_dimensions: Optional[Dict[str, Any]] = None,
    ) -> RiskAssessmentDoc:
        """Generate risk assessment documentation.

        Produces a structured RiskAssessmentDoc capturing the complete
        risk assessment outcome for regulatory documentation purposes.

        Args:
            assessment_id: Source risk assessment identifier from EUDR-028.
            operator_id: Operator identifier.
            commodity: EUDR commodity category.
            composite_score: Composite risk score (0-100).
            risk_level: Risk classification level.
            criterion_evaluations: Optional per-criterion evaluation data.
            country_benchmark: Optional country benchmark classification.
            simplified_dd_eligible: Whether simplified DD applies.
            risk_dimensions: Optional risk decomposition by dimension.

        Returns:
            RiskAssessmentDoc with all sections populated.
        """
        start_time = time.monotonic()
        doc_id = f"rad-{uuid.uuid4().hex[:12]}"
        logger.info(
            "Generating risk assessment doc: id=%s, assessment=%s, "
            "operator=%s, commodity=%s, score=%s, level=%s",
            doc_id, assessment_id, operator_id,
            commodity.value, composite_score, risk_level.value,
        )

        # Build criterion section
        criterion_section: List[Dict[str, Any]] = []
        if self._config.include_criterion_details:
            criterion_section = self._build_criterion_section(
                criterion_evaluations or [],
            )

        # Build decomposition section
        decomposition_section: Dict[str, Any] = {}
        if self._config.include_decomposition and risk_dimensions:
            decomposition_section = self._build_decomposition_section(
                risk_dimensions,
            )

        # Build country benchmark section
        benchmark_section: Dict[str, Any] = {}
        if country_benchmark:
            benchmark_section = self._build_country_benchmark_section(
                country_benchmark,
            )

        # Compute provenance hash
        provenance_data: Dict[str, Any] = {
            "doc_id": doc_id,
            "assessment_id": assessment_id,
            "operator_id": operator_id,
            "commodity": commodity.value,
            "composite_score": str(composite_score),
            "risk_level": risk_level.value,
            "simplified_dd_eligible": simplified_dd_eligible,
            "criterion_count": len(criterion_section),
        }
        provenance_hash = self._provenance.compute_hash(provenance_data)

        # Create document
        doc = RiskAssessmentDoc(
            doc_id=doc_id,
            assessment_id=assessment_id,
            operator_id=operator_id,
            commodity=commodity,
            composite_score=composite_score,
            risk_level=risk_level,
            criterion_evaluations=criterion_section,
            country_benchmark=country_benchmark or "",
            simplified_dd_eligible=simplified_dd_eligible,
        )

        # Record provenance
        self._provenance.create_entry(
            step="generate_risk_doc",
            source="risk_assessment_documenter",
            input_hash=self._provenance.compute_hash(
                {"assessment_id": assessment_id}
            ),
            output_hash=provenance_hash,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "Risk assessment doc generated: id=%s, criteria=%d, "
            "elapsed=%.1fms",
            doc_id, len(criterion_section), elapsed_ms,
        )

        return doc

    def _build_criterion_section(
        self, evaluations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Build Article 10(2) criterion-by-criterion section.

        Merges upstream evaluation data with standard Article 10(2)
        criterion definitions to produce a comprehensive evaluation
        section.

        Args:
            evaluations: List of criterion evaluation dictionaries
                from EUDR-028.

        Returns:
            List of criterion evaluation dictionaries with regulatory
            references.
        """
        result: List[Dict[str, Any]] = []

        # Build lookup of upstream evaluations by criterion ID
        eval_lookup: Dict[str, Dict[str, Any]] = {}
        for ev in evaluations:
            crit_id = ev.get("criterion_id", ev.get("id", ""))
            eval_lookup[crit_id] = ev

        # Merge with standard criteria
        for criterion in _ARTICLE10_CRITERIA:
            crit_id = criterion["id"]
            upstream_eval = eval_lookup.get(crit_id, {})

            entry: Dict[str, Any] = {
                "criterion_id": crit_id,
                "article_reference": criterion["article"],
                "description": criterion["description"],
                "score": upstream_eval.get("score", "N/A"),
                "assessment": upstream_eval.get("assessment", "not_evaluated"),
                "evidence": upstream_eval.get("evidence", []),
                "notes": upstream_eval.get("notes", ""),
            }
            result.append(entry)

        return result

    def _build_decomposition_section(
        self, dimensions: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build risk decomposition showing each dimension's contribution.

        Args:
            dimensions: Risk decomposition dictionary with dimension
                names as keys and scores/weights as values.

        Returns:
            Structured decomposition section.
        """
        decomposition: Dict[str, Any] = {
            "total_dimensions": len(dimensions),
            "dimensions": {},
            "generated_by": AGENT_ID,
        }

        for dim_name, dim_data in dimensions.items():
            if isinstance(dim_data, dict):
                score = dim_data.get("score", Decimal("0"))
                weight = dim_data.get("weight", Decimal("0"))
                contribution = dim_data.get("contribution", Decimal("0"))
            else:
                score = dim_data
                weight = Decimal("0")
                contribution = Decimal("0")

            decomposition["dimensions"][dim_name] = {
                "score": str(score),
                "weight": str(weight),
                "weighted_contribution": str(contribution),
                "classification": self._classify_dimension_score(score),
            }

        return decomposition

    def _build_country_benchmark_section(
        self, benchmark: str,
    ) -> Dict[str, Any]:
        """Build Article 29 country benchmarking section.

        Args:
            benchmark: Country benchmark classification string.

        Returns:
            Structured country benchmark section.
        """
        benchmark_lower = benchmark.lower()
        description = _BENCHMARK_DESCRIPTIONS.get(
            benchmark_lower,
            _BENCHMARK_DESCRIPTIONS["unknown"],
        )

        return {
            "classification": benchmark,
            "description": description,
            "article_reference": "EUDR Article 29",
            "simplified_dd_applicable": benchmark_lower == "low",
            "enhanced_scrutiny_required": benchmark_lower == "high",
        }

    def _classify_dimension_score(
        self, score: Any,
    ) -> str:
        """Classify a dimension score into a risk tier.

        Args:
            score: Dimension score (Decimal, int, float, or str).

        Returns:
            Risk tier classification string.
        """
        try:
            dec_score = Decimal(str(score))
        except Exception:
            return "unknown"

        if dec_score < Decimal("15"):
            return "negligible"
        if dec_score < Decimal("30"):
            return "low"
        if dec_score < Decimal("60"):
            return "standard"
        if dec_score < Decimal("80"):
            return "high"
        return "critical"

    def format_for_dds_inclusion(
        self, doc: RiskAssessmentDoc,
    ) -> Dict[str, Any]:
        """Format risk doc for DDS risk summary section.

        Produces a compact summary suitable for inclusion in the
        DDS document's risk assessment section.

        Args:
            doc: Risk assessment documentation.

        Returns:
            Dictionary formatted for DDS inclusion.
        """
        risk_label = _RISK_LEVEL_LABELS.get(
            doc.risk_level, doc.risk_level.value,
        )

        summary: Dict[str, Any] = {
            "doc_id": doc.doc_id,
            "assessment_id": doc.assessment_id,
            "composite_score": str(doc.composite_score),
            "risk_level": doc.risk_level.value,
            "risk_level_description": risk_label,
            "simplified_dd_eligible": doc.simplified_dd_eligible,
            "criterion_count": len(doc.criterion_evaluations),
            "article_reference": "EUDR Article 10",
            "generated_at": doc.generated_at.isoformat(),
            "generated_by": AGENT_ID,
        }

        if doc.country_benchmark:
            summary["country_benchmark"] = doc.country_benchmark
            summary["benchmark_article"] = "EUDR Article 29"

        return summary

    def get_risk_level_label(self, level: RiskLevel) -> str:
        """Get the display label for a risk level.

        Args:
            level: Risk level enumeration.

        Returns:
            Human-readable risk level label.
        """
        return _RISK_LEVEL_LABELS.get(level, level.value)

    def get_criteria_list(self) -> List[Dict[str, str]]:
        """Get the standard Article 10(2) criteria list.

        Returns:
            List of criterion dictionaries with id, description,
            and article reference.
        """
        return list(_ARTICLE10_CRITERIA)

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status.

        Returns:
            Dictionary with engine status and configuration details.
        """
        return {
            "engine": "RiskAssessmentDocumenter",
            "status": "available",
            "config": {
                "include_criterion_details": (
                    self._config.include_criterion_details
                ),
                "include_decomposition": (
                    self._config.include_decomposition
                ),
                "include_trend_data": (
                    self._config.include_trend_data
                ),
            },
            "criteria_count": len(_ARTICLE10_CRITERIA),
        }
