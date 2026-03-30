# -*- coding: utf-8 -*-
"""
NarrativeGenerationEngine - PACK-003 CSRD Enterprise Engine 4

AI-assisted narrative composition with strict guardrails for CSRD/ESRS
reporting. Drafts section narratives from structured data, performs
deterministic fact-checking against source data, adjusts tone for
different audiences, and validates ESRS disclosure coverage.

Dual-Validation Pattern:
    1. Draft text from structured data (may use LLM for prose)
    2. Deterministically verify every numeric claim against source data
    3. Flag any unverifiable statements before delivery

Audiences:
    - BOARD: Strategic, high-level, forward-looking
    - INVESTOR: Quantitative, risk-focused, comparative
    - REGULATORY: Precise, standards-referenced, complete
    - PUBLIC: Accessible, jargon-free, impact-focused

Zero-Hallucination:
    - Every numeric value in narratives is traced to source data
    - Fact-checking is deterministic string/number matching
    - Readability scoring uses deterministic formula (Flesch-Kincaid)
    - No LLM involvement in numeric validation or scoring

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-003 CSRD Enterprise
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

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
    """Compute a deterministic SHA-256 hash of arbitrary data."""
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

class NarrativeTone(str, Enum):
    """Target audience tone for narrative generation."""

    BOARD = "board"
    INVESTOR = "investor"
    REGULATORY = "regulatory"
    PUBLIC = "public"

class FactCheckStatus(str, Enum):
    """Status of a single fact-check."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    MISMATCH = "mismatch"
    NOT_APPLICABLE = "not_applicable"

class ESRSSection(str, Enum):
    """ESRS standard section identifiers."""

    ESRS_2_GOV = "ESRS_2_GOV"
    ESRS_2_SBM = "ESRS_2_SBM"
    ESRS_2_IRO = "ESRS_2_IRO"
    ESRS_E1 = "ESRS_E1"
    ESRS_E2 = "ESRS_E2"
    ESRS_E3 = "ESRS_E3"
    ESRS_E4 = "ESRS_E4"
    ESRS_E5 = "ESRS_E5"
    ESRS_S1 = "ESRS_S1"
    ESRS_S2 = "ESRS_S2"
    ESRS_S3 = "ESRS_S3"
    ESRS_S4 = "ESRS_S4"
    ESRS_G1 = "ESRS_G1"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """A citation linking narrative text to source data."""

    data_point_ref: str = Field(..., description="Data point reference ID")
    value: Any = Field(..., description="Cited value")
    source: str = Field(..., description="Source of the data point")
    page_or_section: Optional[str] = Field(
        None, description="Page or section reference"
    )

class FactCheckResult(BaseModel):
    """Result of a single fact-check verification."""

    claim: str = Field(..., description="The claim being checked")
    claimed_value: Optional[str] = Field(
        None, description="Value claimed in the narrative"
    )
    source_value: Optional[str] = Field(
        None, description="Value found in source data"
    )
    status: FactCheckStatus = Field(..., description="Verification status")
    source_ref: Optional[str] = Field(
        None, description="Source data reference"
    )

class NarrativeRequest(BaseModel):
    """Request for narrative section generation."""

    section_type: str = Field(
        ..., description="ESRS standard section reference"
    )
    data_points: Dict[str, Any] = Field(
        ..., description="Structured data for narrative composition"
    )
    tone: NarrativeTone = Field(
        NarrativeTone.REGULATORY, description="Target audience tone"
    )
    language: str = Field("en", description="Target language ISO code")
    max_tokens: int = Field(
        2000, ge=100, le=10000, description="Maximum output tokens"
    )
    fact_checking: bool = Field(
        True, description="Enable automatic fact-checking"
    )

class NarrativeResult(BaseModel):
    """Result of narrative generation."""

    narrative_id: str = Field(
        default_factory=_new_uuid, description="Unique narrative ID"
    )
    section_type: str = Field(..., description="ESRS section type")
    narrative_text: str = Field(..., description="Generated narrative text")
    citations: List[Citation] = Field(
        default_factory=list, description="Data citations"
    )
    fact_check_results: List[FactCheckResult] = Field(
        default_factory=list, description="Fact-check verification results"
    )
    word_count: int = Field(0, description="Word count of narrative")
    readability_score: float = Field(
        0.0, description="Flesch-Kincaid readability score"
    )
    compliance_coverage: float = Field(
        0.0, ge=0.0, le=100.0,
        description="ESRS disclosure requirement coverage percentage",
    )
    tone: NarrativeTone = Field(..., description="Tone used")
    language: str = Field("en", description="Language")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")
    generated_at: datetime = Field(
        default_factory=utcnow, description="Generation timestamp"
    )

class RevisionDiff(BaseModel):
    """A single difference between narrative revisions."""

    position: int = Field(..., description="Approximate character position")
    type: str = Field(..., description="addition, deletion, or modification")
    original: str = Field("", description="Original text")
    revised: str = Field("", description="Revised text")

# ---------------------------------------------------------------------------
# ESRS Disclosure Requirements
# ---------------------------------------------------------------------------

_ESRS_REQUIRED_DISCLOSURES: Dict[str, List[str]] = {
    "ESRS_E1": [
        "transition_plan", "ghg_reduction_targets", "scope_1_emissions",
        "scope_2_emissions", "scope_3_emissions", "energy_consumption",
        "energy_mix", "ghg_intensity", "carbon_credits",
        "internal_carbon_pricing", "financial_effects",
    ],
    "ESRS_E2": [
        "pollution_prevention", "substances_of_concern",
        "microplastics", "financial_effects",
    ],
    "ESRS_E3": [
        "water_consumption", "water_withdrawal", "water_discharge",
        "water_stress_areas", "financial_effects",
    ],
    "ESRS_E4": [
        "biodiversity_targets", "land_use", "species_impact",
        "ecosystem_services", "financial_effects",
    ],
    "ESRS_E5": [
        "resource_inflows", "resource_outflows", "waste_generation",
        "circular_economy", "financial_effects",
    ],
    "ESRS_S1": [
        "workforce_composition", "working_conditions", "equal_treatment",
        "training_development", "health_safety", "work_life_balance",
        "remuneration", "collective_bargaining",
    ],
    "ESRS_S2": [
        "value_chain_workers", "due_diligence", "remediation",
        "targets_workers",
    ],
    "ESRS_S3": [
        "affected_communities", "engagement", "remediation_communities",
    ],
    "ESRS_S4": [
        "consumer_health_safety", "information_privacy",
        "responsible_marketing",
    ],
    "ESRS_G1": [
        "governance_structure", "business_conduct", "anti_corruption",
        "political_engagement", "management_of_relationships",
    ],
    "ESRS_2_GOV": [
        "governance_role", "due_diligence_process",
        "risk_management_integration", "sustainability_statement",
    ],
    "ESRS_2_SBM": [
        "business_model", "value_chain", "strategy_sustainability",
        "stakeholder_interests",
    ],
    "ESRS_2_IRO": [
        "impact_identification", "risk_opportunity_assessment",
        "materiality_assessment",
    ],
}

# ---------------------------------------------------------------------------
# Tone Templates
# ---------------------------------------------------------------------------

_TONE_STYLE: Dict[NarrativeTone, Dict[str, str]] = {
    NarrativeTone.BOARD: {
        "opening": "From a strategic perspective, ",
        "connector": "This positions the organization to ",
        "closing": "The Board is recommended to review the attached details.",
        "style": "concise and strategic",
    },
    NarrativeTone.INVESTOR: {
        "opening": "In terms of material ESG performance, ",
        "connector": "This translates to a financial impact of ",
        "closing": "For detailed methodology, refer to the technical appendix.",
        "style": "quantitative and risk-focused",
    },
    NarrativeTone.REGULATORY: {
        "opening": "In accordance with ESRS disclosure requirements, ",
        "connector": "The methodology applied follows ",
        "closing": "This disclosure has been prepared in compliance with applicable standards.",
        "style": "precise and standards-referenced",
    },
    NarrativeTone.PUBLIC: {
        "opening": "Our sustainability efforts this year show that ",
        "connector": "This means that in practical terms, ",
        "closing": "We remain committed to transparency in our sustainability journey.",
        "style": "accessible and impact-focused",
    },
}

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class NarrativeGenerationEngine:
    """AI-assisted narrative composition engine with fact-checking guardrails.

    Generates ESRS-compliant narrative sections from structured data,
    performs deterministic fact verification, and supports multi-audience
    tone adjustment. Uses the dual-validation pattern: draft text, then
    verify every numeric claim against source data.

    Attributes:
        _narratives: History of generated narratives.

    Example:
        >>> engine = NarrativeGenerationEngine()
        >>> request = NarrativeRequest(
        ...     section_type="ESRS_E1",
        ...     data_points={"scope_1_emissions": 5000, "unit": "tCO2e"},
        ...     tone=NarrativeTone.REGULATORY,
        ... )
        >>> result = engine.generate_section(request)
        >>> assert result.word_count > 0
    """

    def __init__(self) -> None:
        """Initialize NarrativeGenerationEngine."""
        self._narratives: Dict[str, NarrativeResult] = {}
        logger.info(
            "NarrativeGenerationEngine v%s initialized", _MODULE_VERSION
        )

    # -- Section Generation -------------------------------------------------

    def generate_section(
        self, request: NarrativeRequest
    ) -> NarrativeResult:
        """Generate a narrative section from structured data.

        Composes narrative text grounded in the provided data points,
        applies the requested tone, and performs fact-checking if enabled.

        Args:
            request: Narrative generation request.

        Returns:
            NarrativeResult with text, citations, and fact-check results.
        """
        start = utcnow()
        logger.info(
            "Generating %s section in %s tone (lang=%s)",
            request.section_type, request.tone.value, request.language,
        )

        # Draft narrative from structured data
        narrative_text = self._compose_narrative(
            request.section_type, request.data_points, request.tone
        )

        # Truncate if over max_tokens (rough: 1 token ~ 4 chars)
        max_chars = request.max_tokens * 4
        if len(narrative_text) > max_chars:
            narrative_text = narrative_text[:max_chars].rsplit(".", 1)[0] + "."

        # Extract citations
        citations = self._extract_citations(
            request.data_points, request.section_type
        )

        # Fact-check
        fact_checks: List[FactCheckResult] = []
        if request.fact_checking:
            fact_checks = self.fact_check_narrative(
                narrative_text, request.data_points
            )

        # Compute readability
        readability = self._flesch_kincaid_score(narrative_text)

        # Compute ESRS coverage
        coverage = self._compute_esrs_coverage(
            request.section_type, request.data_points
        )

        word_count = len(narrative_text.split())

        result = NarrativeResult(
            section_type=request.section_type,
            narrative_text=narrative_text,
            citations=citations,
            fact_check_results=fact_checks,
            word_count=word_count,
            readability_score=readability,
            compliance_coverage=coverage,
            tone=request.tone,
            language=request.language,
            generated_at=start,
        )
        result.provenance_hash = _compute_hash(result)

        self._narratives[result.narrative_id] = result

        logger.info(
            "Narrative generated: %d words, readability=%.1f, coverage=%.1f%%",
            word_count, readability, coverage,
        )
        return result

    def _compose_narrative(
        self,
        section_type: str,
        data_points: Dict[str, Any],
        tone: NarrativeTone,
    ) -> str:
        """Compose narrative text from structured data and tone.

        This method generates deterministic, template-based text from
        data points. In production, an LLM may assist with prose quality,
        but all numeric values come directly from data_points.

        Args:
            section_type: ESRS section identifier.
            data_points: Source data keyed by disclosure name.
            tone: Target audience tone.

        Returns:
            Composed narrative text string.
        """
        style = _TONE_STYLE.get(tone, _TONE_STYLE[NarrativeTone.REGULATORY])
        paragraphs: List[str] = []

        # Opening
        opening = style["opening"]
        section_label = section_type.replace("_", " ")
        paragraphs.append(
            f"{opening}this section presents the disclosures required "
            f"under {section_label}."
        )

        # Data paragraphs
        for key, value in sorted(data_points.items()):
            if key.startswith("_"):
                continue
            label = key.replace("_", " ").title()
            if isinstance(value, (int, float)):
                unit = data_points.get("unit", "")
                unit_str = f" {unit}" if unit else ""
                paragraphs.append(
                    f"The reported {label} for the period was "
                    f"{value:,.2f}{unit_str}."
                )
            elif isinstance(value, str) and len(value) > 0:
                paragraphs.append(f"Regarding {label}: {value}")
            elif isinstance(value, list):
                items = ", ".join(str(v) for v in value[:5])
                paragraphs.append(
                    f"The {label} includes the following: {items}."
                )
            elif isinstance(value, dict):
                sub_items = "; ".join(
                    f"{k}: {v}" for k, v in list(value.items())[:5]
                )
                paragraphs.append(
                    f"The {label} breakdown is as follows: {sub_items}."
                )

        # Year-over-year comparison if both periods present
        current = data_points.get("current_year_value")
        previous = data_points.get("previous_year_value")
        if current is not None and previous is not None:
            try:
                change = float(current) - float(previous)
                pct = (change / float(previous) * 100) if float(previous) != 0 else 0
                direction = "increase" if change > 0 else "decrease"
                paragraphs.append(
                    f"This represents a {abs(pct):.1f}% {direction} "
                    f"compared to the previous reporting period."
                )
            except (TypeError, ValueError):
                pass

        # Closing
        paragraphs.append(style["closing"])

        return "\n\n".join(paragraphs)

    # -- Fact-Checking ------------------------------------------------------

    def fact_check_narrative(
        self, narrative: str, source_data: Dict[str, Any]
    ) -> List[FactCheckResult]:
        """Verify every numeric claim in a narrative against source data.

        Deterministically extracts numbers from the narrative and matches
        them against the provided source data values.

        Args:
            narrative: Narrative text to verify.
            source_data: Source data dictionary.

        Returns:
            List of FactCheckResult for each verified claim.
        """
        results: List[FactCheckResult] = []

        # Extract all numbers from narrative
        number_pattern = r"[\d,]+\.?\d*"
        numbers_in_text = re.findall(number_pattern, narrative)

        # Build lookup of source values
        source_values: Dict[str, str] = {}
        for key, value in source_data.items():
            if isinstance(value, (int, float)):
                formatted = f"{value:,.2f}"
                source_values[key] = formatted
                # Also store without decimals
                source_values[f"{key}_int"] = f"{int(value):,}"

        # Check each number
        for num_str in numbers_in_text:
            num_clean = num_str.replace(",", "")
            try:
                num_val = float(num_clean)
            except ValueError:
                continue

            # Find matching source
            matched = False
            for key, src_val in source_values.items():
                src_clean = src_val.replace(",", "")
                try:
                    src_num = float(src_clean)
                except ValueError:
                    continue

                if abs(num_val - src_num) < 0.01:
                    results.append(FactCheckResult(
                        claim=f"Value {num_str} referenced in narrative",
                        claimed_value=num_str,
                        source_value=src_val,
                        status=FactCheckStatus.VERIFIED,
                        source_ref=key.replace("_int", ""),
                    ))
                    matched = True
                    break

            if not matched and num_val > 0:
                # Check if it might be a percentage or year
                if 1900 < num_val < 2100:
                    results.append(FactCheckResult(
                        claim=f"Year reference {num_str}",
                        claimed_value=num_str,
                        source_value=None,
                        status=FactCheckStatus.NOT_APPLICABLE,
                        source_ref=None,
                    ))
                else:
                    results.append(FactCheckResult(
                        claim=f"Value {num_str} not found in source data",
                        claimed_value=num_str,
                        source_value=None,
                        status=FactCheckStatus.UNVERIFIED,
                        source_ref=None,
                    ))

        logger.info(
            "Fact-check: %d claims checked, %d verified, %d unverified",
            len(results),
            sum(1 for r in results if r.status == FactCheckStatus.VERIFIED),
            sum(1 for r in results if r.status == FactCheckStatus.UNVERIFIED),
        )
        return results

    # -- Tone Adjustment ----------------------------------------------------

    def adjust_tone(
        self, narrative: str, target_tone: NarrativeTone
    ) -> str:
        """Adjust the tone of an existing narrative for a different audience.

        Args:
            narrative: Original narrative text.
            target_tone: Target audience tone.

        Returns:
            Re-styled narrative text.
        """
        style = _TONE_STYLE.get(
            target_tone, _TONE_STYLE[NarrativeTone.REGULATORY]
        )

        # Simple tone adjustment: replace opening and closing
        lines = narrative.split("\n\n")
        if lines:
            lines[0] = style["opening"] + lines[0].split(",", 1)[-1].strip() if "," in lines[0] else lines[0]
        if len(lines) > 1:
            lines[-1] = style["closing"]

        adjusted = "\n\n".join(lines)

        logger.info("Tone adjusted to %s", target_tone.value)
        return adjusted

    # -- Translation --------------------------------------------------------

    def translate_narrative(
        self, narrative: str, target_language: str
    ) -> str:
        """Translate narrative to target language.

        In production, this delegates to a translation service.
        Returns original with language tag for non-English.

        Args:
            narrative: Narrative text in source language.
            target_language: ISO 639-1 language code.

        Returns:
            Translated (or tagged) narrative text.
        """
        if target_language.lower() == "en":
            return narrative

        # Placeholder: in production, delegate to translation API
        translated = (
            f"[{target_language.upper()} TRANSLATION]\n\n{narrative}\n\n"
            f"[NOTE: Translation to {target_language} pending professional review]"
        )

        logger.info("Narrative marked for translation to %s", target_language)
        return translated

    # -- Executive Summary --------------------------------------------------

    def generate_executive_summary(
        self, full_report_data: Dict[str, Any]
    ) -> NarrativeResult:
        """Generate a concise executive summary from full report data.

        Extracts key metrics and composes a board-level summary suitable
        for a 2-3 minute read.

        Args:
            full_report_data: Complete report data with all sections.

        Returns:
            NarrativeResult with executive summary.
        """
        # Extract key metrics
        key_metrics: Dict[str, Any] = {}
        for section, data in full_report_data.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        key_metrics[f"{section}.{key}"] = value

        # Compose summary
        paragraphs: List[str] = [
            "This executive summary presents the key findings from the "
            "sustainability report prepared in accordance with the European "
            "Sustainability Reporting Standards (ESRS).",
        ]

        # Environment summary
        env_keys = [k for k in key_metrics if any(
            e in k for e in ["E1", "E2", "E3", "E4", "E5", "emission", "energy"]
        )]
        if env_keys:
            env_items = [
                f"{k.split('.')[-1].replace('_', ' ').title()}: "
                f"{key_metrics[k]:,.2f}"
                for k in env_keys[:5]
            ]
            paragraphs.append(
                "Environmental Performance: " + "; ".join(env_items) + "."
            )

        # Social summary
        social_keys = [k for k in key_metrics if any(
            s in k for s in ["S1", "S2", "S3", "S4", "workforce", "safety"]
        )]
        if social_keys:
            social_items = [
                f"{k.split('.')[-1].replace('_', ' ').title()}: "
                f"{key_metrics[k]:,.2f}"
                for k in social_keys[:5]
            ]
            paragraphs.append(
                "Social Performance: " + "; ".join(social_items) + "."
            )

        # Governance summary
        gov_keys = [k for k in key_metrics if any(
            g in k for g in ["G1", "governance", "conduct"]
        )]
        if gov_keys:
            gov_items = [
                f"{k.split('.')[-1].replace('_', ' ').title()}: "
                f"{key_metrics[k]:,.2f}"
                for k in gov_keys[:5]
            ]
            paragraphs.append(
                "Governance Performance: " + "; ".join(gov_items) + "."
            )

        if not env_keys and not social_keys and not gov_keys:
            paragraphs.append(
                "The report covers environmental, social, and governance "
                "dimensions of the organization's sustainability performance."
            )

        paragraphs.append(
            "The Board is recommended to review the complete report for "
            "detailed disclosures and action items."
        )

        summary_text = "\n\n".join(paragraphs)

        result = NarrativeResult(
            section_type="EXECUTIVE_SUMMARY",
            narrative_text=summary_text,
            citations=[],
            fact_check_results=[],
            word_count=len(summary_text.split()),
            readability_score=self._flesch_kincaid_score(summary_text),
            compliance_coverage=0.0,
            tone=NarrativeTone.BOARD,
            language="en",
        )
        result.provenance_hash = _compute_hash(result)

        self._narratives[result.narrative_id] = result
        return result

    # -- Revision Tracking --------------------------------------------------

    def track_revisions(
        self, original: str, revised: str
    ) -> Dict[str, Any]:
        """Track differences between narrative revisions.

        Args:
            original: Original narrative text.
            revised: Revised narrative text.

        Returns:
            Dict with revision summary and diff details.
        """
        orig_words = original.split()
        rev_words = revised.split()

        # Simple word-level diff
        additions = 0
        deletions = 0
        modifications = 0

        orig_set = set(orig_words)
        rev_set = set(rev_words)

        added_words = rev_set - orig_set
        removed_words = orig_set - rev_set

        additions = len(added_words)
        deletions = len(removed_words)

        # Count sentences changed
        orig_sentences = re.split(r"[.!?]+", original)
        rev_sentences = re.split(r"[.!?]+", revised)

        orig_sent_set = set(s.strip() for s in orig_sentences if s.strip())
        rev_sent_set = set(s.strip() for s in rev_sentences if s.strip())
        modified_sentences = len(
            (orig_sent_set - rev_sent_set) | (rev_sent_set - orig_sent_set)
        )

        result = {
            "revision_id": _new_uuid(),
            "original_word_count": len(orig_words),
            "revised_word_count": len(rev_words),
            "words_added": additions,
            "words_removed": deletions,
            "sentences_changed": modified_sentences,
            "change_percentage": round(
                (additions + deletions)
                / max(len(orig_words), 1) * 100, 2
            ),
            "original_readability": self._flesch_kincaid_score(original),
            "revised_readability": self._flesch_kincaid_score(revised),
            "tracked_at": utcnow().isoformat(),
            "provenance_hash": _compute_hash(
                {"original_hash": _compute_hash(original),
                 "revised_hash": _compute_hash(revised)}
            ),
        }
        return result

    # -- ESRS Compliance Check ----------------------------------------------

    def check_esrs_compliance(
        self, narrative: str, standard_ref: str
    ) -> Dict[str, Any]:
        """Verify narrative against ESRS disclosure requirements.

        Args:
            narrative: Narrative text to check.
            standard_ref: ESRS standard reference (e.g., 'ESRS_E1').

        Returns:
            Dict with compliance status, coverage, and missing items.
        """
        required = _ESRS_REQUIRED_DISCLOSURES.get(standard_ref, [])

        if not required:
            return {
                "standard_ref": standard_ref,
                "status": "unknown_standard",
                "message": f"No requirements defined for {standard_ref}",
            }

        narrative_lower = narrative.lower()
        covered: List[str] = []
        missing: List[str] = []

        for disclosure in required:
            # Check if the disclosure topic is addressed in the narrative
            search_terms = disclosure.replace("_", " ").split()
            found = any(term in narrative_lower for term in search_terms)
            if found:
                covered.append(disclosure)
            else:
                missing.append(disclosure)

        coverage = (len(covered) / len(required) * 100) if required else 0.0

        result = {
            "standard_ref": standard_ref,
            "total_requirements": len(required),
            "covered": len(covered),
            "missing": len(missing),
            "coverage_percentage": round(coverage, 1),
            "covered_disclosures": covered,
            "missing_disclosures": missing,
            "compliant": coverage >= 80.0,
            "provenance_hash": _compute_hash({
                "standard": standard_ref, "coverage": coverage,
            }),
        }

        logger.info(
            "ESRS compliance check for %s: %.1f%% coverage (%d/%d)",
            standard_ref, coverage, len(covered), len(required),
        )
        return result

    # -- Internal Helpers ---------------------------------------------------

    def _extract_citations(
        self, data_points: Dict[str, Any], section_type: str
    ) -> List[Citation]:
        """Extract citations from data points.

        Args:
            data_points: Source data dictionary.
            section_type: ESRS section type.

        Returns:
            List of Citation objects.
        """
        citations: List[Citation] = []
        for key, value in data_points.items():
            if key.startswith("_"):
                continue
            if isinstance(value, (int, float, str)):
                citations.append(Citation(
                    data_point_ref=f"{section_type}.{key}",
                    value=value,
                    source=f"data_points['{key}']",
                ))
        return citations

    def _compute_esrs_coverage(
        self, section_type: str, data_points: Dict[str, Any]
    ) -> float:
        """Compute ESRS disclosure coverage percentage.

        Args:
            section_type: ESRS section type.
            data_points: Provided data points.

        Returns:
            Coverage percentage (0-100).
        """
        required = _ESRS_REQUIRED_DISCLOSURES.get(section_type, [])
        if not required:
            return 0.0

        data_keys = set(k.lower() for k in data_points.keys())
        covered = sum(
            1 for r in required
            if any(part in data_keys for part in r.lower().split("_"))
        )
        return round(covered / len(required) * 100, 1)

    def _flesch_kincaid_score(self, text: str) -> float:
        """Calculate Flesch-Kincaid readability score.

        Higher scores indicate easier-to-read text:
          90-100: Very easy (5th grade)
          60-70:  Standard (8th-9th grade)
          30-50:  Difficult (college level)
          0-30:   Very difficult (professional)

        Formula: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)

        Args:
            text: Text to score.

        Returns:
            Flesch-Kincaid readability score (0-100+).
        """
        if not text or not text.strip():
            return 0.0

        # Count sentences
        sentences = re.split(r"[.!?]+", text)
        sentences = [s for s in sentences if s.strip()]
        sentence_count = max(len(sentences), 1)

        # Count words
        words = text.split()
        word_count = max(len(words), 1)

        # Estimate syllables (simplified)
        syllable_count = sum(self._count_syllables(w) for w in words)

        # Flesch-Kincaid formula
        score = (
            206.835
            - 1.015 * (word_count / sentence_count)
            - 84.6 * (syllable_count / word_count)
        )

        return round(max(0.0, min(100.0, score)), 1)

    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count for a word.

        Args:
            word: Single word string.

        Returns:
            Estimated syllable count.
        """
        word = word.lower().strip(".,;:!?\"'()")
        if len(word) <= 2:
            return 1

        # Count vowel groups
        vowels = "aeiouy"
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel

        # Adjustments
        if word.endswith("e"):
            count -= 1
        if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
            count += 1

        return max(count, 1)
