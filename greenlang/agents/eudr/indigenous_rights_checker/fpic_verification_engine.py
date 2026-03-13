# -*- coding: utf-8 -*-
"""
FPICVerificationEngine - Feature 2: FPIC Documentation Verification

Deterministic FPIC documentation scoring engine implementing the 10-element
weighted checklist per PRD F2.1. Scores range from 0-100 using Decimal
arithmetic for bit-perfect reproducibility.

FPIC Scoring Formula (PRD Section 6.1, Feature 2):
    FPIC_Score = (
        community_identification * 0.10
        + information_disclosure * 0.15
        + prior_timing * 0.10
        + consultation_process * 0.15
        + community_representation * 0.10
        + consent_record * 0.15
        + absence_of_coercion * 0.10
        + agreement_documentation * 0.05
        + benefit_sharing * 0.05
        + monitoring_provisions * 0.05
    )

Classification:
    CONSENT_OBTAINED:   FPIC_Score >= 80
    CONSENT_PARTIAL:    50 <= FPIC_Score < 80
    CONSENT_MISSING:    FPIC_Score < 50

Performance Target: < 100ms per FPIC assessment.

Example:
    >>> engine = FPICVerificationEngine(config, provenance)
    >>> result = await engine.verify_fpic(
    ...     plot_id="p-001", territory_id="t-001",
    ...     documentation={"community_identified": True, ...},
    ... )
    >>> print(result.fpic_score, result.fpic_status)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 (Feature 2)
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

from greenlang.agents.eudr.indigenous_rights_checker.config import (
    IndigenousRightsCheckerConfig,
)
from greenlang.agents.eudr.indigenous_rights_checker.models import (
    FPICAssessment,
    FPICStatus,
)
from greenlang.agents.eudr.indigenous_rights_checker.provenance import (
    ProvenanceTracker,
)
from greenlang.agents.eudr.indigenous_rights_checker.metrics import (
    record_fpic_assessment,
    observe_fpic_assessment_duration,
)
from greenlang.agents.eudr.indigenous_rights_checker.reference_data.fpic_legal_frameworks import (
    get_fpic_requirements,
    get_consultation_protocol,
)

logger = logging.getLogger(__name__)

# Decimal constants for precision
_D100 = Decimal("100")
_D80 = Decimal("80")
_D50 = Decimal("50")
_D0 = Decimal("0")
_PRECISION = Decimal("0.01")

# ---------------------------------------------------------------------------
# SQL templates
# ---------------------------------------------------------------------------

_SQL_INSERT_ASSESSMENT = """
    INSERT INTO eudr_indigenous_rights_checker.gl_eudr_irc_fpic_assessments (
        assessment_id, plot_id, territory_id, community_id,
        fpic_score, fpic_status,
        community_identification_score, information_disclosure_score,
        prior_timing_score, consultation_process_score,
        community_representation_score, consent_record_score,
        absence_of_coercion_score, agreement_documentation_score,
        benefit_sharing_score, monitoring_provisions_score,
        country_specific_rules, temporal_compliance, coercion_flags,
        validity_start, validity_end, decision_rationale,
        provenance_hash, version, assessed_at
    ) VALUES (
        %(assessment_id)s, %(plot_id)s, %(territory_id)s, %(community_id)s,
        %(fpic_score)s, %(fpic_status)s,
        %(community_identification_score)s, %(information_disclosure_score)s,
        %(prior_timing_score)s, %(consultation_process_score)s,
        %(community_representation_score)s, %(consent_record_score)s,
        %(absence_of_coercion_score)s, %(agreement_documentation_score)s,
        %(benefit_sharing_score)s, %(monitoring_provisions_score)s,
        %(country_specific_rules)s, %(temporal_compliance)s,
        %(coercion_flags)s,
        %(validity_start)s, %(validity_end)s, %(decision_rationale)s,
        %(provenance_hash)s, %(version)s, %(assessed_at)s
    )
"""

_SQL_GET_ASSESSMENTS_FOR_PLOT = """
    SELECT assessment_id, plot_id, territory_id, community_id,
           fpic_score, fpic_status, assessed_at, provenance_hash, version
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_fpic_assessments
    WHERE plot_id = %(plot_id)s
    ORDER BY assessed_at DESC
"""


class FPICVerificationEngine:
    """Deterministic FPIC documentation verification engine.

    Implements the 10-element weighted scoring formula using Decimal
    arithmetic. All scoring is deterministic with zero LLM usage in
    the critical calculation path.

    Attributes:
        _config: Agent configuration with FPIC weights.
        _provenance: Provenance tracker for audit trail.
        _pool: Async database connection pool.

    Example:
        >>> engine = FPICVerificationEngine(config, tracker)
        >>> result = await engine.verify_fpic("p-001", "t-001", docs)
    """

    def __init__(
        self,
        config: IndigenousRightsCheckerConfig,
        provenance: ProvenanceTracker,
    ) -> None:
        """Initialize FPICVerificationEngine.

        Args:
            config: Agent configuration instance.
            provenance: Provenance tracker instance.
        """
        self._config = config
        self._provenance = provenance
        self._pool: Any = None
        # Convert weights to Decimal for precision
        self._weights: Dict[str, Decimal] = {
            k: Decimal(str(v))
            for k, v in config.fpic_weights.items()
        }
        logger.info(
            "FPICVerificationEngine initialized with weights: "
            f"{dict(self._weights)}"
        )

    async def startup(self, pool: Any) -> None:
        """Set the database connection pool."""
        self._pool = pool
        logger.info("FPICVerificationEngine started")

    async def shutdown(self) -> None:
        """Clean up engine resources."""
        self._pool = None

    async def verify_fpic(
        self,
        plot_id: str,
        territory_id: str,
        documentation: Dict[str, Any],
        community_id: Optional[str] = None,
        production_start_date: Optional[date] = None,
        country_code: Optional[str] = None,
    ) -> FPICAssessment:
        """Verify FPIC documentation and produce deterministic score.

        Evaluates 10 FPIC elements using weighted scoring formula.
        All calculations use Decimal arithmetic for bit-perfect
        reproducibility. No LLM calls in the scoring path.

        Args:
            plot_id: Production plot identifier.
            territory_id: Overlapping territory identifier.
            documentation: Dictionary with FPIC documentation evidence.
            community_id: Optional affected community identifier.
            production_start_date: Optional production activity start date.
            country_code: Optional country code for jurisdiction rules.

        Returns:
            FPICAssessment with deterministic score and classification.

        Example:
            >>> result = await engine.verify_fpic(
            ...     plot_id="p-001",
            ...     territory_id="t-001",
            ...     documentation={
            ...         "community_identified": True,
            ...         "identification_verified": True,
            ...         "information_shared": True,
            ...         "disclosure_languages": ["pt", "yanomami"],
            ...         "consent_date": "2024-01-15",
            ...         "production_start": "2024-06-01",
            ...     },
            ...     country_code="BR",
            ... )
        """
        start = time.monotonic()
        assessment_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        # Step 1: Score each of the 10 elements (deterministic)
        element_scores = self._score_all_elements(
            documentation, production_start_date, country_code
        )

        # Step 2: Calculate composite FPIC score (weighted sum)
        fpic_score = self._calculate_composite_score(element_scores)

        # Step 3: Classify FPIC status
        fpic_status = self._classify_fpic_status(fpic_score)

        # Step 4: Check temporal compliance
        temporal_compliance = self._check_temporal_compliance(
            documentation, production_start_date
        )

        # Step 5: Detect coercion indicators
        coercion_flags = self._detect_coercion_indicators(
            documentation, country_code
        )

        # Step 6: Determine validity period
        validity_start, validity_end = self._determine_validity(documentation)

        # Step 7: Get country-specific rules applied
        country_rules = get_consultation_protocol(country_code) if country_code else None

        # Step 8: Generate decision rationale
        rationale = self._generate_rationale(
            fpic_score, fpic_status, element_scores,
            temporal_compliance, coercion_flags,
        )

        # Step 9: Calculate provenance hash
        provenance_hash = self._provenance.compute_data_hash({
            "assessment_id": assessment_id,
            "plot_id": plot_id,
            "territory_id": territory_id,
            "fpic_score": str(fpic_score),
            "element_scores": {k: str(v) for k, v in element_scores.items()},
            "assessed_at": now.isoformat(),
        })

        # Step 10: Build assessment model
        assessment = FPICAssessment(
            assessment_id=assessment_id,
            plot_id=plot_id,
            territory_id=territory_id,
            community_id=community_id,
            fpic_score=fpic_score,
            fpic_status=fpic_status,
            community_identification_score=element_scores["community_identification"],
            information_disclosure_score=element_scores["information_disclosure"],
            prior_timing_score=element_scores["prior_timing"],
            consultation_process_score=element_scores["consultation_process"],
            community_representation_score=element_scores["community_representation"],
            consent_record_score=element_scores["consent_record"],
            absence_of_coercion_score=element_scores["absence_of_coercion"],
            agreement_documentation_score=element_scores["agreement_documentation"],
            benefit_sharing_score=element_scores["benefit_sharing"],
            monitoring_provisions_score=element_scores["monitoring_provisions"],
            country_specific_rules=country_rules,
            temporal_compliance=temporal_compliance,
            coercion_flags=coercion_flags,
            validity_start=validity_start,
            validity_end=validity_end,
            decision_rationale=rationale,
            provenance_hash=provenance_hash,
            version=1,
            assessed_at=now,
        )

        # Record provenance
        self._provenance.record(
            "fpic_assessment", "verify", assessment_id,
            metadata={
                "plot_id": plot_id,
                "territory_id": territory_id,
                "fpic_score": str(fpic_score),
                "fpic_status": fpic_status.value,
            },
        )

        # Persist to database
        await self._persist_assessment(assessment)

        elapsed = time.monotonic() - start
        observe_fpic_assessment_duration(elapsed)
        record_fpic_assessment(fpic_status.value)

        logger.info(
            f"FPIC assessment {assessment_id}: score={fpic_score}, "
            f"status={fpic_status.value}, time={elapsed*1000:.1f}ms"
        )

        return assessment

    def _score_all_elements(
        self,
        documentation: Dict[str, Any],
        production_start_date: Optional[date],
        country_code: Optional[str],
    ) -> Dict[str, Decimal]:
        """Score all 10 FPIC elements deterministically.

        Each element is scored 0-100 based on documentation evidence.

        Args:
            documentation: FPIC documentation evidence.
            production_start_date: Production activity start date.
            country_code: Country code for jurisdiction-specific rules.

        Returns:
            Dictionary mapping element names to Decimal scores (0-100).
        """
        return {
            "community_identification": self._score_community_identification(
                documentation
            ),
            "information_disclosure": self._score_information_disclosure(
                documentation, country_code
            ),
            "prior_timing": self._score_prior_timing(
                documentation, production_start_date
            ),
            "consultation_process": self._score_consultation_process(
                documentation
            ),
            "community_representation": self._score_community_representation(
                documentation
            ),
            "consent_record": self._score_consent_record(documentation),
            "absence_of_coercion": self._score_absence_of_coercion(
                documentation
            ),
            "agreement_documentation": self._score_agreement_documentation(
                documentation
            ),
            "benefit_sharing": self._score_benefit_sharing(documentation),
            "monitoring_provisions": self._score_monitoring_provisions(
                documentation
            ),
        }

    def _calculate_composite_score(
        self, element_scores: Dict[str, Decimal]
    ) -> Decimal:
        """Calculate weighted composite FPIC score.

        Uses Decimal arithmetic for bit-perfect reproducibility.

        Args:
            element_scores: Per-element scores (0-100).

        Returns:
            Composite score (0-100) rounded to 2 decimal places.
        """
        composite = _D0
        for element, score in element_scores.items():
            weight = self._weights.get(element, _D0)
            composite += score * weight
        return composite.quantize(_PRECISION, rounding=ROUND_HALF_UP)

    def _classify_fpic_status(self, score: Decimal) -> FPICStatus:
        """Classify FPIC status based on composite score.

        Per PRD F2.6:
            >= 80: CONSENT_OBTAINED
            50-79: CONSENT_PARTIAL
            < 50: CONSENT_MISSING

        Args:
            score: Composite FPIC score (0-100).

        Returns:
            FPICStatus enum value.
        """
        if score >= _D80:
            return FPICStatus.CONSENT_OBTAINED
        elif score >= _D50:
            return FPICStatus.CONSENT_PARTIAL
        else:
            return FPICStatus.CONSENT_MISSING

    # -----------------------------------------------------------------------
    # Individual element scoring methods (deterministic, zero-hallucination)
    # -----------------------------------------------------------------------

    def _score_community_identification(
        self, doc: Dict[str, Any]
    ) -> Decimal:
        """Score community identification completeness (0-100)."""
        score = _D0
        if doc.get("community_identified"):
            score += Decimal("40")
        if doc.get("identification_verified"):
            score += Decimal("30")
        if doc.get("community_governance_documented"):
            score += Decimal("30")
        return min(score, _D100)

    def _score_information_disclosure(
        self, doc: Dict[str, Any], country_code: Optional[str]
    ) -> Decimal:
        """Score information disclosure adequacy (0-100).

        Checks: project scope, environmental impact, social impact,
        economic impact, alternatives, right to withhold consent.
        """
        score = _D0
        required_docs = [
            "project_description", "environmental_impact",
            "social_impact", "economic_impact",
            "alternatives_analysis", "right_to_withhold",
        ]
        provided = doc.get("disclosure_documents", [])
        for req in required_docs:
            if req in provided:
                score += Decimal("15")

        # Language accessibility check
        languages = doc.get("disclosure_languages", [])
        if len(languages) >= 2:
            score += Decimal("10")
        elif len(languages) >= 1:
            score += Decimal("5")

        return min(score, _D100)

    def _score_prior_timing(
        self, doc: Dict[str, Any], production_start: Optional[date]
    ) -> Decimal:
        """Score temporal compliance (consent before production).

        Full score if consent precedes production by >= min_lead_time_days.
        """
        consent_date_str = doc.get("consent_date")
        if not consent_date_str or not production_start:
            return _D0

        try:
            if isinstance(consent_date_str, str):
                consent_date = date.fromisoformat(consent_date_str)
            else:
                consent_date = consent_date_str

            days_before = (production_start - consent_date).days
            min_lead = self._config.fpic_min_lead_time_days

            if days_before >= min_lead:
                return _D100
            elif days_before > 0:
                # Partial credit proportional to lead time achieved
                ratio = Decimal(str(days_before)) / Decimal(str(min_lead))
                return (ratio * _D100).quantize(_PRECISION)
            else:
                return _D0
        except (ValueError, TypeError):
            return _D0

    def _score_consultation_process(self, doc: Dict[str, Any]) -> Decimal:
        """Score consultation process documentation (0-100)."""
        score = _D0
        if doc.get("consultation_meetings_held"):
            score += Decimal("30")
        if doc.get("consultation_minutes_available"):
            score += Decimal("25")
        if doc.get("consultation_attendees_documented"):
            score += Decimal("20")
        if doc.get("follow_up_actions_documented"):
            score += Decimal("25")
        return min(score, _D100)

    def _score_community_representation(
        self, doc: Dict[str, Any]
    ) -> Decimal:
        """Score community representation legitimacy (0-100)."""
        score = _D0
        if doc.get("representatives_identified"):
            score += Decimal("30")
        if doc.get("mandate_verified"):
            score += Decimal("35")
        if doc.get("community_endorsed_representatives"):
            score += Decimal("35")
        return min(score, _D100)

    def _score_consent_record(self, doc: Dict[str, Any]) -> Decimal:
        """Score consent/objection record (0-100)."""
        score = _D0
        if doc.get("consent_formally_recorded"):
            score += Decimal("40")
        if doc.get("consent_signed_by_representatives"):
            score += Decimal("30")
        if doc.get("consent_witnessed"):
            score += Decimal("15")
        if doc.get("consent_date"):
            score += Decimal("15")
        return min(score, _D100)

    def _score_absence_of_coercion(self, doc: Dict[str, Any]) -> Decimal:
        """Score absence of coercion evidence (0-100).

        Starts at 100 and deducts for each coercion indicator found.
        """
        score = _D100
        coercion_indicators = [
            "time_pressure_detected",
            "active_conflict_during_consent",
            "no_legal_representation",
            "conditional_benefits",
            "military_presence",
        ]
        for indicator in coercion_indicators:
            if doc.get(indicator):
                score -= Decimal("25")
        return max(score, _D0)

    def _score_agreement_documentation(
        self, doc: Dict[str, Any]
    ) -> Decimal:
        """Score agreement documentation completeness (0-100)."""
        score = _D0
        if doc.get("agreement_documented"):
            score += Decimal("50")
        if doc.get("agreement_terms_clear"):
            score += Decimal("25")
        if doc.get("agreement_signed"):
            score += Decimal("25")
        return min(score, _D100)

    def _score_benefit_sharing(self, doc: Dict[str, Any]) -> Decimal:
        """Score benefit-sharing terms documentation (0-100)."""
        score = _D0
        if doc.get("benefit_sharing_terms_defined"):
            score += Decimal("40")
        if doc.get("monetary_benefits_specified"):
            score += Decimal("20")
        if doc.get("non_monetary_benefits_specified"):
            score += Decimal("20")
        if doc.get("payment_schedule_defined"):
            score += Decimal("20")
        return min(score, _D100)

    def _score_monitoring_provisions(self, doc: Dict[str, Any]) -> Decimal:
        """Score ongoing monitoring provisions (0-100)."""
        score = _D0
        if doc.get("monitoring_plan_defined"):
            score += Decimal("40")
        if doc.get("monitoring_frequency_specified"):
            score += Decimal("20")
        if doc.get("community_feedback_mechanism"):
            score += Decimal("20")
        if doc.get("renewal_provisions_defined"):
            score += Decimal("20")
        return min(score, _D100)

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def _check_temporal_compliance(
        self,
        doc: Dict[str, Any],
        production_start: Optional[date],
    ) -> bool:
        """Check if consent was obtained prior to production start."""
        consent_date_str = doc.get("consent_date")
        if not consent_date_str or not production_start:
            return False
        try:
            if isinstance(consent_date_str, str):
                consent_date = date.fromisoformat(consent_date_str)
            else:
                consent_date = consent_date_str
            return consent_date < production_start
        except (ValueError, TypeError):
            return False

    def _detect_coercion_indicators(
        self,
        doc: Dict[str, Any],
        country_code: Optional[str],
    ) -> List[str]:
        """Detect coercion indicators in documentation."""
        flags = []
        # Check disclosure-to-consent timing
        disclosure_date = doc.get("information_disclosure_date")
        consent_date = doc.get("consent_date")
        if disclosure_date and consent_date:
            try:
                d_date = (
                    date.fromisoformat(disclosure_date)
                    if isinstance(disclosure_date, str)
                    else disclosure_date
                )
                c_date = (
                    date.fromisoformat(consent_date)
                    if isinstance(consent_date, str)
                    else consent_date
                )
                days_gap = (c_date - d_date).days
                if days_gap < self._config.fpic_coercion_min_days:
                    flags.append(
                        f"time_pressure: {days_gap} days between "
                        f"disclosure and consent (minimum: "
                        f"{self._config.fpic_coercion_min_days})"
                    )
            except (ValueError, TypeError):
                pass

        if doc.get("active_conflict_during_consent"):
            flags.append("consent_during_active_conflict")
        if doc.get("no_legal_representation"):
            flags.append("no_legal_representation")
        if doc.get("conditional_benefits"):
            flags.append("benefits_conditioned_on_consent")
        if doc.get("military_presence"):
            flags.append("military_or_police_presence")

        return flags

    def _determine_validity(
        self, doc: Dict[str, Any]
    ) -> Tuple[Optional[date], Optional[date]]:
        """Determine consent validity period."""
        consent_date_str = doc.get("consent_date")
        if not consent_date_str:
            return None, None
        try:
            if isinstance(consent_date_str, str):
                start = date.fromisoformat(consent_date_str)
            else:
                start = consent_date_str
            from datetime import timedelta
            end = date(
                start.year + self._config.fpic_validity_years,
                start.month,
                start.day,
            )
            return start, end
        except (ValueError, TypeError):
            return None, None

    def _generate_rationale(
        self,
        fpic_score: Decimal,
        fpic_status: FPICStatus,
        element_scores: Dict[str, Decimal],
        temporal_compliance: bool,
        coercion_flags: List[str],
    ) -> str:
        """Generate human-readable decision rationale."""
        parts = [
            f"FPIC composite score: {fpic_score}/100 "
            f"({fpic_status.value}).",
        ]
        # Identify weakest elements
        weak = [
            (name, score) for name, score in element_scores.items()
            if score < _D50
        ]
        if weak:
            weak_names = ", ".join(
                f"{n} ({s})" for n, s in sorted(weak, key=lambda x: x[1])
            )
            parts.append(f"Weak elements: {weak_names}.")

        if not temporal_compliance:
            parts.append("Temporal compliance NOT met.")

        if coercion_flags:
            parts.append(
                f"Coercion indicators detected: {', '.join(coercion_flags)}."
            )

        return " ".join(parts)

    async def _persist_assessment(self, assessment: FPICAssessment) -> None:
        """Persist FPIC assessment to database."""
        if self._pool is None:
            return

        import json
        params = {
            "assessment_id": assessment.assessment_id,
            "plot_id": assessment.plot_id,
            "territory_id": assessment.territory_id,
            "community_id": assessment.community_id,
            "fpic_score": float(assessment.fpic_score),
            "fpic_status": assessment.fpic_status.value,
            "community_identification_score": float(assessment.community_identification_score),
            "information_disclosure_score": float(assessment.information_disclosure_score),
            "prior_timing_score": float(assessment.prior_timing_score),
            "consultation_process_score": float(assessment.consultation_process_score),
            "community_representation_score": float(assessment.community_representation_score),
            "consent_record_score": float(assessment.consent_record_score),
            "absence_of_coercion_score": float(assessment.absence_of_coercion_score),
            "agreement_documentation_score": float(assessment.agreement_documentation_score),
            "benefit_sharing_score": float(assessment.benefit_sharing_score),
            "monitoring_provisions_score": float(assessment.monitoring_provisions_score),
            "country_specific_rules": assessment.country_specific_rules,
            "temporal_compliance": assessment.temporal_compliance,
            "coercion_flags": json.dumps(assessment.coercion_flags),
            "validity_start": assessment.validity_start,
            "validity_end": assessment.validity_end,
            "decision_rationale": assessment.decision_rationale,
            "provenance_hash": assessment.provenance_hash,
            "version": assessment.version,
            "assessed_at": assessment.assessed_at,
        }

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_INSERT_ASSESSMENT, params)
            await conn.commit()

    async def get_assessments_for_plot(
        self, plot_id: str
    ) -> List[Dict[str, Any]]:
        """Get all FPIC assessments for a given plot.

        Args:
            plot_id: Production plot identifier.

        Returns:
            List of assessment summary dictionaries.
        """
        if self._pool is None:
            return []

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _SQL_GET_ASSESSMENTS_FOR_PLOT,
                    {"plot_id": plot_id},
                )
                rows = await cur.fetchall()

        return [
            {
                "assessment_id": str(row[0]),
                "plot_id": str(row[1]),
                "territory_id": str(row[2]),
                "community_id": str(row[3]) if row[3] else None,
                "fpic_score": float(row[4]),
                "fpic_status": row[5],
                "assessed_at": row[6].isoformat() if row[6] else None,
                "provenance_hash": row[7],
                "version": row[8],
            }
            for row in rows
        ]
