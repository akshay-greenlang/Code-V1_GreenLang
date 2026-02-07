# -*- coding: utf-8 -*-
"""
SOC 2 Self-Assessment Assessor - SEC-009

Implements the Assessor class for conducting SOC 2 Type II readiness
self-assessments. The assessor evaluates each Trust Service Criterion,
collects evidence, calculates maturity scores, and produces a comprehensive
assessment report.

The assessor uses a zero-hallucination approach - all scores are calculated
deterministically based on evidence counts and validation rules, not LLM
inference. LLM assistance is only used for evidence categorization and
recommendation generation.

Example:
    >>> from greenlang.infrastructure.soc2_preparation.self_assessment import Assessor
    >>> from greenlang.infrastructure.soc2_preparation.config import get_config
    >>> config = get_config()
    >>> assessor = Assessor(config)
    >>> assessment = await assessor.run_assessment(
    ...     assessed_by=user_id,
    ...     categories=["security", "availability"]
    ... )
    >>> print(f"Overall score: {assessment.overall_score}%")

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol

from greenlang.infrastructure.soc2_preparation.config import SOC2Config
from greenlang.infrastructure.soc2_preparation.models import (
    Assessment,
    AssessmentCriteria,
    AssessmentStatus,
    ControlStatus,
    Evidence,
    ScoreLevel,
    TrustServiceCategory,
)
from greenlang.infrastructure.soc2_preparation.self_assessment.criteria import (
    TSC_CRITERIA,
    CATEGORY_WEIGHTS,
    get_criteria_by_category,
    CriterionDefinition,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Storage Protocol (for dependency injection)
# ---------------------------------------------------------------------------


class AssessmentStorage(Protocol):
    """Protocol for assessment storage backend."""

    async def save_assessment(self, assessment: Assessment) -> None:
        """Save an assessment to storage."""
        ...

    async def get_assessment(self, assessment_id: uuid.UUID) -> Optional[Assessment]:
        """Retrieve an assessment by ID."""
        ...

    async def get_evidence_for_criterion(
        self, criterion_id: str, tenant_id: Optional[uuid.UUID] = None
    ) -> List[Evidence]:
        """Get all evidence linked to a criterion."""
        ...


class InMemoryStorage:
    """In-memory storage implementation for testing."""

    def __init__(self) -> None:
        self._assessments: Dict[uuid.UUID, Assessment] = {}
        self._evidence: Dict[str, List[Evidence]] = {}

    async def save_assessment(self, assessment: Assessment) -> None:
        """Save assessment to in-memory storage."""
        self._assessments[assessment.id] = assessment

    async def get_assessment(self, assessment_id: uuid.UUID) -> Optional[Assessment]:
        """Retrieve assessment from in-memory storage."""
        return self._assessments.get(assessment_id)

    async def get_evidence_for_criterion(
        self, criterion_id: str, tenant_id: Optional[uuid.UUID] = None
    ) -> List[Evidence]:
        """Get evidence for a criterion."""
        return self._evidence.get(criterion_id, [])


# ---------------------------------------------------------------------------
# Assessor Class
# ---------------------------------------------------------------------------


class Assessor:
    """SOC 2 self-assessment assessor.

    Conducts comprehensive self-assessments against SOC 2 Trust Service
    Criteria. Evaluates each criterion, collects evidence references,
    calculates maturity scores, and produces assessment reports.

    The assessor follows a zero-hallucination principle:
    - Scores are calculated deterministically based on evidence
    - Control status is derived from evidence validation
    - No LLM is used for numeric score calculations

    Attributes:
        config: SOC 2 configuration.
        storage: Storage backend for assessments and evidence.
        enabled_categories: List of enabled TSC categories.

    Example:
        >>> assessor = Assessor(config)
        >>> assessment = await assessor.run_assessment(user_id)
        >>> print(assessment.overall_score)
    """

    def __init__(
        self,
        config: SOC2Config,
        storage: Optional[AssessmentStorage] = None,
    ) -> None:
        """Initialize the assessor.

        Args:
            config: SOC 2 configuration object.
            storage: Optional storage backend. Uses in-memory storage if not provided.
        """
        self.config = config
        self.storage = storage or InMemoryStorage()
        self.enabled_categories = config.enabled_tsc_categories
        logger.info(
            "Assessor initialized with categories: %s",
            self.enabled_categories,
        )

    async def run_assessment(
        self,
        assessed_by: uuid.UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
        categories: Optional[List[str]] = None,
        tenant_id: Optional[uuid.UUID] = None,
    ) -> Assessment:
        """Run a complete self-assessment.

        Evaluates all criteria for the specified categories, calculates
        scores, and produces a comprehensive assessment.

        Args:
            assessed_by: UUID of user initiating the assessment.
            name: Optional assessment name. Auto-generated if not provided.
            description: Optional assessment description.
            categories: Categories to assess. Uses config defaults if not specified.
            tenant_id: Optional tenant ID for multi-tenant isolation.

        Returns:
            Complete Assessment with all criteria evaluated.

        Example:
            >>> assessment = await assessor.run_assessment(
            ...     assessed_by=user_id,
            ...     name="Q1 2026 Readiness Assessment",
            ...     categories=["security", "availability"]
            ... )
        """
        start_time = datetime.now(timezone.utc)
        logger.info("Starting self-assessment for user %s", assessed_by)

        # Determine categories to assess
        assess_categories = categories or self.enabled_categories
        if not assess_categories:
            assess_categories = ["security"]  # Security is always required

        # Create assessment
        assessment = Assessment(
            name=name or f"SOC 2 Self-Assessment {start_time.strftime('%Y-%m-%d %H:%M')}",
            description=description or "",
            status=AssessmentStatus.IN_PROGRESS,
            tsc_categories=[TrustServiceCategory(cat) for cat in assess_categories],
            assessed_by=assessed_by,
        )

        # Assess each category
        all_criteria: List[AssessmentCriteria] = []
        total_score = Decimal("0")
        total_weight = Decimal("0")
        gaps_count = 0
        evidence_count = 0

        for category in assess_categories:
            category_criteria = get_criteria_by_category(category)
            category_weight = Decimal(str(CATEGORY_WEIGHTS.get(category, 1.0)))

            for criterion_id, criterion_def in category_criteria.items():
                # Assess individual criterion
                assessed_criterion = await self.assess_criterion(
                    criterion_id=criterion_id,
                    criterion_def=criterion_def,
                    assessment_id=assessment.id,
                    assessed_by=assessed_by,
                    tenant_id=tenant_id,
                )
                all_criteria.append(assessed_criterion)

                # Accumulate scores
                criterion_score = Decimal(str(assessed_criterion.score))
                total_score += criterion_score * category_weight
                total_weight += category_weight

                # Count gaps and evidence
                if assessed_criterion.gaps_identified:
                    gaps_count += 1
                evidence_count += assessed_criterion.evidence_count

        # Calculate overall score (0-100 scale)
        if total_weight > 0:
            # Score is 0-4, convert to 0-100
            raw_score = (total_score / total_weight) * Decimal("25")
            overall_score = min(raw_score, Decimal("100"))
        else:
            overall_score = Decimal("0")

        # Count compliant criteria (score >= 4)
        compliant_count = sum(
            1 for c in all_criteria if c.score >= ScoreLevel.COMPLIANT
        )

        # Update assessment
        assessment.criteria = all_criteria
        assessment.overall_score = overall_score.quantize(Decimal("0.01"))
        assessment.criteria_assessed = len(all_criteria)
        assessment.criteria_compliant = compliant_count
        assessment.gaps_count = gaps_count
        assessment.evidence_count = evidence_count
        assessment.status = AssessmentStatus.COMPLETED
        assessment.completed_at = datetime.now(timezone.utc)
        assessment.updated_at = datetime.now(timezone.utc)

        # Save assessment
        await self.save_assessment(assessment)

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        logger.info(
            "Assessment completed: id=%s, score=%.2f, criteria=%d, "
            "compliant=%d, gaps=%d, time=%.2fs",
            assessment.id,
            overall_score,
            len(all_criteria),
            compliant_count,
            gaps_count,
            processing_time,
        )

        return assessment

    async def assess_criterion(
        self,
        criterion_id: str,
        criterion_def: CriterionDefinition,
        assessment_id: uuid.UUID,
        assessed_by: uuid.UUID,
        tenant_id: Optional[uuid.UUID] = None,
    ) -> AssessmentCriteria:
        """Assess a single criterion.

        Collects evidence for the criterion, validates control implementation,
        and calculates the maturity score.

        Args:
            criterion_id: The criterion identifier (e.g., "CC6.1").
            criterion_def: The criterion definition from TSC_CRITERIA.
            assessment_id: Parent assessment UUID.
            assessed_by: UUID of assessor.
            tenant_id: Optional tenant ID.

        Returns:
            AssessmentCriteria with score and analysis.
        """
        logger.debug("Assessing criterion %s", criterion_id)

        # Collect evidence for this criterion
        evidence_items = await self.collect_criterion_evidence(
            criterion_id=criterion_id,
            tenant_id=tenant_id,
        )

        # Calculate score based on evidence
        score = self._calculate_criterion_score(
            criterion_id=criterion_id,
            criterion_def=criterion_def,
            evidence=evidence_items,
        )

        # Determine control status from score
        control_status = self._score_to_control_status(score)

        # Identify gaps
        gaps = self._identify_gaps(
            criterion_def=criterion_def,
            evidence=evidence_items,
            score=score,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            criterion_def=criterion_def,
            score=score,
            gaps=gaps,
        )

        return AssessmentCriteria(
            criterion_id=criterion_id,
            assessment_id=assessment_id,
            score=score,
            control_status=control_status,
            evidence_count=len(evidence_items),
            evidence_ids=[e.id for e in evidence_items],
            gaps_identified=gaps,
            recommendations=recommendations,
            notes="",
            assessed_by=assessed_by,
            assessed_at=datetime.now(timezone.utc),
        )

    async def collect_criterion_evidence(
        self,
        criterion_id: str,
        tenant_id: Optional[uuid.UUID] = None,
    ) -> List[Evidence]:
        """Collect all evidence linked to a criterion.

        Retrieves evidence from storage that is mapped to the specified
        criterion. Evidence may be manually uploaded or automatically
        collected.

        Args:
            criterion_id: The criterion identifier.
            tenant_id: Optional tenant ID for filtering.

        Returns:
            List of Evidence items for the criterion.
        """
        evidence = await self.storage.get_evidence_for_criterion(
            criterion_id=criterion_id,
            tenant_id=tenant_id,
        )
        logger.debug(
            "Collected %d evidence items for criterion %s",
            len(evidence),
            criterion_id,
        )
        return evidence

    def _calculate_criterion_score(
        self,
        criterion_id: str,
        criterion_def: CriterionDefinition,
        evidence: List[Evidence],
    ) -> ScoreLevel:
        """Calculate maturity score for a criterion.

        ZERO-HALLUCINATION: This method uses deterministic rules only.
        No LLM inference is used for score calculation.

        Scoring Logic:
        - 0 (NOT_IMPLEMENTED): No evidence
        - 1 (PARTIAL): <50% of required evidence types
        - 2 (IMPLEMENTED): >=50% of required evidence types
        - 3 (TESTED): >=75% of required evidence + test results
        - 4 (COMPLIANT): 100% of required evidence + test results + approval

        Args:
            criterion_id: The criterion identifier.
            criterion_def: The criterion definition.
            evidence: List of collected evidence.

        Returns:
            ScoreLevel indicating maturity.
        """
        if not evidence:
            return ScoreLevel.NOT_IMPLEMENTED

        required_evidence = criterion_def["evidence_requirements"]
        required_count = len(required_evidence)

        if required_count == 0:
            # No specific requirements, base on evidence existence
            return ScoreLevel.IMPLEMENTED if evidence else ScoreLevel.NOT_IMPLEMENTED

        # Calculate coverage
        evidence_count = len(evidence)
        coverage_ratio = evidence_count / required_count

        # Check for test results and verification
        has_test_results = any(
            e.evidence_type in ("report", "log", "metric")
            for e in evidence
        )
        has_verification = any(e.verified_by is not None for e in evidence)

        # Determine score
        if coverage_ratio < 0.5:
            return ScoreLevel.PARTIAL
        elif coverage_ratio < 0.75:
            return ScoreLevel.IMPLEMENTED
        elif coverage_ratio < 1.0 or not has_test_results:
            return ScoreLevel.TESTED
        elif has_verification:
            return ScoreLevel.COMPLIANT
        else:
            return ScoreLevel.TESTED

    def _score_to_control_status(self, score: ScoreLevel) -> ControlStatus:
        """Convert maturity score to control status.

        Args:
            score: The maturity score level.

        Returns:
            Corresponding control status.
        """
        mapping = {
            ScoreLevel.NOT_IMPLEMENTED: ControlStatus.NOT_STARTED,
            ScoreLevel.PARTIAL: ControlStatus.IN_IMPLEMENTATION,
            ScoreLevel.IMPLEMENTED: ControlStatus.IMPLEMENTED,
            ScoreLevel.TESTED: ControlStatus.IN_TESTING,
            ScoreLevel.COMPLIANT: ControlStatus.OPERATING,
        }
        return mapping.get(score, ControlStatus.NOT_STARTED)

    def _identify_gaps(
        self,
        criterion_def: CriterionDefinition,
        evidence: List[Evidence],
        score: ScoreLevel,
    ) -> str:
        """Identify gaps in criterion compliance.

        Args:
            criterion_def: The criterion definition.
            evidence: Collected evidence.
            score: Calculated maturity score.

        Returns:
            Description of identified gaps.
        """
        if score >= ScoreLevel.COMPLIANT:
            return ""

        gaps: List[str] = []

        # Check for missing evidence types
        required_evidence = criterion_def["evidence_requirements"]
        evidence_titles = {e.title.lower() for e in evidence}

        for req in required_evidence:
            # Simple keyword matching for gap identification
            req_lower = req.lower()
            if not any(kw in title for title in evidence_titles for kw in req_lower.split()):
                gaps.append(f"Missing evidence: {req}")

        # Check for control points
        control_points = criterion_def["control_points"]
        if score <= ScoreLevel.PARTIAL:
            gaps.append(
                f"Control implementation incomplete. Required control points: "
                f"{', '.join(control_points[:2])}..."
            )

        # Check for testing
        if score == ScoreLevel.IMPLEMENTED:
            gaps.append("Control testing required to advance maturity.")

        # Check for verification
        if score == ScoreLevel.TESTED:
            gaps.append("Evidence verification required for compliance.")

        return "\n".join(gaps[:5])  # Limit to 5 gaps

    def _generate_recommendations(
        self,
        criterion_def: CriterionDefinition,
        score: ScoreLevel,
        gaps: str,
    ) -> str:
        """Generate recommendations for improving compliance.

        Args:
            criterion_def: The criterion definition.
            score: Current maturity score.
            gaps: Identified gaps.

        Returns:
            Recommendations for improvement.
        """
        if score >= ScoreLevel.COMPLIANT:
            return "Control is compliant. Maintain current practices and monitoring."

        recommendations: List[str] = []

        if score == ScoreLevel.NOT_IMPLEMENTED:
            recommendations.append(
                f"Begin control implementation. Review common controls: "
                f"{', '.join(criterion_def['common_controls'])}."
            )
            recommendations.append(
                "Document policies and procedures for this control area."
            )

        elif score == ScoreLevel.PARTIAL:
            recommendations.append(
                "Complete control implementation and documentation."
            )
            recommendations.append(
                f"Collect required evidence: {', '.join(criterion_def['evidence_requirements'][:3])}."
            )

        elif score == ScoreLevel.IMPLEMENTED:
            recommendations.append(
                "Conduct control testing to validate operating effectiveness."
            )
            recommendations.append(
                "Document test procedures and maintain test results."
            )

        elif score == ScoreLevel.TESTED:
            recommendations.append(
                "Obtain management review and approval of evidence."
            )
            recommendations.append(
                "Ensure evidence completeness before audit."
            )

        # Add risk-based recommendation
        if criterion_def["risk_level"] in ("critical", "high"):
            recommendations.append(
                f"PRIORITY: This is a {criterion_def['risk_level']}-risk criterion. "
                f"Address gaps before audit fieldwork."
            )

        return "\n".join(recommendations)

    async def save_assessment(self, assessment: Assessment) -> None:
        """Save assessment to storage.

        Args:
            assessment: The assessment to save.
        """
        await self.storage.save_assessment(assessment)
        logger.info("Assessment saved: id=%s", assessment.id)

    def calculate_provenance_hash(self, assessment: Assessment) -> str:
        """Calculate SHA-256 hash for assessment audit trail.

        Args:
            assessment: The assessment to hash.

        Returns:
            SHA-256 hash string.
        """
        data = (
            f"{assessment.id}"
            f"{assessment.name}"
            f"{assessment.overall_score}"
            f"{assessment.criteria_assessed}"
            f"{assessment.completed_at.isoformat() if assessment.completed_at else ''}"
        )
        return hashlib.sha256(data.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------------------------


def create_assessor(
    config: Optional[SOC2Config] = None,
    storage: Optional[AssessmentStorage] = None,
) -> Assessor:
    """Create an Assessor instance.

    Factory function for creating Assessor with configuration.

    Args:
        config: Optional SOC2Config. Uses get_config() if not provided.
        storage: Optional storage backend.

    Returns:
        Configured Assessor instance.
    """
    from greenlang.infrastructure.soc2_preparation.config import get_config

    config = config or get_config()
    return Assessor(config=config, storage=storage)


__all__ = [
    "Assessor",
    "AssessmentStorage",
    "InMemoryStorage",
    "create_assessor",
]
