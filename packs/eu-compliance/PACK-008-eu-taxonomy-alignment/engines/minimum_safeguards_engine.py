"""
Minimum Safeguards Engine - PACK-008 EU Taxonomy Alignment

This module verifies compliance with the Minimum Safeguards requirement of the
EU Taxonomy Regulation (Article 18).  The four topics are drawn from the OECD
Guidelines for Multinational Enterprises and the UN Guiding Principles on
Business and Human Rights:

1. Human rights due diligence
2. Anti-corruption and anti-bribery
3. Taxation compliance
4. Fair competition

An activity is taxonomy-aligned only if all four topics pass.

Example:
    >>> engine = MinimumSafeguardsEngine()
    >>> result = engine.verify_safeguards({
    ...     "human_rights_policy": True,
    ...     "anti_corruption_policy": True,
    ...     "tax_compliance_statement": True,
    ...     "fair_competition_policy": True,
    ... })
    >>> print(f"MS pass: {result.overall_pass}")
"""

import hashlib
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SafeguardTopic(str, Enum):
    """The four Minimum Safeguards topics."""
    HUMAN_RIGHTS = "HUMAN_RIGHTS"
    ANTI_CORRUPTION = "ANTI_CORRUPTION"
    TAXATION = "TAXATION"
    FAIR_COMPETITION = "FAIR_COMPETITION"


TOPIC_NAMES: Dict[str, str] = {
    SafeguardTopic.HUMAN_RIGHTS.value: "Human Rights Due Diligence",
    SafeguardTopic.ANTI_CORRUPTION.value: "Anti-Corruption & Anti-Bribery",
    SafeguardTopic.TAXATION.value: "Taxation Compliance",
    SafeguardTopic.FAIR_COMPETITION.value: "Fair Competition",
}


class CheckCategory(str, Enum):
    """Category of safeguard check."""
    PROCEDURAL = "PROCEDURAL"   # Policy / process in place
    OUTCOME = "OUTCOME"          # No violations / sanctions observed


class TopicStatus(str, Enum):
    """Evaluation status for a single topic."""
    PASS = "PASS"
    FAIL = "FAIL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class SafeguardCheck(BaseModel):
    """A single procedural or outcome check within a topic."""

    check_id: str = Field(..., description="Unique check identifier")
    topic: SafeguardTopic = Field(..., description="Safeguard topic")
    category: CheckCategory = Field(..., description="PROCEDURAL or OUTCOME")
    metric_key: str = Field(..., description="Key expected in entity data")
    description: str = Field(..., description="Human-readable description")
    is_mandatory: bool = Field(default=True, description="Mandatory for pass")
    reference: str = Field(
        default="", description="Regulatory reference (OECD/UNGP/EU)"
    )


class CheckResult(BaseModel):
    """Result of evaluating a single safeguard check."""

    check_id: str = Field(..., description="Check identifier")
    topic: SafeguardTopic = Field(..., description="Topic")
    category: CheckCategory = Field(..., description="Check category")
    description: str = Field(..., description="Check description")
    is_met: bool = Field(..., description="Whether check is satisfied")
    has_data: bool = Field(default=True, description="Whether data was provided")
    actual_value: Optional[Any] = Field(None, description="Value provided")
    is_mandatory: bool = Field(default=True, description="Mandatory for pass")


class TopicResult(BaseModel):
    """Evaluation result for one Minimum Safeguards topic."""

    topic: SafeguardTopic = Field(..., description="Topic evaluated")
    topic_name: str = Field(..., description="Full topic name")
    status: TopicStatus = Field(..., description="Topic evaluation status")
    checks_total: int = Field(..., ge=0, description="Total checks")
    checks_passed: int = Field(..., ge=0, description="Checks passed")
    checks_failed: int = Field(..., ge=0, description="Checks failed")
    checks_no_data: int = Field(..., ge=0, description="Checks with no data")
    procedural_pass: bool = Field(
        ..., description="All mandatory procedural checks passed"
    )
    outcome_pass: bool = Field(
        ..., description="All mandatory outcome checks passed"
    )
    check_results: List[CheckResult] = Field(
        default_factory=list, description="Per-check detail"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )


class MSResult(BaseModel):
    """Complete Minimum Safeguards verification result."""

    overall_pass: bool = Field(
        ..., description="True only if all 4 topics pass"
    )
    topics_assessed: int = Field(default=4, description="Number of topics assessed")
    topics_passed: int = Field(..., ge=0, description="Topics passed")
    topics_failed: int = Field(..., ge=0, description="Topics failed")
    topics_no_data: int = Field(..., ge=0, description="Topics with insufficient data")
    failed_topics: List[str] = Field(
        default_factory=list, description="Names of failed topics"
    )
    topic_results: Dict[str, TopicResult] = Field(
        default_factory=dict, description="Per-topic results"
    )
    summary: str = Field(default="", description="Human-readable summary")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    assessed_at: datetime = Field(
        default_factory=datetime.utcnow, description="Assessment timestamp"
    )


# ---------------------------------------------------------------------------
# Check Definitions
# ---------------------------------------------------------------------------

SAFEGUARD_CHECKS: Dict[SafeguardTopic, List[SafeguardCheck]] = {
    # ====================================================================
    # 1. Human Rights
    # ====================================================================
    SafeguardTopic.HUMAN_RIGHTS: [
        # Procedural
        SafeguardCheck(
            check_id="HR-P01",
            topic=SafeguardTopic.HUMAN_RIGHTS,
            category=CheckCategory.PROCEDURAL,
            metric_key="human_rights_policy",
            description="Formal human rights policy publicly available",
            reference="UNGP Principle 16",
        ),
        SafeguardCheck(
            check_id="HR-P02",
            topic=SafeguardTopic.HUMAN_RIGHTS,
            category=CheckCategory.PROCEDURAL,
            metric_key="human_rights_due_diligence_process",
            description="Human rights due diligence process in place",
            reference="UNGP Principles 17-21; OECD GL Chapter IV",
        ),
        SafeguardCheck(
            check_id="HR-P03",
            topic=SafeguardTopic.HUMAN_RIGHTS,
            category=CheckCategory.PROCEDURAL,
            metric_key="grievance_mechanism",
            description="Operational-level grievance mechanism for affected stakeholders",
            reference="UNGP Principle 29",
        ),
        SafeguardCheck(
            check_id="HR-P04",
            topic=SafeguardTopic.HUMAN_RIGHTS,
            category=CheckCategory.PROCEDURAL,
            metric_key="human_rights_training",
            description="Regular human rights training for relevant staff",
            is_mandatory=False,
            reference="UNGP Principle 16(d)",
        ),
        # Outcome
        SafeguardCheck(
            check_id="HR-O01",
            topic=SafeguardTopic.HUMAN_RIGHTS,
            category=CheckCategory.OUTCOME,
            metric_key="no_human_rights_violations",
            description="No confirmed human rights violations in reporting period",
            reference="UNGP Principle 22",
        ),
        SafeguardCheck(
            check_id="HR-O02",
            topic=SafeguardTopic.HUMAN_RIGHTS,
            category=CheckCategory.OUTCOME,
            metric_key="no_forced_labour",
            description="No involvement in forced or child labour",
            reference="ILO Core Conventions",
        ),
    ],
    # ====================================================================
    # 2. Anti-Corruption
    # ====================================================================
    SafeguardTopic.ANTI_CORRUPTION: [
        SafeguardCheck(
            check_id="AC-P01",
            topic=SafeguardTopic.ANTI_CORRUPTION,
            category=CheckCategory.PROCEDURAL,
            metric_key="anti_corruption_policy",
            description="Anti-corruption and anti-bribery policy in place",
            reference="OECD GL Chapter VII",
        ),
        SafeguardCheck(
            check_id="AC-P02",
            topic=SafeguardTopic.ANTI_CORRUPTION,
            category=CheckCategory.PROCEDURAL,
            metric_key="anti_corruption_training",
            description="Anti-corruption training for employees and management",
            reference="OECD GL Chapter VII",
        ),
        SafeguardCheck(
            check_id="AC-P03",
            topic=SafeguardTopic.ANTI_CORRUPTION,
            category=CheckCategory.PROCEDURAL,
            metric_key="whistleblower_mechanism",
            description="Whistleblower protection mechanism in place",
            is_mandatory=False,
            reference="EU Whistleblower Directive 2019/1937",
        ),
        SafeguardCheck(
            check_id="AC-O01",
            topic=SafeguardTopic.ANTI_CORRUPTION,
            category=CheckCategory.OUTCOME,
            metric_key="no_corruption_convictions",
            description="No corruption or bribery convictions in reporting period",
            reference="OECD Anti-Bribery Convention",
        ),
        SafeguardCheck(
            check_id="AC-O02",
            topic=SafeguardTopic.ANTI_CORRUPTION,
            category=CheckCategory.OUTCOME,
            metric_key="no_ongoing_corruption_investigations",
            description="No ongoing regulatory investigations for corruption",
            is_mandatory=False,
            reference="OECD GL Chapter VII",
        ),
    ],
    # ====================================================================
    # 3. Taxation
    # ====================================================================
    SafeguardTopic.TAXATION: [
        SafeguardCheck(
            check_id="TX-P01",
            topic=SafeguardTopic.TAXATION,
            category=CheckCategory.PROCEDURAL,
            metric_key="tax_compliance_statement",
            description="Tax compliance policy or statement published",
            reference="OECD GL Chapter XI",
        ),
        SafeguardCheck(
            check_id="TX-P02",
            topic=SafeguardTopic.TAXATION,
            category=CheckCategory.PROCEDURAL,
            metric_key="country_by_country_reporting",
            description="Country-by-country reporting in place (if applicable)",
            is_mandatory=False,
            reference="OECD BEPS Action 13",
        ),
        SafeguardCheck(
            check_id="TX-P03",
            topic=SafeguardTopic.TAXATION,
            category=CheckCategory.PROCEDURAL,
            metric_key="tax_risk_management",
            description="Tax risk management and governance framework",
            reference="OECD GL Chapter XI",
        ),
        SafeguardCheck(
            check_id="TX-O01",
            topic=SafeguardTopic.TAXATION,
            category=CheckCategory.OUTCOME,
            metric_key="no_tax_evasion_convictions",
            description="No convictions for tax evasion in reporting period",
            reference="OECD GL Chapter XI",
        ),
        SafeguardCheck(
            check_id="TX-O02",
            topic=SafeguardTopic.TAXATION,
            category=CheckCategory.OUTCOME,
            metric_key="not_on_eu_tax_blacklist",
            description="Entity not domiciled in EU list of non-cooperative tax jurisdictions",
            reference="EU Council Conclusions on tax non-cooperative jurisdictions",
        ),
    ],
    # ====================================================================
    # 4. Fair Competition
    # ====================================================================
    SafeguardTopic.FAIR_COMPETITION: [
        SafeguardCheck(
            check_id="FC-P01",
            topic=SafeguardTopic.FAIR_COMPETITION,
            category=CheckCategory.PROCEDURAL,
            metric_key="fair_competition_policy",
            description="Fair competition / antitrust compliance policy in place",
            reference="OECD GL Chapter X",
        ),
        SafeguardCheck(
            check_id="FC-P02",
            topic=SafeguardTopic.FAIR_COMPETITION,
            category=CheckCategory.PROCEDURAL,
            metric_key="competition_compliance_training",
            description="Competition compliance training for relevant staff",
            is_mandatory=False,
            reference="OECD GL Chapter X",
        ),
        SafeguardCheck(
            check_id="FC-O01",
            topic=SafeguardTopic.FAIR_COMPETITION,
            category=CheckCategory.OUTCOME,
            metric_key="no_antitrust_violations",
            description="No antitrust / cartel convictions in reporting period",
            reference="TFEU Articles 101-102",
        ),
        SafeguardCheck(
            check_id="FC-O02",
            topic=SafeguardTopic.FAIR_COMPETITION,
            category=CheckCategory.OUTCOME,
            metric_key="no_ongoing_competition_investigations",
            description="No ongoing major competition authority investigations",
            is_mandatory=False,
            reference="OECD GL Chapter X",
        ),
    ],
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MinimumSafeguardsEngine:
    """
    Minimum Safeguards Engine for PACK-008 EU Taxonomy Alignment.

    This engine verifies that an entity complies with the four Minimum
    Safeguards topics required by Article 18 of the Taxonomy Regulation:
    human rights, anti-corruption, taxation, and fair competition.

    It follows GreenLang's zero-hallucination principle by performing only
    deterministic boolean checks against entity-reported data -- no LLM
    inference in the evaluation path.

    Attributes:
        checks: Check definitions grouped by topic

    Example:
        >>> engine = MinimumSafeguardsEngine()
        >>> result = engine.verify_safeguards({
        ...     "human_rights_policy": True,
        ...     "human_rights_due_diligence_process": True,
        ...     "grievance_mechanism": True,
        ...     "no_human_rights_violations": True,
        ...     "no_forced_labour": True,
        ...     "anti_corruption_policy": True,
        ...     "anti_corruption_training": True,
        ...     "no_corruption_convictions": True,
        ...     "tax_compliance_statement": True,
        ...     "tax_risk_management": True,
        ...     "no_tax_evasion_convictions": True,
        ...     "not_on_eu_tax_blacklist": True,
        ...     "fair_competition_policy": True,
        ...     "no_antitrust_violations": True,
        ... })
        >>> assert result.overall_pass is True
    """

    def __init__(self) -> None:
        """Initialize the Minimum Safeguards Engine."""
        self.checks = SAFEGUARD_CHECKS
        logger.info(
            "Initialized MinimumSafeguardsEngine with %d checks across %d topics",
            sum(len(v) for v in self.checks.values()),
            len(self.checks),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_safeguards(
        self,
        entity_data: Dict[str, Any],
    ) -> MSResult:
        """
        Verify all four Minimum Safeguards topics for an entity.

        Args:
            entity_data: Dictionary of boolean/metric values keyed by check
                         metric_key.

        Returns:
            MSResult with per-topic and overall pass/fail.

        Raises:
            ValueError: If entity_data is None.
        """
        if entity_data is None:
            raise ValueError("entity_data is required")

        start = datetime.utcnow()

        logger.info(
            "Verifying Minimum Safeguards with %d data points", len(entity_data)
        )

        topic_results: Dict[str, TopicResult] = {}
        passed_count = 0
        failed_count = 0
        no_data_count = 0
        failed_topics: List[str] = []

        for topic in SafeguardTopic:
            topic_result = self._evaluate_topic(topic, entity_data)
            topic_results[topic.value] = topic_result

            if topic_result.status == TopicStatus.PASS:
                passed_count += 1
            elif topic_result.status == TopicStatus.FAIL:
                failed_count += 1
                failed_topics.append(topic.value)
            else:
                no_data_count += 1

        overall_pass = (failed_count == 0 and no_data_count == 0)

        summary = self._build_summary(
            overall_pass, passed_count, failed_count, no_data_count, failed_topics
        )

        provenance_hash = self._hash(
            f"MS|{passed_count}|{failed_count}|{no_data_count}"
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000

        result = MSResult(
            overall_pass=overall_pass,
            topics_passed=passed_count,
            topics_failed=failed_count,
            topics_no_data=no_data_count,
            failed_topics=failed_topics,
            topic_results=topic_results,
            summary=summary,
            provenance_hash=provenance_hash,
        )

        logger.info(
            "MS verification: overall=%s passed=%d failed=%d no_data=%d in %.1fms",
            "PASS" if overall_pass else "FAIL",
            passed_count, failed_count, no_data_count, elapsed_ms,
        )

        return result

    def check_human_rights(
        self,
        data: Dict[str, Any],
    ) -> TopicResult:
        """
        Evaluate the Human Rights topic only.

        Args:
            data: Entity data relevant to human rights.

        Returns:
            TopicResult for human rights.
        """
        return self._evaluate_topic(SafeguardTopic.HUMAN_RIGHTS, data)

    def check_anti_corruption(
        self,
        data: Dict[str, Any],
    ) -> TopicResult:
        """
        Evaluate the Anti-Corruption topic only.

        Args:
            data: Entity data relevant to anti-corruption.

        Returns:
            TopicResult for anti-corruption.
        """
        return self._evaluate_topic(SafeguardTopic.ANTI_CORRUPTION, data)

    def check_taxation(
        self,
        data: Dict[str, Any],
    ) -> TopicResult:
        """
        Evaluate the Taxation topic only.

        Args:
            data: Entity data relevant to taxation.

        Returns:
            TopicResult for taxation.
        """
        return self._evaluate_topic(SafeguardTopic.TAXATION, data)

    def check_fair_competition(
        self,
        data: Dict[str, Any],
    ) -> TopicResult:
        """
        Evaluate the Fair Competition topic only.

        Args:
            data: Entity data relevant to fair competition.

        Returns:
            TopicResult for fair competition.
        """
        return self._evaluate_topic(SafeguardTopic.FAIR_COMPETITION, data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate_topic(
        self,
        topic: SafeguardTopic,
        data: Dict[str, Any],
    ) -> TopicResult:
        """
        Evaluate a single Minimum Safeguards topic.

        Args:
            topic: Safeguard topic to evaluate.
            data: Entity data.

        Returns:
            TopicResult with check-level details.
        """
        checks = self.checks.get(topic, [])
        check_results: List[CheckResult] = []
        passed = 0
        failed = 0
        no_data = 0

        for check in checks:
            value = data.get(check.metric_key)

            if value is None:
                cr = CheckResult(
                    check_id=check.check_id,
                    topic=check.topic,
                    category=check.category,
                    description=check.description,
                    is_met=False,
                    has_data=False,
                    actual_value=None,
                    is_mandatory=check.is_mandatory,
                )
                no_data += 1
            else:
                is_met = bool(value)
                cr = CheckResult(
                    check_id=check.check_id,
                    topic=check.topic,
                    category=check.category,
                    description=check.description,
                    is_met=is_met,
                    has_data=True,
                    actual_value=value,
                    is_mandatory=check.is_mandatory,
                )
                if is_met:
                    passed += 1
                else:
                    failed += 1

            check_results.append(cr)

        # Determine procedural and outcome pass
        procedural_pass = all(
            cr.is_met
            for cr in check_results
            if cr.category == CheckCategory.PROCEDURAL and cr.is_mandatory and cr.has_data
        )
        outcome_pass = all(
            cr.is_met
            for cr in check_results
            if cr.category == CheckCategory.OUTCOME and cr.is_mandatory and cr.has_data
        )

        # Mandatory check failure => topic fails
        mandatory_failed = any(
            not cr.is_met and cr.is_mandatory and cr.has_data
            for cr in check_results
        )
        mandatory_no_data = any(
            not cr.has_data and cr.is_mandatory
            for cr in check_results
        )

        if mandatory_failed:
            status = TopicStatus.FAIL
        elif mandatory_no_data:
            status = TopicStatus.INSUFFICIENT_DATA
        else:
            status = TopicStatus.PASS

        # Build recommendations
        recommendations = self._build_recommendations(topic, check_results)

        return TopicResult(
            topic=topic,
            topic_name=TOPIC_NAMES.get(topic.value, topic.value),
            status=status,
            checks_total=len(checks),
            checks_passed=passed,
            checks_failed=failed,
            checks_no_data=no_data,
            procedural_pass=procedural_pass,
            outcome_pass=outcome_pass,
            check_results=check_results,
            recommendations=recommendations,
        )

    @staticmethod
    def _build_recommendations(
        topic: SafeguardTopic,
        results: List[CheckResult],
    ) -> List[str]:
        """Generate recommendations for failed or missing checks."""
        recs: List[str] = []
        for cr in results:
            if not cr.is_met and cr.has_data and cr.is_mandatory:
                recs.append(f"[{topic.value}] Address: {cr.description}")
            elif not cr.has_data and cr.is_mandatory:
                recs.append(f"[{topic.value}] Provide data for: {cr.description}")
        return recs

    @staticmethod
    def _build_summary(
        overall_pass: bool,
        passed: int,
        failed: int,
        no_data: int,
        failed_topics: List[str],
    ) -> str:
        """Build a human-readable summary string."""
        if overall_pass:
            return (
                f"Minimum Safeguards PASSED: all {passed} topics verified. "
                "Entity meets OECD Guidelines and UN Guiding Principles requirements."
            )

        parts: List[str] = []
        if failed > 0:
            parts.append(f"{failed} topic(s) failed ({', '.join(failed_topics)})")
        if no_data > 0:
            parts.append(f"{no_data} topic(s) have insufficient data")

        return (
            f"Minimum Safeguards NOT PASSED: {'; '.join(parts)}. "
            f"{passed}/4 topics verified."
        )

    @staticmethod
    def _hash(data: str) -> str:
        """Return SHA-256 hex digest."""
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
