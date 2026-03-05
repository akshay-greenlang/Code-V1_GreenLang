"""
Minimum Safeguards Engine -- Company-Level 4-Topic Assessment

Implements the fourth step of the EU Taxonomy alignment pipeline: verifying
that a company meets the Minimum Safeguards (Article 18) across four topic
areas: human rights, anti-corruption/bribery, taxation, and fair competition.

The assessment follows the Platform on Sustainable Finance (PSF) Final Report
on Minimum Safeguards (October 2022), which distinguishes between procedural
checks (policies, processes, mechanisms) and outcome checks (no court
rulings/convictions for violations).

Key capabilities:
  - Full 4-topic Minimum Safeguards assessment
  - Per-topic procedural and outcome evaluation
  - Adverse finding recording and tracking
  - Safeguard history and audit trail
  - Summary statistics and compliance scoring

All assessments are deterministic (zero-hallucination).

Reference:
    - Regulation (EU) 2020/852, Article 18
    - Platform on Sustainable Finance: Final Report on Minimum Safeguards (Oct 2022)
    - UN Guiding Principles on Business and Human Rights (UNGPs)
    - OECD Guidelines for Multinational Enterprises (2011, updated 2023)
    - ILO Declaration on Fundamental Principles and Rights at Work
    - UN Convention against Corruption (UNCAC)
    - OECD Anti-Bribery Convention
    - EU Anti-Tax Avoidance Directives (ATAD I & II)

Example:
    >>> engine = MinimumSafeguardsEngine(config)
    >>> result = engine.assess_safeguards("org-1")
    >>> result.all_pass
    True
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    SafeguardTopic,
    TaxonomyAppConfig,
    MINIMUM_SAFEGUARD_TOPICS,
)
from .models import (
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Procedural Check Definitions
# ---------------------------------------------------------------------------

_PROCEDURAL_CHECKS: Dict[str, List[Dict[str, str]]] = {
    "human_rights": [
        {
            "id": "HR-P1",
            "name": "Human rights policy commitment",
            "description": (
                "Formal publicly available policy commitment to respect human "
                "rights, approved by senior management, consistent with UNGPs "
                "Principle 16."
            ),
            "framework": "UNGPs Principle 16",
        },
        {
            "id": "HR-P2",
            "name": "Human rights due diligence process",
            "description": (
                "Ongoing human rights due diligence process that identifies, "
                "prevents, mitigates, and accounts for adverse impacts across "
                "own operations and value chain (UNGPs Principle 17-21)."
            ),
            "framework": "UNGPs Principle 17-21",
        },
        {
            "id": "HR-P3",
            "name": "Operational-level grievance mechanism",
            "description": (
                "Operational-level grievance mechanism accessible to affected "
                "stakeholders, meeting effectiveness criteria (UNGPs Principle 29-31)."
            ),
            "framework": "UNGPs Principle 29-31",
        },
        {
            "id": "HR-P4",
            "name": "Remediation process",
            "description": (
                "Process to provide for or cooperate in remediation of adverse "
                "human rights impacts through legitimate channels (UNGPs Principle 22)."
            ),
            "framework": "UNGPs Principle 22",
        },
    ],
    "anti_corruption": [
        {
            "id": "AC-P1",
            "name": "Anti-corruption policy",
            "description": (
                "Formal anti-bribery and anti-corruption policy covering all "
                "forms of bribery and corruption, consistent with OECD "
                "Guidelines Chapter VII."
            ),
            "framework": "OECD Guidelines Chapter VII",
        },
        {
            "id": "AC-P2",
            "name": "Anti-corruption management system",
            "description": (
                "Internal controls, ethics and compliance programme for "
                "preventing and detecting corruption, aligned with ISO 37001."
            ),
            "framework": "ISO 37001, OECD Guidelines",
        },
        {
            "id": "AC-P3",
            "name": "Anti-corruption training",
            "description": (
                "Regular anti-corruption training for employees, agents, "
                "and business partners in high-risk roles."
            ),
            "framework": "OECD Good Practice Guidance",
        },
        {
            "id": "AC-P4",
            "name": "Whistleblower protection",
            "description": (
                "Protected reporting channels for suspected corruption, "
                "consistent with EU Whistleblower Directive 2019/1937."
            ),
            "framework": "EU Directive 2019/1937",
        },
    ],
    "taxation": [
        {
            "id": "TX-P1",
            "name": "Tax policy commitment",
            "description": (
                "Board-approved commitment to responsible tax practices, "
                "tax compliance, and transparency in all jurisdictions."
            ),
            "framework": "OECD BEPS, GRI 207",
        },
        {
            "id": "TX-P2",
            "name": "Tax governance and oversight",
            "description": (
                "Board or senior management oversight of tax strategy with "
                "clear accountability and risk management."
            ),
            "framework": "GRI 207-1, OECD BEPS",
        },
        {
            "id": "TX-P3",
            "name": "Country-by-country reporting",
            "description": (
                "Public or regulatory country-by-country reporting of tax "
                "payments, revenue, and profit per jurisdiction (DAC6/CbCR)."
            ),
            "framework": "EU DAC6, OECD CbCR",
        },
        {
            "id": "TX-P4",
            "name": "No aggressive tax planning",
            "description": (
                "No use of aggressive tax structures, tax havens, or "
                "artificial arrangements to reduce tax burden."
            ),
            "framework": "EU ATAD I & II",
        },
    ],
    "fair_competition": [
        {
            "id": "FC-P1",
            "name": "Competition compliance policy",
            "description": (
                "Formal antitrust and fair competition policy, consistent "
                "with OECD Guidelines Chapter X."
            ),
            "framework": "OECD Guidelines Chapter X",
        },
        {
            "id": "FC-P2",
            "name": "Competition compliance programme",
            "description": (
                "Compliance programme with training, monitoring, and "
                "reporting mechanisms for competition law adherence."
            ),
            "framework": "EU Competition Law",
        },
        {
            "id": "FC-P3",
            "name": "Competition risk assessment",
            "description": (
                "Periodic assessment of anticompetitive conduct risk, "
                "including cartels, abuse of dominance, and state aid."
            ),
            "framework": "OECD Recommendation on Competition Compliance",
        },
        {
            "id": "FC-P4",
            "name": "Cartel prevention measures",
            "description": (
                "Specific measures to prevent price-fixing, bid-rigging, "
                "market allocation, and output limitation agreements."
            ),
            "framework": "TFEU Articles 101-102",
        },
    ],
}


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------

class ProceduralResult(BaseModel):
    """Result of procedural checks for a Minimum Safeguards topic."""

    topic: str = Field(...)
    total_checks: int = Field(default=0)
    passed_checks: int = Field(default=0)
    failed_checks: int = Field(default=0)
    check_details: List[Dict[str, Any]] = Field(default_factory=list)
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    all_pass: bool = Field(default=False)


class OutcomeResult(BaseModel):
    """Result of outcome checks for a Minimum Safeguards topic."""

    topic: str = Field(...)
    adverse_findings_count: int = Field(default=0)
    has_court_rulings: bool = Field(default=False)
    has_convictions: bool = Field(default=False)
    has_settlements: bool = Field(default=False)
    finding_details: List[Dict[str, Any]] = Field(default_factory=list)
    outcome_pass: bool = Field(default=True, description="True if no adverse outcomes found")
    score: float = Field(default=100.0, ge=0.0, le=100.0)


class TopicResult(BaseModel):
    """Combined assessment result for a single Minimum Safeguards topic."""

    topic: str = Field(...)
    procedural: ProceduralResult = Field(...)
    outcome: OutcomeResult = Field(...)
    topic_pass: bool = Field(default=False)
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    notes: str = Field(default="")
    provenance_hash: str = Field(default="")


class SafeguardResult(BaseModel):
    """Full 4-topic Minimum Safeguards assessment result."""

    assessment_id: str = Field(default_factory=_new_id)
    org_id: str = Field(...)
    all_pass: bool = Field(default=False)
    topic_results: Dict[str, TopicResult] = Field(default_factory=dict)
    failed_topics: List[str] = Field(default_factory=list)
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    assessed_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


class SafeguardSummary(BaseModel):
    """Summary of Minimum Safeguards assessments for an organization."""

    org_id: str = Field(...)
    total_assessments: int = Field(default=0)
    latest_pass: bool = Field(default=False)
    latest_score: float = Field(default=0.0)
    topic_pass_rates: Dict[str, float] = Field(default_factory=dict)
    adverse_findings_total: int = Field(default=0)
    weakest_topic: Optional[str] = Field(None)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# MinimumSafeguardsEngine
# ---------------------------------------------------------------------------

class MinimumSafeguardsEngine:
    """
    Minimum Safeguards Engine for EU Taxonomy Article 18 compliance.

    Assesses company-level compliance across four topic areas per the
    Platform on Sustainable Finance guidance: human rights, anti-corruption,
    taxation, and fair competition.  Each topic has procedural checks
    (policies, processes, mechanisms) and outcome checks (no court
    rulings/convictions).

    Attributes:
        config: Application configuration.
        _org_data: In-memory store of organization MS data keyed by org_id.
        _assessments: In-memory store keyed by assessment_id.
        _adverse_findings: Adverse findings keyed by org_id.
        _history: Assessment history keyed by org_id.

    Example:
        >>> engine = MinimumSafeguardsEngine(config)
        >>> result = engine.assess_safeguards("org-1")
        >>> result.all_pass
        True
    """

    # Topic weights for overall score (equal weighting)
    TOPIC_WEIGHT: float = 0.25

    # Procedural vs outcome score mix within each topic
    PROCEDURAL_WEIGHT: float = 0.60
    OUTCOME_WEIGHT: float = 0.40

    # Pass thresholds
    PROCEDURAL_PASS_THRESHOLD: float = 75.0
    OVERALL_PASS_THRESHOLD: float = 70.0

    # All four topics
    TOPICS: List[str] = [
        "human_rights",
        "anti_corruption",
        "taxation",
        "fair_competition",
    ]

    def __init__(self, config: Optional[TaxonomyAppConfig] = None) -> None:
        """
        Initialize MinimumSafeguardsEngine.

        Args:
            config: Application configuration instance.
        """
        self.config = config or TaxonomyAppConfig()
        self._org_data: Dict[str, Dict[str, Any]] = {}
        self._assessments: Dict[str, SafeguardResult] = {}
        self._adverse_findings: Dict[str, List[Dict[str, Any]]] = {}
        self._history: Dict[str, List[SafeguardResult]] = {}
        logger.info("MinimumSafeguardsEngine initialized")

    # ------------------------------------------------------------------
    # Full 4-Topic Assessment
    # ------------------------------------------------------------------

    def assess_safeguards(self, org_id: str) -> SafeguardResult:
        """
        Perform full 4-topic Minimum Safeguards assessment.

        Evaluates human rights, anti-corruption, taxation, and fair
        competition for both procedural and outcome compliance.

        Args:
            org_id: Organization identifier.

        Returns:
            SafeguardResult with per-topic and overall pass/fail.
        """
        start = datetime.utcnow()
        org_data = self._org_data.get(org_id, {})

        topic_results: Dict[str, TopicResult] = {}
        failed_topics: List[str] = []
        total_score = 0.0

        for topic in self.TOPICS:
            topic_data = org_data.get(topic, {})
            topic_result = self._assess_topic(org_id, topic, topic_data)
            topic_results[topic] = topic_result

            if not topic_result.topic_pass:
                failed_topics.append(topic)

            total_score += topic_result.score * self.TOPIC_WEIGHT

        all_pass = len(failed_topics) == 0 and total_score >= self.OVERALL_PASS_THRESHOLD

        provenance = _sha256(
            f"ms_assess:{org_id}:{all_pass}:{total_score:.2f}:{len(failed_topics)}"
        )

        result = SafeguardResult(
            org_id=org_id,
            all_pass=all_pass,
            topic_results=topic_results,
            failed_topics=failed_topics,
            overall_score=round(total_score, 2),
            provenance_hash=provenance,
        )

        self._assessments[result.assessment_id] = result
        self._history.setdefault(org_id, []).append(result)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "MS assessment for org %s: all_pass=%s, score=%.1f, failed=%s in %.1f ms",
            org_id, all_pass, total_score, failed_topics, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Per-Topic Assessment
    # ------------------------------------------------------------------

    def assess_human_rights(
        self,
        org_id: str,
        due_diligence_data: Dict[str, Any],
    ) -> TopicResult:
        """
        Assess human rights Minimum Safeguards.

        Evaluates UNGPs alignment: policy commitment, HRDD process,
        grievance mechanism, and remediation.

        Args:
            org_id: Organization identifier.
            due_diligence_data: Evidence data with procedural check flags.

        Returns:
            TopicResult for human rights.
        """
        self._org_data.setdefault(org_id, {})["human_rights"] = due_diligence_data
        return self._assess_topic(org_id, "human_rights", due_diligence_data)

    def assess_anti_corruption(
        self,
        org_id: str,
        compliance_data: Dict[str, Any],
    ) -> TopicResult:
        """
        Assess anti-corruption Minimum Safeguards.

        Evaluates anti-bribery policy, management system, training,
        and whistleblower protection.

        Args:
            org_id: Organization identifier.
            compliance_data: Evidence data with compliance check flags.

        Returns:
            TopicResult for anti-corruption.
        """
        self._org_data.setdefault(org_id, {})["anti_corruption"] = compliance_data
        return self._assess_topic(org_id, "anti_corruption", compliance_data)

    def assess_taxation(
        self,
        org_id: str,
        tax_data: Dict[str, Any],
    ) -> TopicResult:
        """
        Assess taxation Minimum Safeguards.

        Evaluates tax policy, governance, CbCR compliance, and absence
        of aggressive tax planning.

        Args:
            org_id: Organization identifier.
            tax_data: Evidence data with tax compliance flags.

        Returns:
            TopicResult for taxation.
        """
        self._org_data.setdefault(org_id, {})["taxation"] = tax_data
        return self._assess_topic(org_id, "taxation", tax_data)

    def assess_fair_competition(
        self,
        org_id: str,
        competition_data: Dict[str, Any],
    ) -> TopicResult:
        """
        Assess fair competition Minimum Safeguards.

        Evaluates competition compliance policy, programme, risk
        assessment, and cartel prevention.

        Args:
            org_id: Organization identifier.
            competition_data: Evidence data with competition law flags.

        Returns:
            TopicResult for fair competition.
        """
        self._org_data.setdefault(org_id, {})["fair_competition"] = competition_data
        return self._assess_topic(org_id, "fair_competition", competition_data)

    # ------------------------------------------------------------------
    # Procedural Checks
    # ------------------------------------------------------------------

    def check_procedural(
        self,
        org_id: str,
        topic: str,
        checks: Dict[str, bool],
    ) -> ProceduralResult:
        """
        Evaluate procedural checks for a given topic.

        Args:
            org_id: Organization identifier.
            topic: MS topic name (human_rights, anti_corruption, etc.).
            checks: Dict mapping check IDs to pass/fail booleans.

        Returns:
            ProceduralResult with per-check and aggregate scores.
        """
        proc_checks = _PROCEDURAL_CHECKS.get(topic, [])
        if not proc_checks:
            return ProceduralResult(
                topic=topic, total_checks=0, passed_checks=0,
                score=0.0, all_pass=False,
            )

        details: List[Dict[str, Any]] = []
        passed = 0

        for check in proc_checks:
            check_id = check["id"]
            check_passed = checks.get(check_id, False)
            if check_passed:
                passed += 1

            details.append({
                "check_id": check_id,
                "name": check["name"],
                "passed": check_passed,
                "framework": check.get("framework", ""),
                "description": check["description"],
            })

        total = len(proc_checks)
        score = (passed / total * 100.0) if total > 0 else 0.0
        all_pass_proc = score >= self.PROCEDURAL_PASS_THRESHOLD

        return ProceduralResult(
            topic=topic,
            total_checks=total,
            passed_checks=passed,
            failed_checks=total - passed,
            check_details=details,
            score=round(score, 2),
            all_pass=all_pass_proc,
        )

    # ------------------------------------------------------------------
    # Outcome Checks
    # ------------------------------------------------------------------

    def check_outcome(
        self,
        org_id: str,
        topic: str,
        findings: Dict[str, Any],
    ) -> OutcomeResult:
        """
        Evaluate outcome checks for a given topic.

        Checks for court rulings, convictions, and settlements related
        to the topic area.

        Args:
            org_id: Organization identifier.
            topic: MS topic name.
            findings: Dict with keys: has_court_rulings, has_convictions,
                has_settlements, details (list).

        Returns:
            OutcomeResult with adverse finding assessment.
        """
        has_court = findings.get("has_court_rulings", False)
        has_convictions = findings.get("has_convictions", False)
        has_settlements = findings.get("has_settlements", False)
        finding_details = findings.get("details", [])

        adverse_count = len(finding_details)

        # Outcome scoring: 100 if clean, reduced for findings
        score = 100.0
        if has_convictions:
            score -= 50.0
        if has_court:
            score -= 30.0
        if has_settlements:
            score -= 20.0
        score = max(score, 0.0)

        outcome_pass = not has_convictions and not has_court

        return OutcomeResult(
            topic=topic,
            adverse_findings_count=adverse_count,
            has_court_rulings=has_court,
            has_convictions=has_convictions,
            has_settlements=has_settlements,
            finding_details=finding_details,
            outcome_pass=outcome_pass,
            score=round(score, 2),
        )

    # ------------------------------------------------------------------
    # Adverse Finding Recording
    # ------------------------------------------------------------------

    def record_adverse_finding(
        self,
        org_id: str,
        topic: str,
        finding_type: str,
        description: str,
        date: str = "",
    ) -> str:
        """
        Record an adverse finding for an organization.

        Args:
            org_id: Organization identifier.
            topic: MS topic name.
            finding_type: Type of finding (court_ruling, conviction, settlement,
                investigation, complaint).
            description: Human-readable finding description.
            date: Date of finding (ISO format string).

        Returns:
            Finding record ID.
        """
        finding_id = _new_id()
        record = {
            "finding_id": finding_id,
            "org_id": org_id,
            "topic": topic,
            "finding_type": finding_type,
            "description": description,
            "date": date or _now().isoformat(),
            "recorded_at": _now().isoformat(),
        }

        self._adverse_findings.setdefault(org_id, []).append(record)

        logger.info(
            "Recorded adverse finding %s for org %s (topic=%s, type=%s)",
            finding_id, org_id, topic, finding_type,
        )
        return finding_id

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_safeguard_history(self, org_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve Minimum Safeguards assessment history for an organization.

        Args:
            org_id: Organization identifier.

        Returns:
            List of assessment summary records.
        """
        assessments = self._history.get(org_id, [])
        results: List[Dict[str, Any]] = []

        for a in assessments:
            results.append({
                "assessment_id": a.assessment_id,
                "all_pass": a.all_pass,
                "overall_score": a.overall_score,
                "failed_topics": a.failed_topics,
                "assessed_at": a.assessed_at.isoformat(),
                "provenance_hash": a.provenance_hash,
            })

        return results

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_safeguard_summary(self, org_id: str) -> SafeguardSummary:
        """
        Get Minimum Safeguards summary for an organization.

        Args:
            org_id: Organization identifier.

        Returns:
            SafeguardSummary with aggregate statistics.
        """
        assessments = self._history.get(org_id, [])
        findings = self._adverse_findings.get(org_id, [])

        total = len(assessments)
        latest_pass = assessments[-1].all_pass if assessments else False
        latest_score = assessments[-1].overall_score if assessments else 0.0

        # Pass rates per topic across all assessments
        topic_pass_counts: Dict[str, int] = {t: 0 for t in self.TOPICS}
        topic_totals: Dict[str, int] = {t: 0 for t in self.TOPICS}

        for a in assessments:
            for topic in self.TOPICS:
                topic_totals[topic] += 1
                if topic in a.topic_results and a.topic_results[topic].topic_pass:
                    topic_pass_counts[topic] += 1

        topic_pass_rates: Dict[str, float] = {}
        for topic in self.TOPICS:
            total_for_topic = topic_totals[topic]
            if total_for_topic > 0:
                topic_pass_rates[topic] = round(
                    topic_pass_counts[topic] / total_for_topic, 4,
                )
            else:
                topic_pass_rates[topic] = 0.0

        # Weakest topic
        weakest = None
        if topic_pass_rates:
            weakest = min(topic_pass_rates, key=topic_pass_rates.get)

        provenance = _sha256(f"ms_summary:{org_id}:{total}:{latest_score}")

        return SafeguardSummary(
            org_id=org_id,
            total_assessments=total,
            latest_pass=latest_pass,
            latest_score=latest_score,
            topic_pass_rates=topic_pass_rates,
            adverse_findings_total=len(findings),
            weakest_topic=weakest,
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _assess_topic(
        self,
        org_id: str,
        topic: str,
        topic_data: Dict[str, Any],
    ) -> TopicResult:
        """
        Internal assessment of a single MS topic.

        Combines procedural and outcome checks into a weighted topic score.

        Args:
            org_id: Organization identifier.
            topic: MS topic name.
            topic_data: Evidence data for the topic.

        Returns:
            TopicResult with combined assessment.
        """
        # Build procedural check dict from evidence
        proc_checks: Dict[str, bool] = {}
        for check in _PROCEDURAL_CHECKS.get(topic, []):
            check_id = check["id"]
            # Look for the check ID directly or a matching flag
            proc_checks[check_id] = topic_data.get(check_id, False)

        # Also map common flag names to check IDs
        flag_mappings = self._get_flag_mappings(topic)
        for flag_name, check_id in flag_mappings.items():
            if flag_name in topic_data and topic_data[flag_name]:
                proc_checks[check_id] = True

        procedural = self.check_procedural(org_id, topic, proc_checks)

        # Outcome checks
        findings = self._adverse_findings.get(org_id, [])
        topic_findings = [f for f in findings if f["topic"] == topic]

        outcome_data = {
            "has_court_rulings": any(
                f["finding_type"] == "court_ruling" for f in topic_findings
            ),
            "has_convictions": any(
                f["finding_type"] == "conviction" for f in topic_findings
            ),
            "has_settlements": any(
                f["finding_type"] == "settlement" for f in topic_findings
            ),
            "details": topic_findings,
        }
        outcome = self.check_outcome(org_id, topic, outcome_data)

        # Combined score
        combined_score = (
            procedural.score * self.PROCEDURAL_WEIGHT
            + outcome.score * self.OUTCOME_WEIGHT
        )

        topic_pass = (
            procedural.all_pass
            and outcome.outcome_pass
            and combined_score >= self.OVERALL_PASS_THRESHOLD
        )

        provenance = _sha256(
            f"ms_topic:{org_id}:{topic}:{topic_pass}:{combined_score:.2f}"
        )

        return TopicResult(
            topic=topic,
            procedural=procedural,
            outcome=outcome,
            topic_pass=topic_pass,
            score=round(combined_score, 2),
            provenance_hash=provenance,
        )

    def _get_flag_mappings(self, topic: str) -> Dict[str, str]:
        """
        Map common evidence flag names to procedural check IDs.

        Returns a dict of {flag_name: check_id} for the given topic.
        """
        mappings: Dict[str, Dict[str, str]] = {
            "human_rights": {
                "policy_commitment": "HR-P1",
                "hrdd_process": "HR-P2",
                "due_diligence": "HR-P2",
                "grievance_mechanism": "HR-P3",
                "remediation_process": "HR-P4",
                "remediation": "HR-P4",
            },
            "anti_corruption": {
                "anti_corruption_policy": "AC-P1",
                "policy": "AC-P1",
                "management_system": "AC-P2",
                "compliance_programme": "AC-P2",
                "training": "AC-P3",
                "anti_corruption_training": "AC-P3",
                "whistleblower_protection": "AC-P4",
                "whistleblower": "AC-P4",
            },
            "taxation": {
                "tax_policy": "TX-P1",
                "policy": "TX-P1",
                "tax_governance": "TX-P2",
                "governance": "TX-P2",
                "cbcr_compliant": "TX-P3",
                "country_by_country": "TX-P3",
                "no_aggressive_planning": "TX-P4",
                "tax_transparency": "TX-P4",
            },
            "fair_competition": {
                "competition_policy": "FC-P1",
                "policy": "FC-P1",
                "compliance_programme": "FC-P2",
                "programme": "FC-P2",
                "risk_assessment": "FC-P3",
                "cartel_prevention": "FC-P4",
            },
        }
        return mappings.get(topic, {})
