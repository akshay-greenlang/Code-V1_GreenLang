"""
Governance Engine -- TCFD Pillar 1: Governance Assessment

Implements the TCFD Governance recommended disclosures:
  - Governance (a): Board's oversight of climate-related risks and opportunities
  - Governance (b): Management's role in assessing and managing climate-related
    risks and opportunities

Evaluates organizational governance maturity across 8 dimensions:
  1. Board Oversight         -- Board-level climate governance structures
  2. Management Roles        -- Dedicated management positions
  3. Climate Competency      -- Board/management climate knowledge
  4. Meeting Frequency       -- Regularity of climate-focused meetings
  5. Reporting Structure     -- Climate reporting to the board
  6. Incentive Alignment     -- Remuneration linked to climate targets
  7. Risk Integration        -- Climate in enterprise risk management
  8. Strategy Integration    -- Climate in strategic planning

Each dimension is scored 1-5 (initial to optimized).  The weighted average
determines the overall maturity level per the TCFD implementation guidance.

Reference:
    - TCFD Final Report, Section C: Governance (June 2017)
    - TCFD Annex: Implementing the Recommendations, Table 1
    - IFRS S2 Paragraphs 5-6 (Governance)

Example:
    >>> engine = GovernanceEngine(config)
    >>> assessment = await engine.assess_governance("org-1", data)
    >>> assessment.overall_maturity
    <GovernanceMaturityLevel.DEFINED: 'defined'>
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    GovernanceMaturityLevel,
    MATURITY_SCORES,
    PILLAR_NAMES,
    SectorType,
    TCFDAppConfig,
    TCFDPillar,
    TCFD_DISCLOSURES,
)
from .models import (
    GovernanceAssessment,
    ManagementRole,
    CreateGovernanceAssessmentRequest,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Governance Maturity Dimension Descriptors
# ---------------------------------------------------------------------------

_MATURITY_DESCRIPTORS: Dict[str, Dict[int, str]] = {
    "board_oversight": {
        1: "No board-level climate governance",
        2: "Ad-hoc board discussion of climate topics",
        3: "Designated board committee considers climate",
        4: "Regular board review of climate strategy and metrics",
        5: "Climate fully integrated into board agenda with KPIs",
    },
    "management_roles": {
        1: "No dedicated climate management roles",
        2: "Part-time climate responsibility assigned",
        3: "Dedicated sustainability/climate officer appointed",
        4: "Cross-functional climate team with clear mandates",
        5: "C-suite climate leadership with enterprise-wide authority",
    },
    "climate_competency": {
        1: "No climate expertise on board or management",
        2: "Basic awareness through ad-hoc training",
        3: "Some board members have climate expertise",
        4: "Structured competency development programme",
        5: "Deep expertise with external advisory support",
    },
    "meeting_frequency": {
        1: "No climate-specific meetings",
        2: "Annual climate review",
        3: "Semi-annual climate reviews",
        4: "Quarterly climate reviews",
        5: "Monthly or more frequent climate updates",
    },
    "reporting_structure": {
        1: "No climate reporting to board",
        2: "Annual sustainability report to board",
        3: "Semi-annual climate dashboard to board",
        4: "Quarterly climate KPI reporting with escalation",
        5: "Real-time climate dashboard with automated alerts",
    },
    "incentive_alignment": {
        1: "No climate-linked remuneration",
        2: "Qualitative climate goals in assessment",
        3: "Climate KPIs in short-term incentive plan",
        4: "Climate metrics in both STI and LTI plans",
        5: "Material portion (>20%) of exec comp linked to climate",
    },
    "risk_integration": {
        1: "Climate not in risk register",
        2: "Climate mentioned in risk register (qualitative)",
        3: "Climate risks scored and tracked in ERM",
        4: "Climate scenario analysis feeds into ERM",
        5: "Climate fully integrated into enterprise risk framework",
    },
    "strategy_integration": {
        1: "Climate not in strategic plan",
        2: "Climate mentioned in sustainability policy",
        3: "Climate considered in strategic planning cycle",
        4: "Climate scenarios inform capital allocation",
        5: "Climate-adjusted strategy with transition plan",
    },
}

_GOVERNANCE_DIMENSIONS: List[str] = [
    "board_oversight", "management_roles", "climate_competency",
    "meeting_frequency", "reporting_structure", "incentive_alignment",
    "risk_integration", "strategy_integration",
]

# ---------------------------------------------------------------------------
# Sector Peer Benchmarks (average maturity scores by sector)
# ---------------------------------------------------------------------------

_SECTOR_BENCHMARKS: Dict[SectorType, Dict[str, Decimal]] = {
    SectorType.ENERGY: {
        "avg_overall": Decimal("3.2"), "avg_board_oversight": Decimal("3.5"),
        "avg_incentive": Decimal("3.0"), "peer_count": Decimal("145"),
    },
    SectorType.BANKING: {
        "avg_overall": Decimal("3.4"), "avg_board_oversight": Decimal("3.8"),
        "avg_incentive": Decimal("3.2"), "peer_count": Decimal("120"),
    },
    SectorType.TRANSPORTATION: {
        "avg_overall": Decimal("2.8"), "avg_board_oversight": Decimal("3.0"),
        "avg_incentive": Decimal("2.5"), "peer_count": Decimal("85"),
    },
    SectorType.MATERIALS_BUILDINGS: {
        "avg_overall": Decimal("2.9"), "avg_board_oversight": Decimal("3.1"),
        "avg_incentive": Decimal("2.6"), "peer_count": Decimal("90"),
    },
    SectorType.AGRICULTURE_FOOD_FOREST: {
        "avg_overall": Decimal("2.4"), "avg_board_oversight": Decimal("2.5"),
        "avg_incentive": Decimal("2.0"), "peer_count": Decimal("60"),
    },
    SectorType.INSURANCE: {
        "avg_overall": Decimal("3.3"), "avg_board_oversight": Decimal("3.6"),
        "avg_incentive": Decimal("3.0"), "peer_count": Decimal("65"),
    },
    SectorType.ASSET_MANAGERS: {
        "avg_overall": Decimal("3.1"), "avg_board_oversight": Decimal("3.4"),
        "avg_incentive": Decimal("2.8"), "peer_count": Decimal("95"),
    },
    SectorType.ASSET_OWNERS: {
        "avg_overall": Decimal("3.0"), "avg_board_oversight": Decimal("3.2"),
        "avg_incentive": Decimal("2.7"), "peer_count": Decimal("80"),
    },
    SectorType.CONSUMER_GOODS: {
        "avg_overall": Decimal("2.6"), "avg_board_oversight": Decimal("2.8"),
        "avg_incentive": Decimal("2.3"), "peer_count": Decimal("110"),
    },
    SectorType.TECHNOLOGY_MEDIA: {
        "avg_overall": Decimal("2.9"), "avg_board_oversight": Decimal("3.0"),
        "avg_incentive": Decimal("2.7"), "peer_count": Decimal("130"),
    },
    SectorType.HEALTHCARE: {
        "avg_overall": Decimal("2.3"), "avg_board_oversight": Decimal("2.4"),
        "avg_incentive": Decimal("2.0"), "peer_count": Decimal("70"),
    },
}


class GovernanceEngine:
    """
    TCFD Pillar 1: Governance assessment engine.

    Evaluates organizational governance maturity across 8 dimensions,
    manages governance roles, tracks board competency, and generates
    Governance (a) and (b) disclosure text.

    Attributes:
        config: Application configuration.
        _assessments: In-memory store keyed by org_id.
        _roles: In-memory store of governance roles keyed by org_id.
    """

    def __init__(self, config: Optional[TCFDAppConfig] = None) -> None:
        """
        Initialize GovernanceEngine.

        Args:
            config: Application configuration.
        """
        self.config = config or TCFDAppConfig()
        self._assessments: Dict[str, List[GovernanceAssessment]] = {}
        self._roles: Dict[str, List[ManagementRole]] = {}
        logger.info("GovernanceEngine initialized")

    # ------------------------------------------------------------------
    # Governance Assessment
    # ------------------------------------------------------------------

    async def assess_governance(
        self,
        org_id: str,
        assessment_data: CreateGovernanceAssessmentRequest,
    ) -> GovernanceAssessment:
        """
        Perform a governance maturity assessment.

        Args:
            org_id: Organization ID.
            assessment_data: Assessment input data.

        Returns:
            GovernanceAssessment with computed maturity scores.
        """
        start = datetime.utcnow()

        maturity_scores = self._compute_dimension_scores(assessment_data)
        overall = self._compute_overall_maturity(maturity_scores)

        assessment = GovernanceAssessment(
            tenant_id="default",
            org_id=org_id,
            assessment_date=date.today(),
            board_oversight_score=assessment_data.board_oversight_score,
            board_committees=assessment_data.board_committees,
            meeting_frequency=assessment_data.meeting_frequency,
            climate_competency_score=assessment_data.climate_competency_score,
            incentive_linkage=assessment_data.incentive_linkage,
            incentive_pct=assessment_data.incentive_pct,
            maturity_scores=maturity_scores,
            overall_maturity=overall,
            notes=assessment_data.notes,
        )

        if org_id not in self._assessments:
            self._assessments[org_id] = []
        self._assessments[org_id].append(assessment)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Governance assessment for org %s: overall=%s in %.1f ms",
            org_id, overall.value, elapsed_ms,
        )
        return assessment

    async def update_governance(
        self,
        org_id: str,
        assessment_id: str,
        updates: Dict[str, Any],
    ) -> GovernanceAssessment:
        """
        Update an existing governance assessment.

        Args:
            org_id: Organization ID.
            assessment_id: Assessment ID to update.
            updates: Field updates to apply.

        Returns:
            Updated GovernanceAssessment.

        Raises:
            ValueError: If assessment not found.
        """
        assessments = self._assessments.get(org_id, [])
        target = None
        for a in assessments:
            if a.id == assessment_id:
                target = a
                break

        if target is None:
            raise ValueError(f"Assessment {assessment_id} not found for org {org_id}")

        data = target.model_dump()
        data.update(updates)
        data["updated_at"] = _now()

        if "maturity_scores" not in updates:
            req = CreateGovernanceAssessmentRequest(**{
                k: data[k] for k in CreateGovernanceAssessmentRequest.model_fields
                if k in data
            })
            data["maturity_scores"] = self._compute_dimension_scores(req)
            data["overall_maturity"] = self._compute_overall_maturity(data["maturity_scores"])

        updated = GovernanceAssessment(**data)

        self._assessments[org_id] = [
            updated if a.id == assessment_id else a
            for a in self._assessments[org_id]
        ]

        logger.info("Updated governance assessment %s for org %s", assessment_id, org_id)
        return updated

    async def get_governance_history(
        self,
        org_id: str,
    ) -> List[GovernanceAssessment]:
        """
        Retrieve governance assessment history for an organization.

        Args:
            org_id: Organization ID.

        Returns:
            List of GovernanceAssessment sorted by date.
        """
        assessments = self._assessments.get(org_id, [])
        return sorted(assessments, key=lambda a: a.assessment_date)

    async def calculate_maturity_score(
        self,
        assessment: GovernanceAssessment,
    ) -> Dict[str, int]:
        """
        Recalculate maturity scores for all 8 dimensions.

        Args:
            assessment: Existing assessment to recalculate.

        Returns:
            Dict of dimension -> score (1-5).
        """
        req = CreateGovernanceAssessmentRequest(
            board_oversight_score=assessment.board_oversight_score,
            board_committees=assessment.board_committees,
            meeting_frequency=assessment.meeting_frequency,
            climate_competency_score=assessment.climate_competency_score,
            incentive_linkage=assessment.incentive_linkage,
            incentive_pct=assessment.incentive_pct,
        )
        return self._compute_dimension_scores(req)

    # ------------------------------------------------------------------
    # Role Management
    # ------------------------------------------------------------------

    async def manage_roles(
        self,
        org_id: str,
        roles: List[Dict[str, Any]],
    ) -> List[ManagementRole]:
        """
        Create or update governance roles for an organization.

        Args:
            org_id: Organization ID.
            roles: List of role data dicts.

        Returns:
            List of ManagementRole objects.
        """
        created_roles: List[ManagementRole] = []
        for role_data in roles:
            role = ManagementRole(tenant_id="default", org_id=org_id, **role_data)
            created_roles.append(role)

        self._roles[org_id] = created_roles
        logger.info("Managed %d governance roles for org %s", len(created_roles), org_id)
        return created_roles

    # ------------------------------------------------------------------
    # Board Competency Assessment
    # ------------------------------------------------------------------

    async def assess_board_competency(
        self,
        org_id: str,
        skills_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Assess board-level climate competency.

        Evaluates climate knowledge, experience, and training across
        board members to produce a competency profile.

        Args:
            org_id: Organization ID.
            skills_data: Dict with keys: board_size, climate_experts,
                training_hours, certifications, external_advisors.

        Returns:
            Dict with competency score, gaps, and recommendations.
        """
        board_size = skills_data.get("board_size", 1)
        climate_experts = skills_data.get("climate_experts", 0)
        training_hours = skills_data.get("training_hours", 0)
        certifications = skills_data.get("certifications", 0)
        external_advisors = skills_data.get("external_advisors", 0)

        expert_ratio = climate_experts / max(board_size, 1)
        training_score = min(training_hours / 20, 5)
        cert_score = min(certifications / 3, 5)
        advisor_score = min(external_advisors * 2, 5)

        composite = (expert_ratio * 5 + training_score + cert_score + advisor_score) / 4
        composite = min(round(composite, 1), 5.0)

        gaps: List[str] = []
        if expert_ratio < 0.2:
            gaps.append("Less than 20% of board has climate expertise")
        if training_hours < 10:
            gaps.append("Average training below 10 hours per director")
        if certifications == 0:
            gaps.append("No board members hold climate-related certifications")

        recommendations: List[str] = []
        if gaps:
            recommendations.append("Consider climate competency training programme for board")
            if expert_ratio < 0.2:
                recommendations.append("Recruit board member with climate/ESG expertise")
            if external_advisors == 0:
                recommendations.append("Engage external climate advisory panel")

        logger.info(
            "Board competency assessment for org %s: score=%.1f, gaps=%d",
            org_id, composite, len(gaps),
        )

        return {
            "org_id": org_id,
            "competency_score": composite,
            "expert_ratio": round(expert_ratio, 2),
            "training_score": round(training_score, 1),
            "certification_score": round(cert_score, 1),
            "advisory_score": round(advisor_score, 1),
            "gaps": gaps,
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------
    # Incentive Linkage Tracking
    # ------------------------------------------------------------------

    async def track_incentive_linkage(
        self,
        org_id: str,
        incentive_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Track climate-linked remuneration for executives.

        Args:
            org_id: Organization ID.
            incentive_data: Dict with keys: executives_total, linked_count,
                sti_pct, lti_pct, kpi_examples.

        Returns:
            Dict with incentive coverage analysis.
        """
        executives_total = incentive_data.get("executives_total", 1)
        linked_count = incentive_data.get("linked_count", 0)
        sti_pct = Decimal(str(incentive_data.get("sti_pct", 0)))
        lti_pct = Decimal(str(incentive_data.get("lti_pct", 0)))
        kpi_examples = incentive_data.get("kpi_examples", [])

        coverage_ratio = linked_count / max(executives_total, 1)
        total_linkage_pct = sti_pct + lti_pct

        maturity_score = 1
        if total_linkage_pct > Decimal("0"):
            maturity_score = 2
        if total_linkage_pct >= Decimal("5"):
            maturity_score = 3
        if total_linkage_pct >= Decimal("10") and lti_pct > Decimal("0"):
            maturity_score = 4
        if total_linkage_pct >= Decimal("20") and coverage_ratio >= 0.8:
            maturity_score = 5

        logger.info(
            "Incentive linkage for org %s: coverage=%.0f%%, linkage=%.1f%%, maturity=%d",
            org_id, coverage_ratio * 100, total_linkage_pct, maturity_score,
        )

        return {
            "org_id": org_id,
            "executives_total": executives_total,
            "linked_count": linked_count,
            "coverage_ratio": round(coverage_ratio, 2),
            "sti_pct": str(sti_pct),
            "lti_pct": str(lti_pct),
            "total_linkage_pct": str(total_linkage_pct),
            "kpi_examples": kpi_examples,
            "maturity_score": maturity_score,
        }

    # ------------------------------------------------------------------
    # Disclosure Generation
    # ------------------------------------------------------------------

    async def generate_governance_disclosure(
        self,
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Generate TCFD Governance (a) and (b) disclosure text.

        Pulls from the latest assessment and governance roles to
        produce structured disclosure content.

        Args:
            org_id: Organization ID.
            year: Reporting year.

        Returns:
            Dict with gov_a and gov_b disclosure sections.
        """
        assessments = self._assessments.get(org_id, [])
        latest = assessments[-1] if assessments else None
        roles = self._roles.get(org_id, [])

        gov_a_content = self._build_gov_a_text(latest, roles)
        gov_b_content = self._build_gov_b_text(latest, roles)

        return {
            "org_id": org_id,
            "reporting_year": year,
            "gov_a": {
                "ref": "Governance (a)",
                "title": "Board Oversight",
                "content": gov_a_content,
                "compliance_score": self._score_disclosure("gov_a", latest),
            },
            "gov_b": {
                "ref": "Governance (b)",
                "title": "Management Role",
                "content": gov_b_content,
                "compliance_score": self._score_disclosure("gov_b", latest),
            },
            "maturity": latest.overall_maturity.value if latest else "initial",
        }

    # ------------------------------------------------------------------
    # Peer Benchmarking
    # ------------------------------------------------------------------

    async def benchmark_governance(
        self,
        org_id: str,
        sector: SectorType,
    ) -> Dict[str, Any]:
        """
        Benchmark governance maturity against sector peers.

        Args:
            org_id: Organization ID.
            sector: Sector for peer comparison.

        Returns:
            Dict with benchmark comparison.
        """
        assessments = self._assessments.get(org_id, [])
        latest = assessments[-1] if assessments else None

        benchmark = _SECTOR_BENCHMARKS.get(sector, {})
        peer_avg = benchmark.get("avg_overall", Decimal("2.5"))
        peer_count = benchmark.get("peer_count", Decimal("50"))

        org_score = Decimal("1")
        if latest and latest.maturity_scores:
            weights = self.config.maturity_weights
            weighted_sum = Decimal("0")
            total_weight = Decimal("0")
            for dim, score in latest.maturity_scores.items():
                w = weights.get(dim, Decimal("0.125"))
                weighted_sum += Decimal(str(score)) * w
                total_weight += w
            if total_weight > 0:
                org_score = (weighted_sum / total_weight).quantize(
                    Decimal("0.1"), rounding=ROUND_HALF_UP,
                )

        percentile = int(min(max((org_score / Decimal("5")) * 100, 0), 100))

        above_average = org_score > peer_avg

        logger.info(
            "Benchmark for org %s in %s: score=%.1f vs peer_avg=%.1f, percentile=%d",
            org_id, sector.value, org_score, peer_avg, percentile,
        )

        return {
            "org_id": org_id,
            "sector": sector.value,
            "org_maturity_score": str(org_score),
            "peer_average_score": str(peer_avg),
            "peer_count": str(peer_count),
            "above_average": above_average,
            "estimated_percentile": percentile,
            "dimension_comparison": self._compare_dimensions(latest, benchmark),
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _compute_dimension_scores(
        self,
        data: CreateGovernanceAssessmentRequest,
    ) -> Dict[str, int]:
        """Compute all 8 maturity dimension scores from assessment input."""
        scores: Dict[str, int] = {}

        scores["board_oversight"] = data.board_oversight_score

        scores["management_roles"] = 1

        scores["climate_competency"] = data.climate_competency_score

        freq = data.meeting_frequency
        if freq == 0:
            scores["meeting_frequency"] = 1
        elif freq <= 1:
            scores["meeting_frequency"] = 2
        elif freq <= 2:
            scores["meeting_frequency"] = 3
        elif freq <= 4:
            scores["meeting_frequency"] = 4
        else:
            scores["meeting_frequency"] = 5

        committees = len(data.board_committees)
        if committees == 0:
            scores["reporting_structure"] = 1
        elif committees == 1:
            scores["reporting_structure"] = 3
        else:
            scores["reporting_structure"] = min(committees + 2, 5)

        if not data.incentive_linkage:
            scores["incentive_alignment"] = 1
        elif data.incentive_pct < Decimal("5"):
            scores["incentive_alignment"] = 2
        elif data.incentive_pct < Decimal("10"):
            scores["incentive_alignment"] = 3
        elif data.incentive_pct < Decimal("20"):
            scores["incentive_alignment"] = 4
        else:
            scores["incentive_alignment"] = 5

        scores["risk_integration"] = min(data.board_oversight_score, data.climate_competency_score)
        scores["strategy_integration"] = min(
            data.board_oversight_score,
            max(data.climate_competency_score - 1, 1),
        )

        return scores

    def _compute_overall_maturity(self, scores: Dict[str, int]) -> GovernanceMaturityLevel:
        """Compute overall maturity from weighted dimension scores."""
        weights = self.config.maturity_weights
        weighted_sum = Decimal("0")
        total_weight = Decimal("0")

        for dim in _GOVERNANCE_DIMENSIONS:
            score = scores.get(dim, 1)
            w = weights.get(dim, Decimal("0.125"))
            weighted_sum += Decimal(str(score)) * w
            total_weight += w

        if total_weight > 0:
            avg = weighted_sum / total_weight
        else:
            avg = Decimal("1")

        if avg >= Decimal("4.5"):
            return GovernanceMaturityLevel.OPTIMIZED
        elif avg >= Decimal("3.5"):
            return GovernanceMaturityLevel.MANAGED
        elif avg >= Decimal("2.5"):
            return GovernanceMaturityLevel.DEFINED
        elif avg >= Decimal("1.5"):
            return GovernanceMaturityLevel.DEVELOPING
        else:
            return GovernanceMaturityLevel.INITIAL

    @staticmethod
    def _build_gov_a_text(
        assessment: Optional[GovernanceAssessment],
        roles: List[ManagementRole],
    ) -> str:
        """Build Governance (a) disclosure text."""
        if assessment is None:
            return "Governance assessment not yet completed."

        parts: List[str] = []
        parts.append(
            f"The board exercises oversight of climate-related risks and opportunities "
            f"through {len(assessment.board_committees)} dedicated committee(s)"
        )
        if assessment.board_committees:
            parts.append(f" ({', '.join(assessment.board_committees)})")
        parts.append(".")

        parts.append(
            f" Climate matters are reviewed in {assessment.meeting_frequency} "
            f"board meeting(s) per year."
        )

        if assessment.incentive_linkage:
            parts.append(
                f" Executive remuneration is linked to climate performance, "
                f"with {assessment.incentive_pct}% of compensation tied to "
                f"climate-related KPIs."
            )

        return "".join(parts)

    @staticmethod
    def _build_gov_b_text(
        assessment: Optional[GovernanceAssessment],
        roles: List[ManagementRole],
    ) -> str:
        """Build Governance (b) disclosure text."""
        if assessment is None:
            return "Management role assessment not yet completed."

        parts: List[str] = []
        if roles:
            accountable = [r for r in roles if r.climate_accountability]
            parts.append(
                f"The organization has {len(roles)} management role(s) with "
                f"climate responsibilities, of which {len(accountable)} have "
                f"explicit climate accountability."
            )
            for role in accountable[:3]:
                parts.append(
                    f" The {role.role_title} is responsible for "
                    f"{role.responsibility_description[:200]}."
                )
        else:
            parts.append(
                "Management roles with climate responsibilities "
                "are being formalized."
            )

        maturity = assessment.overall_maturity.value if assessment else "initial"
        parts.append(
            f" The organization's governance maturity is assessed as '{maturity}'."
        )
        return "".join(parts)

    @staticmethod
    def _score_disclosure(
        disclosure_ref: str,
        assessment: Optional[GovernanceAssessment],
    ) -> int:
        """Score a disclosure section (0-100)."""
        if assessment is None:
            return 0

        base = 20
        if assessment.board_committees:
            base += 15
        if assessment.meeting_frequency > 0:
            base += 10
        if assessment.incentive_linkage:
            base += 15
        if assessment.climate_competency_score >= 3:
            base += 15
        if assessment.board_oversight_score >= 3:
            base += 15
        if assessment.maturity_scores:
            base += 10

        return min(base, 100)

    @staticmethod
    def _compare_dimensions(
        assessment: Optional[GovernanceAssessment],
        benchmark: Dict[str, Decimal],
    ) -> Dict[str, str]:
        """Compare org dimensions to sector benchmark."""
        if assessment is None or not assessment.maturity_scores:
            return {}

        comparison: Dict[str, str] = {}
        for dim, score in assessment.maturity_scores.items():
            bench_key = f"avg_{dim}"
            peer_val = benchmark.get(bench_key, Decimal("2.5"))
            if Decimal(str(score)) > peer_val:
                comparison[dim] = "above_average"
            elif Decimal(str(score)) == peer_val:
                comparison[dim] = "average"
            else:
                comparison[dim] = "below_average"
        return comparison
