"""
Risk Management Engine -- TCFD Pillar 3: Risk Management Processes

Implements the TCFD Risk Management recommended disclosures:
  - RM (a): Risk identification and assessment processes
  - RM (b): Risk management processes
  - RM (c): Integration into overall risk management (ERM)

Provides:
  - Risk register creation and lifecycle management
  - 5x5 risk matrix scoring (likelihood x impact = 1..25)
  - Risk rating derivation (low / medium / high / critical)
  - Risk prioritization with weighted scoring
  - Risk response planning and tracking
  - Key Risk Indicator (KRI) monitoring with thresholds
  - Enterprise Risk Management (ERM) integration assessment
  - Risk register review and update workflow
  - Risk heat map data generation
  - Risk management summary statistics
  - RM (a/b/c) disclosure content generation

All calculations are deterministic (zero-hallucination).

Reference:
    - TCFD Final Report, Section E: Risk Management (June 2017)
    - TCFD Annex: Implementing the Recommendations, Table 4
    - IFRS S2 Paragraph 25 (Risk Management)
    - ISO 31000:2018 Risk Management Guidelines
    - COSO ERM Framework (2017)

Example:
    >>> engine = RiskManagementEngine(config)
    >>> register = await engine.create_risk_register("org-1")
    >>> entry = await engine.add_risk_entry("org-1", risk_data)
    >>> heat_map = await engine.get_risk_heat_map("org-1")
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    DEFAULT_RISK_MATRIX,
    IMPACT_SCORES,
    LIKELIHOOD_SCORES,
    RISK_MATRIX_THRESHOLDS,
    RiskImpact,
    RiskLikelihood,
    RiskResponse,
    RiskType,
    TCFDAppConfig,
    TCFDPillar,
    TimeHorizon,
)
from .models import (
    ClimateRisk,
    CreateRiskManagementRecordRequest,
    ERMIntegration,
    RiskAssessment,
    RiskIndicator,
    RiskManagementRecord,
    RiskRegisterEntry,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RISK_MATRIX_LABELS: Dict[int, Dict[int, str]] = {
    1: {1: "low", 2: "low", 3: "low", 4: "medium", 5: "medium"},
    2: {1: "low", 2: "medium", 3: "medium", 4: "high", 5: "high"},
    3: {1: "low", 2: "medium", 3: "medium", 4: "high", 5: "high"},
    4: {1: "medium", 2: "high", 3: "high", 4: "critical", 5: "critical"},
    5: {1: "medium", 2: "high", 3: "high", 4: "critical", 5: "critical"},
}

RISK_COLORS: Dict[str, str] = {
    "low": "#22c55e",
    "medium": "#f59e0b",
    "high": "#f97316",
    "critical": "#ef4444",
}

RISK_VELOCITY_MAP: Dict[str, int] = {
    "immediate": 5,
    "fast": 4,
    "medium": 3,
    "slow": 2,
    "gradual": 1,
}

RISK_PERSISTENCE_MAP: Dict[str, int] = {
    "temporary": 1,
    "short_term": 2,
    "medium_term": 3,
    "long_term": 4,
    "permanent": 5,
}


class RiskManagementEngine:
    """
    TCFD Pillar 3: Risk Management engine covering RM (a/b/c) disclosures.

    Manages the climate risk register, performs 5x5 matrix assessments,
    tracks Key Risk Indicators (KRIs), evaluates ERM integration,
    and generates disclosure content.

    Attributes:
        config: Application configuration.
        _registers: In-memory risk register keyed by org_id.
        _records: In-memory risk management records keyed by org_id.
        _indicators: In-memory KRI store keyed by org_id.
        _erm: In-memory ERM integration store keyed by org_id.
    """

    def __init__(self, config: Optional[TCFDAppConfig] = None) -> None:
        """Initialize RiskManagementEngine."""
        self.config = config or TCFDAppConfig()
        self._registers: Dict[str, List[RiskRegisterEntry]] = {}
        self._records: Dict[str, List[RiskManagementRecord]] = {}
        self._indicators: Dict[str, List[RiskIndicator]] = {}
        self._erm: Dict[str, ERMIntegration] = {}
        logger.info("RiskManagementEngine initialized")

    # ------------------------------------------------------------------
    # Risk Register Management -- RM (a)
    # ------------------------------------------------------------------

    async def create_risk_register(self, org_id: str) -> Dict[str, Any]:
        """
        Create or reset the risk register for an organization.

        Args:
            org_id: Organization ID.

        Returns:
            Dict with register creation confirmation.
        """
        self._registers[org_id] = []
        self._records[org_id] = []
        self._indicators[org_id] = []
        logger.info("Created risk register for org %s", org_id)
        return {
            "org_id": org_id,
            "status": "created",
            "entries": 0,
            "created_at": _now().isoformat(),
        }

    async def add_risk_entry(
        self,
        org_id: str,
        risk: ClimateRisk,
        response: RiskResponse = RiskResponse.MITIGATE,
        response_actions: Optional[List[str]] = None,
        owner: Optional[str] = None,
        review_frequency: str = "quarterly",
    ) -> RiskRegisterEntry:
        """
        Add a climate risk to the risk register with assessment.

        Args:
            org_id: Organization ID.
            risk: Climate risk to register.
            response: Risk response strategy.
            response_actions: Planned response actions.
            owner: Risk owner.
            review_frequency: How often to review.

        Returns:
            Created RiskRegisterEntry.
        """
        start = datetime.utcnow()

        likelihood_score = LIKELIHOOD_SCORES.get(risk.likelihood, 3)
        impact_score = IMPACT_SCORES.get(risk.impact, 3)

        assessment = RiskAssessment(
            tenant_id="default",
            risk_id=risk.id,
            likelihood=risk.likelihood,
            likelihood_score=likelihood_score,
            impact=risk.impact,
            impact_score=impact_score,
        )

        next_review = self._calculate_next_review(review_frequency)

        entry = RiskRegisterEntry(
            tenant_id="default",
            org_id=org_id,
            risk_id=risk.id,
            risk_name=risk.name,
            risk_type=risk.risk_type,
            assessment=assessment,
            response=response,
            response_actions=response_actions or [],
            owner=owner,
            review_frequency=review_frequency,
            next_review_date=next_review,
            status="active",
        )

        if org_id not in self._registers:
            self._registers[org_id] = []
        self._registers[org_id].append(entry)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Added risk '%s' to register for org %s, score=%d (%s) in %.1f ms",
            risk.name, org_id, assessment.risk_score,
            assessment.risk_rating, elapsed_ms,
        )
        return entry

    async def update_risk_entry(
        self,
        org_id: str,
        entry_id: str,
        updates: Dict[str, Any],
    ) -> RiskRegisterEntry:
        """
        Update an existing risk register entry.

        Args:
            org_id: Organization ID.
            entry_id: Register entry ID.
            updates: Field updates to apply.

        Returns:
            Updated RiskRegisterEntry.

        Raises:
            ValueError: If entry not found.
        """
        entries = self._registers.get(org_id, [])
        for i, entry in enumerate(entries):
            if entry.id == entry_id:
                data = entry.model_dump()
                data.update(updates)
                data["updated_at"] = _now()
                updated = RiskRegisterEntry(**data)
                self._registers[org_id][i] = updated
                logger.info("Updated risk entry %s for org %s", entry_id, org_id)
                return updated
        raise ValueError(f"Risk register entry {entry_id} not found")

    async def list_risk_register(
        self,
        org_id: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RiskRegisterEntry]:
        """
        List risk register entries with optional filters.

        Args:
            org_id: Organization ID.
            filters: Optional filters (risk_type, status, rating).

        Returns:
            Filtered list of register entries.
        """
        entries = list(self._registers.get(org_id, []))

        if filters:
            if "risk_type" in filters:
                rt = RiskType(filters["risk_type"])
                entries = [e for e in entries if e.risk_type == rt]
            if "status" in filters:
                entries = [e for e in entries if e.status == filters["status"]]
            if "rating" in filters:
                entries = [
                    e for e in entries
                    if e.assessment and e.assessment.risk_rating == filters["rating"]
                ]
            if "owner" in filters:
                entries = [e for e in entries if e.owner == filters["owner"]]

        return entries

    # ------------------------------------------------------------------
    # Risk Assessment -- RM (a) continued
    # ------------------------------------------------------------------

    async def assess_risk(
        self,
        org_id: str,
        risk_id: str,
        likelihood: RiskLikelihood,
        impact: RiskImpact,
        velocity: str = "medium",
        persistence: str = "medium",
        assessed_by: Optional[str] = None,
    ) -> RiskAssessment:
        """
        Assess a specific risk using the 5x5 matrix.

        Args:
            org_id: Organization ID.
            risk_id: Climate risk ID.
            likelihood: Likelihood rating.
            impact: Impact rating.
            velocity: Speed of onset.
            persistence: Duration of impact.
            assessed_by: Assessor name or role.

        Returns:
            RiskAssessment with computed score and rating.
        """
        likelihood_score = LIKELIHOOD_SCORES.get(likelihood, 3)
        impact_score = IMPACT_SCORES.get(impact, 3)

        assessment = RiskAssessment(
            tenant_id="default",
            risk_id=risk_id,
            likelihood=likelihood,
            likelihood_score=likelihood_score,
            impact=impact,
            impact_score=impact_score,
            velocity=velocity,
            persistence=persistence,
            assessed_by=assessed_by,
        )

        self._update_register_assessment(org_id, risk_id, assessment)

        logger.info(
            "Assessed risk %s: score=%d, rating=%s",
            risk_id, assessment.risk_score, assessment.risk_rating,
        )
        return assessment

    async def calculate_risk_score(
        self,
        likelihood: RiskLikelihood,
        impact: RiskImpact,
    ) -> Dict[str, Any]:
        """
        Calculate risk score from likelihood and impact.

        Uses the DEFAULT_RISK_MATRIX from config.py.

        Args:
            likelihood: Likelihood rating.
            impact: Impact rating.

        Returns:
            Dict with score, rating, and color.
        """
        l_idx = LIKELIHOOD_SCORES.get(likelihood, 3) - 1
        i_idx = IMPACT_SCORES.get(impact, 3) - 1

        score = DEFAULT_RISK_MATRIX[l_idx][i_idx]
        rating = self._score_to_rating(score)
        color = RISK_COLORS.get(rating, "#999999")

        return {
            "likelihood": likelihood.value,
            "impact": impact.value,
            "score": score,
            "max_score": 25,
            "rating": rating,
            "color": color,
        }

    # ------------------------------------------------------------------
    # Risk Prioritization
    # ------------------------------------------------------------------

    async def prioritize_risks(
        self,
        org_id: str,
        weighting: Optional[Dict[str, Decimal]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Prioritize risks using weighted composite scoring.

        Composite = risk_score * score_weight + velocity * velocity_weight
                   + persistence * persistence_weight + financial_impact_weight

        Args:
            org_id: Organization ID.
            weighting: Optional custom weights.

        Returns:
            Ranked list of risks with priority scores.
        """
        entries = self._registers.get(org_id, [])
        if not entries:
            return []

        weights = weighting or {
            "score": Decimal("0.40"),
            "velocity": Decimal("0.20"),
            "persistence": Decimal("0.15"),
            "financial": Decimal("0.25"),
        }

        scored: List[Dict[str, Any]] = []
        max_financial = max(
            (self._get_financial_impact(org_id, e.risk_id) for e in entries),
            default=Decimal("1"),
        )
        if max_financial <= 0:
            max_financial = Decimal("1")

        for entry in entries:
            if entry.assessment is None:
                continue

            score_norm = Decimal(str(entry.assessment.risk_score)) / Decimal("25")
            velocity_norm = Decimal(str(
                RISK_VELOCITY_MAP.get(entry.assessment.velocity, 3)
            )) / Decimal("5")
            persist_norm = Decimal(str(
                RISK_PERSISTENCE_MAP.get(entry.assessment.persistence, 3)
            )) / Decimal("5")
            financial = self._get_financial_impact(org_id, entry.risk_id)
            financial_norm = financial / max_financial

            composite = (
                score_norm * weights["score"]
                + velocity_norm * weights["velocity"]
                + persist_norm * weights["persistence"]
                + financial_norm * weights["financial"]
            )

            scored.append({
                "entry_id": entry.id,
                "risk_id": entry.risk_id,
                "risk_name": entry.risk_name,
                "risk_type": entry.risk_type.value,
                "risk_score": entry.assessment.risk_score,
                "risk_rating": entry.assessment.risk_rating,
                "velocity": entry.assessment.velocity,
                "persistence": entry.assessment.persistence,
                "composite_priority": str(composite.quantize(Decimal("0.001"))),
                "owner": entry.owner or "unassigned",
                "status": entry.status,
            })

        scored.sort(key=lambda x: Decimal(x["composite_priority"]), reverse=True)
        for i, item in enumerate(scored):
            item["rank"] = i + 1

        logger.info("Prioritized %d risks for org %s", len(scored), org_id)
        return scored

    # ------------------------------------------------------------------
    # Risk Response -- RM (b)
    # ------------------------------------------------------------------

    async def define_risk_response(
        self,
        org_id: str,
        risk_id: str,
        response_type: RiskResponse,
        actions: List[str],
        cost_usd: Decimal = Decimal("0"),
        owner: Optional[str] = None,
    ) -> RiskManagementRecord:
        """
        Define risk response plan for a specific risk.

        Args:
            org_id: Organization ID.
            risk_id: Climate risk ID.
            response_type: Response strategy.
            actions: List of response actions.
            cost_usd: Estimated cost of response.
            owner: Response owner.

        Returns:
            RiskManagementRecord with response plan.
        """
        entry = self._find_register_entry(org_id, risk_id)
        assessment = entry.assessment if entry else None

        residual_score = None
        if assessment:
            residual_score = self._estimate_residual_score(
                assessment.risk_score, response_type,
            )

        record = RiskManagementRecord(
            tenant_id="default",
            org_id=org_id,
            risk_id=risk_id,
            identification_process="climate_risk_register",
            assessment_methodology="5x5_likelihood_impact_matrix",
            assessment=assessment,
            response_type=response_type,
            response_actions=actions,
            response_cost_usd=cost_usd,
            residual_risk_score=residual_score,
            owner=owner,
            review_date=self._calculate_next_review("quarterly"),
        )

        if org_id not in self._records:
            self._records[org_id] = []
        self._records[org_id].append(record)

        if entry:
            entry_updates = {
                "response": response_type,
                "response_actions": actions,
                "owner": owner or entry.owner,
            }
            await self.update_risk_entry(org_id, entry.id, entry_updates)

        logger.info(
            "Defined %s response for risk %s, cost=$%.0f, residual=%s",
            response_type.value, risk_id, cost_usd,
            residual_score if residual_score else "N/A",
        )
        return record

    # ------------------------------------------------------------------
    # Key Risk Indicators (KRI) -- RM (b) continued
    # ------------------------------------------------------------------

    async def register_kri(
        self,
        org_id: str,
        indicator_name: str,
        current_value: Decimal,
        threshold_amber: Decimal,
        threshold_red: Decimal,
        unit: str = "",
        related_risk_id: Optional[str] = None,
    ) -> RiskIndicator:
        """
        Register a Key Risk Indicator for ongoing monitoring.

        Args:
            org_id: Organization ID.
            indicator_name: KRI name.
            current_value: Current measured value.
            threshold_amber: Amber warning threshold.
            threshold_red: Red alert threshold.
            unit: Measurement unit.
            related_risk_id: Associated risk ID.

        Returns:
            Created RiskIndicator.
        """
        status = self._evaluate_kri_status(current_value, threshold_amber, threshold_red)

        indicator = RiskIndicator(
            tenant_id="default",
            org_id=org_id,
            indicator_name=indicator_name,
            current_value=current_value,
            threshold_amber=threshold_amber,
            threshold_red=threshold_red,
            unit=unit,
            status=status,
            trend="stable",
            related_risk_id=related_risk_id,
        )

        if org_id not in self._indicators:
            self._indicators[org_id] = []
        self._indicators[org_id].append(indicator)

        logger.info(
            "Registered KRI '%s' for org %s: value=%.1f, status=%s",
            indicator_name, org_id, current_value, status,
        )
        return indicator

    async def update_kri(
        self,
        org_id: str,
        indicator_id: str,
        new_value: Decimal,
    ) -> RiskIndicator:
        """
        Update a KRI with a new value and recalculate status and trend.

        Args:
            org_id: Organization ID.
            indicator_id: Indicator ID.
            new_value: New measured value.

        Returns:
            Updated RiskIndicator.

        Raises:
            ValueError: If indicator not found.
        """
        indicators = self._indicators.get(org_id, [])
        for i, ind in enumerate(indicators):
            if ind.id == indicator_id:
                old_value = ind.current_value
                new_status = self._evaluate_kri_status(
                    new_value, ind.threshold_amber, ind.threshold_red,
                )
                trend = self._evaluate_kri_trend(old_value, new_value)

                data = ind.model_dump()
                data["current_value"] = new_value
                data["status"] = new_status
                data["trend"] = trend
                data["updated_at"] = _now()
                updated = RiskIndicator(**data)
                self._indicators[org_id][i] = updated

                logger.info(
                    "KRI '%s' updated: %.1f -> %.1f, status=%s, trend=%s",
                    ind.indicator_name, old_value, new_value, new_status, trend,
                )
                return updated

        raise ValueError(f"Indicator {indicator_id} not found")

    async def track_risk_indicators(
        self,
        org_id: str,
    ) -> Dict[str, Any]:
        """
        Get summary of all KRIs for an organization.

        Args:
            org_id: Organization ID.

        Returns:
            Dict with KRI dashboard data.
        """
        indicators = self._indicators.get(org_id, [])
        by_status = {"green": 0, "amber": 0, "red": 0}
        by_trend = {"improving": 0, "stable": 0, "deteriorating": 0}

        for ind in indicators:
            by_status[ind.status] = by_status.get(ind.status, 0) + 1
            by_trend[ind.trend] = by_trend.get(ind.trend, 0) + 1

        breached = [
            {
                "name": ind.indicator_name,
                "value": str(ind.current_value),
                "threshold_red": str(ind.threshold_red),
                "unit": ind.unit,
            }
            for ind in indicators
            if ind.status == "red"
        ]

        return {
            "org_id": org_id,
            "total_indicators": len(indicators),
            "by_status": by_status,
            "by_trend": by_trend,
            "breached_indicators": breached,
            "breach_count": len(breached),
        }

    # ------------------------------------------------------------------
    # ERM Integration -- RM (c)
    # ------------------------------------------------------------------

    async def map_erm_integration(
        self,
        org_id: str,
        erm_framework: str = "COSO",
        climate_risk_in_erm: bool = False,
        integration_level: str = "partial",
        risk_appetite_defined: bool = False,
        climate_risk_appetite_statement: Optional[str] = None,
        board_risk_committee_oversight: bool = False,
        reporting_frequency: str = "quarterly",
    ) -> ERMIntegration:
        """
        Map climate risk management into enterprise risk management.

        Args:
            org_id: Organization ID.
            erm_framework: ERM framework in use (COSO, ISO 31000).
            climate_risk_in_erm: Whether climate is in ERM.
            integration_level: none, partial, full.
            risk_appetite_defined: Whether risk appetite is defined.
            climate_risk_appetite_statement: Statement text.
            board_risk_committee_oversight: Board oversight flag.
            reporting_frequency: How often reported.

        Returns:
            ERMIntegration record.
        """
        erm = ERMIntegration(
            tenant_id="default",
            org_id=org_id,
            erm_framework=erm_framework,
            climate_risk_in_erm=climate_risk_in_erm,
            integration_level=integration_level,
            risk_appetite_defined=risk_appetite_defined,
            climate_risk_appetite_statement=climate_risk_appetite_statement,
            board_risk_committee_oversight=board_risk_committee_oversight,
            reporting_frequency=reporting_frequency,
            last_review_date=date.today(),
        )

        self._erm[org_id] = erm
        logger.info(
            "ERM integration mapped for org %s: framework=%s, level=%s",
            org_id, erm_framework, integration_level,
        )
        return erm

    async def assess_erm_maturity(self, org_id: str) -> Dict[str, Any]:
        """
        Assess the maturity of climate-ERM integration.

        Scores across 6 dimensions: framework, integration, appetite,
        oversight, reporting, and review process.

        Args:
            org_id: Organization ID.

        Returns:
            Dict with ERM maturity assessment.
        """
        erm = self._erm.get(org_id)
        if erm is None:
            return {
                "org_id": org_id,
                "maturity_score": 0,
                "maturity_level": "none",
                "message": "No ERM integration configured",
            }

        score = 0
        dimensions: Dict[str, int] = {}

        # Framework adoption
        fw_score = 1
        if erm.erm_framework in ("COSO", "ISO 31000"):
            fw_score = 3
        dimensions["framework"] = fw_score
        score += fw_score

        # Integration level
        int_map = {"none": 1, "partial": 3, "full": 5}
        int_score = int_map.get(erm.integration_level, 1)
        dimensions["integration"] = int_score
        score += int_score

        # Risk appetite
        appetite_score = 1
        if erm.risk_appetite_defined:
            appetite_score = 3
        if erm.climate_risk_appetite_statement:
            appetite_score = 5
        dimensions["risk_appetite"] = appetite_score
        score += appetite_score

        # Board oversight
        oversight_score = 1
        if erm.board_risk_committee_oversight:
            oversight_score = 5
        dimensions["board_oversight"] = oversight_score
        score += oversight_score

        # Reporting frequency
        freq_map = {"annual": 2, "semi_annual": 3, "quarterly": 4, "monthly": 5}
        report_score = freq_map.get(erm.reporting_frequency, 2)
        dimensions["reporting"] = report_score
        score += report_score

        # Climate-in-ERM flag
        erm_score = 5 if erm.climate_risk_in_erm else 1
        dimensions["climate_in_erm"] = erm_score
        score += erm_score

        avg_score = Decimal(str(score)) / Decimal("6")
        maturity = self._score_to_erm_maturity(avg_score)

        return {
            "org_id": org_id,
            "maturity_score": str(avg_score.quantize(Decimal("0.1"))),
            "max_score": "5.0",
            "maturity_level": maturity,
            "dimensions": dimensions,
            "recommendations": self._erm_recommendations(dimensions),
        }

    # ------------------------------------------------------------------
    # Risk Register Review
    # ------------------------------------------------------------------

    async def review_risk_register(self, org_id: str) -> Dict[str, Any]:
        """
        Perform a scheduled review of the risk register.

        Identifies overdue reviews, stale entries, and entries
        requiring re-assessment.

        Args:
            org_id: Organization ID.

        Returns:
            Dict with review findings and actions required.
        """
        entries = self._registers.get(org_id, [])
        today = date.today()

        overdue_reviews: List[Dict[str, Any]] = []
        stale_entries: List[Dict[str, Any]] = []
        active_count = 0
        mitigated_count = 0
        closed_count = 0

        for entry in entries:
            if entry.status == "active":
                active_count += 1
            elif entry.status == "mitigated":
                mitigated_count += 1
            elif entry.status == "closed":
                closed_count += 1

            if entry.next_review_date and entry.next_review_date < today:
                overdue_reviews.append({
                    "entry_id": entry.id,
                    "risk_name": entry.risk_name,
                    "next_review_date": entry.next_review_date.isoformat(),
                    "days_overdue": (today - entry.next_review_date).days,
                    "owner": entry.owner or "unassigned",
                })

            last_updated = entry.updated_at.date() if entry.updated_at else entry.created_at.date()
            days_since_update = (today - last_updated).days
            if days_since_update > 180 and entry.status == "active":
                stale_entries.append({
                    "entry_id": entry.id,
                    "risk_name": entry.risk_name,
                    "days_since_update": days_since_update,
                    "owner": entry.owner or "unassigned",
                })

        return {
            "org_id": org_id,
            "review_date": today.isoformat(),
            "total_entries": len(entries),
            "active": active_count,
            "mitigated": mitigated_count,
            "closed": closed_count,
            "overdue_reviews": overdue_reviews,
            "overdue_count": len(overdue_reviews),
            "stale_entries": stale_entries,
            "stale_count": len(stale_entries),
            "action_required": len(overdue_reviews) > 0 or len(stale_entries) > 0,
        }

    # ------------------------------------------------------------------
    # Risk Heat Map
    # ------------------------------------------------------------------

    async def get_risk_heat_map(self, org_id: str) -> Dict[str, Any]:
        """
        Generate risk heat map data for visualization.

        Produces a 5x5 grid with counts and risk details per cell.

        Args:
            org_id: Organization ID.

        Returns:
            Dict with heat map grid, cell data, and summary.
        """
        entries = self._registers.get(org_id, [])

        grid: Dict[str, List[Dict[str, Any]]] = {}
        for l in range(1, 6):
            for i_val in range(1, 6):
                key = f"{l}_{i_val}"
                grid[key] = []

        for entry in entries:
            if entry.assessment is None or entry.status == "closed":
                continue
            l_score = entry.assessment.likelihood_score
            i_score = entry.assessment.impact_score
            key = f"{l_score}_{i_score}"
            grid[key].append({
                "risk_id": entry.risk_id,
                "risk_name": entry.risk_name,
                "risk_type": entry.risk_type.value,
                "score": entry.assessment.risk_score,
                "rating": entry.assessment.risk_rating,
                "owner": entry.owner or "unassigned",
            })

        cells: List[Dict[str, Any]] = []
        for l in range(1, 6):
            for i_val in range(1, 6):
                key = f"{l}_{i_val}"
                score = l * i_val
                rating = self._score_to_rating(score)
                cells.append({
                    "likelihood": l,
                    "impact": i_val,
                    "score": score,
                    "rating": rating,
                    "color": RISK_COLORS.get(rating, "#999999"),
                    "risk_count": len(grid[key]),
                    "risks": grid[key],
                })

        return {
            "org_id": org_id,
            "grid_size": "5x5",
            "cells": cells,
            "total_active_risks": sum(c["risk_count"] for c in cells),
            "likelihood_labels": [ll.value for ll in RiskLikelihood],
            "impact_labels": [il.value for il in RiskImpact],
        }

    # ------------------------------------------------------------------
    # Risk Management Summary
    # ------------------------------------------------------------------

    async def get_risk_management_summary(self, org_id: str) -> Dict[str, Any]:
        """
        Get comprehensive risk management summary statistics.

        Args:
            org_id: Organization ID.

        Returns:
            Dict with register stats, response distribution,
            KRI status, and ERM integration level.
        """
        entries = self._registers.get(org_id, [])
        records = self._records.get(org_id, [])
        indicators = self._indicators.get(org_id, [])

        by_type: Dict[str, int] = {}
        by_rating: Dict[str, int] = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        by_response: Dict[str, int] = {}

        for entry in entries:
            rt_val = entry.risk_type.value
            by_type[rt_val] = by_type.get(rt_val, 0) + 1
            if entry.assessment:
                by_rating[entry.assessment.risk_rating] = (
                    by_rating.get(entry.assessment.risk_rating, 0) + 1
                )
            resp_val = entry.response.value
            by_response[resp_val] = by_response.get(resp_val, 0) + 1

        kri_red = sum(1 for ind in indicators if ind.status == "red")
        kri_amber = sum(1 for ind in indicators if ind.status == "amber")
        kri_green = sum(1 for ind in indicators if ind.status == "green")

        erm = self._erm.get(org_id)
        erm_level = erm.integration_level if erm else "not_configured"

        total_response_cost = sum(r.response_cost_usd for r in records)

        return {
            "org_id": org_id,
            "register_size": len(entries),
            "active_risks": sum(1 for e in entries if e.status == "active"),
            "mitigated_risks": sum(1 for e in entries if e.status == "mitigated"),
            "by_risk_type": by_type,
            "by_risk_rating": by_rating,
            "by_response_strategy": by_response,
            "total_response_cost_usd": str(total_response_cost),
            "management_records": len(records),
            "kri_total": len(indicators),
            "kri_red": kri_red,
            "kri_amber": kri_amber,
            "kri_green": kri_green,
            "erm_integration_level": erm_level,
        }

    # ------------------------------------------------------------------
    # Disclosure Generation -- RM (a/b/c)
    # ------------------------------------------------------------------

    async def generate_rm_disclosure(
        self,
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Generate TCFD Risk Management (a), (b), and (c) disclosure content.

        Args:
            org_id: Organization ID.
            year: Reporting year.

        Returns:
            Dict with rm_a, rm_b, rm_c disclosure sections.
        """
        entries = self._registers.get(org_id, [])
        records = self._records.get(org_id, [])
        indicators = self._indicators.get(org_id, [])
        erm = self._erm.get(org_id)

        active = [e for e in entries if e.status == "active"]
        physical = [e for e in active if e.risk_type.value.startswith("physical_")]
        transition = [e for e in active if e.risk_type.value.startswith("transition_")]
        high_critical = [
            e for e in active
            if e.assessment and e.assessment.risk_rating in ("high", "critical")
        ]

        # RM (a): Risk Identification and Assessment
        rm_a_content = (
            f"The organization employs a structured climate risk identification "
            f"process covering both physical and transition risks. "
            f"A total of {len(active)} active climate risk(s) have been identified, "
            f"comprising {len(physical)} physical and {len(transition)} transition risk(s). "
            f"Risks are assessed using a 5x5 likelihood-impact matrix, yielding "
            f"scores from 1 (low) to 25 (critical). "
            f"Of the active risks, {len(high_critical)} are rated high or critical."
        )

        # RM (b): Risk Management Processes
        response_distribution = {}
        for e in active:
            resp_val = e.response.value
            response_distribution[resp_val] = response_distribution.get(resp_val, 0) + 1

        total_response_cost = sum(r.response_cost_usd for r in records)
        rm_b_content = (
            f"Climate risks are managed through defined response strategies: "
            f"{', '.join(f'{k} ({v})' for k, v in response_distribution.items())}. "
            f"Total budgeted response cost is ${total_response_cost:,.0f}. "
            f"The organization monitors {len(indicators)} Key Risk Indicator(s) "
            f"with amber and red threshold alerting."
        )

        # RM (c): ERM Integration
        if erm:
            rm_c_content = (
                f"Climate risk management is integrated into the organization's "
                f"enterprise risk management framework ({erm.erm_framework}) "
                f"at a '{erm.integration_level}' level. "
                f"Board risk committee oversight: "
                f"{'Yes' if erm.board_risk_committee_oversight else 'No'}. "
                f"Climate risk appetite defined: "
                f"{'Yes' if erm.risk_appetite_defined else 'No'}. "
                f"Reporting frequency: {erm.reporting_frequency}."
            )
        else:
            rm_c_content = (
                "ERM integration has not been formally configured. "
                "The organization should map climate risk processes into "
                "its enterprise risk management framework."
            )

        compliance_a = self._score_rm_a(entries, physical, transition)
        compliance_b = self._score_rm_b(records, indicators)
        compliance_c = self._score_rm_c(erm)

        return {
            "org_id": org_id,
            "reporting_year": year,
            "rm_a": {
                "ref": "Risk Management (a)",
                "title": "Risk Identification and Assessment",
                "content": rm_a_content,
                "compliance_score": compliance_a,
                "risk_count": len(active),
                "high_critical_count": len(high_critical),
            },
            "rm_b": {
                "ref": "Risk Management (b)",
                "title": "Risk Management Processes",
                "content": rm_b_content,
                "compliance_score": compliance_b,
                "response_distribution": response_distribution,
                "total_response_cost": str(total_response_cost),
                "kri_count": len(indicators),
            },
            "rm_c": {
                "ref": "Risk Management (c)",
                "title": "ERM Integration",
                "content": rm_c_content,
                "compliance_score": compliance_c,
                "erm_framework": erm.erm_framework if erm else "not_configured",
                "integration_level": erm.integration_level if erm else "none",
            },
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_rating(score: int) -> str:
        """Convert numeric risk score (1-25) to risk rating band."""
        for band, thresholds in RISK_MATRIX_THRESHOLDS.items():
            if thresholds["min"] <= score <= thresholds["max"]:
                return band
        return "medium"

    @staticmethod
    def _calculate_next_review(frequency: str) -> date:
        """Calculate next review date based on frequency."""
        today = date.today()
        days_map = {
            "monthly": 30,
            "quarterly": 90,
            "semi_annual": 180,
            "annual": 365,
        }
        days = days_map.get(frequency, 90)
        return date.fromordinal(today.toordinal() + days)

    @staticmethod
    def _evaluate_kri_status(
        value: Decimal,
        amber: Decimal,
        red: Decimal,
    ) -> str:
        """Evaluate KRI status based on thresholds."""
        if value >= red:
            return "red"
        if value >= amber:
            return "amber"
        return "green"

    @staticmethod
    def _evaluate_kri_trend(old_value: Decimal, new_value: Decimal) -> str:
        """Evaluate KRI trend from old to new value."""
        if new_value < old_value:
            return "improving"
        if new_value > old_value:
            return "deteriorating"
        return "stable"

    @staticmethod
    def _estimate_residual_score(
        inherent_score: int,
        response: RiskResponse,
    ) -> int:
        """Estimate residual risk score after response implementation."""
        reduction_factors = {
            RiskResponse.MITIGATE: Decimal("0.50"),
            RiskResponse.ADAPT: Decimal("0.60"),
            RiskResponse.TRANSFER: Decimal("0.40"),
            RiskResponse.ACCEPT: Decimal("1.00"),
            RiskResponse.AVOID: Decimal("0.10"),
        }
        factor = reduction_factors.get(response, Decimal("1.00"))
        residual = int(
            (Decimal(str(inherent_score)) * factor).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP,
            )
        )
        return max(residual, 1)

    @staticmethod
    def _score_to_erm_maturity(avg: Decimal) -> str:
        """Map average ERM score to maturity level."""
        if avg >= Decimal("4.5"):
            return "optimized"
        if avg >= Decimal("3.5"):
            return "managed"
        if avg >= Decimal("2.5"):
            return "defined"
        if avg >= Decimal("1.5"):
            return "developing"
        return "initial"

    @staticmethod
    def _erm_recommendations(dimensions: Dict[str, int]) -> List[str]:
        """Generate ERM improvement recommendations based on dimension scores."""
        recs: List[str] = []
        if dimensions.get("integration", 0) < 4:
            recs.append(
                "Fully integrate climate risks into the enterprise risk register "
                "and risk governance structure."
            )
        if dimensions.get("risk_appetite", 0) < 4:
            recs.append(
                "Define a formal climate risk appetite statement aligned with "
                "organizational strategy and regulatory expectations."
            )
        if dimensions.get("board_oversight", 0) < 4:
            recs.append(
                "Ensure the board risk committee has explicit oversight of "
                "climate-related risks and reviews them regularly."
            )
        if dimensions.get("reporting", 0) < 4:
            recs.append(
                "Increase climate risk reporting frequency to at least quarterly "
                "with KRI dashboard integration."
            )
        if dimensions.get("climate_in_erm", 0) < 4:
            recs.append(
                "Formally incorporate climate risk categories into the enterprise "
                "risk taxonomy and assessment methodology."
            )
        if not recs:
            recs.append("ERM integration is at a mature level. Continue monitoring.")
        return recs

    def _find_register_entry(
        self, org_id: str, risk_id: str,
    ) -> Optional[RiskRegisterEntry]:
        """Find a register entry by risk_id."""
        for entry in self._registers.get(org_id, []):
            if entry.risk_id == risk_id:
                return entry
        return None

    def _update_register_assessment(
        self, org_id: str, risk_id: str, assessment: RiskAssessment,
    ) -> None:
        """Update the assessment on a register entry."""
        entries = self._registers.get(org_id, [])
        for i, entry in enumerate(entries):
            if entry.risk_id == risk_id:
                data = entry.model_dump()
                data["assessment"] = assessment.model_dump()
                data["updated_at"] = _now()
                self._registers[org_id][i] = RiskRegisterEntry(**data)
                return

    def _get_financial_impact(self, org_id: str, risk_id: str) -> Decimal:
        """Get financial impact for a risk from records or default zero."""
        records = self._records.get(org_id, [])
        for rec in records:
            if rec.risk_id == risk_id:
                return rec.response_cost_usd
        return Decimal("0")

    @staticmethod
    def _score_rm_a(
        entries: List[RiskRegisterEntry],
        physical: List[RiskRegisterEntry],
        transition: List[RiskRegisterEntry],
    ) -> int:
        """Score RM (a) disclosure completeness (0-100)."""
        score = 0
        if entries:
            score += 20
        if physical:
            score += 15
        if transition:
            score += 15
        assessed = [e for e in entries if e.assessment is not None]
        if assessed:
            score += 20
        if len(assessed) == len(entries) and entries:
            score += 10
        owners_assigned = [e for e in entries if e.owner]
        if len(owners_assigned) >= len(entries) * 0.8:
            score += 10
        types_covered = set(e.risk_type for e in entries)
        score += min(len(types_covered) * 5, 10)
        return min(score, 100)

    @staticmethod
    def _score_rm_b(
        records: List[RiskManagementRecord],
        indicators: List[RiskIndicator],
    ) -> int:
        """Score RM (b) disclosure completeness (0-100)."""
        score = 0
        if records:
            score += 25
            response_types = set(r.response_type for r in records)
            score += min(len(response_types) * 5, 15)
            with_actions = [r for r in records if r.response_actions]
            if with_actions:
                score += 15
            with_cost = [r for r in records if r.response_cost_usd > 0]
            if with_cost:
                score += 10
        if indicators:
            score += 15
            if len(indicators) >= 3:
                score += 10
        return min(score, 100)

    @staticmethod
    def _score_rm_c(erm: Optional[ERMIntegration]) -> int:
        """Score RM (c) disclosure completeness (0-100)."""
        if erm is None:
            return 0
        score = 15
        if erm.climate_risk_in_erm:
            score += 20
        int_map = {"none": 0, "partial": 10, "full": 20}
        score += int_map.get(erm.integration_level, 0)
        if erm.risk_appetite_defined:
            score += 15
        if erm.climate_risk_appetite_statement:
            score += 10
        if erm.board_risk_committee_oversight:
            score += 15
        freq_map = {"annual": 2, "semi_annual": 3, "quarterly": 5, "monthly": 5}
        score += freq_map.get(erm.reporting_frequency, 2)
        return min(score, 100)
