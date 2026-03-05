"""
Temperature Scoring Engine -- SBTi Temperature Rating Methodology

Implements the SBTi Temperature Rating methodology v2.0 for computing
implied temperature scores at company and portfolio level.  Covers
company-level scoring (short-term, mid-term, long-term), scope-level
breakdown, portfolio temperature aggregation (WATS, TETS, MOTS, EOTS,
ECOTS, AOTS), peer ranking, and what-if scenario analysis.

All numeric calculations are deterministic (zero-hallucination).

Reference:
    - SBTi Temperature Rating Methodology v2.0 (2023)
    - SBTi Portfolio Coverage approach
    - CDP-SBTi Temperature Rating methodology paper
    - SBTi ITR (Implied Temperature Rise) methodology

Example:
    >>> from services.config import SBTiAppConfig
    >>> engine = TemperatureScoringEngine(SBTiAppConfig())
    >>> score = engine.calculate_company_score("org-1")
    >>> print(score.company_score)
    1.5
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    AmbitionLevel,
    SBTiAppConfig,
    SBTiSector,
    TargetScope,
    TEMPERATURE_SCORING_DEFAULTS,
    ValidationStatus,
)
from .models import (
    Organization,
    Target,
    TemperatureScore,
    PortfolioTemperature,
    ScopeTemperatureScore,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ScopeScore(BaseModel):
    """Temperature score for a single scope with detail."""

    scope: str = Field(...)
    score: float = Field(default=3.2, ge=1.0, le=6.0)
    has_target: bool = Field(default=False)
    target_ambition: Optional[str] = Field(None)
    annual_reduction_rate: float = Field(default=0.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class CompanyScoreResult(BaseModel):
    """Complete company-level temperature score result."""

    org_id: str = Field(...)
    company_score: float = Field(default=3.2, ge=1.0, le=6.0)
    short_term_score: float = Field(default=3.2)
    mid_term_score: float = Field(default=3.2)
    long_term_score: float = Field(default=3.2)
    scope_breakdown: List[ScopeScore] = Field(default_factory=list)
    has_validated_target: bool = Field(default=False)
    methodology_version: str = Field(default="v2.0")
    provenance_hash: str = Field(default="")
    scored_at: datetime = Field(default_factory=_now)


class PortfolioScoreResult(BaseModel):
    """Portfolio-level temperature aggregation result."""

    org_id: str = Field(...)
    portfolio_score: float = Field(default=3.2, ge=1.0, le=6.0)
    aggregation_method: str = Field(default="WATS")
    holdings_count: int = Field(default=0)
    holdings_with_targets: int = Field(default=0)
    coverage_pct: float = Field(default=0.0)
    short_term_score: float = Field(default=3.2)
    long_term_score: float = Field(default=3.2)
    alignment_1_5c_pct: float = Field(default=0.0)
    alignment_wb2c_pct: float = Field(default=0.0)
    alignment_not_aligned_pct: float = Field(default=0.0)
    sector_breakdown: List[Dict[str, Any]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    scored_at: datetime = Field(default_factory=_now)


class PeerRanking(BaseModel):
    """Peer temperature score ranking."""

    org_id: str = Field(...)
    sector: str = Field(default="general")
    company_score: float = Field(default=3.2)
    sector_avg_score: float = Field(default=3.2)
    sector_median_score: float = Field(default=3.2)
    percentile_rank: float = Field(default=50.0)
    total_peers: int = Field(default=0)
    better_than_count: int = Field(default=0)
    message: str = Field(default="")


class WhatIfResult(BaseModel):
    """What-if scenario analysis for temperature score improvement."""

    org_id: str = Field(...)
    current_score: float = Field(default=3.2)
    scenario_score: float = Field(default=3.2)
    improvement: float = Field(default=0.0)
    scenario_description: str = Field(default="")
    actions: List[Dict[str, Any]] = Field(default_factory=list)


class ScoreHistory(BaseModel):
    """Historical temperature scores for trend tracking."""

    org_id: str = Field(...)
    scores: List[Dict[str, Any]] = Field(default_factory=list)
    trend_direction: str = Field(
        default="flat", description="improving, deteriorating, flat",
    )
    avg_score: float = Field(default=3.2)


# ---------------------------------------------------------------------------
# TemperatureScoringEngine
# ---------------------------------------------------------------------------

class TemperatureScoringEngine:
    """
    SBTi Temperature Rating calculation engine.

    Computes implied temperature scores for companies based on their
    target ambition and validates against SBTi benchmarks. Supports
    company-level scoring, portfolio aggregation, peer ranking, and
    what-if scenario analysis.

    Attributes:
        config: Application configuration.
        _organizations: In-memory organization store keyed by org_id.
        _targets: In-memory target store keyed by org_id.
        _scores: Computed temperature scores keyed by org_id.
        _peer_data: Sector peer score data.

    Example:
        >>> engine = TemperatureScoringEngine(SBTiAppConfig())
        >>> score = engine.calculate_company_score("org-1")
    """

    # Default temperature score for companies with no target
    DEFAULT_NO_TARGET_SCORE: float = 3.2

    # Score mapping from ambition to temperature
    AMBITION_SCORE_MAP: Dict[str, float] = {
        "1.5c": 1.5,
        "well_below_2c": 1.75,
        "2c": 2.0,
        "not_aligned": 3.2,
    }

    # Time horizon weights per ITR methodology v2.0
    SHORT_TERM_WEIGHT: float = 0.4
    MID_TERM_WEIGHT: float = 0.2
    LONG_TERM_WEIGHT: float = 0.4

    # Scope weights
    SCOPE_1_2_WEIGHT: float = 0.6
    SCOPE_3_WEIGHT: float = 0.4

    def __init__(self, config: Optional[SBTiAppConfig] = None) -> None:
        """Initialize the TemperatureScoringEngine."""
        self.config = config or SBTiAppConfig()
        self._organizations: Dict[str, Organization] = {}
        self._targets: Dict[str, List[Dict[str, Any]]] = {}
        self._scores: Dict[str, CompanyScoreResult] = {}
        self._peer_data: Dict[str, List[float]] = {}
        logger.info("TemperatureScoringEngine initialized (default=%.1fC)",
                     self.DEFAULT_NO_TARGET_SCORE)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_organization(self, org: Organization) -> None:
        """Register an organization for temperature scoring."""
        self._organizations[org.id] = org

    def register_target(
        self, org_id: str, target_data: Dict[str, Any],
    ) -> None:
        """
        Register a target for temperature scoring.

        Args:
            org_id: Organization identifier.
            target_data: Dict with keys: id, target_type, scope, ambition_level,
                        base_year, target_year, annual_reduction_rate,
                        validation_status.
        """
        self._targets.setdefault(org_id, []).append(target_data)

    def register_peer_scores(
        self, sector: str, scores: List[float],
    ) -> None:
        """Register peer temperature scores for a sector."""
        self._peer_data[sector] = scores

    # ------------------------------------------------------------------
    # Company-Level Scoring
    # ------------------------------------------------------------------

    def calculate_company_score(self, org_id: str) -> CompanyScoreResult:
        """
        Calculate the temperature score for a company.

        Maps target ambition to implied temperature using the SBTi ITR
        methodology v2.0. Companies without targets receive the default
        3.2C score (current policies trajectory).

        Args:
            org_id: Organization identifier.

        Returns:
            CompanyScoreResult with short/mid/long-term scores.
        """
        start = datetime.utcnow()

        org = self._organizations.get(org_id)
        targets = self._targets.get(org_id, [])

        if not targets:
            result = CompanyScoreResult(
                org_id=org_id,
                company_score=self.DEFAULT_NO_TARGET_SCORE,
                short_term_score=self.DEFAULT_NO_TARGET_SCORE,
                mid_term_score=self.DEFAULT_NO_TARGET_SCORE,
                long_term_score=self.DEFAULT_NO_TARGET_SCORE,
                has_validated_target=False,
                provenance_hash=_sha256(
                    f"temp_score:{org_id}:{self.DEFAULT_NO_TARGET_SCORE}"
                ),
            )
            self._scores[org_id] = result
            return result

        # Calculate scope-level scores
        scope_scores = self._calculate_scope_scores(targets)

        # Calculate time-horizon scores
        short_term = self._score_time_horizon(targets, "near_term")
        mid_term = self._score_time_horizon(targets, "near_term")
        long_term = self._score_time_horizon(targets, "long_term")

        # Composite company score = weighted average of time horizons
        company_score = (
            short_term * self.SHORT_TERM_WEIGHT
            + mid_term * self.MID_TERM_WEIGHT
            + long_term * self.LONG_TERM_WEIGHT
        )

        has_validated = any(
            t.get("validation_status") == "validated"
            for t in targets
        )

        provenance = _sha256(
            f"temp_score:{org_id}:{company_score}:{short_term}:{long_term}"
        )

        result = CompanyScoreResult(
            org_id=org_id,
            company_score=round(company_score, 2),
            short_term_score=round(short_term, 2),
            mid_term_score=round(mid_term, 2),
            long_term_score=round(long_term, 2),
            scope_breakdown=scope_scores,
            has_validated_target=has_validated,
            provenance_hash=provenance,
        )

        self._scores[org_id] = result

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Temperature score for org %s: %.2fC (ST=%.2f MT=%.2f LT=%.2f) "
            "in %.1f ms",
            org_id, company_score, short_term, mid_term, long_term,
            elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Portfolio-Level Scoring
    # ------------------------------------------------------------------

    def calculate_portfolio_score(
        self,
        org_id: str,
        holdings: List[Dict[str, Any]],
        method: str = "WATS",
    ) -> PortfolioScoreResult:
        """
        Calculate a portfolio-level temperature score.

        Aggregates company-level scores weighted by exposure using
        one of the SBTi-supported aggregation methods:
        - WATS: Weighted Average Temperature Score
        - TETS: Total Emissions Weighted Temperature Score
        - MOTS: Market Owned Temperature Score
        - EOTS: Enterprise Owned Temperature Score
        - ECOTS: Enterprise Value + Cash Owned Temperature Score
        - AOTS: Revenue Allocated Temperature Score

        Args:
            org_id: FI organization identifier.
            holdings: List of dicts with keys: company_name, exposure_usd,
                     financed_emissions, temperature_score, has_target,
                     sector.
            method: Aggregation method (default WATS).

        Returns:
            PortfolioScoreResult with weighted portfolio temperature.
        """
        start = datetime.utcnow()

        if not holdings:
            return PortfolioScoreResult(
                org_id=org_id,
                aggregation_method=method,
            )

        total_weight = 0.0
        weighted_score_sum = 0.0
        with_targets = 0
        alignment_1_5c = 0
        alignment_wb2c = 0
        alignment_not_aligned = 0
        short_term_sum = 0.0
        long_term_sum = 0.0

        sector_scores: Dict[str, List[float]] = {}

        for h in holdings:
            score = h.get("temperature_score", self.DEFAULT_NO_TARGET_SCORE)
            weight = self._get_weight(h, method)
            total_weight += weight
            weighted_score_sum += score * weight

            if h.get("has_target", False):
                with_targets += 1

            # Alignment buckets
            if score <= 1.5:
                alignment_1_5c += 1
            elif score <= 2.0:
                alignment_wb2c += 1
            else:
                alignment_not_aligned += 1

            # Short/long approximation
            short_term_sum += score * weight
            long_term_sum += score * weight

            sector = h.get("sector", "general")
            sector_scores.setdefault(sector, []).append(score)

        portfolio_score = (
            weighted_score_sum / total_weight
            if total_weight > 0 else self.DEFAULT_NO_TARGET_SCORE
        )
        short_term = short_term_sum / total_weight if total_weight > 0 else 3.2
        long_term = long_term_sum / total_weight if total_weight > 0 else 3.2

        n = len(holdings)
        coverage_pct = (with_targets / n * 100.0) if n > 0 else 0.0

        # Sector breakdown
        sector_breakdown: List[Dict[str, Any]] = []
        for sector, scores in sector_scores.items():
            avg = sum(scores) / len(scores) if scores else 3.2
            sector_breakdown.append({
                "sector": sector,
                "avg_score": round(avg, 2),
                "holdings_count": len(scores),
            })

        provenance = _sha256(
            f"portfolio_temp:{org_id}:{portfolio_score}:{n}:{method}"
        )

        result = PortfolioScoreResult(
            org_id=org_id,
            portfolio_score=round(portfolio_score, 2),
            aggregation_method=method,
            holdings_count=n,
            holdings_with_targets=with_targets,
            coverage_pct=round(coverage_pct, 2),
            short_term_score=round(short_term, 2),
            long_term_score=round(long_term, 2),
            alignment_1_5c_pct=round(alignment_1_5c / n * 100 if n > 0 else 0, 2),
            alignment_wb2c_pct=round(alignment_wb2c / n * 100 if n > 0 else 0, 2),
            alignment_not_aligned_pct=round(
                alignment_not_aligned / n * 100 if n > 0 else 0, 2,
            ),
            sector_breakdown=sector_breakdown,
            provenance_hash=provenance,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Portfolio score for org %s: %.2fC (%s), %d holdings, "
            "%.0f%% coverage in %.1f ms",
            org_id, portfolio_score, method, n, coverage_pct, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Peer Ranking
    # ------------------------------------------------------------------

    def get_peer_ranking(self, org_id: str) -> PeerRanking:
        """
        Rank an organization's temperature score against sector peers.

        Args:
            org_id: Organization identifier.

        Returns:
            PeerRanking with percentile and comparison.
        """
        score_result = self._scores.get(org_id)
        org = self._organizations.get(org_id)

        company_score = (
            score_result.company_score if score_result
            else self.DEFAULT_NO_TARGET_SCORE
        )
        sector = org.sector.value if org else "general"
        peers = self._peer_data.get(sector, [])

        if not peers:
            return PeerRanking(
                org_id=org_id,
                sector=sector,
                company_score=company_score,
                message="No peer data available for ranking.",
            )

        sorted_peers = sorted(peers)
        total = len(sorted_peers)
        better_than = sum(1 for p in sorted_peers if p > company_score)

        percentile = (better_than / total * 100.0) if total > 0 else 50.0
        avg = sum(sorted_peers) / total if total > 0 else 3.2
        median_idx = total // 2
        median = sorted_peers[median_idx] if total > 0 else 3.2

        return PeerRanking(
            org_id=org_id,
            sector=sector,
            company_score=round(company_score, 2),
            sector_avg_score=round(avg, 2),
            sector_median_score=round(median, 2),
            percentile_rank=round(percentile, 2),
            total_peers=total,
            better_than_count=better_than,
            message=(
                f"Company score of {company_score:.2f}C is better than "
                f"{better_than} of {total} sector peers "
                f"(top {percentile:.0f}th percentile)."
            ),
        )

    # ------------------------------------------------------------------
    # What-If Scenario Analysis
    # ------------------------------------------------------------------

    def run_what_if(
        self,
        org_id: str,
        scenario: Dict[str, Any],
    ) -> WhatIfResult:
        """
        Run a what-if scenario to estimate temperature score improvement.

        Scenarios can include: setting a 1.5C target, validating an
        existing target, adding Scope 3 target, etc.

        Args:
            org_id: Organization identifier.
            scenario: Dict with keys: scenario_type, new_ambition,
                     scope, reduction_pct.

        Returns:
            WhatIfResult with projected score improvement.
        """
        current_result = self._scores.get(org_id)
        current_score = (
            current_result.company_score if current_result
            else self.DEFAULT_NO_TARGET_SCORE
        )

        scenario_type = scenario.get("scenario_type", "set_target")
        new_ambition = scenario.get("new_ambition", "1.5c")
        scope = scenario.get("scope", "scope_1_2")

        # Calculate scenario score
        ambition_score = self.AMBITION_SCORE_MAP.get(new_ambition, 3.2)

        if scenario_type == "set_target":
            if scope in ("scope_1_2", "scope_1"):
                scenario_score = (
                    ambition_score * self.SCOPE_1_2_WEIGHT
                    + current_score * self.SCOPE_3_WEIGHT
                )
            elif scope == "scope_3":
                scenario_score = (
                    current_score * self.SCOPE_1_2_WEIGHT
                    + ambition_score * self.SCOPE_3_WEIGHT
                )
            else:
                scenario_score = ambition_score
        elif scenario_type == "validate_target":
            # Validation typically improves confidence, slight score improvement
            scenario_score = current_score * 0.95
        elif scenario_type == "set_net_zero":
            scenario_score = min(ambition_score, 1.5)
        else:
            scenario_score = current_score

        improvement = current_score - scenario_score

        actions: List[Dict[str, Any]] = []
        if scenario_type == "set_target":
            actions.append({
                "action": f"Set {new_ambition} target for {scope}",
                "impact": f"Reduces score by {improvement:.2f}C",
            })
        if scenario_type == "validate_target":
            actions.append({
                "action": "Validate existing target with SBTi",
                "impact": "Improves confidence and score marginally",
            })

        return WhatIfResult(
            org_id=org_id,
            current_score=round(current_score, 2),
            scenario_score=round(scenario_score, 2),
            improvement=round(improvement, 2),
            scenario_description=f"Scenario: {scenario_type} ({new_ambition} for {scope})",
            actions=actions,
        )

    # ------------------------------------------------------------------
    # Score History
    # ------------------------------------------------------------------

    def get_score_history(self, org_id: str) -> ScoreHistory:
        """
        Get historical temperature score trend.

        Returns stored scores for trend analysis. Since this is an
        in-memory engine, history is limited to scores computed in
        the current session.

        Args:
            org_id: Organization identifier.

        Returns:
            ScoreHistory with trend direction.
        """
        score = self._scores.get(org_id)
        if not score:
            return ScoreHistory(org_id=org_id)

        # Single data point for in-memory
        scores = [{
            "date": score.scored_at.isoformat(),
            "score": score.company_score,
            "short_term": score.short_term_score,
            "long_term": score.long_term_score,
        }]

        return ScoreHistory(
            org_id=org_id,
            scores=scores,
            trend_direction="flat",
            avg_score=score.company_score,
        )

    # ------------------------------------------------------------------
    # Domain Model Conversion
    # ------------------------------------------------------------------

    def to_domain_model(self, org_id: str) -> TemperatureScore:
        """
        Convert computed score to the TemperatureScore domain model.

        Args:
            org_id: Organization identifier.

        Returns:
            TemperatureScore Pydantic domain model.
        """
        result = self._scores.get(org_id)
        if result is None:
            result = self.calculate_company_score(org_id)

        scope_breakdown = [
            ScopeTemperatureScore(
                scope=TargetScope(s.scope) if s.scope in [e.value for e in TargetScope] else TargetScope.SCOPE_1_2,
                score=Decimal(str(s.score)),
                has_target=s.has_target,
                target_ambition=(
                    AmbitionLevel(s.target_ambition)
                    if s.target_ambition and s.target_ambition in [e.value for e in AmbitionLevel]
                    else None
                ),
                confidence=Decimal(str(s.confidence)),
            )
            for s in result.scope_breakdown
        ]

        return TemperatureScore(
            tenant_id="default",
            org_id=org_id,
            company_score=Decimal(str(result.company_score)),
            short_term_score=Decimal(str(result.short_term_score)),
            mid_term_score=Decimal(str(result.mid_term_score)),
            long_term_score=Decimal(str(result.long_term_score)),
            scope_breakdown=scope_breakdown,
            methodology_version=result.methodology_version,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_scope_scores(
        self, targets: List[Dict[str, Any]],
    ) -> List[ScopeScore]:
        """Calculate temperature scores broken down by scope."""
        scope_map: Dict[str, Dict[str, Any]] = {}

        for target in targets:
            scope = target.get("scope", "scope_1_2")
            ambition = target.get("ambition_level", "not_aligned")
            rate = target.get("annual_reduction_rate", 0.0)

            score = self.AMBITION_SCORE_MAP.get(ambition, self.DEFAULT_NO_TARGET_SCORE)

            if scope not in scope_map or score < scope_map[scope]["score"]:
                scope_map[scope] = {
                    "scope": scope,
                    "score": score,
                    "has_target": True,
                    "target_ambition": ambition,
                    "annual_reduction_rate": rate,
                    "confidence": 0.8 if target.get("validation_status") == "validated" else 0.6,
                }

        # Default scopes not covered
        for scope in ["scope_1_2", "scope_3"]:
            if scope not in scope_map:
                scope_map[scope] = {
                    "scope": scope,
                    "score": self.DEFAULT_NO_TARGET_SCORE,
                    "has_target": False,
                    "target_ambition": None,
                    "annual_reduction_rate": 0.0,
                    "confidence": 0.5,
                }

        return [ScopeScore(**v) for v in scope_map.values()]

    def _score_time_horizon(
        self, targets: List[Dict[str, Any]], horizon: str,
    ) -> float:
        """
        Calculate score for a specific time horizon.

        Args:
            targets: List of target dicts.
            horizon: "near_term", "long_term", or "net_zero".

        Returns:
            Temperature score for the horizon.
        """
        horizon_targets = [
            t for t in targets
            if t.get("target_type") == horizon
        ]

        if not horizon_targets:
            # Check for net_zero targets as long-term fallback
            if horizon == "long_term":
                nz = [t for t in targets if t.get("target_type") == "net_zero"]
                if nz:
                    horizon_targets = nz

        if not horizon_targets:
            return self.DEFAULT_NO_TARGET_SCORE

        # Use the best (most ambitious) target for this horizon
        best_score = self.DEFAULT_NO_TARGET_SCORE
        for t in horizon_targets:
            ambition = t.get("ambition_level", "not_aligned")
            score = self.AMBITION_SCORE_MAP.get(ambition, self.DEFAULT_NO_TARGET_SCORE)

            # Weighted by scope
            scope = t.get("scope", "scope_1_2")
            if scope in ("scope_1_2", "scope_1"):
                weighted = score * self.SCOPE_1_2_WEIGHT + self.DEFAULT_NO_TARGET_SCORE * self.SCOPE_3_WEIGHT
            elif scope == "scope_3":
                weighted = self.DEFAULT_NO_TARGET_SCORE * self.SCOPE_1_2_WEIGHT + score * self.SCOPE_3_WEIGHT
            elif scope == "scope_1_2_3":
                weighted = score
            else:
                weighted = score

            if weighted < best_score:
                best_score = weighted

        return round(best_score, 2)

    def _get_weight(
        self, holding: Dict[str, Any], method: str,
    ) -> float:
        """
        Get the weighting factor for a holding based on aggregation method.

        Args:
            holding: Holding dict with financial data.
            method: Aggregation method (WATS, TETS, etc.).

        Returns:
            Weight for the holding.
        """
        if method == "WATS":
            return holding.get("exposure_usd", 1.0)
        elif method == "TETS":
            return holding.get("financed_emissions", 1.0)
        elif method == "MOTS":
            return holding.get("market_cap_usd", 1.0)
        elif method == "EOTS":
            return holding.get("enterprise_value_usd", 1.0)
        elif method == "ECOTS":
            return holding.get("evic_usd", 1.0)
        elif method == "AOTS":
            return holding.get("revenue_usd", 1.0)
        else:
            return holding.get("exposure_usd", 1.0)
