# -*- coding: utf-8 -*-
"""
ImpactDataBridge - Impact Data Intake from Investees and Third Parties
=======================================================================

This module connects PACK-011 (SFDR Article 9) with investee impact data
sources and third-party impact verification providers. Article 9 products
must demonstrate that all investments are sustainable, which requires
measuring real-world impact aligned with the stated sustainable investment
objective. This bridge handles SDG alignment data intake, impact KPI
tracking, verification workflows, and aggregated impact reporting.

Architecture:
    PACK-011 SFDR Art 9 --> ImpactDataBridge --> Investee Data / Third Parties
                                  |
                                  v
    SDG Alignment, Impact KPIs, Verification, Contribution Analysis

Regulatory Context:
    SFDR Article 9 products must have sustainable investment as their
    objective. This means they must measure and report on the contribution
    of investments to the sustainable objective, including SDG alignment
    where applicable. The RTS Annex III/V templates require impact metrics
    and SDG contribution disclosure.

Example:
    >>> config = ImpactDataConfig(sdg_targets=[7, 13])
    >>> bridge = ImpactDataBridge(config)
    >>> impact = bridge.assess_portfolio_impact(holdings)
    >>> print(f"SDG alignment: {impact.sdg_alignment_score:.1f}%")

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-011 SFDR Article 9
Version: 1.0.0
Status: Production Ready
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Helpers
# =============================================================================


def _utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _hash_data(data: Any) -> str:
    """Compute a SHA-256 hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


# =============================================================================
# Agent Stub
# =============================================================================


class _AgentStub:
    """Deferred agent loader for lazy initialization."""

    def __init__(self, agent_id: str, module_path: str, class_name: str) -> None:
        self.agent_id = agent_id
        self.module_path = module_path
        self.class_name = class_name
        self._instance: Optional[Any] = None

    def load(self) -> Any:
        """Load and return the agent instance."""
        if self._instance is not None:
            return self._instance
        try:
            import importlib
            mod = importlib.import_module(self.module_path)
            cls = getattr(mod, self.class_name)
            self._instance = cls()
            return self._instance
        except Exception as exc:
            logger.warning(
                "AgentStub: failed to load %s from %s: %s",
                self.agent_id, self.module_path, exc,
            )
            return None

    @property
    def is_loaded(self) -> bool:
        """Whether the agent has been loaded."""
        return self._instance is not None


# =============================================================================
# Enums
# =============================================================================


class SDGGoal(str, Enum):
    """UN Sustainable Development Goals (1-17)."""
    NO_POVERTY = "sdg_1"
    ZERO_HUNGER = "sdg_2"
    GOOD_HEALTH = "sdg_3"
    QUALITY_EDUCATION = "sdg_4"
    GENDER_EQUALITY = "sdg_5"
    CLEAN_WATER = "sdg_6"
    AFFORDABLE_ENERGY = "sdg_7"
    DECENT_WORK = "sdg_8"
    INDUSTRY_INNOVATION = "sdg_9"
    REDUCED_INEQUALITIES = "sdg_10"
    SUSTAINABLE_CITIES = "sdg_11"
    RESPONSIBLE_CONSUMPTION = "sdg_12"
    CLIMATE_ACTION = "sdg_13"
    LIFE_BELOW_WATER = "sdg_14"
    LIFE_ON_LAND = "sdg_15"
    PEACE_JUSTICE = "sdg_16"
    PARTNERSHIPS = "sdg_17"


class ImpactCategory(str, Enum):
    """Impact measurement category."""
    ENVIRONMENTAL = "environmental"
    SOCIAL = "social"
    GOVERNANCE = "governance"
    CLIMATE = "climate"
    BIODIVERSITY = "biodiversity"
    HEALTH = "health"
    EDUCATION = "education"
    FINANCIAL_INCLUSION = "financial_inclusion"


class VerificationStatus(str, Enum):
    """Impact data verification status."""
    VERIFIED = "verified"
    SELF_REPORTED = "self_reported"
    THIRD_PARTY_ESTIMATED = "third_party_estimated"
    UNVERIFIED = "unverified"
    PENDING = "pending"


class ImpactDirection(str, Enum):
    """Direction of impact contribution."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class ContributionLevel(str, Enum):
    """Level of contribution to sustainable objective."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"
    NEGATIVE = "negative"


# =============================================================================
# Data Models
# =============================================================================


class ImpactDataConfig(BaseModel):
    """Configuration for the Impact Data Bridge."""
    sdg_targets: List[int] = Field(
        default_factory=list,
        description="SDG goal numbers linked to sustainable objective",
    )
    impact_categories: List[str] = Field(
        default_factory=lambda: ["environmental", "climate"],
        description="Impact categories to assess",
    )
    impact_kpis: List[str] = Field(
        default_factory=lambda: [
            "ghg_avoided_tco2e",
            "renewable_energy_mwh",
            "jobs_created",
            "people_reached",
        ],
        description="Key Performance Indicators for impact measurement",
    )
    min_sdg_alignment_pct: float = Field(
        default=50.0, ge=0.0, le=100.0,
        description="Minimum SDG alignment percentage for Article 9",
    )
    require_verification: bool = Field(
        default=False,
        description="Require third-party verification of impact data",
    )
    verification_provider: str = Field(
        default="", description="Third-party impact verification provider"
    )
    data_freshness_days: int = Field(
        default=365, ge=1,
        description="Maximum age of impact data in days",
    )
    enable_contribution_analysis: bool = Field(
        default=True,
        description="Enable contribution-to-objective analysis",
    )
    enable_provenance: bool = Field(
        default=True, description="Enable provenance hash tracking"
    )


class ImpactKPI(BaseModel):
    """A single impact Key Performance Indicator."""
    kpi_id: str = Field(default="", description="KPI identifier")
    kpi_name: str = Field(default="", description="KPI display name")
    value: float = Field(default=0.0, description="KPI value")
    unit: str = Field(default="", description="Unit of measurement")
    period: str = Field(default="", description="Measurement period")
    target_value: float = Field(default=0.0, description="Target value")
    achievement_pct: float = Field(
        default=0.0, ge=0.0, description="Achievement percentage"
    )
    data_source: str = Field(default="", description="Data source")
    verification_status: str = Field(
        default="unverified", description="Verification status"
    )


class SDGContribution(BaseModel):
    """SDG contribution assessment for a holding."""
    sdg_goal: int = Field(default=0, ge=1, le=17, description="SDG goal number")
    sdg_name: str = Field(default="", description="SDG goal name")
    contribution_level: str = Field(
        default="none", description="Contribution level"
    )
    direction: str = Field(default="neutral", description="Impact direction")
    revenue_aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Revenue aligned with this SDG (%)",
    )
    evidence: List[str] = Field(
        default_factory=list, description="Supporting evidence"
    )
    data_quality: str = Field(
        default="estimated", description="Data quality assessment"
    )


class InvesteeImpactData(BaseModel):
    """Impact data for a single investee company."""
    isin: str = Field(default="", description="Investee ISIN")
    company_name: str = Field(default="", description="Company name")
    sector: str = Field(default="", description="Company sector")
    weight_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Portfolio weight (%)"
    )
    impact_category: str = Field(
        default="environmental", description="Primary impact category"
    )
    sdg_contributions: List[SDGContribution] = Field(
        default_factory=list, description="SDG contribution assessments"
    )
    impact_kpis: List[ImpactKPI] = Field(
        default_factory=list, description="Impact KPIs"
    )
    overall_impact_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Overall impact score"
    )
    contribution_to_objective: str = Field(
        default="none", description="Contribution level to sustainable objective"
    )
    verification_status: str = Field(
        default="unverified", description="Verification status"
    )
    data_date: str = Field(default="", description="Data collection date")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class SDGAlignmentData(BaseModel):
    """Aggregated SDG alignment data for the portfolio."""
    sdg_goal: int = Field(default=0, description="SDG goal number")
    sdg_name: str = Field(default="", description="SDG goal name")
    aligned_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Portfolio weight aligned with this SDG (%)",
    )
    contributing_holdings: int = Field(
        default=0, description="Number of holdings contributing"
    )
    total_revenue_aligned_eur: float = Field(
        default=0.0, description="Total revenue aligned (EUR)"
    )
    average_contribution_level: str = Field(
        default="none", description="Average contribution level"
    )
    direction: str = Field(default="positive", description="Impact direction")


class ImpactVerification(BaseModel):
    """Result of impact data verification."""
    verification_id: str = Field(default="", description="Verification ID")
    provider: str = Field(default="", description="Verification provider")
    verified_at: str = Field(default="", description="Verification timestamp")
    total_holdings_verified: int = Field(
        default=0, description="Holdings verified"
    )
    total_holdings: int = Field(default=0, description="Total holdings")
    verification_rate_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Verification rate (%)"
    )
    verified_impact_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Impact score of verified portion",
    )
    unverified_impact_score: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Impact score of unverified portion",
    )
    findings: List[Dict[str, Any]] = Field(
        default_factory=list, description="Verification findings"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )
    overall_status: str = Field(
        default="pending", description="Overall verification status"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class PortfolioImpactResult(BaseModel):
    """Aggregated portfolio-level impact assessment."""
    portfolio_name: str = Field(default="", description="Portfolio name")
    reference_date: str = Field(default="", description="Reference date")
    total_holdings: int = Field(default=0, description="Total holdings")
    assessed_holdings: int = Field(
        default=0, description="Holdings with impact data"
    )
    coverage_pct: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Impact data coverage (%)"
    )

    # Aggregate impact
    overall_impact_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Portfolio impact score"
    )
    sdg_alignment_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="SDG alignment score (%)"
    )
    contribution_to_objective_pct: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Portfolio contributing to sustainable objective (%)",
    )

    # Per-SDG
    sdg_alignment: List[SDGAlignmentData] = Field(
        default_factory=list, description="Per-SDG alignment data"
    )

    # Per-holding
    investee_impacts: List[InvesteeImpactData] = Field(
        default_factory=list, description="Per-investee impact data"
    )

    # Aggregate KPIs
    aggregate_kpis: List[ImpactKPI] = Field(
        default_factory=list, description="Portfolio-level aggregate KPIs"
    )

    # Verification
    verification: Optional[ImpactVerification] = Field(
        default=None, description="Verification result"
    )

    errors: List[str] = Field(default_factory=list, description="Errors")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    calculated_at: str = Field(default="", description="Calculation timestamp")
    execution_time_ms: float = Field(default=0.0, description="Execution time")


# =============================================================================
# SDG Reference Data
# =============================================================================


SDG_NAMES: Dict[int, str] = {
    1: "No Poverty",
    2: "Zero Hunger",
    3: "Good Health and Well-Being",
    4: "Quality Education",
    5: "Gender Equality",
    6: "Clean Water and Sanitation",
    7: "Affordable and Clean Energy",
    8: "Decent Work and Economic Growth",
    9: "Industry, Innovation and Infrastructure",
    10: "Reduced Inequalities",
    11: "Sustainable Cities and Communities",
    12: "Responsible Consumption and Production",
    13: "Climate Action",
    14: "Life Below Water",
    15: "Life on Land",
    16: "Peace, Justice and Strong Institutions",
    17: "Partnerships for the Goals",
}


# =============================================================================
# Impact Data Bridge
# =============================================================================


class ImpactDataBridge:
    """Bridge for impact data intake, SDG alignment, and verification.

    Manages investee-level impact data collection, SDG contribution
    assessment, portfolio-level impact aggregation, and third-party
    verification workflows for Article 9 products.

    Attributes:
        config: Bridge configuration.
        _agents: Deferred agent stubs.

    Example:
        >>> bridge = ImpactDataBridge(ImpactDataConfig(sdg_targets=[7, 13]))
        >>> result = bridge.assess_portfolio_impact(holdings)
        >>> print(f"Impact score: {result.overall_impact_score:.1f}")
    """

    def __init__(self, config: Optional[ImpactDataConfig] = None) -> None:
        """Initialize the Impact Data Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or ImpactDataConfig()
        self.logger = logger

        self._agents: Dict[str, _AgentStub] = {
            "data_quality_profiler": _AgentStub(
                "GL-DATA-010",
                "greenlang.agents.data.data_quality_profiler",
                "DataQualityProfiler",
            ),
        }

        self.logger.info(
            "ImpactDataBridge initialized: sdg_targets=%s, categories=%s, "
            "kpis=%d, verification=%s",
            self.config.sdg_targets,
            self.config.impact_categories,
            len(self.config.impact_kpis),
            self.config.require_verification,
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def assess_portfolio_impact(
        self,
        holdings: List[Dict[str, Any]],
    ) -> PortfolioImpactResult:
        """Assess portfolio-level impact for Article 9 reporting.

        Processes each holding's impact data, calculates SDG alignment,
        aggregates KPIs, and optionally runs verification.

        Args:
            holdings: Portfolio holding records with impact data.

        Returns:
            PortfolioImpactResult with full impact assessment.
        """
        start_time = time.time()
        errors: List[str] = []
        warnings: List[str] = []

        # Process individual investee impacts
        investee_impacts: List[InvesteeImpactData] = []
        for holding in holdings:
            impact = self._assess_investee_impact(holding)
            investee_impacts.append(impact)

        # Calculate SDG alignment
        sdg_alignment = self._calculate_sdg_alignment(investee_impacts)

        # Aggregate KPIs
        aggregate_kpis = self._aggregate_kpis(investee_impacts)

        # Calculate portfolio scores
        assessed = [i for i in investee_impacts if i.overall_impact_score > 0]
        coverage_pct = (len(assessed) / len(holdings) * 100.0) if holdings else 0.0

        overall_score = self._calculate_weighted_impact_score(investee_impacts)
        sdg_score = self._calculate_sdg_alignment_score(sdg_alignment)
        contribution_pct = self._calculate_contribution_pct(investee_impacts)

        # Coverage warning for Article 9
        if coverage_pct < 80.0:
            warnings.append(
                f"Impact data coverage ({coverage_pct:.1f}%) below recommended 80% "
                "for Article 9 products"
            )

        # SDG alignment warning
        if sdg_score < self.config.min_sdg_alignment_pct:
            warnings.append(
                f"SDG alignment ({sdg_score:.1f}%) below minimum threshold "
                f"({self.config.min_sdg_alignment_pct:.1f}%)"
            )

        # Verification
        verification = None
        if self.config.require_verification:
            verification = self._run_verification(investee_impacts)

        elapsed_ms = (time.time() - start_time) * 1000

        result = PortfolioImpactResult(
            portfolio_name="Article 9 Portfolio",
            reference_date=_utcnow().strftime("%Y-%m-%d"),
            total_holdings=len(holdings),
            assessed_holdings=len(assessed),
            coverage_pct=coverage_pct,
            overall_impact_score=overall_score,
            sdg_alignment_score=sdg_score,
            contribution_to_objective_pct=contribution_pct,
            sdg_alignment=sdg_alignment,
            investee_impacts=investee_impacts,
            aggregate_kpis=aggregate_kpis,
            verification=verification,
            errors=errors,
            warnings=warnings,
            calculated_at=_utcnow().isoformat(),
            execution_time_ms=elapsed_ms,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _hash_data(
                result.model_dump(
                    exclude={"provenance_hash", "investee_impacts", "verification"}
                )
            )

        self.logger.info(
            "ImpactDataBridge: assessed %d/%d holdings, impact=%.1f, "
            "sdg=%.1f%%, contribution=%.1f%%, elapsed=%.1fms",
            len(assessed), len(holdings), overall_score,
            sdg_score, contribution_pct, elapsed_ms,
        )
        return result

    def assess_investee(
        self,
        holding: Dict[str, Any],
    ) -> InvesteeImpactData:
        """Assess impact for a single investee.

        Args:
            holding: Holding record with impact data fields.

        Returns:
            InvesteeImpactData with impact assessment.
        """
        return self._assess_investee_impact(holding)

    def get_sdg_mapping(self) -> Dict[int, str]:
        """Get SDG goal number to name mapping.

        Returns:
            Dict mapping SDG numbers to goal names.
        """
        return dict(SDG_NAMES)

    def validate_impact_data(
        self,
        holdings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate impact data completeness for Article 9 requirements.

        Args:
            holdings: Portfolio holding records.

        Returns:
            Validation result with per-field coverage analysis.
        """
        total = len(holdings)
        if total == 0:
            return {"valid": False, "reason": "No holdings provided"}

        fields = [
            "impact_score", "sdg_contributions", "impact_kpis",
            "verification_status", "impact_category",
        ]

        field_coverage: Dict[str, float] = {}
        for field in fields:
            covered = sum(1 for h in holdings if h.get(field) is not None)
            field_coverage[field] = (covered / total) * 100.0

        overall_coverage = sum(field_coverage.values()) / len(field_coverage)
        is_valid = overall_coverage >= 50.0

        return {
            "valid": is_valid,
            "overall_coverage_pct": overall_coverage,
            "field_coverage": field_coverage,
            "total_holdings": total,
            "validated_at": _utcnow().isoformat(),
        }

    # -------------------------------------------------------------------------
    # Private Methods
    # -------------------------------------------------------------------------

    def _assess_investee_impact(
        self,
        holding: Dict[str, Any],
    ) -> InvesteeImpactData:
        """Assess impact for a single investee holding."""
        isin = str(holding.get("isin", ""))
        name = str(holding.get("name", holding.get("company_name", "")))

        # Parse SDG contributions
        sdg_raw = holding.get("sdg_contributions", [])
        sdg_contributions: List[SDGContribution] = []
        for sdg_entry in sdg_raw:
            if isinstance(sdg_entry, dict):
                goal = int(sdg_entry.get("sdg_goal", 0))
                sdg_contributions.append(SDGContribution(
                    sdg_goal=goal if 1 <= goal <= 17 else 1,
                    sdg_name=SDG_NAMES.get(goal, ""),
                    contribution_level=str(sdg_entry.get("level", "none")),
                    direction=str(sdg_entry.get("direction", "neutral")),
                    revenue_aligned_pct=float(sdg_entry.get("revenue_aligned_pct", 0.0)),
                    evidence=sdg_entry.get("evidence", []),
                ))

        # Auto-assess against configured SDG targets
        if not sdg_contributions and self.config.sdg_targets:
            for target in self.config.sdg_targets:
                sdg_contributions.append(SDGContribution(
                    sdg_goal=target,
                    sdg_name=SDG_NAMES.get(target, ""),
                    contribution_level="none",
                    direction="neutral",
                ))

        # Parse KPIs
        kpi_raw = holding.get("impact_kpis", [])
        impact_kpis: List[ImpactKPI] = []
        for kpi_entry in kpi_raw:
            if isinstance(kpi_entry, dict):
                value = float(kpi_entry.get("value", 0.0))
                target = float(kpi_entry.get("target", 0.0))
                achievement = (value / target * 100.0) if target > 0 else 0.0
                impact_kpis.append(ImpactKPI(
                    kpi_id=str(kpi_entry.get("id", "")),
                    kpi_name=str(kpi_entry.get("name", "")),
                    value=value,
                    unit=str(kpi_entry.get("unit", "")),
                    period=str(kpi_entry.get("period", "")),
                    target_value=target,
                    achievement_pct=achievement,
                    data_source=str(kpi_entry.get("source", "")),
                    verification_status=str(kpi_entry.get("verified", "unverified")),
                ))

        # Calculate impact score
        impact_score = float(holding.get("impact_score", 0.0))
        if impact_score == 0.0 and sdg_contributions:
            impact_score = self._estimate_impact_score(sdg_contributions)

        # Determine contribution level
        contribution = self._determine_contribution_level(
            impact_score, sdg_contributions,
        )

        result = InvesteeImpactData(
            isin=isin,
            company_name=name,
            sector=str(holding.get("sector", "")),
            weight_pct=float(holding.get("weight", holding.get("weight_pct", 0.0))),
            impact_category=str(holding.get("impact_category", "environmental")),
            sdg_contributions=sdg_contributions,
            impact_kpis=impact_kpis,
            overall_impact_score=impact_score,
            contribution_to_objective=contribution,
            verification_status=str(holding.get("verification_status", "unverified")),
            data_date=str(holding.get("data_date", _utcnow().strftime("%Y-%m-%d"))),
        )

        if self.config.enable_provenance:
            result.provenance_hash = _hash_data(
                result.model_dump(exclude={"provenance_hash"})
            )

        return result

    def _calculate_sdg_alignment(
        self,
        investee_impacts: List[InvesteeImpactData],
    ) -> List[SDGAlignmentData]:
        """Calculate per-SDG alignment across the portfolio."""
        sdg_data: Dict[int, Dict[str, Any]] = {}

        for impact in investee_impacts:
            for sdg in impact.sdg_contributions:
                goal = sdg.sdg_goal
                if goal not in sdg_data:
                    sdg_data[goal] = {
                        "aligned_weight": 0.0,
                        "contributing_count": 0,
                        "total_revenue": 0.0,
                        "levels": [],
                    }
                if sdg.contribution_level in ("high", "medium"):
                    sdg_data[goal]["aligned_weight"] += impact.weight_pct
                    sdg_data[goal]["contributing_count"] += 1
                sdg_data[goal]["levels"].append(sdg.contribution_level)

        result: List[SDGAlignmentData] = []
        for goal, data in sorted(sdg_data.items()):
            avg_level = self._average_contribution_level(data["levels"])
            result.append(SDGAlignmentData(
                sdg_goal=goal,
                sdg_name=SDG_NAMES.get(goal, ""),
                aligned_pct=data["aligned_weight"],
                contributing_holdings=data["contributing_count"],
                average_contribution_level=avg_level,
                direction="positive",
            ))

        return result

    def _aggregate_kpis(
        self,
        investee_impacts: List[InvesteeImpactData],
    ) -> List[ImpactKPI]:
        """Aggregate impact KPIs across all investees."""
        kpi_totals: Dict[str, Dict[str, Any]] = {}

        for impact in investee_impacts:
            for kpi in impact.impact_kpis:
                if kpi.kpi_name not in kpi_totals:
                    kpi_totals[kpi.kpi_name] = {
                        "total": 0.0,
                        "count": 0,
                        "unit": kpi.unit,
                        "target_total": 0.0,
                    }
                kpi_totals[kpi.kpi_name]["total"] += kpi.value
                kpi_totals[kpi.kpi_name]["count"] += 1
                kpi_totals[kpi.kpi_name]["target_total"] += kpi.target_value

        result: List[ImpactKPI] = []
        for name, data in kpi_totals.items():
            total = data["total"]
            target = data["target_total"]
            result.append(ImpactKPI(
                kpi_id=f"agg_{name}",
                kpi_name=name,
                value=total,
                unit=data["unit"],
                target_value=target,
                achievement_pct=(total / target * 100.0) if target > 0 else 0.0,
                data_source="aggregated",
            ))

        return result

    def _calculate_weighted_impact_score(
        self,
        investee_impacts: List[InvesteeImpactData],
    ) -> float:
        """Calculate portfolio weighted average impact score."""
        total_weight = 0.0
        weighted_score = 0.0
        for impact in investee_impacts:
            w = impact.weight_pct
            if w > 0:
                weighted_score += w * impact.overall_impact_score
                total_weight += w
        return (weighted_score / total_weight) if total_weight > 0 else 0.0

    def _calculate_sdg_alignment_score(
        self,
        sdg_alignment: List[SDGAlignmentData],
    ) -> float:
        """Calculate overall SDG alignment score."""
        if not sdg_alignment:
            return 0.0
        target_sdgs = set(self.config.sdg_targets) if self.config.sdg_targets else set()
        if not target_sdgs:
            return sum(s.aligned_pct for s in sdg_alignment) / len(sdg_alignment)

        relevant = [s for s in sdg_alignment if s.sdg_goal in target_sdgs]
        if not relevant:
            return 0.0
        return sum(s.aligned_pct for s in relevant) / len(relevant)

    def _calculate_contribution_pct(
        self,
        investee_impacts: List[InvesteeImpactData],
    ) -> float:
        """Calculate percentage of portfolio contributing to objective."""
        contributing_weight = sum(
            i.weight_pct for i in investee_impacts
            if i.contribution_to_objective in ("high", "medium")
        )
        total_weight = sum(i.weight_pct for i in investee_impacts)
        return (contributing_weight / total_weight * 100.0) if total_weight > 0 else 0.0

    def _estimate_impact_score(
        self,
        sdg_contributions: List[SDGContribution],
    ) -> float:
        """Estimate impact score from SDG contributions."""
        level_scores = {
            "high": 85.0, "medium": 60.0, "low": 30.0,
            "none": 0.0, "negative": -20.0,
        }
        if not sdg_contributions:
            return 0.0
        scores = [
            level_scores.get(s.contribution_level, 0.0)
            for s in sdg_contributions
        ]
        return max(0.0, sum(scores) / len(scores))

    def _determine_contribution_level(
        self,
        impact_score: float,
        sdg_contributions: List[SDGContribution],
    ) -> str:
        """Determine contribution level to sustainable objective."""
        if impact_score >= 70.0:
            return ContributionLevel.HIGH.value
        elif impact_score >= 40.0:
            return ContributionLevel.MEDIUM.value
        elif impact_score >= 10.0:
            return ContributionLevel.LOW.value
        else:
            return ContributionLevel.NONE.value

    def _average_contribution_level(
        self,
        levels: List[str],
    ) -> str:
        """Calculate average contribution level."""
        level_scores = {
            "high": 3, "medium": 2, "low": 1, "none": 0, "negative": -1,
        }
        score_levels = {3: "high", 2: "medium", 1: "low", 0: "none", -1: "negative"}
        if not levels:
            return "none"
        avg = sum(level_scores.get(l, 0) for l in levels) / len(levels)
        rounded = round(avg)
        return score_levels.get(rounded, "none")

    def _run_verification(
        self,
        investee_impacts: List[InvesteeImpactData],
    ) -> ImpactVerification:
        """Run impact data verification workflow."""
        total = len(investee_impacts)
        verified = sum(
            1 for i in investee_impacts
            if i.verification_status == "verified"
        )
        verified_scores = [
            i.overall_impact_score for i in investee_impacts
            if i.verification_status == "verified"
        ]
        unverified_scores = [
            i.overall_impact_score for i in investee_impacts
            if i.verification_status != "verified"
        ]

        findings: List[Dict[str, Any]] = []
        if verified < total:
            findings.append({
                "type": "coverage_gap",
                "message": f"{total - verified} holdings lack third-party verification",
                "severity": "medium",
            })

        recommendations: List[str] = []
        if verified / total < 0.5 if total > 0 else True:
            recommendations.append(
                "Engage third-party verification provider for remaining holdings"
            )

        verification = ImpactVerification(
            verification_id=f"VER-{_utcnow().strftime('%Y%m%d%H%M%S')}",
            provider=self.config.verification_provider or "internal",
            verified_at=_utcnow().isoformat(),
            total_holdings_verified=verified,
            total_holdings=total,
            verification_rate_pct=(verified / total * 100.0) if total > 0 else 0.0,
            verified_impact_score=(
                sum(verified_scores) / len(verified_scores)
                if verified_scores else 0.0
            ),
            unverified_impact_score=(
                sum(unverified_scores) / len(unverified_scores)
                if unverified_scores else 0.0
            ),
            findings=findings,
            recommendations=recommendations,
            overall_status="verified" if verified == total else "partial",
        )

        if self.config.enable_provenance:
            verification.provenance_hash = _hash_data(
                verification.model_dump(exclude={"provenance_hash"})
            )

        return verification
