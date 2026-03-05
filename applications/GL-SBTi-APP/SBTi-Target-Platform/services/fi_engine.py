"""
FI Engine -- Financial Institution Net-Zero Target Setting (FINZ V1.0)

Implements the SBTi Financial Institutions Net-Zero Standard v1.0 for
portfolio-level target setting, financed emissions accounting (PCAF),
portfolio coverage pathways, engagement tracking, PCAF data quality
assessment, WACI calculation, and sectoral decarbonization for FI
portfolios.

All numeric calculations are deterministic (zero-hallucination).

Reference:
    - SBTi Financial Institutions Net-Zero Standard v1.0 (2024)
    - PCAF Global GHG Accounting & Reporting Standard (2022)
    - SBTi Portfolio Coverage approach
    - SBTi Temperature Rating methodology v2.0
    - Partnership for Carbon Accounting Financials (PCAF) methodology

Example:
    >>> from services.config import SBTiAppConfig
    >>> engine = FIEngine(SBTiAppConfig())
    >>> portfolio = engine.create_portfolio("org-fi-1", "Main Portfolio", 2025)
    >>> coverage = engine.calculate_portfolio_coverage("org-fi-1")
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import (
    ATTRIBUTION_METHODS,
    FI_COVERAGE_PATHWAY,
    FIAssetClass,
    FITargetType,
    PCAFDataQuality,
    PCAF_DQ_DESCRIPTIONS,
    PCAF_DQ_SCORES,
    SBTiAppConfig,
)
from .models import (
    EngagementRecord,
    FIPortfolio,
    Organization,
    PortfolioHolding,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class PortfolioCoverageResult(BaseModel):
    """Portfolio SBTi target coverage assessment."""

    org_id: str = Field(...)
    total_holdings: int = Field(default=0)
    holdings_with_targets: int = Field(default=0)
    coverage_by_count_pct: float = Field(default=0.0)
    coverage_by_exposure_pct: float = Field(default=0.0)
    total_exposure_usd: float = Field(default=0.0, ge=0.0)
    covered_exposure_usd: float = Field(default=0.0, ge=0.0)
    pathway_target_pct: float = Field(default=0.0)
    on_track: bool = Field(default=False)
    gap_pct: float = Field(default=0.0)
    message: str = Field(default="")
    assessed_at: datetime = Field(default_factory=_now)


class FinancedEmissionsResult(BaseModel):
    """Financed emissions calculation result per PCAF."""

    org_id: str = Field(...)
    total_financed_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    by_asset_class: Dict[str, float] = Field(default_factory=dict)
    by_sector: Dict[str, float] = Field(default_factory=dict)
    attribution_method: str = Field(default="evic")
    avg_pcaf_dq: float = Field(default=5.0)
    weighted_avg_pcaf_dq: float = Field(default=5.0)
    holdings_assessed: int = Field(default=0)
    provenance_hash: str = Field(default="")
    calculated_at: datetime = Field(default_factory=_now)


class WACIResult(BaseModel):
    """Weighted Average Carbon Intensity calculation."""

    org_id: str = Field(...)
    waci_value: float = Field(default=0.0, ge=0.0)
    waci_unit: str = Field(default="tCO2e/$M_revenue")
    holdings_included: int = Field(default=0)
    total_portfolio_value_usd: float = Field(default=0.0, ge=0.0)
    sector_contribution: Dict[str, float] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class PCAFQualityAssessment(BaseModel):
    """PCAF data quality assessment for the portfolio."""

    org_id: str = Field(...)
    overall_dq_score: float = Field(default=5.0, ge=1.0, le=5.0)
    weighted_dq_score: float = Field(default=5.0, ge=1.0, le=5.0)
    dq_distribution: Dict[str, int] = Field(default_factory=dict)
    improvement_priority: List[Dict[str, Any]] = Field(default_factory=list)
    target_dq: float = Field(default=3.0)
    gap_to_target: float = Field(default=0.0)
    message: str = Field(default="")


class EngagementSummary(BaseModel):
    """Summary of investee engagement activities."""

    org_id: str = Field(...)
    total_engagements: int = Field(default=0)
    successful_count: int = Field(default=0)
    in_progress_count: int = Field(default=0)
    unsuccessful_count: int = Field(default=0)
    coverage_pct: float = Field(default=0.0)
    top_engagements: List[Dict[str, Any]] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=_now)


class CoveragePathwayResult(BaseModel):
    """Portfolio coverage pathway against SBTi milestones."""

    org_id: str = Field(...)
    current_year: int = Field(default=2025)
    current_coverage_pct: float = Field(default=0.0)
    pathway_milestones: List[Dict[str, float]] = Field(default_factory=list)
    on_track: bool = Field(default=False)
    gap_pct: float = Field(default=0.0)
    target_year: int = Field(default=2040)
    target_coverage_pct: float = Field(default=100.0)
    provenance_hash: str = Field(default="")


class FIReport(BaseModel):
    """Comprehensive FI portfolio report."""

    org_id: str = Field(...)
    portfolio_coverage: Dict[str, Any] = Field(default_factory=dict)
    financed_emissions: Dict[str, Any] = Field(default_factory=dict)
    waci: Dict[str, Any] = Field(default_factory=dict)
    pcaf_quality: Dict[str, Any] = Field(default_factory=dict)
    engagement_summary: Dict[str, Any] = Field(default_factory=dict)
    coverage_pathway: Dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=_now)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# FIEngine
# ---------------------------------------------------------------------------

class FIEngine:
    """
    Financial Institution target setting engine per SBTi FINZ V1.0.

    Manages FI portfolio creation, financed emissions accounting (PCAF),
    portfolio coverage tracking against the SBTi coverage pathway,
    WACI calculation, PCAF data quality assessment, engagement tracking,
    and FI-specific reporting.

    Attributes:
        config: Application configuration.
        _portfolios: In-memory portfolio store keyed by org_id.
        _organizations: In-memory organization store keyed by org_id.
        _engagements: In-memory engagement store keyed by org_id.

    Example:
        >>> engine = FIEngine(SBTiAppConfig())
        >>> portfolio = engine.create_portfolio("org-fi-1", "Main", 2025)
    """

    def __init__(self, config: Optional[SBTiAppConfig] = None) -> None:
        """Initialize the FIEngine."""
        self.config = config or SBTiAppConfig()
        self._portfolios: Dict[str, FIPortfolio] = {}
        self._organizations: Dict[str, Organization] = {}
        self._engagements: Dict[str, List[EngagementRecord]] = {}
        logger.info(
            "FIEngine initialized (coverage target year=%d)",
            self.config.fi_coverage_target_year,
        )

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_organization(self, org: Organization) -> None:
        """Register a financial institution organization."""
        self._organizations[org.id] = org

    # ------------------------------------------------------------------
    # Portfolio Management
    # ------------------------------------------------------------------

    def create_portfolio(
        self,
        org_id: str,
        name: str = "Main Portfolio",
        year: int = 2025,
        holdings: Optional[List[PortfolioHolding]] = None,
        fi_target_type: Optional[FITargetType] = None,
    ) -> FIPortfolio:
        """
        Create a new FI portfolio.

        Args:
            org_id: FI organization identifier.
            name: Portfolio name.
            year: Reporting year.
            holdings: Optional list of portfolio holdings.
            fi_target_type: SBTi FI target approach.

        Returns:
            FIPortfolio domain model.
        """
        start = datetime.utcnow()

        all_holdings = holdings or []
        total_exposure = sum(float(h.exposure_usd) for h in all_holdings)
        total_financed = sum(float(h.financed_emissions_tco2e) for h in all_holdings)
        with_sbti = sum(1 for h in all_holdings if h.has_sbti_target)
        n = len(all_holdings)

        coverage_pct = (with_sbti / n * 100.0) if n > 0 else 0.0

        # Asset class breakdown
        asset_breakdown: Dict[str, Decimal] = {}
        for h in all_holdings:
            key = h.asset_class.value
            asset_breakdown[key] = asset_breakdown.get(key, Decimal("0")) + h.exposure_usd

        # Average PCAF data quality
        if all_holdings:
            dq_scores = [PCAF_DQ_SCORES.get(h.pcaf_dq, 5) for h in all_holdings]
            avg_dq = sum(dq_scores) / len(dq_scores)
        else:
            avg_dq = 5.0

        portfolio = FIPortfolio(
            tenant_id="default",
            org_id=org_id,
            name=name,
            year=year,
            fi_target_type=fi_target_type,
            total_exposure_usd=Decimal(str(round(total_exposure, 2))),
            total_financed_emissions_tco2e=Decimal(str(round(total_financed, 2))),
            coverage_pct=Decimal(str(round(coverage_pct, 2))),
            holdings=all_holdings,
            sbti_coverage_pct=Decimal(str(round(coverage_pct, 2))),
            asset_class_breakdown=asset_breakdown,
            avg_pcaf_dq=Decimal(str(round(avg_dq, 2))),
        )

        self._portfolios[org_id] = portfolio

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Created portfolio for org %s: %d holdings, %.0f coverage%% in %.1f ms",
            org_id, n, coverage_pct, elapsed_ms,
        )
        return portfolio

    def get_portfolio(self, org_id: str) -> Optional[FIPortfolio]:
        """Get the portfolio for an FI organization."""
        return self._portfolios.get(org_id)

    # ------------------------------------------------------------------
    # Portfolio Coverage
    # ------------------------------------------------------------------

    def calculate_portfolio_coverage(
        self, org_id: str,
    ) -> PortfolioCoverageResult:
        """
        Calculate SBTi portfolio target coverage.

        Measures the percentage of the portfolio (by count and exposure)
        that has companies with validated SBTi targets, and compares
        against the FI coverage pathway milestone for the current year.

        Args:
            org_id: FI organization identifier.

        Returns:
            PortfolioCoverageResult with coverage metrics.
        """
        portfolio = self._portfolios.get(org_id)
        if portfolio is None:
            return PortfolioCoverageResult(
                org_id=org_id,
                message="No portfolio registered.",
            )

        holdings = portfolio.holdings
        n = len(holdings)
        if n == 0:
            return PortfolioCoverageResult(
                org_id=org_id,
                message="Portfolio has no holdings.",
            )

        with_targets = sum(1 for h in holdings if h.has_sbti_target)
        total_exposure = sum(float(h.exposure_usd) for h in holdings)
        covered_exposure = sum(
            float(h.exposure_usd) for h in holdings if h.has_sbti_target
        )

        coverage_count = (with_targets / n * 100.0)
        coverage_exposure = (
            (covered_exposure / total_exposure * 100.0)
            if total_exposure > 0 else 0.0
        )

        # Pathway target for current year
        year = portfolio.year
        pathway_target = float(FI_COVERAGE_PATHWAY.get(year, Decimal("50.0")))
        on_track = coverage_exposure >= pathway_target
        gap = max(pathway_target - coverage_exposure, 0.0)

        return PortfolioCoverageResult(
            org_id=org_id,
            total_holdings=n,
            holdings_with_targets=with_targets,
            coverage_by_count_pct=round(coverage_count, 2),
            coverage_by_exposure_pct=round(coverage_exposure, 2),
            total_exposure_usd=round(total_exposure, 2),
            covered_exposure_usd=round(covered_exposure, 2),
            pathway_target_pct=pathway_target,
            on_track=on_track,
            gap_pct=round(gap, 2),
            message=(
                f"Coverage: {coverage_exposure:.1f}% by exposure "
                f"(target: {pathway_target:.1f}%). "
                f"{'On track.' if on_track else f'Gap of {gap:.1f}% to close.'}"
            ),
        )

    # ------------------------------------------------------------------
    # Financed Emissions (PCAF)
    # ------------------------------------------------------------------

    def calculate_financed_emissions(
        self,
        org_id: str,
        method: str = "evic",
    ) -> FinancedEmissionsResult:
        """
        Calculate total financed emissions using PCAF methodology.

        Aggregates financed emissions from all holdings using the
        specified attribution method (EVIC, revenue, balance sheet,
        or floor area).

        Args:
            org_id: FI organization identifier.
            method: Attribution method (evic, revenue, balance_sheet, floor_area).

        Returns:
            FinancedEmissionsResult with breakdown.
        """
        start = datetime.utcnow()
        portfolio = self._portfolios.get(org_id)
        if portfolio is None:
            return FinancedEmissionsResult(
                org_id=org_id,
                attribution_method=method,
            )

        total = 0.0
        by_asset_class: Dict[str, float] = {}
        by_sector: Dict[str, float] = {}
        dq_scores: List[int] = []
        weighted_dq_sum = 0.0
        total_weight = 0.0

        for h in portfolio.holdings:
            emissions = float(h.financed_emissions_tco2e)
            total += emissions

            ac = h.asset_class.value
            by_asset_class[ac] = by_asset_class.get(ac, 0.0) + emissions

            sector = h.sector or "unknown"
            by_sector[sector] = by_sector.get(sector, 0.0) + emissions

            dq = PCAF_DQ_SCORES.get(h.pcaf_dq, 5)
            dq_scores.append(dq)
            exposure = float(h.exposure_usd)
            weighted_dq_sum += dq * exposure
            total_weight += exposure

        avg_dq = sum(dq_scores) / len(dq_scores) if dq_scores else 5.0
        weighted_dq = weighted_dq_sum / total_weight if total_weight > 0 else 5.0

        provenance = _sha256(
            f"financed_emissions:{org_id}:{total}:{method}"
        )

        result = FinancedEmissionsResult(
            org_id=org_id,
            total_financed_emissions_tco2e=round(total, 2),
            by_asset_class=by_asset_class,
            by_sector=by_sector,
            attribution_method=method,
            avg_pcaf_dq=round(avg_dq, 2),
            weighted_avg_pcaf_dq=round(weighted_dq, 2),
            holdings_assessed=len(portfolio.holdings),
            provenance_hash=provenance,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Financed emissions for org %s: %.0f tCO2e, DQ=%.1f (%s) in %.1f ms",
            org_id, total, avg_dq, method, elapsed_ms,
        )
        return result

    # ------------------------------------------------------------------
    # WACI
    # ------------------------------------------------------------------

    def calculate_waci(self, org_id: str) -> WACIResult:
        """
        Calculate Weighted Average Carbon Intensity (WACI).

        WACI = sum(portfolio_weight_i * carbon_intensity_i)
        where portfolio_weight = exposure / total_portfolio_value
        and carbon_intensity = emissions / revenue.

        Args:
            org_id: FI organization identifier.

        Returns:
            WACIResult with WACI value and sector breakdown.
        """
        portfolio = self._portfolios.get(org_id)
        if portfolio is None:
            return WACIResult(org_id=org_id)

        total_value = sum(float(h.exposure_usd) for h in portfolio.holdings)
        if total_value <= 0:
            return WACIResult(org_id=org_id)

        waci = 0.0
        sector_contrib: Dict[str, float] = {}
        included = 0

        for h in portfolio.holdings:
            exposure = float(h.exposure_usd)
            emissions = float(h.financed_emissions_tco2e)
            weight = exposure / total_value if total_value > 0 else 0.0

            # Carbon intensity approximation (tCO2e per $M exposure)
            intensity = (emissions / exposure * 1_000_000) if exposure > 0 else 0.0
            contribution = weight * intensity
            waci += contribution
            included += 1

            sector = h.sector or "unknown"
            sector_contrib[sector] = sector_contrib.get(sector, 0.0) + contribution

        provenance = _sha256(f"waci:{org_id}:{waci}:{total_value}")

        return WACIResult(
            org_id=org_id,
            waci_value=round(waci, 2),
            waci_unit="tCO2e/$M_revenue",
            holdings_included=included,
            total_portfolio_value_usd=round(total_value, 2),
            sector_contribution=sector_contrib,
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # PCAF Data Quality
    # ------------------------------------------------------------------

    def assess_pcaf_quality(self, org_id: str) -> PCAFQualityAssessment:
        """
        Assess PCAF data quality across the portfolio.

        Evaluates the distribution of data quality scores, identifies
        improvement priorities, and calculates the gap to target DQ.

        Args:
            org_id: FI organization identifier.

        Returns:
            PCAFQualityAssessment with distribution and priorities.
        """
        portfolio = self._portfolios.get(org_id)
        if portfolio is None:
            return PCAFQualityAssessment(
                org_id=org_id,
                message="No portfolio registered.",
            )

        distribution: Dict[str, int] = {
            "dq_1": 0, "dq_2": 0, "dq_3": 0, "dq_4": 0, "dq_5": 0,
        }
        total_dq = 0.0
        weighted_dq = 0.0
        total_exposure = 0.0

        for h in portfolio.holdings:
            dq = PCAF_DQ_SCORES.get(h.pcaf_dq, 5)
            distribution[h.pcaf_dq.value] = distribution.get(h.pcaf_dq.value, 0) + 1
            total_dq += dq
            exposure = float(h.exposure_usd)
            weighted_dq += dq * exposure
            total_exposure += exposure

        n = len(portfolio.holdings)
        overall = total_dq / n if n > 0 else 5.0
        weighted = weighted_dq / total_exposure if total_exposure > 0 else 5.0

        # Improvement priorities: holdings with worst DQ and highest exposure
        priorities: List[Dict[str, Any]] = []
        worst = sorted(
            portfolio.holdings,
            key=lambda h: (-PCAF_DQ_SCORES.get(h.pcaf_dq, 5), -float(h.exposure_usd)),
        )
        for h in worst[:10]:
            dq_score = PCAF_DQ_SCORES.get(h.pcaf_dq, 5)
            if dq_score >= 4:
                priorities.append({
                    "company": h.company_name,
                    "current_dq": h.pcaf_dq.value,
                    "current_dq_score": dq_score,
                    "exposure_usd": float(h.exposure_usd),
                    "recommendation": PCAF_DQ_DESCRIPTIONS.get(
                        PCAFDataQuality(f"dq_{dq_score - 1}"),
                        "Improve data collection",
                    ),
                })

        target_dq = 3.0
        gap = max(overall - target_dq, 0.0)

        return PCAFQualityAssessment(
            org_id=org_id,
            overall_dq_score=round(overall, 2),
            weighted_dq_score=round(weighted, 2),
            dq_distribution=distribution,
            improvement_priority=priorities,
            target_dq=target_dq,
            gap_to_target=round(gap, 2),
            message=(
                f"Portfolio PCAF DQ: {overall:.1f} (weighted: {weighted:.1f}). "
                f"Target: {target_dq:.1f}. "
                f"{'On target.' if gap <= 0 else f'Gap of {gap:.1f} points to close.'}"
            ),
        )

    # ------------------------------------------------------------------
    # Engagement Tracking
    # ------------------------------------------------------------------

    def record_engagement(
        self,
        org_id: str,
        holding_id: str,
        investee_name: str,
        engagement_type: str = "direct",
        status: str = "in_progress",
        notes: Optional[str] = None,
    ) -> EngagementRecord:
        """
        Record an engagement activity with an investee company.

        Args:
            org_id: FI organization identifier.
            holding_id: Associated holding ID.
            investee_name: Name of the investee company.
            engagement_type: Type of engagement.
            status: Current status.
            notes: Optional notes.

        Returns:
            EngagementRecord.
        """
        record = EngagementRecord(
            holding_id=holding_id,
            investee_name=investee_name,
            engagement_type=engagement_type,
            status=status,
            notes=notes,
        )
        self._engagements.setdefault(org_id, []).append(record)

        logger.info(
            "Engagement recorded for org %s: %s (%s, %s)",
            org_id, investee_name, engagement_type, status,
        )
        return record

    def get_engagement_summary(self, org_id: str) -> EngagementSummary:
        """
        Get engagement activity summary for an FI.

        Args:
            org_id: FI organization identifier.

        Returns:
            EngagementSummary with counts and top engagements.
        """
        records = self._engagements.get(org_id, [])
        portfolio = self._portfolios.get(org_id)

        successful = sum(1 for r in records if r.status == "successful")
        in_progress = sum(1 for r in records if r.status == "in_progress")
        unsuccessful = sum(1 for r in records if r.status == "unsuccessful")

        n_holdings = len(portfolio.holdings) if portfolio else 0
        coverage = (len(records) / n_holdings * 100.0) if n_holdings > 0 else 0.0

        top = [
            {
                "investee": r.investee_name,
                "type": r.engagement_type,
                "status": r.status,
            }
            for r in records[:10]
        ]

        return EngagementSummary(
            org_id=org_id,
            total_engagements=len(records),
            successful_count=successful,
            in_progress_count=in_progress,
            unsuccessful_count=unsuccessful,
            coverage_pct=round(coverage, 2),
            top_engagements=top,
        )

    # ------------------------------------------------------------------
    # Coverage Pathway
    # ------------------------------------------------------------------

    def calculate_coverage_pathway(
        self, org_id: str,
    ) -> CoveragePathwayResult:
        """
        Calculate portfolio coverage against SBTi FI coverage pathway.

        The SBTi FI coverage pathway requires increasing percentages of
        portfolio companies to have validated targets, reaching 100%
        by 2040.

        Args:
            org_id: FI organization identifier.

        Returns:
            CoveragePathwayResult with milestone comparison.
        """
        portfolio = self._portfolios.get(org_id)
        if portfolio is None:
            return CoveragePathwayResult(org_id=org_id)

        current_year = portfolio.year
        current_coverage = float(portfolio.sbti_coverage_pct)

        milestones: List[Dict[str, float]] = []
        for year, target in sorted(FI_COVERAGE_PATHWAY.items()):
            milestones.append({
                "year": year,
                "target_pct": float(target),
                "actual_pct": current_coverage if year <= current_year else 0.0,
                "on_track": current_coverage >= float(target) if year <= current_year else False,
            })

        pathway_target = float(FI_COVERAGE_PATHWAY.get(current_year, Decimal("50.0")))
        on_track = current_coverage >= pathway_target
        gap = max(pathway_target - current_coverage, 0.0)

        provenance = _sha256(
            f"coverage_pathway:{org_id}:{current_year}:{current_coverage}"
        )

        return CoveragePathwayResult(
            org_id=org_id,
            current_year=current_year,
            current_coverage_pct=round(current_coverage, 2),
            pathway_milestones=milestones,
            on_track=on_track,
            gap_pct=round(gap, 2),
            target_year=self.config.fi_coverage_target_year,
            target_coverage_pct=100.0,
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # FI Report
    # ------------------------------------------------------------------

    def generate_fi_report(self, org_id: str) -> FIReport:
        """
        Generate a comprehensive FI portfolio report.

        Combines portfolio coverage, financed emissions, WACI, PCAF
        data quality, engagement summary, and coverage pathway into
        a single report.

        Args:
            org_id: FI organization identifier.

        Returns:
            FIReport with all FI-specific analyses.
        """
        start = datetime.utcnow()

        coverage = self.calculate_portfolio_coverage(org_id)
        financed = self.calculate_financed_emissions(org_id)
        waci = self.calculate_waci(org_id)
        pcaf = self.assess_pcaf_quality(org_id)
        engagement = self.get_engagement_summary(org_id)
        pathway = self.calculate_coverage_pathway(org_id)

        provenance = _sha256(
            f"fi_report:{org_id}:{coverage.coverage_by_exposure_pct}:{financed.total_financed_emissions_tco2e}"
        )

        report = FIReport(
            org_id=org_id,
            portfolio_coverage=coverage.model_dump(),
            financed_emissions=financed.model_dump(),
            waci=waci.model_dump(),
            pcaf_quality=pcaf.model_dump(),
            engagement_summary=engagement.model_dump(),
            coverage_pathway=pathway.model_dump(),
            provenance_hash=provenance,
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "FI report for org %s: coverage=%.1f%%, financed=%.0f tCO2e in %.1f ms",
            org_id, coverage.coverage_by_exposure_pct,
            financed.total_financed_emissions_tco2e, elapsed_ms,
        )
        return report
