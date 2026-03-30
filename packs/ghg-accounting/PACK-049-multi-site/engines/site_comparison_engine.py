# -*- coding: utf-8 -*-
"""
PACK-049 GHG Multi-Site Management Pack - Site Comparison Engine
=================================================================

Benchmarks and ranks facilities within a multi-site portfolio using
normalised key performance indicators (KPIs).  Constructs peer groups,
calculates descriptive statistics (mean, median, percentiles, std-dev),
ranks sites within their peer group, identifies best-practice facilities,
and quantifies the improvement potential across the portfolio.

KPI Calculation:
    kpi_value = numerator / denominator
    Example:  emissions_per_m2 = total_tCO2e / gross_internal_area_m2

Statistics:
    Mean:       mu    = SUM(x_i) / n
    Median:     p50   = interpolated 50th percentile
    Std Dev:    sigma = sqrt(SUM((x_i - mu)^2) / n)    (population)
    Percentiles: linear interpolation between adjacent sorted values

Ranking:
    Sites sorted ascending by KPI (lower = better for emissions intensity).
    Percentile: rank_percentile = (rank - 1) / (n - 1) * 100   (for n > 1)

Improvement Potential:
    gap_to_best   = site_kpi - best_kpi
    improvement_potential = SUM(gap_to_best_i) across all sites

Provenance:
    SHA-256 hash on every ComparisonResult.

Regulatory References:
    - GHG Protocol Corporate Standard (2004, rev 2015) - Tracking emissions over time
    - ISO 14064-1:2018 Clause 5 - Normalised indicators
    - EU CSRD / ESRS E1-4 - GHG intensity per net revenue
    - GRESB (2024) - Real estate sustainability benchmarking
    - CRREM (2024) - Carbon Risk Real Estate Monitor
    - EnergyStar Portfolio Manager - Building benchmarking

Zero-Hallucination:
    - All statistics computed with Decimal arithmetic
    - Percentile calculations use explicit linear interpolation
    - No LLM involvement in ranking or scoring
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-049 GHG Multi-Site Management
Engine:  7 of 10
Status:  Production Ready
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

_ZERO = Decimal("0")
_ONE = Decimal("1")
_TWO = Decimal("2")
_HUNDRED = Decimal("100")
_DP6 = Decimal("0.000001")
_DP10 = Decimal("0.0000000001")

def _safe_divide(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Divide with zero-guard."""
    if denominator == _ZERO:
        return _ZERO
    return (numerator / denominator).quantize(_DP6, rounding=ROUND_HALF_UP)

def _quantise(value: Decimal, precision: Decimal = _DP6) -> Decimal:
    """Quantise a Decimal to the requested precision."""
    return value.quantize(precision, rounding=ROUND_HALF_UP)

def _decimal_sqrt(value: Decimal) -> Decimal:
    """
    Integer-arithmetic Newton's method square root for Decimal.

    Returns a Decimal with 6 decimal places of precision.
    """
    if value < _ZERO:
        raise ValueError("Cannot compute square root of a negative number")
    if value == _ZERO:
        return _ZERO
    # Use Python's Decimal.sqrt() which is exact for arbitrary precision
    # then quantise to our standard precision
    result = value.sqrt()
    return _quantise(result, _DP6)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class KPIType(str, Enum):
    """Supported key performance indicator types."""
    EMISSIONS_PER_M2 = "EMISSIONS_PER_M2"
    EMISSIONS_PER_FTE = "EMISSIONS_PER_FTE"
    EMISSIONS_PER_UNIT = "EMISSIONS_PER_UNIT"
    EMISSIONS_PER_REVENUE = "EMISSIONS_PER_REVENUE"
    ENERGY_PER_M2 = "ENERGY_PER_M2"
    ENERGY_PER_FTE = "ENERGY_PER_FTE"
    WASTE_PER_FTE = "WASTE_PER_FTE"
    WATER_PER_M2 = "WATER_PER_M2"
    SCOPE1_PER_M2 = "SCOPE1_PER_M2"
    SCOPE2_PER_M2 = "SCOPE2_PER_M2"
    CUSTOM = "CUSTOM"

class RankDirection(str, Enum):
    """Direction of ranking: ASCENDING means lower is better."""
    ASCENDING = "ASCENDING"
    DESCENDING = "DESCENDING"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class SiteKPI(BaseModel):
    """A single KPI observation for a site."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    site_id: str = Field(..., description="Site identifier")
    kpi_type: str = Field(..., description="KPI type (e.g. EMISSIONS_PER_M2)")
    numerator: Decimal = Field(..., description="Numerator value (e.g. tCO2e)")
    denominator: Decimal = Field(..., description="Denominator value (e.g. m2)")
    value: Decimal = Field(_ZERO, description="Computed KPI value")
    unit: str = Field("", description="Unit of the KPI (e.g. tCO2e/m2)")
    period: str = Field("", description="Reporting period (e.g. 2025)")
    quality_score: Decimal = Field(
        Decimal("3"), ge=_ONE, le=Decimal("5"),
        description="Data quality score (1=best, 5=worst)",
    )

    def model_post_init(self, __context: Any) -> None:
        """Compute KPI value from numerator / denominator."""
        if self.value == _ZERO and self.denominator != _ZERO:
            self.value = _safe_divide(self.numerator, self.denominator)

class PeerGroup(BaseModel):
    """A group of comparable sites for benchmarking."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    group_id: str = Field(default_factory=_new_uuid, description="Peer group identifier")
    group_name: str = Field("", description="Human-readable peer group name")
    criteria: Dict[str, Any] = Field(
        default_factory=dict,
        description="Criteria used to construct the peer group (e.g. facility_type, region)",
    )
    member_site_ids: List[str] = Field(
        default_factory=list, description="Site IDs in this peer group"
    )

class SiteStatistics(BaseModel):
    """Descriptive statistics for a set of KPI values."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mean: Decimal = Field(_ZERO, description="Arithmetic mean")
    median: Decimal = Field(_ZERO, description="Median (50th percentile)")
    std_dev: Decimal = Field(_ZERO, description="Population standard deviation")
    p10: Decimal = Field(_ZERO, description="10th percentile")
    p25: Decimal = Field(_ZERO, description="25th percentile")
    p75: Decimal = Field(_ZERO, description="75th percentile")
    p90: Decimal = Field(_ZERO, description="90th percentile")
    min_val: Decimal = Field(_ZERO, description="Minimum value")
    max_val: Decimal = Field(_ZERO, description="Maximum value")
    count: int = Field(0, description="Number of observations")

class RankingResult(BaseModel):
    """Ranking of a single site within its peer group."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    site_id: str = Field(..., description="Site identifier")
    kpi_type: str = Field(..., description="KPI type used for ranking")
    rank: int = Field(1, ge=1, description="Rank within peer group (1 = best)")
    percentile: Decimal = Field(_ZERO, description="Percentile position (0 = best, 100 = worst)")
    value: Decimal = Field(_ZERO, description="Site KPI value")
    peer_group_mean: Decimal = Field(_ZERO, description="Peer group mean")
    peer_group_median: Decimal = Field(_ZERO, description="Peer group median")
    gap_to_best: Decimal = Field(_ZERO, description="Gap to best performer (value - min)")

class ComparisonResult(BaseModel):
    """Full comparison result for a peer group and KPI type."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    result_id: str = Field(default_factory=_new_uuid, description="Result identifier")
    peer_group: PeerGroup = Field(..., description="Peer group used for comparison")
    kpi_type: str = Field(..., description="KPI type compared")
    rankings: List[RankingResult] = Field(
        default_factory=list, description="Per-site ranking results"
    )
    statistics: SiteStatistics = Field(
        default_factory=SiteStatistics, description="Descriptive statistics"
    )
    best_practice_sites: List[str] = Field(
        default_factory=list, description="Site IDs of best performers"
    )
    improvement_potential: Decimal = Field(
        _ZERO, description="Total improvement potential if all sites matched best (tCO2e)"
    )
    created_at: datetime = Field(default_factory=utcnow, description="Timestamp")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = (
                f"{self.result_id}|{self.kpi_type}|"
                f"{self.peer_group.group_id}|{len(self.rankings)}|"
                f"{self.statistics.mean}|{self.statistics.median}|"
                f"{self.improvement_potential}"
            )
            self.provenance_hash = _compute_hash(payload)

class TrendPoint(BaseModel):
    """A single data point in a trend series."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    period: str = Field(..., description="Period label (e.g. '2023', 'Q1-2025')")
    value: Decimal = Field(..., description="KPI value for the period")
    rank: Optional[int] = Field(None, description="Rank within peer group for the period")

class SiteTrend(BaseModel):
    """Trend data for a site across multiple periods."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    site_id: str = Field(..., description="Site identifier")
    kpi_type: str = Field(..., description="KPI type")
    data_points: List[TrendPoint] = Field(default_factory=list, description="Trend data points")
    direction: str = Field("STABLE", description="IMPROVING, WORSENING, or STABLE")
    change_pct: Decimal = Field(_ZERO, description="Percentage change from first to last period")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            payload = f"{self.site_id}|{self.kpi_type}|{len(self.data_points)}|{self.change_pct}"
            self.provenance_hash = _compute_hash(payload)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SiteComparisonEngine:
    """
    Benchmarks and ranks facilities using normalised KPIs.

    Constructs peer groups, computes descriptive statistics, ranks sites,
    identifies best practices, and quantifies portfolio-wide improvement
    potential.

    All calculations use Decimal arithmetic.  Every result carries a
    SHA-256 provenance hash.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        rounding_precision: Decimal = _DP6,
        default_direction: str = RankDirection.ASCENDING.value,
        outlier_std_threshold: Decimal = Decimal("2"),
    ) -> None:
        """
        Initialise the SiteComparisonEngine.

        Args:
            rounding_precision: Decimal quantisation precision.
            default_direction: Default ranking direction (ASCENDING = lower is better).
            outlier_std_threshold: Std-dev multiplier for outlier detection.
        """
        self._precision = rounding_precision
        self._direction = default_direction
        self._outlier_threshold = outlier_std_threshold
        logger.info(
            "SiteComparisonEngine v%s initialised (direction=%s, outlier=%s)",
            _MODULE_VERSION, default_direction, outlier_std_threshold,
        )

    # ----------------------------------------------- peer group construction
    def build_peer_group(
        self,
        sites: List[Dict[str, Any]],
        criteria: Dict[str, Any],
    ) -> PeerGroup:
        """
        Construct a peer group by filtering sites matching all criteria.

        Each criterion key must match a field in the site dict.
        Supported match types:
            - Exact string match
            - List membership (criterion value is a list)
            - Numeric range (criterion value is a dict with min/max)

        Args:
            sites: List of site dicts with metadata fields.
            criteria: Filter criteria dict (e.g. {"facility_type": "OFFICE"}).

        Returns:
            PeerGroup with matching site IDs.

        Raises:
            ValueError: If fewer than 2 sites match criteria.
        """
        logger.info("Building peer group with criteria: %s from %d sites", criteria, len(sites))

        matching_ids: List[str] = []
        for site in sites:
            site_id = site.get("site_id", "")
            if not site_id:
                continue
            if self._site_matches_criteria(site, criteria):
                matching_ids.append(site_id)

        if len(matching_ids) < 2:
            raise ValueError(
                f"Peer group requires at least 2 sites, only {len(matching_ids)} matched criteria"
            )

        group_name = self._derive_group_name(criteria)

        peer_group = PeerGroup(
            group_name=group_name,
            criteria=criteria,
            member_site_ids=sorted(matching_ids),
        )

        logger.info(
            "Peer group built: id=%s name='%s' members=%d",
            peer_group.group_id, group_name, len(matching_ids),
        )
        return peer_group

    # ----------------------------------------------- KPI calculation
    def calculate_site_kpis(
        self,
        site_id: str,
        emissions: Dict[str, Decimal],
        denominators: Dict[str, Decimal],
        kpi_types: List[str],
        period: str = "",
    ) -> List[SiteKPI]:
        """
        Calculate KPIs for a single site across multiple KPI types.

        Args:
            site_id: Site identifier.
            emissions: Emissions by scope / type (e.g. {"total": Decimal("100")}).
            denominators: Denominator values (e.g. {"m2": Decimal("5000")}).
            kpi_types: List of KPI types to calculate.
            period: Reporting period label.

        Returns:
            List of SiteKPI, one per requested kpi_type.
        """
        logger.info(
            "Calculating KPIs for site %s: types=%s period=%s",
            site_id, kpi_types, period,
        )
        results: List[SiteKPI] = []

        for kpi_type in kpi_types:
            numerator, denominator, unit = self._resolve_kpi_components(
                kpi_type, emissions, denominators
            )
            value = _safe_divide(numerator, denominator)

            kpi = SiteKPI(
                site_id=site_id,
                kpi_type=kpi_type,
                numerator=numerator,
                denominator=denominator,
                value=value,
                unit=unit,
                period=period,
            )
            results.append(kpi)

        return results

    # ----------------------------------------------- statistics
    def calculate_statistics(
        self,
        values: List[Decimal],
    ) -> SiteStatistics:
        """
        Calculate descriptive statistics for a list of KPI values.

        Uses population standard deviation (not sample).
        Percentiles computed via linear interpolation between adjacent
        sorted values using the formula:
            index = percentile/100 * (n - 1)
            lower = sorted_values[floor(index)]
            upper = sorted_values[ceil(index)]
            result = lower + (upper - lower) * frac(index)

        Args:
            values: List of Decimal KPI values.

        Returns:
            SiteStatistics with mean, median, std_dev, percentiles.

        Raises:
            ValueError: If values list is empty.
        """
        if not values:
            raise ValueError("Cannot calculate statistics for an empty list")

        n = len(values)
        sorted_vals = sorted(values)

        # Mean
        total = sum(sorted_vals)
        mean = _quantise(total / Decimal(str(n)), self._precision)

        # Variance and std dev (population)
        variance_sum = sum((v - mean) ** 2 for v in sorted_vals)
        variance = _quantise(variance_sum / Decimal(str(n)), self._precision)
        std_dev = _decimal_sqrt(variance)

        # Percentiles
        median = self._interpolated_percentile(sorted_vals, Decimal("50"))
        p10 = self._interpolated_percentile(sorted_vals, Decimal("10"))
        p25 = self._interpolated_percentile(sorted_vals, Decimal("25"))
        p75 = self._interpolated_percentile(sorted_vals, Decimal("75"))
        p90 = self._interpolated_percentile(sorted_vals, Decimal("90"))

        stats = SiteStatistics(
            mean=mean,
            median=median,
            std_dev=std_dev,
            p10=p10,
            p25=p25,
            p75=p75,
            p90=p90,
            min_val=sorted_vals[0],
            max_val=sorted_vals[-1],
            count=n,
        )

        logger.debug(
            "Statistics: mean=%s median=%s std=%s n=%d",
            mean, median, std_dev, n,
        )
        return stats

    # ----------------------------------------------- ranking
    def rank_sites(
        self,
        peer_group: PeerGroup,
        site_kpis: Dict[str, SiteKPI],
        direction: Optional[str] = None,
    ) -> List[RankingResult]:
        """
        Rank sites within a peer group based on their KPI values.

        Args:
            peer_group: PeerGroup defining member sites.
            site_kpis: Mapping of site_id -> SiteKPI for the comparison KPI.
            direction: ASCENDING (lower=better) or DESCENDING (higher=better).
                       Defaults to engine default.

        Returns:
            List of RankingResult sorted by rank (best first).
        """
        direction = direction or self._direction
        logger.info(
            "Ranking sites: peer_group=%s members=%d direction=%s",
            peer_group.group_id, len(peer_group.member_site_ids), direction,
        )

        # Collect values for member sites
        member_values: List[Tuple[str, Decimal]] = []
        for site_id in peer_group.member_site_ids:
            kpi = site_kpis.get(site_id)
            if kpi is not None:
                member_values.append((site_id, kpi.value))
            else:
                logger.warning("Site %s in peer group but has no KPI data, excluded", site_id)

        if not member_values:
            return []

        # Sort: ASCENDING = lower is better, DESCENDING = higher is better
        ascending = direction.upper() == RankDirection.ASCENDING.value
        member_values.sort(key=lambda x: x[1], reverse=not ascending)

        # Calculate group statistics
        all_values = [v for _, v in member_values]
        stats = self.calculate_statistics(all_values)
        best_value = member_values[0][1] if member_values else _ZERO

        n = len(member_values)
        rankings: List[RankingResult] = []

        for rank_idx, (site_id, value) in enumerate(member_values):
            rank = rank_idx + 1
            if n > 1:
                percentile = _quantise(
                    Decimal(str(rank_idx)) / Decimal(str(n - 1)) * _HUNDRED,
                    self._precision,
                )
            else:
                percentile = _ZERO

            gap = _quantise(abs(value - best_value), self._precision)

            ranking = RankingResult(
                site_id=site_id,
                kpi_type=site_kpis[site_id].kpi_type if site_id in site_kpis else "",
                rank=rank,
                percentile=percentile,
                value=value,
                peer_group_mean=stats.mean,
                peer_group_median=stats.median,
                gap_to_best=gap,
            )
            rankings.append(ranking)

        return rankings

    # ----------------------------------------------- best practices
    def identify_best_practices(
        self,
        rankings: List[RankingResult],
        top_n: int = 3,
    ) -> List[str]:
        """
        Identify the top-performing sites from a ranking list.

        Args:
            rankings: Sorted list of RankingResult (best first).
            top_n: Number of top performers to return.

        Returns:
            List of site IDs of best practice performers.
        """
        if not rankings:
            return []
        top_n = min(top_n, len(rankings))
        best = [r.site_id for r in rankings[:top_n]]
        logger.info("Best practice sites (top %d): %s", top_n, best)
        return best

    # ----------------------------------------------- improvement potential
    def calculate_improvement_potential(
        self,
        site_kpi: SiteKPI,
        peer_group_median: Decimal,
    ) -> Decimal:
        """
        Calculate the improvement potential for a site relative to the peer median.

        A positive result means the site has higher intensity than the median
        (assuming ASCENDING direction = lower is better).

        improvement = site_value - median (clamped to zero minimum)

        If the site is already at or below the median the potential is zero.

        Args:
            site_kpi: The site's KPI observation.
            peer_group_median: Peer group median KPI value.

        Returns:
            Improvement potential as a Decimal (non-negative).
        """
        gap = site_kpi.value - peer_group_median
        potential = max(gap, _ZERO)
        return _quantise(potential, self._precision)

    # ----------------------------------------------- full comparison
    def compare_sites(
        self,
        sites_kpi_map: Dict[str, SiteKPI],
        kpi_type: str,
        peer_group: PeerGroup,
        direction: Optional[str] = None,
        top_n: int = 3,
    ) -> ComparisonResult:
        """
        Full comparison pipeline: rank, statistics, best practices, improvement.

        Args:
            sites_kpi_map: Mapping of site_id -> SiteKPI for the given kpi_type.
            kpi_type: The KPI type being compared.
            peer_group: Peer group to compare within.
            direction: Ranking direction (defaults to engine default).
            top_n: Number of best-practice sites to identify.

        Returns:
            ComparisonResult with rankings, statistics, and improvement potential.
        """
        logger.info(
            "Comparing sites: kpi=%s peer_group=%s members=%d",
            kpi_type, peer_group.group_id, len(peer_group.member_site_ids),
        )

        # Filter KPIs to peer group members
        member_kpis: Dict[str, SiteKPI] = {
            sid: kpi for sid, kpi in sites_kpi_map.items()
            if sid in peer_group.member_site_ids
        }

        if not member_kpis:
            logger.warning("No KPI data found for peer group members")
            return ComparisonResult(
                peer_group=peer_group,
                kpi_type=kpi_type,
            )

        # Rank
        rankings = self.rank_sites(peer_group, member_kpis, direction)

        # Statistics
        all_values = [kpi.value for kpi in member_kpis.values()]
        statistics = self.calculate_statistics(all_values)

        # Best practices
        best_practices = self.identify_best_practices(rankings, top_n)

        # Total improvement potential
        total_improvement = _ZERO
        for site_id, kpi in member_kpis.items():
            potential = self.calculate_improvement_potential(kpi, statistics.median)
            total_improvement += potential

        total_improvement = _quantise(total_improvement, self._precision)

        result = ComparisonResult(
            peer_group=peer_group,
            kpi_type=kpi_type,
            rankings=rankings,
            statistics=statistics,
            best_practice_sites=best_practices,
            improvement_potential=total_improvement,
        )

        logger.info(
            "Comparison complete: kpi=%s best=%s improvement=%s hash=%s",
            kpi_type, best_practices, total_improvement, result.provenance_hash[:12],
        )
        return result

    # ----------------------------------------------- trend
    def get_trend(
        self,
        site_id: str,
        kpi_type: str,
        kpis_by_period: Dict[str, SiteKPI],
    ) -> SiteTrend:
        """
        Calculate trend direction and change for a site across periods.

        Periods are sorted lexicographically (works for YYYY and YYYY-QQ).
        Direction is determined by comparing first and last values:
            IMPROVING:  last < first (for ascending direction = lower is better)
            WORSENING:  last > first
            STABLE:     last == first

        Args:
            site_id: Site identifier.
            kpi_type: KPI type being tracked.
            kpis_by_period: Mapping of period label -> SiteKPI.

        Returns:
            SiteTrend with data points, direction, and change percentage.
        """
        logger.info(
            "Calculating trend: site=%s kpi=%s periods=%d",
            site_id, kpi_type, len(kpis_by_period),
        )
        if not kpis_by_period:
            return SiteTrend(
                site_id=site_id,
                kpi_type=kpi_type,
                direction="STABLE",
                change_pct=_ZERO,
            )

        sorted_periods = sorted(kpis_by_period.keys())
        data_points: List[TrendPoint] = []
        for period in sorted_periods:
            kpi = kpis_by_period[period]
            data_points.append(TrendPoint(period=period, value=kpi.value))

        first_val = data_points[0].value
        last_val = data_points[-1].value

        if first_val == _ZERO:
            change_pct = _ZERO if last_val == _ZERO else _HUNDRED
        else:
            change_pct = _quantise(
                (last_val - first_val) / abs(first_val) * _HUNDRED,
                self._precision,
            )

        ascending = self._direction == RankDirection.ASCENDING.value
        if ascending:
            if last_val < first_val:
                direction = "IMPROVING"
            elif last_val > first_val:
                direction = "WORSENING"
            else:
                direction = "STABLE"
        else:
            if last_val > first_val:
                direction = "IMPROVING"
            elif last_val < first_val:
                direction = "WORSENING"
            else:
                direction = "STABLE"

        trend = SiteTrend(
            site_id=site_id,
            kpi_type=kpi_type,
            data_points=data_points,
            direction=direction,
            change_pct=change_pct,
        )

        logger.info(
            "Trend: site=%s direction=%s change=%s%% hash=%s",
            site_id, direction, change_pct, trend.provenance_hash[:12],
        )
        return trend

    # ----------------------------------------------- multi-site trend
    def get_portfolio_trend(
        self,
        sites_kpis_by_period: Dict[str, Dict[str, SiteKPI]],
        kpi_type: str,
    ) -> Dict[str, SiteTrend]:
        """
        Calculate trends for all sites in the portfolio.

        Args:
            sites_kpis_by_period: Mapping site_id -> {period -> SiteKPI}.
            kpi_type: KPI type to track.

        Returns:
            Dict mapping site_id -> SiteTrend.
        """
        logger.info("Portfolio trend: %d sites, kpi=%s", len(sites_kpis_by_period), kpi_type)
        trends: Dict[str, SiteTrend] = {}

        for site_id, kpis_by_period in sites_kpis_by_period.items():
            trend = self.get_trend(site_id, kpi_type, kpis_by_period)
            trends[site_id] = trend

        return trends

    # ----------------------------------------------- outlier detection
    def detect_outliers(
        self,
        rankings: List[RankingResult],
        statistics: SiteStatistics,
    ) -> List[Dict[str, Any]]:
        """
        Identify statistical outliers in a ranking using std-dev threshold.

        A site is an outlier if:
            |value - mean| > threshold * std_dev

        Args:
            rankings: List of RankingResult.
            statistics: Descriptive statistics for the peer group.

        Returns:
            List of outlier dicts with site_id, value, z_score, direction.
        """
        outliers: List[Dict[str, Any]] = []
        if statistics.std_dev == _ZERO:
            return outliers

        for r in rankings:
            z_score = _safe_divide(abs(r.value - statistics.mean), statistics.std_dev)
            if z_score > self._outlier_threshold:
                direction = "HIGH" if r.value > statistics.mean else "LOW"
                outliers.append({
                    "site_id": r.site_id,
                    "value": r.value,
                    "z_score": z_score,
                    "mean": statistics.mean,
                    "std_dev": statistics.std_dev,
                    "direction": direction,
                })
                logger.warning(
                    "Outlier detected: site=%s value=%s z=%s direction=%s",
                    r.site_id, r.value, z_score, direction,
                )

        return outliers

    # ----------------------------------------------- multi-KPI comparison
    def compare_sites_multi_kpi(
        self,
        sites_kpi_map: Dict[str, List[SiteKPI]],
        kpi_types: List[str],
        peer_group: PeerGroup,
    ) -> Dict[str, ComparisonResult]:
        """
        Run comparison for multiple KPI types in a single call.

        Args:
            sites_kpi_map: Mapping site_id -> List[SiteKPI].
            kpi_types: List of KPI types to compare.
            peer_group: Peer group to compare within.

        Returns:
            Dict mapping kpi_type -> ComparisonResult.
        """
        logger.info(
            "Multi-KPI comparison: %d types, peer_group=%s",
            len(kpi_types), peer_group.group_id,
        )
        results: Dict[str, ComparisonResult] = {}

        for kpi_type in kpi_types:
            # Extract the specific KPI for each site
            type_map: Dict[str, SiteKPI] = {}
            for site_id, kpis in sites_kpi_map.items():
                for kpi in kpis:
                    if kpi.kpi_type == kpi_type:
                        type_map[site_id] = kpi
                        break

            result = self.compare_sites(type_map, kpi_type, peer_group)
            results[kpi_type] = result

        return results

    # ----------------------------------------------- gap analysis
    def calculate_portfolio_gap(
        self,
        comparison_results: Dict[str, ComparisonResult],
    ) -> Dict[str, Any]:
        """
        Calculate aggregate improvement gap across multiple KPI comparisons.

        Args:
            comparison_results: Dict of kpi_type -> ComparisonResult.

        Returns:
            Summary dict with total improvement potential per KPI type.
        """
        summary: Dict[str, Any] = {
            "kpi_count": len(comparison_results),
            "gaps": {},
            "total_improvement_potential": _ZERO,
        }

        for kpi_type, result in comparison_results.items():
            summary["gaps"][kpi_type] = {
                "improvement_potential": result.improvement_potential,
                "best_practice_count": len(result.best_practice_sites),
                "peer_count": len(result.rankings),
                "median": result.statistics.median,
                "mean": result.statistics.mean,
            }
            summary["total_improvement_potential"] += result.improvement_potential

        summary["total_improvement_potential"] = _quantise(
            summary["total_improvement_potential"], self._precision
        )

        provenance_payload = f"gap|{len(comparison_results)}|{summary['total_improvement_potential']}"
        summary["provenance_hash"] = _compute_hash(provenance_payload)

        return summary

    # ---------------------------------------------------------------------------
    # Private Helpers
    # ---------------------------------------------------------------------------

    def _interpolated_percentile(
        self,
        sorted_values: List[Decimal],
        percentile: Decimal,
    ) -> Decimal:
        """
        Linear interpolation percentile calculation.

        index = percentile / 100 * (n - 1)
        lower = sorted_values[floor(index)]
        upper = sorted_values[ceil(index)]
        result = lower + (upper - lower) * frac(index)

        Args:
            sorted_values: Pre-sorted list of Decimal values.
            percentile: Percentile to calculate (0-100).

        Returns:
            Interpolated percentile value.
        """
        n = len(sorted_values)
        if n == 1:
            return sorted_values[0]

        # Compute exact index position
        idx = percentile / _HUNDRED * Decimal(str(n - 1))
        lower_idx = int(idx)
        upper_idx = min(lower_idx + 1, n - 1)
        frac = idx - Decimal(str(lower_idx))

        lower_val = sorted_values[lower_idx]
        upper_val = sorted_values[upper_idx]

        result = lower_val + (upper_val - lower_val) * frac
        return _quantise(result, self._precision)

    def _site_matches_criteria(
        self,
        site: Dict[str, Any],
        criteria: Dict[str, Any],
    ) -> bool:
        """
        Check if a site matches all criteria.

        Supports:
            - Exact match (string/number equality)
            - List membership (value in criterion list)
            - Range match (dict with 'min' and/or 'max')

        Args:
            site: Site metadata dict.
            criteria: Filter criteria.

        Returns:
            True if site matches all criteria.
        """
        for key, criterion in criteria.items():
            site_val = site.get(key)
            if site_val is None:
                return False

            if isinstance(criterion, list):
                if site_val not in criterion:
                    return False
            elif isinstance(criterion, dict):
                crit_min = criterion.get("min")
                crit_max = criterion.get("max")
                try:
                    val_dec = Decimal(str(site_val))
                    if crit_min is not None and val_dec < Decimal(str(crit_min)):
                        return False
                    if crit_max is not None and val_dec > Decimal(str(crit_max)):
                        return False
                except Exception:
                    return False
            else:
                if site_val != criterion:
                    return False

        return True

    def _derive_group_name(self, criteria: Dict[str, Any]) -> str:
        """Derive a human-readable group name from criteria."""
        parts: List[str] = []
        for key, value in sorted(criteria.items()):
            if isinstance(value, list):
                parts.append(f"{key}=[{','.join(str(v) for v in value)}]")
            elif isinstance(value, dict):
                range_str = ""
                if "min" in value:
                    range_str += f">={value['min']}"
                if "max" in value:
                    range_str += f"<={value['max']}"
                parts.append(f"{key}{range_str}")
            else:
                parts.append(f"{key}={value}")
        return " | ".join(parts) if parts else "All Sites"

    def _resolve_kpi_components(
        self,
        kpi_type: str,
        emissions: Dict[str, Decimal],
        denominators: Dict[str, Decimal],
    ) -> Tuple[Decimal, Decimal, str]:
        """
        Resolve numerator, denominator, and unit for a KPI type.

        Args:
            kpi_type: KPI type string.
            emissions: Emissions data dict.
            denominators: Denominator data dict.

        Returns:
            Tuple of (numerator, denominator, unit_string).
        """
        kpi_mapping: Dict[str, Tuple[str, str, str]] = {
            KPIType.EMISSIONS_PER_M2.value: ("total", "m2", "tCO2e/m2"),
            KPIType.EMISSIONS_PER_FTE.value: ("total", "fte", "tCO2e/FTE"),
            KPIType.EMISSIONS_PER_UNIT.value: ("total", "units", "tCO2e/unit"),
            KPIType.EMISSIONS_PER_REVENUE.value: ("total", "revenue", "tCO2e/mEUR"),
            KPIType.ENERGY_PER_M2.value: ("energy", "m2", "kWh/m2"),
            KPIType.ENERGY_PER_FTE.value: ("energy", "fte", "kWh/FTE"),
            KPIType.WASTE_PER_FTE.value: ("waste", "fte", "t/FTE"),
            KPIType.WATER_PER_M2.value: ("water", "m2", "m3/m2"),
            KPIType.SCOPE1_PER_M2.value: ("scope1", "m2", "tCO2e/m2"),
            KPIType.SCOPE2_PER_M2.value: ("scope2", "m2", "tCO2e/m2"),
        }

        if kpi_type in kpi_mapping:
            num_key, den_key, unit = kpi_mapping[kpi_type]
            numerator = emissions.get(num_key, _ZERO)
            denominator = denominators.get(den_key, _ZERO)
            return numerator, denominator, unit

        # CUSTOM: use 'custom_numerator' and 'custom_denominator'
        numerator = emissions.get("custom_numerator", emissions.get("total", _ZERO))
        denominator = denominators.get("custom_denominator", denominators.get("m2", _ZERO))
        return numerator, denominator, "custom"

# ---------------------------------------------------------------------------
# Pydantic v2 model rebuild (required with `from __future__ import annotations`)
# ---------------------------------------------------------------------------

SiteKPI.model_rebuild()
PeerGroup.model_rebuild()
SiteStatistics.model_rebuild()
RankingResult.model_rebuild()
ComparisonResult.model_rebuild()
TrendPoint.model_rebuild()
SiteTrend.model_rebuild()
