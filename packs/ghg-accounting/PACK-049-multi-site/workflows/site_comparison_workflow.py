# -*- coding: utf-8 -*-
"""
Site Comparison Workflow
====================================

5-phase workflow for cross-site GHG performance comparison covering
peer group construction, KPI calculation, ranking, gap analysis, and
best practice reporting within PACK-049 Multi-Site Management.

Phases:
    1. PeerGroupBuild       -- Build peer groups by facility type, sector,
                               region, size band, or custom criteria.
    2. KPICalculate         -- Calculate intensity KPIs per site (tCO2e/sqm,
                               tCO2e/FTE, tCO2e/revenue, tCO2e/unit).
    3. Rank                 -- Rank sites within each peer group by each KPI.
    4. GapAnalysis          -- Compute gap-to-best-practice for each site.
    5. BestPracticeReport   -- Generate comparison report with league tables.

Regulatory Basis:
    GHG Protocol Corporate Standard (Ch. 9) -- Performance tracking
    ISO 14064-1:2018 (Cl. 5.3) -- Intensity metrics
    CSRD / ESRS E1-6 -- Transition targets

Author: GreenLang Team
Version: 49.0.0
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)
_MODULE_VERSION = "1.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ComparisonPhase(str, Enum):
    PEER_GROUP_BUILD = "peer_group_build"
    KPI_CALCULATE = "kpi_calculate"
    RANK = "rank"
    GAP_ANALYSIS = "gap_analysis"
    BEST_PRACTICE_REPORT = "best_practice_report"


class PeerGroupCriteria(str, Enum):
    FACILITY_TYPE = "facility_type"
    SECTOR = "sector"
    REGION = "region"
    SIZE_BAND = "size_band"
    BUSINESS_UNIT = "business_unit"
    CUSTOM = "custom"


class KPIType(str, Enum):
    TCO2E_PER_SQM = "tco2e_per_sqm"
    TCO2E_PER_FTE = "tco2e_per_fte"
    TCO2E_PER_REVENUE = "tco2e_per_revenue"
    TCO2E_PER_UNIT = "tco2e_per_unit"
    TCO2E_PER_HOUR = "tco2e_per_hour"
    ABSOLUTE_TCO2E = "absolute_tco2e"


class PerformanceBand(str, Enum):
    TOP_QUARTILE = "top_quartile"
    SECOND_QUARTILE = "second_quartile"
    THIRD_QUARTILE = "third_quartile"
    BOTTOM_QUARTILE = "bottom_quartile"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    phase_name: str = Field(...)
    phase_number: int = Field(default=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class SiteMetrics(BaseModel):
    """Site-level metrics input for comparison."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_id: str = Field(...)
    site_name: str = Field("")
    facility_type: str = Field("")
    sector: str = Field("")
    region: str = Field("")
    business_unit: str = Field("")
    total_tco2e: Decimal = Field(Decimal("0"))
    scope_1_tco2e: Decimal = Field(Decimal("0"))
    scope_2_tco2e: Decimal = Field(Decimal("0"))
    scope_3_tco2e: Decimal = Field(Decimal("0"))
    floor_area_sqm: Decimal = Field(Decimal("0"))
    headcount: Decimal = Field(Decimal("0"))
    revenue: Decimal = Field(Decimal("0"))
    production_output: Decimal = Field(Decimal("0"))
    production_unit: str = Field("")
    operating_hours_yr: Decimal = Field(Decimal("0"))


class PeerGroup(BaseModel):
    """A constructed peer group."""
    group_id: str = Field(default_factory=_new_uuid)
    group_name: str = Field(...)
    criteria: PeerGroupCriteria = Field(PeerGroupCriteria.FACILITY_TYPE)
    criteria_value: str = Field("")
    site_ids: List[str] = Field(default_factory=list)
    site_count: int = Field(0)


class SiteKPI(BaseModel):
    """Calculated KPI for a site."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_id: str = Field(...)
    site_name: str = Field("")
    kpi_type: KPIType = Field(...)
    kpi_value: Decimal = Field(Decimal("0"))
    kpi_unit: str = Field("")
    numerator: Decimal = Field(Decimal("0"))
    denominator: Decimal = Field(Decimal("0"))


class SiteRanking(BaseModel):
    """Ranking of a site within its peer group for a KPI."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_id: str = Field(...)
    site_name: str = Field("")
    group_id: str = Field("")
    kpi_type: KPIType = Field(...)
    rank: int = Field(0)
    total_in_group: int = Field(0)
    percentile: Decimal = Field(Decimal("0"))
    performance_band: PerformanceBand = Field(PerformanceBand.SECOND_QUARTILE)
    kpi_value: Decimal = Field(Decimal("0"))


class GapAnalysisItem(BaseModel):
    """Gap-to-best-practice for a site."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    site_id: str = Field(...)
    site_name: str = Field("")
    group_id: str = Field("")
    kpi_type: KPIType = Field(...)
    site_value: Decimal = Field(Decimal("0"))
    best_in_group: Decimal = Field(Decimal("0"))
    group_median: Decimal = Field(Decimal("0"))
    gap_to_best: Decimal = Field(Decimal("0"))
    gap_to_best_pct: Decimal = Field(Decimal("0"))
    gap_to_median: Decimal = Field(Decimal("0"))
    gap_to_median_pct: Decimal = Field(Decimal("0"))
    reduction_potential_tco2e: Decimal = Field(Decimal("0"))


class BestPracticeEntry(BaseModel):
    """Best practice entry for a peer group."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    group_id: str = Field("")
    group_name: str = Field("")
    kpi_type: KPIType = Field(...)
    best_site_id: str = Field("")
    best_site_name: str = Field("")
    best_value: Decimal = Field(Decimal("0"))
    median_value: Decimal = Field(Decimal("0"))
    worst_value: Decimal = Field(Decimal("0"))
    std_dev: Decimal = Field(Decimal("0"))


class SiteComparisonInput(BaseModel):
    """Input for the site comparison workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    organisation_id: str = Field(...)
    reporting_year: int = Field(...)
    site_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    peer_group_criteria: List[str] = Field(
        default_factory=lambda: ["facility_type"],
        description="Criteria for building peer groups"
    )
    kpi_types: List[str] = Field(
        default_factory=lambda: ["tco2e_per_sqm", "tco2e_per_fte"],
    )
    skip_phases: List[str] = Field(default_factory=list)


class SiteComparisonResult(BaseModel):
    """Output from the site comparison workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    workflow_id: str = Field(default_factory=_new_uuid)
    organisation_id: str = Field("")
    reporting_year: int = Field(0)
    status: WorkflowStatus = Field(WorkflowStatus.PENDING)
    phase_results: List[PhaseResult] = Field(default_factory=list)
    peer_groups: List[PeerGroup] = Field(default_factory=list)
    site_kpis: List[SiteKPI] = Field(default_factory=list)
    rankings: List[SiteRanking] = Field(default_factory=list)
    gap_analysis: List[GapAnalysisItem] = Field(default_factory=list)
    best_practices: List[BestPracticeEntry] = Field(default_factory=list)
    total_reduction_potential_tco2e: Decimal = Field(Decimal("0"))
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    duration_seconds: float = Field(0.0)
    provenance_hash: str = Field("")
    started_at: str = Field("")
    completed_at: str = Field("")


# =============================================================================
# WORKFLOW CLASS
# =============================================================================


class SiteComparisonWorkflow:
    """
    5-phase site comparison workflow for multi-site GHG benchmarking.

    Builds peer groups, computes KPIs, ranks sites, analyses gaps to
    best practice, and generates comparison reports.

    Example:
        >>> wf = SiteComparisonWorkflow()
        >>> inp = SiteComparisonInput(
        ...     organisation_id="ORG-001", reporting_year=2025,
        ...     site_metrics=[...],
        ... )
        >>> result = wf.execute(inp)
    """

    PHASE_ORDER: List[ComparisonPhase] = [
        ComparisonPhase.PEER_GROUP_BUILD,
        ComparisonPhase.KPI_CALCULATE,
        ComparisonPhase.RANK,
        ComparisonPhase.GAP_ANALYSIS,
        ComparisonPhase.BEST_PRACTICE_REPORT,
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self._metrics: Dict[str, SiteMetrics] = {}
        self._peer_groups: List[PeerGroup] = []
        self._kpis: Dict[str, List[SiteKPI]] = {}  # site_id -> KPIs
        self._rankings: List[SiteRanking] = []

    def execute(self, input_data: SiteComparisonInput) -> SiteComparisonResult:
        start = _utcnow()
        result = SiteComparisonResult(
            organisation_id=input_data.organisation_id,
            reporting_year=input_data.reporting_year,
            status=WorkflowStatus.RUNNING, started_at=start.isoformat(),
        )

        phase_methods = {
            ComparisonPhase.PEER_GROUP_BUILD: self._phase_peer_group_build,
            ComparisonPhase.KPI_CALCULATE: self._phase_kpi_calculate,
            ComparisonPhase.RANK: self._phase_rank,
            ComparisonPhase.GAP_ANALYSIS: self._phase_gap_analysis,
            ComparisonPhase.BEST_PRACTICE_REPORT: self._phase_best_practice_report,
        }

        for idx, phase in enumerate(self.PHASE_ORDER, 1):
            if phase.value in input_data.skip_phases:
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx, status=PhaseStatus.SKIPPED,
                ))
                continue
            phase_start = _utcnow()
            try:
                phase_out = phase_methods[phase](input_data, result)
                elapsed = (_utcnow() - phase_start).total_seconds()
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
                    outputs=phase_out, provenance_hash=_compute_hash(str(phase_out)),
                ))
            except Exception as exc:
                elapsed = (_utcnow() - phase_start).total_seconds()
                logger.error("Phase %s failed: %s", phase.value, exc, exc_info=True)
                result.phase_results.append(PhaseResult(
                    phase_name=phase.value, phase_number=idx,
                    status=PhaseStatus.FAILED, duration_seconds=elapsed, errors=[str(exc)],
                ))
                result.status = WorkflowStatus.FAILED
                result.errors.append(f"Phase {phase.value}: {exc}")
                break

        if result.status != WorkflowStatus.FAILED:
            result.status = WorkflowStatus.COMPLETED
        end = _utcnow()
        result.completed_at = end.isoformat()
        result.duration_seconds = (end - start).total_seconds()
        result.provenance_hash = _compute_hash(
            f"{result.workflow_id}|{result.organisation_id}|{result.completed_at}"
        )
        return result

    # -----------------------------------------------------------------
    # PHASE 1 -- PEER GROUP BUILD
    # -----------------------------------------------------------------

    def _phase_peer_group_build(
        self, input_data: SiteComparisonInput, result: SiteComparisonResult,
    ) -> Dict[str, Any]:
        """Build peer groups by criteria."""
        logger.info("Phase 1 -- Peer Group Build")
        metrics: Dict[str, SiteMetrics] = {}
        for raw in input_data.site_metrics:
            sid = raw.get("site_id", _new_uuid())
            sm = SiteMetrics(
                site_id=sid,
                site_name=raw.get("site_name", ""),
                facility_type=raw.get("facility_type", ""),
                sector=raw.get("sector", ""),
                region=raw.get("region", ""),
                business_unit=raw.get("business_unit", ""),
                total_tco2e=self._dec(raw.get("total_tco2e", "0")),
                scope_1_tco2e=self._dec(raw.get("scope_1_tco2e", "0")),
                scope_2_tco2e=self._dec(raw.get("scope_2_tco2e", "0")),
                scope_3_tco2e=self._dec(raw.get("scope_3_tco2e", "0")),
                floor_area_sqm=self._dec(raw.get("floor_area_sqm", "0")),
                headcount=self._dec(raw.get("headcount", "0")),
                revenue=self._dec(raw.get("revenue", "0")),
                production_output=self._dec(raw.get("production_output", "0")),
                production_unit=raw.get("production_unit", ""),
                operating_hours_yr=self._dec(raw.get("operating_hours_yr", "0")),
            )
            metrics[sid] = sm
        self._metrics = metrics

        groups: List[PeerGroup] = []
        for criteria_str in input_data.peer_group_criteria:
            try:
                criteria = PeerGroupCriteria(criteria_str)
            except ValueError:
                continue

            buckets: Dict[str, List[str]] = {}
            for sid, sm in metrics.items():
                val = self._get_criteria_value(sm, criteria)
                if val:
                    buckets.setdefault(val, []).append(sid)

            for val, sids in buckets.items():
                if len(sids) < 2:
                    continue
                pg = PeerGroup(
                    group_name=f"{criteria.value}:{val}",
                    criteria=criteria,
                    criteria_value=val,
                    site_ids=sids,
                    site_count=len(sids),
                )
                groups.append(pg)

        self._peer_groups = groups
        result.peer_groups = groups

        logger.info("Built %d peer groups from %d sites", len(groups), len(metrics))
        return {"peer_groups": len(groups), "total_sites": len(metrics)}

    def _get_criteria_value(self, sm: SiteMetrics, criteria: PeerGroupCriteria) -> str:
        mapping = {
            PeerGroupCriteria.FACILITY_TYPE: sm.facility_type,
            PeerGroupCriteria.SECTOR: sm.sector,
            PeerGroupCriteria.REGION: sm.region,
            PeerGroupCriteria.BUSINESS_UNIT: sm.business_unit,
        }
        return mapping.get(criteria, "")

    # -----------------------------------------------------------------
    # PHASE 2 -- KPI CALCULATE
    # -----------------------------------------------------------------

    def _phase_kpi_calculate(
        self, input_data: SiteComparisonInput, result: SiteComparisonResult,
    ) -> Dict[str, Any]:
        """Calculate intensity KPIs per site."""
        logger.info("Phase 2 -- KPI Calculate")
        all_kpis: List[SiteKPI] = []
        kpis_by_site: Dict[str, List[SiteKPI]] = {}

        kpi_types = []
        for kt_str in input_data.kpi_types:
            try:
                kpi_types.append(KPIType(kt_str))
            except ValueError:
                pass
        if not kpi_types:
            kpi_types = [KPIType.TCO2E_PER_SQM, KPIType.TCO2E_PER_FTE]

        for sid, sm in self._metrics.items():
            site_kpis: List[SiteKPI] = []
            for kt in kpi_types:
                num, den, unit = self._get_kpi_components(sm, kt)
                if den > Decimal("0"):
                    value = (num / den).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
                else:
                    value = Decimal("0")

                kpi = SiteKPI(
                    site_id=sid, site_name=sm.site_name,
                    kpi_type=kt, kpi_value=value, kpi_unit=unit,
                    numerator=num, denominator=den,
                )
                site_kpis.append(kpi)
                all_kpis.append(kpi)

            kpis_by_site[sid] = site_kpis

        self._kpis = kpis_by_site
        result.site_kpis = all_kpis

        logger.info("Calculated %d KPIs across %d sites", len(all_kpis), len(self._metrics))
        return {"kpis_calculated": len(all_kpis), "kpi_types": len(kpi_types)}

    def _get_kpi_components(
        self, sm: SiteMetrics, kpi_type: KPIType
    ) -> Tuple[Decimal, Decimal, str]:
        """Get numerator, denominator, and unit string for a KPI."""
        if kpi_type == KPIType.TCO2E_PER_SQM:
            return sm.total_tco2e, sm.floor_area_sqm, "tCO2e/sqm"
        elif kpi_type == KPIType.TCO2E_PER_FTE:
            return sm.total_tco2e, sm.headcount, "tCO2e/FTE"
        elif kpi_type == KPIType.TCO2E_PER_REVENUE:
            return sm.total_tco2e, sm.revenue, "tCO2e/revenue"
        elif kpi_type == KPIType.TCO2E_PER_UNIT:
            return sm.total_tco2e, sm.production_output, f"tCO2e/{sm.production_unit or 'unit'}"
        elif kpi_type == KPIType.TCO2E_PER_HOUR:
            return sm.total_tco2e, sm.operating_hours_yr, "tCO2e/hr"
        else:
            return sm.total_tco2e, Decimal("1"), "tCO2e"

    # -----------------------------------------------------------------
    # PHASE 3 -- RANK
    # -----------------------------------------------------------------

    def _phase_rank(
        self, input_data: SiteComparisonInput, result: SiteComparisonResult,
    ) -> Dict[str, Any]:
        """Rank sites within peer groups by each KPI."""
        logger.info("Phase 3 -- Rank")
        rankings: List[SiteRanking] = []

        for pg in self._peer_groups:
            kpi_types_in_use = set()
            for sid in pg.site_ids:
                for kpi in self._kpis.get(sid, []):
                    kpi_types_in_use.add(kpi.kpi_type)

            for kt in kpi_types_in_use:
                # Collect KPI values for group
                group_vals: List[Tuple[str, str, Decimal]] = []
                for sid in pg.site_ids:
                    for kpi in self._kpis.get(sid, []):
                        if kpi.kpi_type == kt:
                            group_vals.append((sid, kpi.site_name, kpi.kpi_value))

                # Sort ascending (lower intensity = better)
                group_vals.sort(key=lambda x: x[2])
                total = len(group_vals)

                for rank_idx, (sid, sname, val) in enumerate(group_vals, 1):
                    percentile = Decimal("0")
                    if total > 1:
                        percentile = (
                            Decimal(str(total - rank_idx)) / Decimal(str(total - 1)) * Decimal("100")
                        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

                    if percentile >= Decimal("75"):
                        band = PerformanceBand.TOP_QUARTILE
                    elif percentile >= Decimal("50"):
                        band = PerformanceBand.SECOND_QUARTILE
                    elif percentile >= Decimal("25"):
                        band = PerformanceBand.THIRD_QUARTILE
                    else:
                        band = PerformanceBand.BOTTOM_QUARTILE

                    rankings.append(SiteRanking(
                        site_id=sid, site_name=sname, group_id=pg.group_id,
                        kpi_type=kt, rank=rank_idx, total_in_group=total,
                        percentile=percentile, performance_band=band,
                        kpi_value=val,
                    ))

        self._rankings = rankings
        result.rankings = rankings

        logger.info("Generated %d rankings", len(rankings))
        return {"rankings_generated": len(rankings)}

    # -----------------------------------------------------------------
    # PHASE 4 -- GAP ANALYSIS
    # -----------------------------------------------------------------

    def _phase_gap_analysis(
        self, input_data: SiteComparisonInput, result: SiteComparisonResult,
    ) -> Dict[str, Any]:
        """Compute gap-to-best-practice for each site."""
        logger.info("Phase 4 -- Gap Analysis")
        gaps: List[GapAnalysisItem] = []
        total_reduction = Decimal("0")

        for pg in self._peer_groups:
            kpi_types_in_use = set()
            for sid in pg.site_ids:
                for kpi in self._kpis.get(sid, []):
                    kpi_types_in_use.add(kpi.kpi_type)

            for kt in kpi_types_in_use:
                group_vals: Dict[str, Decimal] = {}
                for sid in pg.site_ids:
                    for kpi in self._kpis.get(sid, []):
                        if kpi.kpi_type == kt:
                            group_vals[sid] = kpi.kpi_value

                if not group_vals:
                    continue

                sorted_vals = sorted(group_vals.values())
                best = sorted_vals[0] if sorted_vals else Decimal("0")
                median_idx = len(sorted_vals) // 2
                median = sorted_vals[median_idx] if sorted_vals else Decimal("0")

                for sid, val in group_vals.items():
                    sm = self._metrics.get(sid)
                    gap_best = val - best
                    gap_median = val - median
                    gap_best_pct = Decimal("0")
                    gap_median_pct = Decimal("0")
                    if val > Decimal("0"):
                        gap_best_pct = (gap_best / val * Decimal("100")).quantize(
                            Decimal("0.01"), rounding=ROUND_HALF_UP
                        )
                        gap_median_pct = (gap_median / val * Decimal("100")).quantize(
                            Decimal("0.01"), rounding=ROUND_HALF_UP
                        )

                    # Estimate reduction potential
                    reduction = Decimal("0")
                    if sm and val > best and best > Decimal("0"):
                        denominator = self._get_denominator(sm, kt)
                        if denominator > Decimal("0"):
                            reduction = (gap_best * denominator).quantize(
                                Decimal("0.01"), rounding=ROUND_HALF_UP
                            )
                            total_reduction += reduction

                    gaps.append(GapAnalysisItem(
                        site_id=sid,
                        site_name=sm.site_name if sm else "",
                        group_id=pg.group_id,
                        kpi_type=kt,
                        site_value=val,
                        best_in_group=best,
                        group_median=median,
                        gap_to_best=gap_best.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
                        gap_to_best_pct=gap_best_pct,
                        gap_to_median=gap_median.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
                        gap_to_median_pct=gap_median_pct,
                        reduction_potential_tco2e=reduction,
                    ))

        result.gap_analysis = gaps
        result.total_reduction_potential_tco2e = total_reduction.quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        logger.info("Gap analysis: %d items, %.2f tCO2e reduction potential",
                     len(gaps), float(total_reduction))
        return {
            "gap_items": len(gaps),
            "total_reduction_potential_tco2e": float(total_reduction),
        }

    def _get_denominator(self, sm: SiteMetrics, kpi_type: KPIType) -> Decimal:
        mapping = {
            KPIType.TCO2E_PER_SQM: sm.floor_area_sqm,
            KPIType.TCO2E_PER_FTE: sm.headcount,
            KPIType.TCO2E_PER_REVENUE: sm.revenue,
            KPIType.TCO2E_PER_UNIT: sm.production_output,
            KPIType.TCO2E_PER_HOUR: sm.operating_hours_yr,
            KPIType.ABSOLUTE_TCO2E: Decimal("1"),
        }
        return mapping.get(kpi_type, Decimal("1"))

    # -----------------------------------------------------------------
    # PHASE 5 -- BEST PRACTICE REPORT
    # -----------------------------------------------------------------

    def _phase_best_practice_report(
        self, input_data: SiteComparisonInput, result: SiteComparisonResult,
    ) -> Dict[str, Any]:
        """Generate best practice entries per peer group."""
        logger.info("Phase 5 -- Best Practice Report")
        entries: List[BestPracticeEntry] = []

        for pg in self._peer_groups:
            kpi_types_in_use = set()
            for sid in pg.site_ids:
                for kpi in self._kpis.get(sid, []):
                    kpi_types_in_use.add(kpi.kpi_type)

            for kt in kpi_types_in_use:
                group_vals: List[Tuple[str, str, Decimal]] = []
                for sid in pg.site_ids:
                    for kpi in self._kpis.get(sid, []):
                        if kpi.kpi_type == kt:
                            sm = self._metrics.get(sid)
                            group_vals.append((sid, sm.site_name if sm else "", kpi.kpi_value))

                if not group_vals:
                    continue

                group_vals.sort(key=lambda x: x[2])
                values = [v[2] for v in group_vals]
                best = group_vals[0]
                worst = group_vals[-1]
                median = values[len(values) // 2]

                # Standard deviation
                mean = sum(values) / Decimal(str(len(values)))
                variance = sum((v - mean) ** 2 for v in values) / Decimal(str(len(values)))
                # Integer sqrt approximation for Decimal
                std_dev = variance.sqrt() if hasattr(variance, 'sqrt') else Decimal("0")

                entries.append(BestPracticeEntry(
                    group_id=pg.group_id,
                    group_name=pg.group_name,
                    kpi_type=kt,
                    best_site_id=best[0],
                    best_site_name=best[1],
                    best_value=best[2],
                    median_value=median,
                    worst_value=worst[2],
                    std_dev=std_dev.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP),
                ))

        result.best_practices = entries

        logger.info("Generated %d best practice entries", len(entries))
        return {"best_practice_entries": len(entries)}

    # -----------------------------------------------------------------
    # HELPERS
    # -----------------------------------------------------------------

    def _dec(self, value: Any) -> Decimal:
        if value is None:
            return Decimal("0")
        try:
            return Decimal(str(value))
        except Exception:
            return Decimal("0")


__all__ = [
    "SiteComparisonWorkflow",
    "SiteComparisonInput",
    "SiteComparisonResult",
    "ComparisonPhase",
    "PeerGroupCriteria",
    "KPIType",
    "PerformanceBand",
    "SiteMetrics",
    "PeerGroup",
    "SiteKPI",
    "SiteRanking",
    "GapAnalysisItem",
    "BestPracticeEntry",
    "PhaseResult",
    "PhaseStatus",
    "WorkflowStatus",
]
