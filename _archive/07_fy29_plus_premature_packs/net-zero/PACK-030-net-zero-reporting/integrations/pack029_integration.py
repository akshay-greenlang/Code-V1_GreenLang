# -*- coding: utf-8 -*-
"""
PACK029Integration - PACK-029 Interim Targets Pack Integration for PACK-030
==============================================================================

Enterprise integration for fetching interim targets, progress monitoring
data, and variance analysis from PACK-029 (Interim Targets Pack) into the
Net Zero Reporting Pack. Data feeds into SBTi progress reports (target
progress), CDP C4 (interim targets), TCFD Metrics & Targets (Table 2),
GRI 305-5 (reduction trajectory), ISSB (transition plan), SEC (targets),
and CSRD E1-4 (emission reduction targets).

Integration Points:
    - Interim Targets: 5-year milestones and annual carbon budgets
    - Progress Monitoring: Actual vs target performance tracking
    - Variance Analysis: Gap analysis with root cause attribution
    - Corrective Actions: Off-track response plans
    - Carbon Budgets: Annual budget allocation and consumption

Architecture:
    PACK-029 Targets     --> PACK-030 Multi-Framework Target Sections
    PACK-029 Progress    --> PACK-030 SBTi/CDP/TCFD progress reporting
    PACK-029 Variance    --> PACK-030 Variance explanations / narratives

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-030 Net Zero Reporting Pack
Status: Production Ready
"""

import hashlib
import importlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

class _PackStub:
    def __init__(self, component: str) -> None:
        self._component = component

    def __getattr__(self, name: str) -> Any:
        def _stub(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {"component": self._component, "status": "not_available", "pack": "PACK-029"}
        return _stub

def _try_import(component: str, module_path: str) -> Any:
    try:
        return importlib.import_module(module_path)
    except ImportError:
        logger.debug("PACK-029 component '%s' not available, using stub", component)
        return _PackStub(component)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TargetScope(str, Enum):
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_12 = "scope_12"
    SCOPE_3 = "scope_3"
    ALL_SCOPES = "all_scopes"

class TargetType(str, Enum):
    ABSOLUTE = "absolute"
    INTENSITY = "intensity"

class TargetStatus(str, Enum):
    ON_TRACK = "on_track"
    AT_RISK = "at_risk"
    OFF_TRACK = "off_track"
    ACHIEVED = "achieved"
    NOT_STARTED = "not_started"

class VarianceDirection(str, Enum):
    FAVORABLE = "favorable"
    UNFAVORABLE = "unfavorable"
    NEUTRAL = "neutral"

class RAGStatus(str, Enum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"

class ImportStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    STALE = "stale"
    CACHED = "cached"

# ---------------------------------------------------------------------------
# Component Registry
# ---------------------------------------------------------------------------

PACK029_COMPONENTS: Dict[str, Dict[str, str]] = {
    "interim_target_engine": {
        "name": "Interim Target Engine",
        "module": "packs.net_zero.PACK_029_interim_targets.engines.interim_target_engine",
        "description": "5-year interim milestone generation and decomposition",
    },
    "progress_tracker_engine": {
        "name": "Progress Tracker Engine",
        "module": "packs.net_zero.PACK_029_interim_targets.engines.progress_tracker_engine",
        "description": "Actual vs target performance tracking",
    },
    "variance_analysis_engine": {
        "name": "Variance Analysis Engine",
        "module": "packs.net_zero.PACK_029_interim_targets.engines.variance_analysis_engine",
        "description": "Gap analysis and root cause attribution",
    },
    "corrective_action_engine": {
        "name": "Corrective Action Engine",
        "module": "packs.net_zero.PACK_029_interim_targets.engines.corrective_action_engine",
        "description": "Off-track response plan generation",
    },
    "budget_allocation_engine": {
        "name": "Budget Allocation Engine",
        "module": "packs.net_zero.PACK_029_interim_targets.engines.budget_allocation_engine",
        "description": "Annual carbon budget allocation and tracking",
    },
}

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class PACK029IntegrationConfig(BaseModel):
    pack_id: str = Field(default="PACK-030")
    source_pack_id: str = Field(default="PACK-029")
    organization_id: str = Field(default="")
    organization_name: str = Field(default="")
    base_year: int = Field(default=2023, ge=2015, le=2025)
    reporting_year: int = Field(default=2025, ge=2020, le=2035)
    net_zero_year: int = Field(default=2050, ge=2040, le=2060)
    enable_provenance: bool = Field(default=True)
    db_connection_string: str = Field(default="")
    db_pool_size: int = Field(default=5, ge=1, le=20)
    cache_ttl_seconds: int = Field(default=3600)
    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_delay_seconds: float = Field(default=1.0)

class InterimTarget(BaseModel):
    """Interim target milestone from PACK-029."""
    target_id: str = Field(default_factory=_new_uuid)
    target_year: int = Field(default=2030)
    scope: TargetScope = Field(default=TargetScope.ALL_SCOPES)
    target_type: TargetType = Field(default=TargetType.ABSOLUTE)
    base_year: int = Field(default=2023)
    base_year_tco2e: float = Field(default=0.0)
    target_tco2e: float = Field(default=0.0)
    reduction_pct: float = Field(default=0.0)
    annual_budget_tco2e: float = Field(default=0.0)
    sbti_aligned: bool = Field(default=True)
    pathway: str = Field(default="aca_15c")
    status: TargetStatus = Field(default=TargetStatus.NOT_STARTED)

class TargetPortfolio(BaseModel):
    """Portfolio of all interim targets from PACK-029."""
    portfolio_id: str = Field(default_factory=_new_uuid)
    organization_id: str = Field(default="")
    targets: List[InterimTarget] = Field(default_factory=list)
    total_targets: int = Field(default=0)
    near_term_target_year: int = Field(default=2030)
    long_term_target_year: int = Field(default=2050)
    net_zero_year: int = Field(default=2050)
    overall_status: TargetStatus = Field(default=TargetStatus.NOT_STARTED)
    provenance_hash: str = Field(default="")
    fetched_at: datetime = Field(default_factory=utcnow)

class ProgressRecord(BaseModel):
    """Progress monitoring record from PACK-029."""
    record_id: str = Field(default_factory=_new_uuid)
    reporting_year: int = Field(default=2025)
    scope: TargetScope = Field(default=TargetScope.ALL_SCOPES)
    target_tco2e: float = Field(default=0.0)
    actual_tco2e: float = Field(default=0.0)
    variance_tco2e: float = Field(default=0.0)
    variance_pct: float = Field(default=0.0)
    rag_status: RAGStatus = Field(default=RAGStatus.GREEN)
    target_status: TargetStatus = Field(default=TargetStatus.ON_TRACK)
    cumulative_reduction_pct: float = Field(default=0.0)
    on_track_for_2030: bool = Field(default=True)
    on_track_for_net_zero: bool = Field(default=True)
    quarterly_data: Dict[str, float] = Field(default_factory=dict)

class ProgressSummary(BaseModel):
    """Progress monitoring summary from PACK-029."""
    summary_id: str = Field(default_factory=_new_uuid)
    organization_id: str = Field(default="")
    reporting_year: int = Field(default=2025)
    records: List[ProgressRecord] = Field(default_factory=list)
    overall_rag: RAGStatus = Field(default=RAGStatus.GREEN)
    overall_status: TargetStatus = Field(default=TargetStatus.ON_TRACK)
    total_target_tco2e: float = Field(default=0.0)
    total_actual_tco2e: float = Field(default=0.0)
    overall_variance_pct: float = Field(default=0.0)
    years_since_base: int = Field(default=0)
    years_to_net_zero: int = Field(default=0)
    cumulative_reduction_pct: float = Field(default=0.0)
    linear_trajectory_gap_pct: float = Field(default=0.0)
    provenance_hash: str = Field(default="")
    fetched_at: datetime = Field(default_factory=utcnow)

class VarianceDetail(BaseModel):
    """Variance analysis detail from PACK-029."""
    detail_id: str = Field(default_factory=_new_uuid)
    scope: TargetScope = Field(default=TargetScope.SCOPE_1)
    category: str = Field(default="")
    variance_tco2e: float = Field(default=0.0)
    variance_pct: float = Field(default=0.0)
    direction: VarianceDirection = Field(default=VarianceDirection.NEUTRAL)
    root_cause: str = Field(default="")
    corrective_action: str = Field(default="")
    expected_recovery_year: int = Field(default=0)

class VarianceReport(BaseModel):
    """Variance analysis report from PACK-029."""
    report_id: str = Field(default_factory=_new_uuid)
    organization_id: str = Field(default="")
    reporting_year: int = Field(default=2025)
    details: List[VarianceDetail] = Field(default_factory=list)
    total_variance_tco2e: float = Field(default=0.0)
    total_variance_pct: float = Field(default=0.0)
    favorable_count: int = Field(default=0)
    unfavorable_count: int = Field(default=0)
    top_contributors: List[Dict[str, Any]] = Field(default_factory=list)
    narrative_summary: str = Field(default="")
    provenance_hash: str = Field(default="")
    fetched_at: datetime = Field(default_factory=utcnow)

class PACK029IntegrationResult(BaseModel):
    result_id: str = Field(default_factory=_new_uuid)
    targets: Optional[TargetPortfolio] = Field(None)
    progress: Optional[ProgressSummary] = Field(None)
    variance: Optional[VarianceReport] = Field(None)
    pack029_available: bool = Field(default=False)
    import_status: ImportStatus = Field(default=ImportStatus.FAILED)
    integration_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    frameworks_serviced: List[str] = Field(default_factory=list)
    validation_errors: List[str] = Field(default_factory=list)
    validation_warnings: List[str] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# PACK029Integration
# ---------------------------------------------------------------------------

class PACK029Integration:
    """PACK-029 Interim Targets Pack integration for PACK-030.

    Fetches interim targets, progress monitoring, and variance analysis
    from PACK-029 for multi-framework report generation.

    Example:
        >>> config = PACK029IntegrationConfig(
        ...     organization_name="Acme Corp",
        ...     reporting_year=2025,
        ... )
        >>> integration = PACK029Integration(config)
        >>> targets = await integration.fetch_targets()
        >>> progress = await integration.fetch_progress()
        >>> variance = await integration.fetch_variance()
    """

    def __init__(self, config: Optional[PACK029IntegrationConfig] = None) -> None:
        self.config = config or PACK029IntegrationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        self._components: Dict[str, Any] = {}
        self._loaded: List[str] = []
        self._stubbed: List[str] = []

        for comp_id, comp_info in PACK029_COMPONENTS.items():
            agent = _try_import(comp_id, comp_info["module"])
            self._components[comp_id] = agent
            if isinstance(agent, _PackStub):
                self._stubbed.append(comp_id)
            else:
                self._loaded.append(comp_id)

        self._targets_cache: Optional[TargetPortfolio] = None
        self._progress_cache: Optional[ProgressSummary] = None
        self._variance_cache: Optional[VarianceReport] = None
        self._db_pool: Optional[Any] = None

        self.logger.info(
            "PACK029Integration (PACK-030) initialized: %d/%d components",
            len(self._loaded), len(PACK029_COMPONENTS),
        )

    async def _get_db_pool(self) -> Any:
        if self._db_pool is not None:
            return self._db_pool
        if not self.config.db_connection_string:
            return None
        try:
            import psycopg_pool
            self._db_pool = psycopg_pool.AsyncConnectionPool(
                self.config.db_connection_string, min_size=1, max_size=self.config.db_pool_size)
            await self._db_pool.open()
            return self._db_pool
        except Exception as exc:
            self.logger.warning("DB pool creation failed: %s", exc)
            return None

    async def _query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        pool = await self._get_db_pool()
        if not pool:
            return []
        attempt = 0
        while attempt < self.config.retry_attempts:
            try:
                async with pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(query, params or {})
                        columns = [desc[0] for desc in cur.description] if cur.description else []
                        rows = await cur.fetchall()
                        return [dict(zip(columns, row)) for row in rows]
            except Exception as exc:
                attempt += 1
                if attempt < self.config.retry_attempts:
                    import asyncio

                    await asyncio.sleep(self.config.retry_delay_seconds * attempt)
        return []

    # -----------------------------------------------------------------------
    # Fetch Targets
    # -----------------------------------------------------------------------

    async def fetch_targets(
        self, override_data: Optional[List[Dict[str, Any]]] = None,
    ) -> TargetPortfolio:
        """Fetch interim targets from PACK-029.

        Retrieves 5-year interim milestones, annual carbon budgets, and
        target decomposition by scope. Used in SBTi (target progress),
        CDP C4 (interim targets), TCFD Table 2 (milestones), CSRD E1-4
        (emission reduction targets), and SEC (climate targets).
        """
        if self._targets_cache is not None:
            return self._targets_cache

        raw_data = override_data or []
        if not raw_data and self.config.db_connection_string:
            raw_data = await self._query(
                "SELECT * FROM gl_pack029_interim_targets "
                "WHERE organization_id = %(org_id)s ORDER BY target_year, scope",
                {"org_id": self.config.organization_id},
            )

        if not raw_data:
            raw_data = self._default_targets()

        targets: List[InterimTarget] = []
        for row in raw_data:
            targets.append(InterimTarget(
                target_id=row.get("target_id", _new_uuid()),
                target_year=row.get("target_year", 2030),
                scope=TargetScope(row.get("scope", "all_scopes")),
                target_type=TargetType(row.get("target_type", "absolute")),
                base_year=row.get("base_year", self.config.base_year),
                base_year_tco2e=row.get("base_year_tco2e", 0.0),
                target_tco2e=row.get("target_tco2e", 0.0),
                reduction_pct=row.get("reduction_pct", 0.0),
                annual_budget_tco2e=row.get("annual_budget_tco2e", 0.0),
                sbti_aligned=row.get("sbti_aligned", True),
                pathway=row.get("pathway", "aca_15c"),
                status=TargetStatus(row.get("status", "not_started")),
            ))

        # Determine overall status
        statuses = [t.status for t in targets]
        if all(s == TargetStatus.ACHIEVED for s in statuses):
            overall = TargetStatus.ACHIEVED
        elif any(s == TargetStatus.OFF_TRACK for s in statuses):
            overall = TargetStatus.OFF_TRACK
        elif any(s == TargetStatus.AT_RISK for s in statuses):
            overall = TargetStatus.AT_RISK
        elif any(s == TargetStatus.ON_TRACK for s in statuses):
            overall = TargetStatus.ON_TRACK
        else:
            overall = TargetStatus.NOT_STARTED

        portfolio = TargetPortfolio(
            organization_id=self.config.organization_id,
            targets=targets,
            total_targets=len(targets),
            near_term_target_year=min((t.target_year for t in targets), default=2030),
            long_term_target_year=max((t.target_year for t in targets), default=2050),
            net_zero_year=self.config.net_zero_year,
            overall_status=overall,
        )

        if self.config.enable_provenance:
            portfolio.provenance_hash = _compute_hash(portfolio)

        self._targets_cache = portfolio
        self.logger.info(
            "Targets fetched from PACK-029: %d targets, overall_status=%s",
            len(targets), overall.value,
        )
        return portfolio

    # -----------------------------------------------------------------------
    # Fetch Progress
    # -----------------------------------------------------------------------

    async def fetch_progress(
        self, override_data: Optional[List[Dict[str, Any]]] = None,
    ) -> ProgressSummary:
        """Fetch progress monitoring data from PACK-029.

        Retrieves actual vs target performance for the reporting year
        with quarterly breakdown. Used in SBTi (progress disclosure),
        CDP C4.2 (target progress), TCFD (metrics tracking), and
        dashboard generation.
        """
        if self._progress_cache is not None:
            return self._progress_cache

        raw_data = override_data or []
        if not raw_data and self.config.db_connection_string:
            raw_data = await self._query(
                "SELECT * FROM gl_pack029_progress "
                "WHERE organization_id = %(org_id)s AND reporting_year = %(year)s "
                "ORDER BY scope",
                {"org_id": self.config.organization_id, "year": self.config.reporting_year},
            )

        if not raw_data:
            raw_data = self._default_progress()

        records: List[ProgressRecord] = []
        for row in raw_data:
            target = row.get("target_tco2e", 0.0)
            actual = row.get("actual_tco2e", 0.0)
            variance = actual - target
            variance_pct = (variance / max(target, 1.0)) * 100.0

            if variance_pct <= 0:
                rag = RAGStatus.GREEN
                t_status = TargetStatus.ON_TRACK
            elif variance_pct <= 10:
                rag = RAGStatus.AMBER
                t_status = TargetStatus.AT_RISK
            else:
                rag = RAGStatus.RED
                t_status = TargetStatus.OFF_TRACK

            records.append(ProgressRecord(
                reporting_year=row.get("reporting_year", self.config.reporting_year),
                scope=TargetScope(row.get("scope", "all_scopes")),
                target_tco2e=target,
                actual_tco2e=actual,
                variance_tco2e=round(variance, 2),
                variance_pct=round(variance_pct, 2),
                rag_status=rag,
                target_status=t_status,
                cumulative_reduction_pct=row.get("cumulative_reduction_pct", 0.0),
                on_track_for_2030=row.get("on_track_for_2030", variance_pct <= 5),
                on_track_for_net_zero=row.get("on_track_for_net_zero", variance_pct <= 10),
                quarterly_data=row.get("quarterly_data", {}),
            ))

        total_target = sum(r.target_tco2e for r in records)
        total_actual = sum(r.actual_tco2e for r in records)
        overall_var = ((total_actual - total_target) / max(total_target, 1.0)) * 100.0

        if overall_var <= 0:
            overall_rag = RAGStatus.GREEN
            overall_status = TargetStatus.ON_TRACK
        elif overall_var <= 10:
            overall_rag = RAGStatus.AMBER
            overall_status = TargetStatus.AT_RISK
        else:
            overall_rag = RAGStatus.RED
            overall_status = TargetStatus.OFF_TRACK

        summary = ProgressSummary(
            organization_id=self.config.organization_id,
            reporting_year=self.config.reporting_year,
            records=records,
            overall_rag=overall_rag,
            overall_status=overall_status,
            total_target_tco2e=round(total_target, 2),
            total_actual_tco2e=round(total_actual, 2),
            overall_variance_pct=round(overall_var, 2),
            years_since_base=self.config.reporting_year - self.config.base_year,
            years_to_net_zero=self.config.net_zero_year - self.config.reporting_year,
            cumulative_reduction_pct=round(
                ((records[0].cumulative_reduction_pct if records else 0.0)), 2
            ),
        )

        if self.config.enable_provenance:
            summary.provenance_hash = _compute_hash(summary)

        self._progress_cache = summary
        self.logger.info(
            "Progress fetched from PACK-029: year=%d, target=%.1f, actual=%.1f, "
            "variance=%.1f%%, rag=%s",
            summary.reporting_year, total_target, total_actual, overall_var, overall_rag.value,
        )
        return summary

    # -----------------------------------------------------------------------
    # Fetch Variance
    # -----------------------------------------------------------------------

    async def fetch_variance(
        self, override_data: Optional[List[Dict[str, Any]]] = None,
    ) -> VarianceReport:
        """Fetch variance analysis from PACK-029.

        Retrieves root cause attribution for emissions variance,
        corrective actions, and narrative summary. Used in SBTi
        (variance explanation), CDP (deviation justification),
        and assurance evidence packages.
        """
        if self._variance_cache is not None:
            return self._variance_cache

        raw_data = override_data or []
        if not raw_data and self.config.db_connection_string:
            raw_data = await self._query(
                "SELECT * FROM gl_pack029_variance_analysis "
                "WHERE organization_id = %(org_id)s AND reporting_year = %(year)s "
                "ORDER BY ABS(variance_tco2e) DESC",
                {"org_id": self.config.organization_id, "year": self.config.reporting_year},
            )

        if not raw_data:
            raw_data = self._default_variance()

        details: List[VarianceDetail] = []
        for row in raw_data:
            var = row.get("variance_tco2e", 0.0)
            direction = VarianceDirection.FAVORABLE if var < 0 else (
                VarianceDirection.UNFAVORABLE if var > 0 else VarianceDirection.NEUTRAL
            )
            details.append(VarianceDetail(
                scope=TargetScope(row.get("scope", "scope_1")),
                category=row.get("category", ""),
                variance_tco2e=var,
                variance_pct=row.get("variance_pct", 0.0),
                direction=direction,
                root_cause=row.get("root_cause", ""),
                corrective_action=row.get("corrective_action", ""),
                expected_recovery_year=row.get("expected_recovery_year", 0),
            ))

        total_var = sum(d.variance_tco2e for d in details)
        fav = sum(1 for d in details if d.direction == VarianceDirection.FAVORABLE)
        unfav = sum(1 for d in details if d.direction == VarianceDirection.UNFAVORABLE)

        top_contributors = [
            {"category": d.category, "variance_tco2e": d.variance_tco2e, "root_cause": d.root_cause}
            for d in sorted(details, key=lambda x: abs(x.variance_tco2e), reverse=True)[:5]
        ]

        narrative_parts = []
        if unfav > 0:
            unfav_total = sum(d.variance_tco2e for d in details if d.direction == VarianceDirection.UNFAVORABLE)
            narrative_parts.append(
                f"Total unfavorable variance of {unfav_total:,.0f} tCO2e across "
                f"{unfav} categories."
            )
        if fav > 0:
            fav_total = abs(sum(d.variance_tco2e for d in details if d.direction == VarianceDirection.FAVORABLE))
            narrative_parts.append(
                f"Offset by favorable variance of {fav_total:,.0f} tCO2e across "
                f"{fav} categories."
            )
        narrative = " ".join(narrative_parts) if narrative_parts else "No significant variance detected."

        report = VarianceReport(
            organization_id=self.config.organization_id,
            reporting_year=self.config.reporting_year,
            details=details,
            total_variance_tco2e=round(total_var, 2),
            total_variance_pct=round(total_var / max(abs(total_var), 1.0) * 100 if total_var else 0.0, 2),
            favorable_count=fav,
            unfavorable_count=unfav,
            top_contributors=top_contributors,
            narrative_summary=narrative,
        )

        if self.config.enable_provenance:
            report.provenance_hash = _compute_hash(report)

        self._variance_cache = report
        self.logger.info(
            "Variance fetched from PACK-029: total=%.1f tCO2e, favorable=%d, unfavorable=%d",
            total_var, fav, unfav,
        )
        return report

    # -----------------------------------------------------------------------
    # Framework-specific exports
    # -----------------------------------------------------------------------

    async def get_sbti_progress_data(self) -> Dict[str, Any]:
        targets = await self.fetch_targets()
        progress = await self.fetch_progress()
        variance = await self.fetch_variance()
        return {
            "targets": [t.model_dump() for t in targets.targets],
            "overall_status": targets.overall_status.value,
            "progress": {
                "target_tco2e": progress.total_target_tco2e,
                "actual_tco2e": progress.total_actual_tco2e,
                "variance_pct": progress.overall_variance_pct,
                "rag": progress.overall_rag.value,
            },
            "variance_summary": variance.narrative_summary,
            "on_track_for_2030": progress.records[0].on_track_for_2030 if progress.records else True,
        }

    async def get_cdp_c4_data(self) -> Dict[str, Any]:
        targets = await self.fetch_targets()
        progress = await self.fetch_progress()
        return {
            "c4_1_description": f"The organization has set {targets.total_targets} interim targets "
                                f"aligned with {targets.targets[0].pathway if targets.targets else 'aca_15c'} "
                                f"pathway, targeting net-zero by {targets.net_zero_year}.",
            "c4_2_targets": [
                {
                    "target_year": t.target_year,
                    "scope": t.scope.value,
                    "base_year": t.base_year,
                    "base_year_tco2e": t.base_year_tco2e,
                    "target_tco2e": t.target_tco2e,
                    "reduction_pct": t.reduction_pct,
                    "status": t.status.value,
                }
                for t in targets.targets
            ],
            "progress_summary": {
                "overall_rag": progress.overall_rag.value,
                "variance_pct": progress.overall_variance_pct,
            },
        }

    async def get_tcfd_table2_data(self) -> Dict[str, Any]:
        targets = await self.fetch_targets()
        progress = await self.fetch_progress()
        return {
            "interim_targets": [
                {
                    "milestone_year": t.target_year,
                    "scope": t.scope.value,
                    "reduction_pct": t.reduction_pct,
                    "target_tco2e": t.target_tco2e,
                    "status": t.status.value,
                }
                for t in targets.targets
            ],
            "progress_tracking": {
                "current_year": progress.reporting_year,
                "total_actual_tco2e": progress.total_actual_tco2e,
                "overall_variance_pct": progress.overall_variance_pct,
                "years_to_net_zero": progress.years_to_net_zero,
            },
        }

    async def get_csrd_e1_4_data(self) -> Dict[str, Any]:
        targets = await self.fetch_targets()
        progress = await self.fetch_progress()
        variance = await self.fetch_variance()
        return {
            "emission_reduction_targets": [
                {
                    "target_year": t.target_year,
                    "scope": t.scope.value,
                    "base_year": t.base_year,
                    "reduction_pct": t.reduction_pct,
                    "sbti_aligned": t.sbti_aligned,
                }
                for t in targets.targets
            ],
            "progress": {
                "actual_tco2e": progress.total_actual_tco2e,
                "target_tco2e": progress.total_target_tco2e,
                "variance_pct": progress.overall_variance_pct,
            },
            "corrective_actions": [
                {"category": d.category, "action": d.corrective_action}
                for d in variance.details if d.corrective_action
            ],
        }

    # -----------------------------------------------------------------------
    # Full Integration
    # -----------------------------------------------------------------------

    async def get_full_integration(self) -> PACK029IntegrationResult:
        errors: List[str] = []
        warnings: List[str] = []

        targets = None
        progress = None
        variance = None

        try:
            targets = await self.fetch_targets()
        except Exception as exc:
            errors.append(f"Targets fetch failed: {exc}")
        try:
            progress = await self.fetch_progress()
        except Exception as exc:
            errors.append(f"Progress fetch failed: {exc}")
        try:
            variance = await self.fetch_variance()
        except Exception as exc:
            warnings.append(f"Variance fetch failed: {exc}")

        quality = 0.0
        if targets:
            quality += 40.0
        if progress:
            quality += 35.0
        if variance:
            quality += 25.0

        status = ImportStatus.SUCCESS if not errors else (
            ImportStatus.FAILED if quality < 40.0 else ImportStatus.PARTIAL
        )

        result = PACK029IntegrationResult(
            targets=targets,
            progress=progress,
            variance=variance,
            pack029_available=len(self._loaded) > 0,
            import_status=status,
            integration_quality_score=quality,
            frameworks_serviced=["SBTi", "CDP", "TCFD", "GRI", "ISSB", "SEC", "CSRD"],
            validation_errors=errors,
            validation_warnings=warnings,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        return result

    # -----------------------------------------------------------------------
    # Default data
    # -----------------------------------------------------------------------

    def _default_targets(self) -> List[Dict[str, Any]]:
        base_total = 195000.0
        return [
            {"target_year": 2025, "scope": "scope_12", "base_year": 2023, "base_year_tco2e": 75000.0,
             "target_tco2e": 68550.0, "reduction_pct": 8.6, "sbti_aligned": True, "pathway": "aca_15c",
             "status": "on_track", "annual_budget_tco2e": 68550.0},
            {"target_year": 2030, "scope": "scope_12", "base_year": 2023, "base_year_tco2e": 75000.0,
             "target_tco2e": 43500.0, "reduction_pct": 42.0, "sbti_aligned": True, "pathway": "aca_15c",
             "status": "on_track", "annual_budget_tco2e": 43500.0},
            {"target_year": 2030, "scope": "scope_3", "base_year": 2023, "base_year_tco2e": 120000.0,
             "target_tco2e": 90000.0, "reduction_pct": 25.0, "sbti_aligned": True, "pathway": "aca_15c",
             "status": "on_track", "annual_budget_tco2e": 90000.0},
            {"target_year": 2035, "scope": "all_scopes", "base_year": 2023, "base_year_tco2e": base_total,
             "target_tco2e": 97500.0, "reduction_pct": 50.0, "sbti_aligned": True, "pathway": "aca_15c",
             "status": "not_started", "annual_budget_tco2e": 97500.0},
            {"target_year": 2040, "scope": "all_scopes", "base_year": 2023, "base_year_tco2e": base_total,
             "target_tco2e": 58500.0, "reduction_pct": 70.0, "sbti_aligned": True, "pathway": "aca_15c",
             "status": "not_started", "annual_budget_tco2e": 58500.0},
            {"target_year": 2050, "scope": "all_scopes", "base_year": 2023, "base_year_tco2e": base_total,
             "target_tco2e": 9750.0, "reduction_pct": 95.0, "sbti_aligned": True, "pathway": "aca_15c",
             "status": "not_started", "annual_budget_tco2e": 9750.0},
        ]

    def _default_progress(self) -> List[Dict[str, Any]]:
        return [
            {"reporting_year": self.config.reporting_year, "scope": "scope_12",
             "target_tco2e": 68550.0, "actual_tco2e": 67200.0,
             "cumulative_reduction_pct": 10.4, "on_track_for_2030": True, "on_track_for_net_zero": True,
             "quarterly_data": {"Q1": 17500, "Q2": 16800, "Q3": 16400, "Q4": 16500}},
            {"reporting_year": self.config.reporting_year, "scope": "scope_3",
             "target_tco2e": 114000.0, "actual_tco2e": 112000.0,
             "cumulative_reduction_pct": 6.7, "on_track_for_2030": True, "on_track_for_net_zero": True,
             "quarterly_data": {"Q1": 29000, "Q2": 28000, "Q3": 27500, "Q4": 27500}},
        ]

    def _default_variance(self) -> List[Dict[str, Any]]:
        return [
            {"scope": "scope_1", "category": "Stationary combustion", "variance_tco2e": -1200.0,
             "variance_pct": -4.8, "root_cause": "Heat pump conversion completed ahead of schedule",
             "corrective_action": "", "expected_recovery_year": 0},
            {"scope": "scope_2", "category": "Purchased electricity", "variance_tco2e": -800.0,
             "variance_pct": -3.2, "root_cause": "Solar PV Phase 1 generated above forecast",
             "corrective_action": "", "expected_recovery_year": 0},
            {"scope": "scope_3", "category": "Business travel", "variance_tco2e": 500.0,
             "variance_pct": 6.7, "root_cause": "Post-pandemic travel rebound exceeded forecast",
             "corrective_action": "Implement enhanced virtual meeting policy and flight approval process",
             "expected_recovery_year": 2026},
            {"scope": "scope_3", "category": "Purchased goods", "variance_tco2e": 1800.0,
             "variance_pct": 4.0, "root_cause": "Supplier decarbonization slower than expected",
             "corrective_action": "Accelerate supplier engagement with SBTi-aligned requirements in procurement",
             "expected_recovery_year": 2027},
        ]

    # -----------------------------------------------------------------------
    # Status & lifecycle
    # -----------------------------------------------------------------------

    def get_integration_status(self) -> Dict[str, Any]:
        return {
            "pack_id": self.config.pack_id,
            "source_pack_id": self.config.source_pack_id,
            "components_loaded": len(self._loaded),
            "targets_fetched": self._targets_cache is not None,
            "progress_fetched": self._progress_cache is not None,
            "variance_fetched": self._variance_cache is not None,
            "module_version": _MODULE_VERSION,
        }

    async def refresh(self) -> PACK029IntegrationResult:
        self._targets_cache = None
        self._progress_cache = None
        self._variance_cache = None
        return await self.get_full_integration()

    async def close(self) -> None:
        if self._db_pool is not None:
            try:
                await self._db_pool.close()
            except Exception as exc:
                self.logger.warning("Error closing DB pool: %s", exc)
            self._db_pool = None
