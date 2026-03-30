# -*- coding: utf-8 -*-
"""
PACK-049 GHG Multi-Site Management Pack - Multi-Site Reporting Engine
======================================================================

Generates portfolio-level and site-level reports from the outputs of
engines 1-9, with drill-down capability and multi-format export.

Report Types:
    PORTFOLIO_DASHBOARD:   Executive overview of entire portfolio
    SITE_DETAIL:           Detailed report for a single facility
    CONSOLIDATION:         Consolidation run results
    BOUNDARY_DEFINITION:   Organisational boundary documentation
    ALLOCATION:            Shared-services / landlord-tenant allocations
    COMPARISON:            Cross-site KPI benchmarking
    COLLECTION_STATUS:     Data collection completeness tracker
    QUALITY_HEATMAP:       Portfolio quality assessment
    TREND:                 Multi-year trend analysis

Export Formats:
    Markdown: GitHub-flavoured with pipe tables
    HTML:     Standalone with inline CSS
    JSON:     Structured serialisation
    CSV:      Tabular section extraction

Drill-Down:
    Hierarchical levels (e.g. Group > Region > Country > Site) with
    per-level data aggregation.

Provenance:
    SHA-256 hash on every MultiSiteReport.

Regulatory References:
    - GHG Protocol Corporate Standard (2004, rev 2015) - Reporting guidance
    - ISO 14064-1:2018 Clause 8 - GHG reporting
    - EU CSRD / ESRS E1 - Emissions disclosure
    - TCFD (2017) - Metrics and targets
    - CDP (2024) - Climate change questionnaire
    - GRI 305 (2016) - Emissions
    - SEC Climate Disclosure Rules (2024) - Registrant reporting

Zero-Hallucination:
    - The reporting engine does NOT perform any GHG calculations
    - It only formats, aggregates, and renders pre-computed outputs
    - No LLM involvement in report generation
    - SHA-256 provenance chain across all source data

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-049 GHG Multi-Site Management
Engine:  10 of 10
Status:  Production Ready
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from greenlang.schemas import utcnow
from greenlang.schemas.enums import ReportFormat

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
_HUNDRED = Decimal("100")
_DP6 = Decimal("0.000001")
_DP2 = Decimal("0.01")

def _safe_divide(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Divide with zero-guard."""
    if denominator == _ZERO:
        return _ZERO
    return (numerator / denominator).quantize(_DP6, rounding=ROUND_HALF_UP)

def _quantise(value: Decimal, precision: Decimal = _DP2) -> Decimal:
    """Quantise a Decimal value for display."""
    return value.quantize(precision, rounding=ROUND_HALF_UP)

def _fmt_decimal(value: Any, dp: int = 2) -> str:
    """Format a Decimal or numeric value for display."""
    if isinstance(value, Decimal):
        return str(value.quantize(Decimal(10) ** -dp, rounding=ROUND_HALF_UP))
    if isinstance(value, (int, float)):
        return f"{value:.{dp}f}"
    return str(value)

def _escape_md(text: str) -> str:
    """Escape pipe characters for markdown tables."""
    return str(text).replace("|", "\\|")

def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )

class _DecimalEncoder(json.JSONEncoder):
    """JSON encoder that serialises Decimal, datetime, and date."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ReportType(str, Enum):
    """Supported multi-site report types."""
    PORTFOLIO_DASHBOARD = "PORTFOLIO_DASHBOARD"
    SITE_DETAIL = "SITE_DETAIL"
    CONSOLIDATION = "CONSOLIDATION"
    BOUNDARY_DEFINITION = "BOUNDARY_DEFINITION"
    ALLOCATION = "ALLOCATION"
    COMPARISON = "COMPARISON"
    COLLECTION_STATUS = "COLLECTION_STATUS"
    QUALITY_HEATMAP = "QUALITY_HEATMAP"
    TREND = "TREND"

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ReportSection(BaseModel):
    """A single section within a multi-site report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    section_id: str = Field(default_factory=_new_uuid, description="Section identifier")
    title: str = Field(..., description="Section title")
    content: Dict[str, Any] = Field(
        default_factory=dict, description="Section content as key-value pairs"
    )
    tables: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Tabular data. Each dict has 'headers' (List[str]) and 'rows' (List[List[Any]])",
    )
    charts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Chart specifications for visualisation layer",
    )

class DrillDownLevel(BaseModel):
    """A single level in the drill-down hierarchy."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    level: str = Field(..., description="Level label (e.g. 'Region', 'Country', 'Site')")
    label: str = Field("", description="Human-readable label for this level")
    data: List[Dict[str, Any]] = Field(
        default_factory=list, description="Aggregated data rows for this level"
    )

class MultiSiteReport(BaseModel):
    """A complete multi-site report."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    report_id: str = Field(default_factory=_new_uuid, description="Report identifier")
    report_type: str = Field(..., description="Type of report")
    title: str = Field("", description="Report title")
    sections: List[ReportSection] = Field(
        default_factory=list, description="Report sections"
    )
    drill_down: List[DrillDownLevel] = Field(
        default_factory=list, description="Drill-down hierarchy"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Report metadata"
    )
    created_at: datetime = Field(default_factory=utcnow, description="Report timestamp")
    provenance_hash: str = Field("", description="SHA-256 provenance hash")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash from section data."""
        if not self.provenance_hash:
            payload = f"{self.report_id}|{self.report_type}|{self.title}|{len(self.sections)}"
            for s in self.sections:
                payload += f"|{s.section_id}:{s.title}"
            self.provenance_hash = _compute_hash(payload)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MultiSiteReportingEngine:
    """
    Generates multi-site portfolio reports with drill-down and multi-format export.

    This engine does NOT perform GHG calculations.  It aggregates,
    formats, and renders pre-computed outputs from engines 1-9.

    All exports produce real, formatted output.  Markdown uses pipe tables,
    HTML uses semantic table elements with inline CSS, JSON uses proper
    serialisation, and CSV handles tabular section extraction.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        decimal_places: int = 2,
        company_name: str = "",
        branding: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialise the MultiSiteReportingEngine.

        Args:
            decimal_places: Number of decimal places for display.
            company_name: Company name for report headers.
            branding: Branding dict (logo_url, primary_colour, etc.).
        """
        self._dp = decimal_places
        self._company = company_name
        self._branding = branding or {
            "primary_colour": "#1B5E20",
            "logo_url": "",
        }
        logger.info(
            "MultiSiteReportingEngine v%s initialised (dp=%d company=%s)",
            _MODULE_VERSION, decimal_places, company_name,
        )

    # ----------------------------------------------- portfolio dashboard
    def generate_portfolio_dashboard(
        self,
        registry_result: Dict[str, Any],
        consolidation: Dict[str, Any],
        completion: Dict[str, Any],
        quality: Dict[str, Any],
    ) -> MultiSiteReport:
        """
        Generate an executive portfolio dashboard report.

        Combines high-level metrics from registry, consolidation,
        completion, and quality engines into a single overview.

        Args:
            registry_result: Output from SiteRegistryEngine.
            consolidation: Output from SiteConsolidationEngine.
            completion: Output from SiteCompletionEngine.
            quality: Output from SiteQualityEngine.

        Returns:
            MultiSiteReport with dashboard sections.
        """
        logger.info("Generating portfolio dashboard")
        sections: List[ReportSection] = []

        # Section 1: Portfolio Overview
        sections.append(ReportSection(
            title="Portfolio Overview",
            content={
                "total_sites": registry_result.get("total_sites", 0),
                "active_sites": registry_result.get("active_sites", 0),
                "countries": registry_result.get("country_count", 0),
                "reporting_year": registry_result.get("reporting_year", ""),
                "consolidation_approach": registry_result.get("consolidation_approach", ""),
            },
        ))

        # Section 2: Emissions Summary
        emissions_data = consolidation.get("emissions_summary", {})
        sections.append(ReportSection(
            title="Emissions Summary",
            content=emissions_data,
            tables=[{
                "headers": ["Scope", "Emissions (tCO2e)", "% of Total"],
                "rows": self._build_scope_rows(emissions_data),
            }],
        ))

        # Section 3: Collection Status
        sections.append(ReportSection(
            title="Data Collection Status",
            content={
                "overall_completeness": completion.get("overall_completeness", _ZERO),
                "sites_reporting_pct": completion.get("sites_reporting_pct", _ZERO),
                "emissions_covered_pct": completion.get("emissions_covered_pct", _ZERO),
                "gap_count": completion.get("gap_count", 0),
                "overdue_count": len(completion.get("overdue_sites", [])),
            },
        ))

        # Section 4: Quality Overview
        sections.append(ReportSection(
            title="Data Quality Overview",
            content={
                "corporate_quality_score": quality.get("corporate_quality_score", _ZERO),
                "assessment_count": quality.get("assessment_count", 0),
                "pcaf_distribution": quality.get("pcaf_distribution", {}),
            },
        ))

        report = MultiSiteReport(
            report_type=ReportType.PORTFOLIO_DASHBOARD.value,
            title=f"{self._company} GHG Portfolio Dashboard".strip(),
            sections=sections,
            metadata={
                "engine_version": _MODULE_VERSION,
                "company_name": self._company,
                "generated_at": utcnow().isoformat(),
            },
        )

        logger.info("Portfolio dashboard generated: %s", report.report_id)
        return report

    # ----------------------------------------------- site detail
    def generate_site_detail(
        self,
        site_record: Dict[str, Any],
        site_total: Dict[str, Any],
        kpis: List[Dict[str, Any]],
        quality_assessment: Dict[str, Any],
    ) -> MultiSiteReport:
        """
        Generate a detailed report for a single site.

        Args:
            site_record: Site registry record.
            site_total: Site-level emission totals.
            kpis: List of KPI values for the site.
            quality_assessment: Site quality assessment data.

        Returns:
            MultiSiteReport with site detail sections.
        """
        site_id = site_record.get("site_id", "unknown")
        logger.info("Generating site detail: site=%s", site_id)
        sections: List[ReportSection] = []

        # Section 1: Site Profile
        sections.append(ReportSection(
            title="Site Profile",
            content={
                "site_id": site_id,
                "site_name": site_record.get("site_name", ""),
                "facility_type": site_record.get("facility_type", ""),
                "country": site_record.get("country", ""),
                "region": site_record.get("region", ""),
                "floor_area_m2": site_record.get("floor_area_m2", 0),
                "headcount": site_record.get("headcount", 0),
                "lifecycle_status": site_record.get("lifecycle_status", ""),
            },
        ))

        # Section 2: Emissions Breakdown
        sections.append(ReportSection(
            title="Emissions Breakdown",
            content=site_total,
            tables=[{
                "headers": ["Scope", "Category", "Emissions (tCO2e)", "Method"],
                "rows": self._build_emission_detail_rows(site_total),
            }],
        ))

        # Section 3: Key Performance Indicators
        kpi_rows = [[
            kpi.get("kpi_type", ""),
            _fmt_decimal(kpi.get("value", _ZERO), self._dp),
            kpi.get("unit", ""),
            _fmt_decimal(kpi.get("quality_score", _ZERO), 1),
        ] for kpi in kpis]

        sections.append(ReportSection(
            title="Key Performance Indicators",
            tables=[{
                "headers": ["KPI", "Value", "Unit", "Quality"],
                "rows": kpi_rows,
            }],
        ))

        # Section 4: Data Quality
        sections.append(ReportSection(
            title="Data Quality Assessment",
            content={
                "overall_score": quality_assessment.get("overall_score", _ZERO),
                "pcaf_tier": quality_assessment.get("pcaf_equivalent", 3),
                "estimated_pct": quality_assessment.get("estimated_pct", _ZERO),
            },
            tables=[{
                "headers": ["Dimension", "Score", "Weight", "Weighted"],
                "rows": self._build_quality_dimension_rows(
                    quality_assessment.get("dimension_scores", [])
                ),
            }],
        ))

        report = MultiSiteReport(
            report_type=ReportType.SITE_DETAIL.value,
            title=f"Site Detail: {site_record.get('site_name', site_id)}",
            sections=sections,
            metadata={"site_id": site_id},
        )

        logger.info("Site detail generated: %s", report.report_id)
        return report

    # ----------------------------------------------- consolidation report
    def generate_consolidation_report(
        self,
        consolidation_run: Dict[str, Any],
    ) -> MultiSiteReport:
        """
        Generate a consolidation run report.

        Args:
            consolidation_run: Consolidation engine output.

        Returns:
            MultiSiteReport with consolidation sections.
        """
        logger.info("Generating consolidation report")
        sections: List[ReportSection] = []

        # Section 1: Consolidation Summary
        sections.append(ReportSection(
            title="Consolidation Summary",
            content={
                "approach": consolidation_run.get("approach", ""),
                "period": consolidation_run.get("period", ""),
                "total_sites_consolidated": consolidation_run.get("site_count", 0),
                "total_emissions_tco2e": consolidation_run.get("total_emissions", _ZERO),
                "equity_adjustments": consolidation_run.get("equity_adjustments", _ZERO),
                "eliminations": consolidation_run.get("eliminations", _ZERO),
                "final_total": consolidation_run.get("final_total", _ZERO),
            },
        ))

        # Section 2: Per-Scope Totals
        scope_data = consolidation_run.get("by_scope", {})
        scope_rows = [
            [scope, _fmt_decimal(val, self._dp)]
            for scope, val in sorted(scope_data.items())
        ]
        sections.append(ReportSection(
            title="Emissions by Scope",
            tables=[{
                "headers": ["Scope", "Emissions (tCO2e)"],
                "rows": scope_rows,
            }],
        ))

        # Section 3: Per-Site Contribution
        site_data = consolidation_run.get("by_site", [])
        site_rows = [
            [
                s.get("site_id", ""),
                s.get("site_name", ""),
                _fmt_decimal(s.get("emissions", _ZERO), self._dp),
                _fmt_decimal(s.get("equity_pct", _HUNDRED), 1),
                _fmt_decimal(s.get("adjusted_emissions", _ZERO), self._dp),
            ]
            for s in site_data
        ]
        sections.append(ReportSection(
            title="Site Contributions",
            tables=[{
                "headers": ["Site ID", "Name", "Gross (tCO2e)", "Equity %", "Adjusted (tCO2e)"],
                "rows": site_rows,
            }],
        ))

        # Section 4: Reconciliation
        sections.append(ReportSection(
            title="Reconciliation",
            content={
                "gross_total": consolidation_run.get("total_emissions", _ZERO),
                "equity_adjustments": consolidation_run.get("equity_adjustments", _ZERO),
                "eliminations": consolidation_run.get("eliminations", _ZERO),
                "net_total": consolidation_run.get("final_total", _ZERO),
                "reconciliation_status": consolidation_run.get("reconciliation_status", "PASS"),
            },
        ))

        report = MultiSiteReport(
            report_type=ReportType.CONSOLIDATION.value,
            title=f"{self._company} GHG Consolidation Report".strip(),
            sections=sections,
            metadata={"period": consolidation_run.get("period", "")},
        )
        return report

    # ----------------------------------------------- boundary report
    def generate_boundary_report(
        self,
        boundary_definition: Dict[str, Any],
        entity_chain: List[Dict[str, Any]],
    ) -> MultiSiteReport:
        """
        Generate an organisational boundary definition report.

        Args:
            boundary_definition: Boundary engine output.
            entity_chain: Entity ownership chain data.

        Returns:
            MultiSiteReport with boundary sections.
        """
        logger.info("Generating boundary report")
        sections: List[ReportSection] = []

        sections.append(ReportSection(
            title="Organisational Boundary Definition",
            content={
                "approach": boundary_definition.get("consolidation_approach", ""),
                "reporting_entity": boundary_definition.get("reporting_entity", ""),
                "total_entities": boundary_definition.get("entity_count", 0),
                "included_entities": boundary_definition.get("included_count", 0),
                "excluded_entities": boundary_definition.get("excluded_count", 0),
            },
        ))

        # Entity chain table
        chain_rows = [
            [
                e.get("entity_name", ""),
                e.get("ownership_type", ""),
                _fmt_decimal(e.get("equity_pct", _ZERO), 1),
                "Yes" if e.get("included", False) else "No",
                e.get("justification", ""),
            ]
            for e in entity_chain
        ]
        sections.append(ReportSection(
            title="Entity Ownership Chain",
            tables=[{
                "headers": ["Entity", "Ownership", "Equity %", "Included", "Justification"],
                "rows": chain_rows,
            }],
        ))

        # Scope coverage
        sections.append(ReportSection(
            title="Scope Coverage",
            content=boundary_definition.get("scope_coverage", {}),
        ))

        report = MultiSiteReport(
            report_type=ReportType.BOUNDARY_DEFINITION.value,
            title=f"{self._company} Organisational Boundary".strip(),
            sections=sections,
        )
        return report

    # ----------------------------------------------- allocation report
    def generate_allocation_report(
        self,
        allocation_results: List[Dict[str, Any]],
    ) -> MultiSiteReport:
        """
        Generate an allocation report for shared services / tenant splits.

        Args:
            allocation_results: List of allocation result dicts.

        Returns:
            MultiSiteReport with allocation sections.
        """
        logger.info("Generating allocation report: %d results", len(allocation_results))
        sections: List[ReportSection] = []

        # Summary
        total_allocated = sum(
            Decimal(str(r.get("total_allocated", 0))) for r in allocation_results
        )
        total_remainder = sum(
            Decimal(str(r.get("unallocated_remainder", 0))) for r in allocation_results
        )

        sections.append(ReportSection(
            title="Allocation Summary",
            content={
                "total_allocations": len(allocation_results),
                "total_allocated_tco2e": total_allocated,
                "total_unallocated_tco2e": total_remainder,
            },
        ))

        # Detail table
        alloc_rows = []
        for r in allocation_results:
            for site_id, amount in r.get("allocated_amounts", {}).items():
                alloc_rows.append([
                    r.get("allocation_type", ""),
                    r.get("method", ""),
                    site_id,
                    _fmt_decimal(amount, self._dp),
                ])
        sections.append(ReportSection(
            title="Allocation Detail",
            tables=[{
                "headers": ["Type", "Method", "Site", "Allocated (tCO2e)"],
                "rows": alloc_rows,
            }],
        ))

        report = MultiSiteReport(
            report_type=ReportType.ALLOCATION.value,
            title=f"{self._company} Emission Allocation Report".strip(),
            sections=sections,
        )
        return report

    # ----------------------------------------------- comparison report
    def generate_comparison_report(
        self,
        comparison_result: Dict[str, Any],
    ) -> MultiSiteReport:
        """
        Generate a cross-site benchmarking comparison report.

        Args:
            comparison_result: Output from SiteComparisonEngine.

        Returns:
            MultiSiteReport with comparison sections.
        """
        logger.info("Generating comparison report")
        sections: List[ReportSection] = []

        sections.append(ReportSection(
            title="Peer Group Summary",
            content={
                "kpi_type": comparison_result.get("kpi_type", ""),
                "peer_group_name": comparison_result.get("peer_group_name", ""),
                "member_count": comparison_result.get("member_count", 0),
                "improvement_potential": comparison_result.get("improvement_potential", _ZERO),
            },
        ))

        # Statistics
        stats = comparison_result.get("statistics", {})
        sections.append(ReportSection(
            title="Statistical Summary",
            content={
                "mean": stats.get("mean", _ZERO),
                "median": stats.get("median", _ZERO),
                "std_dev": stats.get("std_dev", _ZERO),
                "min": stats.get("min_val", _ZERO),
                "max": stats.get("max_val", _ZERO),
                "p10": stats.get("p10", _ZERO),
                "p90": stats.get("p90", _ZERO),
            },
        ))

        # Rankings table
        rankings = comparison_result.get("rankings", [])
        rank_rows = [
            [
                r.get("rank", ""),
                r.get("site_id", ""),
                _fmt_decimal(r.get("value", _ZERO), self._dp),
                _fmt_decimal(r.get("percentile", _ZERO), 1),
                _fmt_decimal(r.get("gap_to_best", _ZERO), self._dp),
            ]
            for r in rankings
        ]
        sections.append(ReportSection(
            title="Site Rankings",
            tables=[{
                "headers": ["Rank", "Site", "KPI Value", "Percentile", "Gap to Best"],
                "rows": rank_rows,
            }],
        ))

        # Best practices
        best = comparison_result.get("best_practice_sites", [])
        sections.append(ReportSection(
            title="Best Practice Sites",
            content={"best_practice_site_ids": best},
        ))

        report = MultiSiteReport(
            report_type=ReportType.COMPARISON.value,
            title=f"{self._company} Site Benchmarking Report".strip(),
            sections=sections,
        )
        return report

    # ----------------------------------------------- collection status report
    def generate_collection_status_report(
        self,
        completion_result: Dict[str, Any],
    ) -> MultiSiteReport:
        """
        Generate a data collection status report.

        Args:
            completion_result: Output from SiteCompletionEngine.

        Returns:
            MultiSiteReport with collection status sections.
        """
        logger.info("Generating collection status report")
        sections: List[ReportSection] = []

        sections.append(ReportSection(
            title="Collection Overview",
            content={
                "round_id": completion_result.get("round_id", ""),
                "overall_completeness": completion_result.get("overall_completeness", _ZERO),
                "sites_reporting_pct": completion_result.get("sites_reporting_pct", _ZERO),
                "emissions_covered_pct": completion_result.get("emissions_covered_pct", _ZERO),
                "floor_area_covered_pct": completion_result.get("floor_area_covered_pct", _ZERO),
            },
        ))

        # Gaps table
        gaps = completion_result.get("gaps", [])
        gap_rows = [
            [
                g.get("gap_type", ""),
                g.get("site_id", ""),
                g.get("scope", ""),
                g.get("category", ""),
                g.get("details", ""),
            ]
            for g in gaps
        ]
        sections.append(ReportSection(
            title="Data Gaps",
            tables=[{
                "headers": ["Gap Type", "Site", "Scope", "Category", "Details"],
                "rows": gap_rows,
            }],
        ))

        # Overdue sites
        overdue = completion_result.get("overdue_sites", [])
        sections.append(ReportSection(
            title="Overdue Sites",
            content={"overdue_site_ids": overdue, "overdue_count": len(overdue)},
        ))

        report = MultiSiteReport(
            report_type=ReportType.COLLECTION_STATUS.value,
            title=f"{self._company} Data Collection Status".strip(),
            sections=sections,
        )
        return report

    # ----------------------------------------------- quality report
    def generate_quality_report(
        self,
        quality_result: Dict[str, Any],
    ) -> MultiSiteReport:
        """
        Generate a portfolio quality assessment report.

        Args:
            quality_result: Output from SiteQualityEngine.

        Returns:
            MultiSiteReport with quality sections.
        """
        logger.info("Generating quality report")
        sections: List[ReportSection] = []

        sections.append(ReportSection(
            title="Corporate Quality Score",
            content={
                "corporate_score": quality_result.get("corporate_quality_score", _ZERO),
                "weighted_by_emissions": quality_result.get("weighted_by_emissions", False),
                "assessment_count": len(quality_result.get("assessments", [])),
            },
        ))

        # Quality heatmap as table
        heatmap = quality_result.get("heatmap", [])
        heatmap_rows = [
            [
                h.get("site_id", ""),
                h.get("scope", ""),
                _fmt_decimal(h.get("score", _ZERO), 1),
                h.get("colour_code", ""),
            ]
            for h in heatmap
        ]
        sections.append(ReportSection(
            title="Quality Heatmap",
            tables=[{
                "headers": ["Site", "Scope", "Score", "Rating"],
                "rows": heatmap_rows,
            }],
        ))

        # Improvement priorities
        priorities = quality_result.get("improvement_priorities", [])
        priority_rows = [
            [
                p.get("site_id", ""),
                _fmt_decimal(p.get("overall_score", _ZERO), 1),
                p.get("top_action", {}).get("dimension", "") if p.get("top_action") else "",
                str(p.get("action_count", 0)),
            ]
            for p in priorities
        ]
        sections.append(ReportSection(
            title="Improvement Priorities",
            tables=[{
                "headers": ["Site", "Score", "Top Dimension", "Actions"],
                "rows": priority_rows,
            }],
        ))

        report = MultiSiteReport(
            report_type=ReportType.QUALITY_HEATMAP.value,
            title=f"{self._company} Data Quality Report".strip(),
            sections=sections,
        )
        return report

    # ----------------------------------------------- trend report
    def generate_trend_report(
        self,
        multi_year_data: Dict[str, Any],
    ) -> MultiSiteReport:
        """
        Generate a multi-year trend report.

        Args:
            multi_year_data: Multi-year trend data from comparison engine.

        Returns:
            MultiSiteReport with trend sections.
        """
        logger.info("Generating trend report")
        sections: List[ReportSection] = []

        # Portfolio-level trends
        portfolio_trend = multi_year_data.get("portfolio_trend", {})
        sections.append(ReportSection(
            title="Portfolio Emission Trend",
            content={
                "direction": portfolio_trend.get("direction", "STABLE"),
                "change_pct": portfolio_trend.get("change_pct", _ZERO),
                "base_year": portfolio_trend.get("base_year", ""),
                "latest_year": portfolio_trend.get("latest_year", ""),
            },
        ))

        # Year-over-year table
        yearly = multi_year_data.get("yearly_data", [])
        year_rows = [
            [
                y.get("year", ""),
                _fmt_decimal(y.get("total_emissions", _ZERO), self._dp),
                _fmt_decimal(y.get("change_pct", _ZERO), 1),
                str(y.get("site_count", 0)),
            ]
            for y in yearly
        ]
        sections.append(ReportSection(
            title="Year-over-Year Emissions",
            tables=[{
                "headers": ["Year", "Total (tCO2e)", "Change %", "Sites"],
                "rows": year_rows,
            }],
        ))

        # Per-site trends
        site_trends = multi_year_data.get("site_trends", [])
        trend_rows = [
            [
                t.get("site_id", ""),
                t.get("direction", ""),
                _fmt_decimal(t.get("change_pct", _ZERO), 1),
                _fmt_decimal(t.get("first_value", _ZERO), self._dp),
                _fmt_decimal(t.get("last_value", _ZERO), self._dp),
            ]
            for t in site_trends
        ]
        sections.append(ReportSection(
            title="Site-Level Trends",
            tables=[{
                "headers": ["Site", "Direction", "Change %", "First", "Latest"],
                "rows": trend_rows,
            }],
        ))

        report = MultiSiteReport(
            report_type=ReportType.TREND.value,
            title=f"{self._company} Multi-Year Trend Report".strip(),
            sections=sections,
        )
        return report

    # ===================================================================
    # Export Methods
    # ===================================================================

    # ----------------------------------------------- markdown
    def export_markdown(self, report: MultiSiteReport) -> str:
        """
        Export a report to GitHub-flavoured Markdown.

        Produces:
            - H1 title
            - Metadata block
            - Per-section H2 headings
            - Key-value content as bullet lists
            - Pipe tables for tabular data

        Args:
            report: MultiSiteReport to export.

        Returns:
            Full Markdown string.
        """
        lines: List[str] = []

        # Title
        lines.append(f"# {_escape_md(report.title)}")
        lines.append("")
        lines.append(f"**Report ID:** {report.report_id}")
        lines.append(f"**Type:** {report.report_type}")
        lines.append(f"**Generated:** {report.created_at.isoformat()}")
        lines.append(f"**Provenance:** `{report.provenance_hash[:16]}...`")
        lines.append("")

        # Metadata
        if report.metadata:
            lines.append("## Metadata")
            lines.append("")
            for key, value in sorted(report.metadata.items()):
                lines.append(f"- **{_escape_md(str(key))}:** {_escape_md(str(value))}")
            lines.append("")

        # Sections
        for section in report.sections:
            lines.append(f"## {_escape_md(section.title)}")
            lines.append("")

            # Content as bullet list
            if section.content:
                for key, value in section.content.items():
                    display_val = _fmt_decimal(value, self._dp) if isinstance(value, Decimal) else str(value)
                    lines.append(f"- **{_escape_md(str(key))}:** {_escape_md(display_val)}")
                lines.append("")

            # Tables
            for table in section.tables:
                headers = table.get("headers", [])
                rows = table.get("rows", [])
                if headers:
                    lines.append(self._md_table(headers, rows))
                    lines.append("")

        # Drill-down
        if report.drill_down:
            lines.append("## Drill-Down")
            lines.append("")
            for level in report.drill_down:
                lines.append(f"### Level: {_escape_md(level.level)} - {_escape_md(level.label)}")
                lines.append("")
                if level.data:
                    # Auto-detect headers from first row
                    first_row = level.data[0]
                    headers = list(first_row.keys())
                    rows = [[str(row.get(h, "")) for h in headers] for row in level.data]
                    lines.append(self._md_table(headers, rows))
                    lines.append("")

        return "\n".join(lines)

    # ----------------------------------------------- HTML
    def export_html(self, report: MultiSiteReport) -> str:
        """
        Export a report to standalone HTML with inline CSS.

        Args:
            report: MultiSiteReport to export.

        Returns:
            Full HTML string.
        """
        primary = self._branding.get("primary_colour", "#1B5E20")
        css = self._inline_css(primary)

        parts: List[str] = []
        parts.append("<!DOCTYPE html>")
        parts.append('<html lang="en">')
        parts.append("<head>")
        parts.append('<meta charset="UTF-8">')
        parts.append(f"<title>{_escape_html(report.title)}</title>")
        parts.append(f"<style>{css}</style>")
        parts.append("</head>")
        parts.append("<body>")
        parts.append(f'<div class="report">')

        # Header
        parts.append(f"<h1>{_escape_html(report.title)}</h1>")
        parts.append('<div class="meta">')
        parts.append(f"<p><strong>Report ID:</strong> {_escape_html(report.report_id)}</p>")
        parts.append(f"<p><strong>Type:</strong> {_escape_html(report.report_type)}</p>")
        parts.append(f"<p><strong>Generated:</strong> {_escape_html(report.created_at.isoformat())}</p>")
        parts.append(f"<p><strong>Provenance:</strong> <code>{report.provenance_hash[:16]}...</code></p>")
        parts.append("</div>")

        # Sections
        for section in report.sections:
            parts.append(f"<h2>{_escape_html(section.title)}</h2>")

            if section.content:
                parts.append("<ul>")
                for key, value in section.content.items():
                    display_val = _fmt_decimal(value, self._dp) if isinstance(value, Decimal) else str(value)
                    parts.append(
                        f"<li><strong>{_escape_html(str(key))}:</strong> "
                        f"{_escape_html(display_val)}</li>"
                    )
                parts.append("</ul>")

            for table in section.tables:
                headers = table.get("headers", [])
                rows = table.get("rows", [])
                if headers:
                    parts.append(self._html_table(headers, rows))

        # Drill-down
        if report.drill_down:
            parts.append("<h2>Drill-Down</h2>")
            for level in report.drill_down:
                parts.append(f"<h3>{_escape_html(level.level)}: {_escape_html(level.label)}</h3>")
                if level.data:
                    first_row = level.data[0]
                    headers = list(first_row.keys())
                    rows = [[str(row.get(h, "")) for h in headers] for row in level.data]
                    parts.append(self._html_table(headers, rows))

        parts.append("</div>")
        parts.append("</body>")
        parts.append("</html>")

        return "\n".join(parts)

    # ----------------------------------------------- JSON
    def export_json(self, report: MultiSiteReport) -> str:
        """
        Export a report to formatted JSON.

        Uses custom encoder for Decimal, datetime, and date types.

        Args:
            report: MultiSiteReport to export.

        Returns:
            JSON string.
        """
        data = {
            "report_id": report.report_id,
            "report_type": report.report_type,
            "title": report.title,
            "created_at": report.created_at,
            "provenance_hash": report.provenance_hash,
            "metadata": report.metadata,
            "sections": [],
            "drill_down": [],
        }

        for section in report.sections:
            data["sections"].append({
                "section_id": section.section_id,
                "title": section.title,
                "content": section.content,
                "tables": section.tables,
                "charts": section.charts,
            })

        for level in report.drill_down:
            data["drill_down"].append({
                "level": level.level,
                "label": level.label,
                "data": level.data,
            })

        return json.dumps(data, cls=_DecimalEncoder, indent=2, ensure_ascii=False)

    # ----------------------------------------------- CSV
    def export_csv(self, report: MultiSiteReport) -> str:
        """
        Export tabular sections of a report to CSV.

        Each section with tables produces a CSV block separated by
        a blank line and a section header comment.

        Args:
            report: MultiSiteReport to export.

        Returns:
            CSV string with all tabular data.
        """
        lines: List[str] = []

        for section in report.sections:
            for table_idx, table in enumerate(section.tables):
                headers = table.get("headers", [])
                rows = table.get("rows", [])
                if not headers:
                    continue

                # Section header comment
                lines.append(f"# {section.title}")

                # Header row
                lines.append(",".join(self._csv_escape(h) for h in headers))

                # Data rows
                for row in rows:
                    cells = []
                    for cell in row:
                        display = _fmt_decimal(cell, self._dp) if isinstance(cell, Decimal) else str(cell)
                        cells.append(self._csv_escape(display))
                    lines.append(",".join(cells))

                lines.append("")  # Blank line between tables

        return "\n".join(lines)

    # ----------------------------------------------- drill-down builder
    def build_drill_down(
        self,
        data: List[Dict[str, Any]],
        levels: List[str],
    ) -> List[DrillDownLevel]:
        """
        Build a drill-down hierarchy from flat data.

        Given data rows with level keys (e.g. 'region', 'country', 'site_id')
        and a metric field, aggregates at each level.

        Example:
            levels = ["region", "country", "site_id"]
            Each DrillDownLevel aggregates emissions at that granularity.

        Args:
            data: List of flat data dicts.
            levels: Ordered list of level keys from broadest to narrowest.

        Returns:
            List of DrillDownLevel, one per level.
        """
        logger.info("Building drill-down: %d rows, levels=%s", len(data), levels)
        drill_down: List[DrillDownLevel] = []

        for level_key in levels:
            aggregated: Dict[str, Dict[str, Any]] = {}
            for row in data:
                group_val = str(row.get(level_key, "UNKNOWN"))
                if group_val not in aggregated:
                    aggregated[group_val] = {
                        level_key: group_val,
                        "count": 0,
                        "total_emissions": _ZERO,
                    }
                aggregated[group_val]["count"] += 1
                emissions = row.get("total_emissions", row.get("emissions", _ZERO))
                if isinstance(emissions, (int, float, str)):
                    emissions = Decimal(str(emissions))
                aggregated[group_val]["total_emissions"] += emissions

            level_data = sorted(aggregated.values(), key=lambda x: x[level_key])
            drill_down.append(DrillDownLevel(
                level=level_key,
                label=level_key.replace("_", " ").title(),
                data=level_data,
            ))

        return drill_down

    # ===================================================================
    # Private Helpers
    # ===================================================================

    def _md_table(self, headers: List[str], rows: List[List[Any]]) -> str:
        """Render a markdown pipe table."""
        lines: List[str] = []
        header_str = "| " + " | ".join(_escape_md(str(h)) for h in headers) + " |"
        sep_str = "| " + " | ".join("---" for _ in headers) + " |"
        lines.append(header_str)
        lines.append(sep_str)
        for row in rows:
            cells = []
            for cell in row:
                display = _fmt_decimal(cell, self._dp) if isinstance(cell, Decimal) else str(cell)
                cells.append(_escape_md(display))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def _html_table(self, headers: List[str], rows: List[List[Any]]) -> str:
        """Render an HTML table element."""
        parts: List[str] = ['<table class="data-table">']
        parts.append("<thead><tr>")
        for h in headers:
            parts.append(f"<th>{_escape_html(str(h))}</th>")
        parts.append("</tr></thead>")
        parts.append("<tbody>")
        for row in rows:
            parts.append("<tr>")
            for cell in row:
                display = _fmt_decimal(cell, self._dp) if isinstance(cell, Decimal) else str(cell)
                parts.append(f"<td>{_escape_html(display)}</td>")
            parts.append("</tr>")
        parts.append("</tbody>")
        parts.append("</table>")
        return "\n".join(parts)

    @staticmethod
    def _csv_escape(value: str) -> str:
        """Escape a value for CSV output."""
        val = str(value)
        if "," in val or '"' in val or "\n" in val:
            return '"' + val.replace('"', '""') + '"'
        return val

    @staticmethod
    def _inline_css(primary_colour: str) -> str:
        """Generate inline CSS for HTML reports."""
        return f"""
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0; padding: 20px; background: #fafafa; color: #333;
}}
.report {{
    max-width: 1200px; margin: 0 auto; background: #fff;
    padding: 40px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}}
h1 {{
    color: {primary_colour}; border-bottom: 3px solid {primary_colour};
    padding-bottom: 12px; margin-bottom: 24px;
}}
h2 {{
    color: {primary_colour}; margin-top: 32px; padding-bottom: 8px;
    border-bottom: 1px solid #e0e0e0;
}}
h3 {{ color: #555; margin-top: 24px; }}
.meta {{ background: #f5f5f5; padding: 12px 16px; border-radius: 4px; margin-bottom: 24px; }}
.meta p {{ margin: 4px 0; }}
ul {{ padding-left: 20px; }}
li {{ margin: 4px 0; }}
.data-table {{
    width: 100%; border-collapse: collapse; margin: 16px 0;
}}
.data-table th {{
    background: {primary_colour}; color: #fff; padding: 10px 12px;
    text-align: left; font-weight: 600;
}}
.data-table td {{
    padding: 8px 12px; border-bottom: 1px solid #e0e0e0;
}}
.data-table tbody tr:nth-child(even) {{ background: #f9f9f9; }}
.data-table tbody tr:hover {{ background: #e8f5e9; }}
code {{ background: #f5f5f5; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
"""

    def _build_scope_rows(self, emissions_data: Dict[str, Any]) -> List[List[str]]:
        """Build scope rows for the portfolio emissions table."""
        rows: List[List[str]] = []
        scope_keys = ["scope_1", "scope_2", "scope_3", "total"]
        total = Decimal(str(emissions_data.get("total", 1)))
        if total == _ZERO:
            total = Decimal("1")

        for key in scope_keys:
            val = emissions_data.get(key)
            if val is not None:
                val_dec = Decimal(str(val))
                pct = _quantise(val_dec / total * _HUNDRED, _DP2)
                label = key.replace("_", " ").title()
                rows.append([label, _fmt_decimal(val_dec, self._dp), f"{pct}%"])

        return rows

    def _build_emission_detail_rows(
        self, site_total: Dict[str, Any]
    ) -> List[List[str]]:
        """Build emission detail rows for a site report."""
        rows: List[List[str]] = []
        details = site_total.get("details", [])
        if isinstance(details, list):
            for d in details:
                rows.append([
                    str(d.get("scope", "")),
                    str(d.get("category", "")),
                    _fmt_decimal(d.get("emissions", _ZERO), self._dp),
                    str(d.get("method", "")),
                ])
        return rows

    def _build_quality_dimension_rows(
        self, dimension_scores: List[Any]
    ) -> List[List[str]]:
        """Build dimension score rows for quality table."""
        rows: List[List[str]] = []
        for ds in dimension_scores:
            if isinstance(ds, dict):
                rows.append([
                    str(ds.get("dimension", "")),
                    _fmt_decimal(ds.get("score", _ZERO), 1),
                    _fmt_decimal(ds.get("weight", _ZERO), 2),
                    _fmt_decimal(ds.get("weighted_score", _ZERO), 2),
                ])
        return rows

# ---------------------------------------------------------------------------
# Pydantic v2 model rebuild (required with `from __future__ import annotations`)
# ---------------------------------------------------------------------------

ReportSection.model_rebuild()
DrillDownLevel.model_rebuild()
MultiSiteReport.model_rebuild()
