"""
CrossRegulationMappingReport - CBAM cross-regulation data mapping template.

This module implements the cross-regulation mapping report for PACK-005 CBAM
Complete. It generates reports showing how CBAM data flows to and supports
compliance with 6 related regulations: CSRD, CDP, SBTi, EU Taxonomy, EU ETS,
and EUDR. Includes data reuse statistics and consistency checks.

Example:
    >>> template = CrossRegulationMappingReport()
    >>> data = CrossRegulationData(
    ...     data_flow_overview=DataFlowOverview(total_data_points=150, ...),
    ...     csrd_mapping=CSRDMapping(mapped_fields=[...]),
    ...     ...
    ... )
    >>> markdown = template.render_markdown(data.model_dump())
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
#  Pydantic Input Models
# ---------------------------------------------------------------------------

class DataFlowOverview(BaseModel):
    """Overview of data flows across regulations."""

    total_data_points: int = Field(0, ge=0, description="Total CBAM data points")
    regulations_supported: int = Field(6, ge=0, description="Number of regulations supported")
    total_mappings: int = Field(0, ge=0, description="Total cross-regulation mappings")
    reuse_ratio: float = Field(0.0, ge=0.0, description="Data reuse ratio")
    last_sync_date: str = Field("", description="Last data synchronization date")


class MappedField(BaseModel):
    """Individual data field mapping between CBAM and a target regulation."""

    cbam_field: str = Field("", description="CBAM source field")
    cbam_value_example: str = Field("", description="Example CBAM value")
    target_field: str = Field("", description="Target regulation field")
    target_section: str = Field("", description="Target regulation section/reference")
    mapping_type: str = Field("direct", description="Mapping type: direct, derived, aggregated")
    status: str = Field("mapped", description="Status: mapped, pending, conflict")
    notes: str = Field("", description="Mapping notes")


class CSRDMapping(BaseModel):
    """CBAM to CSRD ESRS E1 mapping data."""

    total_mapped: int = Field(0, ge=0, description="Total fields mapped")
    coverage_pct: float = Field(0.0, ge=0.0, le=100.0, description="Coverage percentage")
    mapped_fields: List[MappedField] = Field(default_factory=list, description="Field mappings")


class CDPMapping(BaseModel):
    """CBAM to CDP sections mapping data."""

    total_mapped: int = Field(0, ge=0, description="Total fields mapped")
    coverage_pct: float = Field(0.0, ge=0.0, le=100.0, description="Coverage percentage")
    sections_covered: List[str] = Field(default_factory=list, description="CDP sections: C6, C7, C11, C12")
    mapped_fields: List[MappedField] = Field(default_factory=list, description="Field mappings")


class SBTiMapping(BaseModel):
    """CBAM to SBTi Scope 3 Category 1 mapping data."""

    total_mapped: int = Field(0, ge=0, description="Total fields mapped")
    coverage_pct: float = Field(0.0, ge=0.0, le=100.0, description="Coverage percentage")
    scope3_categories_covered: List[str] = Field(
        default_factory=list, description="Scope 3 categories covered"
    )
    mapped_fields: List[MappedField] = Field(default_factory=list, description="Field mappings")


class TaxonomyMapping(BaseModel):
    """CBAM to EU Taxonomy climate mitigation mapping data."""

    total_mapped: int = Field(0, ge=0, description="Total fields mapped")
    coverage_pct: float = Field(0.0, ge=0.0, le=100.0, description="Coverage percentage")
    sectors_aligned: List[str] = Field(default_factory=list, description="Aligned sectors")
    mapped_fields: List[MappedField] = Field(default_factory=list, description="Field mappings")


class ETSMapping(BaseModel):
    """CBAM to EU ETS free allocation mapping data."""

    total_mapped: int = Field(0, ge=0, description="Total fields mapped")
    coverage_pct: float = Field(0.0, ge=0.0, le=100.0, description="Coverage percentage")
    benchmark_references: List[str] = Field(default_factory=list, description="Benchmark references")
    phase_out_schedule: List[Dict[str, Any]] = Field(
        default_factory=list, description="Free allocation phase-out: year, pct"
    )
    mapped_fields: List[MappedField] = Field(default_factory=list, description="Field mappings")


class EUDRMapping(BaseModel):
    """CBAM to EUDR fertilizer supply chain mapping data."""

    total_mapped: int = Field(0, ge=0, description="Total fields mapped")
    coverage_pct: float = Field(0.0, ge=0.0, le=100.0, description="Coverage percentage")
    supply_chain_links: List[str] = Field(
        default_factory=list, description="Supply chain traceability links"
    )
    mapped_fields: List[MappedField] = Field(default_factory=list, description="Field mappings")


class DataReuseStatistic(BaseModel):
    """Data reuse statistics per regulation."""

    regulation: str = Field("", description="Regulation name")
    data_points_shared: int = Field(0, ge=0, description="Data points shared from CBAM")
    data_points_unique: int = Field(0, ge=0, description="Unique data points needed")
    reuse_pct: float = Field(0.0, ge=0.0, le=100.0, description="Reuse percentage")
    efficiency_gain_hours: float = Field(0.0, ge=0.0, description="Estimated hours saved")


class ConsistencyCheck(BaseModel):
    """Cross-regulation consistency check result."""

    check_id: str = Field("", description="Check identifier")
    regulation_a: str = Field("", description="First regulation")
    regulation_b: str = Field("", description="Second regulation")
    field_name: str = Field("", description="Field being compared")
    value_a: str = Field("", description="Value in regulation A")
    value_b: str = Field("", description="Value in regulation B")
    status: str = Field("consistent", description="consistent, conflict, warning")
    severity: str = Field("info", description="info, warning, error")
    resolution: str = Field("", description="Resolution if conflict")


class CrossRegulationData(BaseModel):
    """Complete input data for cross-regulation mapping report."""

    data_flow_overview: DataFlowOverview = Field(default_factory=DataFlowOverview)
    csrd_mapping: CSRDMapping = Field(default_factory=CSRDMapping)
    cdp_mapping: CDPMapping = Field(default_factory=CDPMapping)
    sbti_mapping: SBTiMapping = Field(default_factory=SBTiMapping)
    taxonomy_mapping: TaxonomyMapping = Field(default_factory=TaxonomyMapping)
    ets_mapping: ETSMapping = Field(default_factory=ETSMapping)
    eudr_mapping: EUDRMapping = Field(default_factory=EUDRMapping)
    reuse_statistics: List[DataReuseStatistic] = Field(default_factory=list)
    consistency_checks: List[ConsistencyCheck] = Field(default_factory=list)


# ---------------------------------------------------------------------------
#  Template Implementation
# ---------------------------------------------------------------------------

class CrossRegulationMappingReport:
    """
    CBAM cross-regulation data mapping report template.

    Generates reports mapping CBAM data to 6 related EU and international
    regulations (CSRD, CDP, SBTi, EU Taxonomy, EU ETS, EUDR) with reuse
    statistics and consistency checks.

    Attributes:
        config: Optional configuration dictionary.
        pack_id: Pack identifier (PACK-005).
        template_name: Template name for metadata.
        version: Template version.

    Example:
        >>> template = CrossRegulationMappingReport()
        >>> md = template.render_markdown(data)
        >>> assert "CSRD Mapping" in md
    """

    PACK_ID = "PACK-005"
    TEMPLATE_NAME = "cross_regulation_mapping_report"
    VERSION = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize CrossRegulationMappingReport.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public render methods
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """
        Render the cross-regulation mapping report as Markdown.

        Args:
            data: Report data dictionary matching CrossRegulationData schema.

        Returns:
            GitHub-flavored Markdown string with provenance hash.
        """
        sections: List[str] = []
        sections.append(self._md_header())
        sections.append(self._md_data_flow_overview(data))
        sections.append(self._md_regulation_mapping(data, "csrd_mapping", "2", "CSRD Mapping (ESRS E1)"))
        sections.append(self._md_regulation_mapping(data, "cdp_mapping", "3", "CDP Mapping (C6, C7, C11, C12)"))
        sections.append(self._md_regulation_mapping(data, "sbti_mapping", "4", "SBTi Mapping (Scope 3 Cat 1)"))
        sections.append(self._md_regulation_mapping(data, "taxonomy_mapping", "5", "EU Taxonomy Mapping"))
        sections.append(self._md_ets_mapping(data))
        sections.append(self._md_regulation_mapping(data, "eudr_mapping", "7", "EUDR Mapping (Fertilizer Supply Chain)"))
        sections.append(self._md_reuse_statistics(data))
        sections.append(self._md_consistency_checks(data))

        content = "\n\n".join(sections)
        provenance_hash = self._compute_provenance_hash(content)
        footer = self._md_footer(provenance_hash)
        content += "\n\n" + footer
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the cross-regulation mapping report as self-contained HTML.

        Args:
            data: Report data dictionary matching CrossRegulationData schema.

        Returns:
            Complete HTML document with inline CSS and provenance hash.
        """
        sections: List[str] = []
        sections.append(self._html_header())
        sections.append(self._html_data_flow_overview(data))
        sections.append(self._html_regulation_mapping(data, "csrd_mapping", "2", "CSRD Mapping (ESRS E1)"))
        sections.append(self._html_regulation_mapping(data, "cdp_mapping", "3", "CDP Mapping (C6, C7, C11, C12)"))
        sections.append(self._html_regulation_mapping(data, "sbti_mapping", "4", "SBTi Mapping (Scope 3 Cat 1)"))
        sections.append(self._html_regulation_mapping(data, "taxonomy_mapping", "5", "EU Taxonomy Mapping"))
        sections.append(self._html_ets_mapping(data))
        sections.append(self._html_regulation_mapping(data, "eudr_mapping", "7", "EUDR Mapping (Fertilizer Supply Chain)"))
        sections.append(self._html_reuse_statistics(data))
        sections.append(self._html_consistency_checks(data))

        body = "\n".join(sections)
        provenance_hash = self._compute_provenance_hash(body)
        return self._wrap_html(
            title="CBAM Cross-Regulation Mapping Report",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the cross-regulation mapping report as structured JSON.

        Args:
            data: Report data dictionary matching CrossRegulationData schema.

        Returns:
            Dictionary with all sections, metadata, and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "cbam_cross_regulation_mapping",
            "pack_id": self.PACK_ID,
            "template_name": self.TEMPLATE_NAME,
            "version": self.VERSION,
            "generated_at": self.generated_at,
            "data_flow_overview": self._json_overview(data),
            "csrd_mapping": self._json_mapping_section(data, "csrd_mapping"),
            "cdp_mapping": self._json_mapping_section(data, "cdp_mapping"),
            "sbti_mapping": self._json_mapping_section(data, "sbti_mapping"),
            "taxonomy_mapping": self._json_mapping_section(data, "taxonomy_mapping"),
            "ets_mapping": self._json_ets_mapping(data),
            "eudr_mapping": self._json_mapping_section(data, "eudr_mapping"),
            "reuse_statistics": self._json_reuse_statistics(data),
            "consistency_checks": self._json_consistency_checks(data),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._compute_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown section builders
    # ------------------------------------------------------------------ #

    def _md_header(self) -> str:
        """Build Markdown report header."""
        return (
            "# CBAM Cross-Regulation Mapping Report\n\n"
            f"**Pack:** {self.PACK_ID} | "
            f"**Template:** {self.TEMPLATE_NAME} | "
            f"**Version:** {self.VERSION}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_data_flow_overview(self, data: Dict[str, Any]) -> str:
        """Build Markdown data flow overview section."""
        dfo = data.get("data_flow_overview", {})

        overview = (
            "## 1. Data Flow Overview\n\n"
            "```\n"
            "                    +----------+\n"
            "                    |   CBAM   |\n"
            "                    | Data Hub |\n"
            "                    +----+-----+\n"
            "                         |\n"
            "         +-------+-------+-------+-------+-------+\n"
            "         |       |       |       |       |       |\n"
            "      +--v--+ +--v--+ +--v--+ +--v--+ +--v--+ +--v--+\n"
            "      |CSRD | | CDP | |SBTi | |Taxon| | ETS | |EUDR |\n"
            "      +-----+ +-----+ +-----+ +-----+ +-----+ +-----+\n"
            "```\n\n"
            "| Metric | Value |\n"
            "|--------|-------|\n"
            f"| Total CBAM Data Points | {self._fmt_int(dfo.get('total_data_points', 0))} |\n"
            f"| Regulations Supported | {dfo.get('regulations_supported', 6)} |\n"
            f"| Total Mappings | {self._fmt_int(dfo.get('total_mappings', 0))} |\n"
            f"| Data Reuse Ratio | {dfo.get('reuse_ratio', 0.0):.1f}x |\n"
            f"| Last Sync | {dfo.get('last_sync_date', 'N/A')} |"
        )

        return overview

    def _md_regulation_mapping(
        self, data: Dict[str, Any], key: str, num: str, title: str
    ) -> str:
        """Build generic Markdown regulation mapping section."""
        mapping = data.get(key, {})
        fields = mapping.get("mapped_fields", [])

        summary = (
            f"## {num}. {title}\n\n"
            f"**Mapped Fields:** {mapping.get('total_mapped', 0)} | "
            f"**Coverage:** {mapping.get('coverage_pct', 0.0):.1f}%\n\n"
        )

        if not fields:
            return summary + "*No field mappings defined.*"

        table = (
            "| CBAM Field | Target Field | Section | Type | Status | Notes |\n"
            "|------------|-------------|---------|------|--------|-------|\n"
        )

        rows: List[str] = []
        for f in fields:
            status = f.get("status", "mapped")
            status_fmt = f"**{status.upper()}**" if status == "conflict" else status.upper()
            rows.append(
                f"| {f.get('cbam_field', '')} | "
                f"{f.get('target_field', '')} | "
                f"{f.get('target_section', '')} | "
                f"{f.get('mapping_type', 'direct')} | "
                f"{status_fmt} | "
                f"{f.get('notes', '') or '-'} |"
            )

        return summary + table + "\n".join(rows)

    def _md_ets_mapping(self, data: Dict[str, Any]) -> str:
        """Build Markdown EU ETS mapping section with phase-out schedule."""
        mapping = data.get("ets_mapping", {})
        fields = mapping.get("mapped_fields", [])
        schedule = mapping.get("phase_out_schedule", [])

        base = self._md_regulation_mapping(
            data, "ets_mapping", "6", "EU ETS Mapping (Free Allocation)"
        )

        if not schedule:
            return base

        phase_out = (
            "\n\n### Free Allocation Phase-Out Schedule\n\n"
            "| Year | Free Allocation (%) |\n"
            "|------|--------------------|\n"
        )

        rows: List[str] = []
        for entry in schedule:
            year = entry.get("year", "")
            pct = entry.get("pct", 0.0)
            rows.append(f"| {year} | {pct:.1f}% |")

        return base + phase_out + "\n".join(rows)

    def _md_reuse_statistics(self, data: Dict[str, Any]) -> str:
        """Build Markdown data reuse statistics section."""
        stats = data.get("reuse_statistics", [])

        header = (
            "## 8. Data Reuse Statistics\n\n"
            "| Regulation | Shared Points | Unique Points | Reuse (%) | Hours Saved |\n"
            "|------------|---------------|---------------|-----------|------------|\n"
        )

        rows: List[str] = []
        total_shared = 0
        total_unique = 0
        total_hours = 0.0

        for s in stats:
            shared = s.get("data_points_shared", 0)
            unique = s.get("data_points_unique", 0)
            hours = s.get("efficiency_gain_hours", 0.0)
            total_shared += shared
            total_unique += unique
            total_hours += hours

            rows.append(
                f"| {s.get('regulation', '')} | "
                f"{self._fmt_int(shared)} | "
                f"{self._fmt_int(unique)} | "
                f"{s.get('reuse_pct', 0.0):.1f}% | "
                f"{hours:.1f}h |"
            )

        if not rows:
            return header + "| *No statistics available* | | | | |"

        total_reuse = (total_shared / max(total_shared + total_unique, 1)) * 100
        rows.append(
            f"| **TOTAL** | **{self._fmt_int(total_shared)}** | "
            f"**{self._fmt_int(total_unique)}** | "
            f"**{total_reuse:.1f}%** | **{total_hours:.1f}h** |"
        )

        return header + "\n".join(rows)

    def _md_consistency_checks(self, data: Dict[str, Any]) -> str:
        """Build Markdown consistency checks section."""
        checks = data.get("consistency_checks", [])

        header = (
            "## 9. Consistency Checks\n\n"
            "| ID | Reg A | Reg B | Field | Value A | Value B | Status | Severity | Resolution |\n"
            "|----|-------|-------|-------|---------|---------|--------|----------|------------|\n"
        )

        rows: List[str] = []
        for c in checks:
            status = c.get("status", "consistent")
            severity = c.get("severity", "info")
            status_fmt = f"**{status.upper()}**" if status == "conflict" else status.upper()
            resolution = c.get("resolution", "") or "-"
            rows.append(
                f"| {c.get('check_id', '')} | "
                f"{c.get('regulation_a', '')} | "
                f"{c.get('regulation_b', '')} | "
                f"{c.get('field_name', '')} | "
                f"{c.get('value_a', '')} | "
                f"{c.get('value_b', '')} | "
                f"{status_fmt} | "
                f"{severity.upper()} | "
                f"{resolution} |"
            )

        if not rows:
            return header + "| *No checks performed* | | | | | | | | |"

        # Summary
        conflicts = sum(1 for c in checks if c.get("status") == "conflict")
        warnings = sum(1 for c in checks if c.get("status") == "warning")
        consistent = sum(1 for c in checks if c.get("status") == "consistent")

        summary = (
            f"\n\n**Summary:** {consistent} consistent | "
            f"{warnings} warnings | {conflicts} conflicts"
        )

        return header + "\n".join(rows) + summary

    def _md_footer(self, provenance_hash: str) -> str:
        """Build Markdown provenance footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            f"*Template: {self.TEMPLATE_NAME} v{self.VERSION} | "
            f"Pack: {self.PACK_ID}*\n\n"
            f"*Provenance Hash: `{provenance_hash}`*"
        )

    # ------------------------------------------------------------------ #
    #  HTML section builders
    # ------------------------------------------------------------------ #

    def _html_header(self) -> str:
        """Build HTML report header."""
        return (
            '<div class="report-header">'
            '<h1>CBAM Cross-Regulation Mapping Report</h1>'
            f'<div class="meta-item">Pack: {self.PACK_ID} | '
            f'Template: {self.TEMPLATE_NAME} | Version: {self.VERSION}</div>'
            f'<div class="meta-item">Generated: {self.generated_at}</div>'
            '</div>'
        )

    def _html_data_flow_overview(self, data: Dict[str, Any]) -> str:
        """Build HTML data flow overview section."""
        dfo = data.get("data_flow_overview", {})

        regs = ["CSRD", "CDP", "SBTi", "Taxonomy", "EU ETS", "EUDR"]
        colors = ["#2ecc71", "#3498db", "#9b59b6", "#1abc9c", "#f39c12", "#e67e22"]

        nodes = ""
        for reg, color in zip(regs, colors):
            nodes += (
                f'<div style="background:{color};color:#fff;padding:12px 16px;'
                f'border-radius:8px;text-align:center;font-weight:bold;font-size:14px">'
                f'{reg}</div>'
            )

        kpis = (
            f'<div class="kpi-grid">'
            f'<div class="kpi-card"><div class="kpi-label">Total Data Points</div>'
            f'<div class="kpi-value">{self._fmt_int(dfo.get("total_data_points", 0))}</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">Regulations</div>'
            f'<div class="kpi-value">{dfo.get("regulations_supported", 6)}</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">Total Mappings</div>'
            f'<div class="kpi-value">{self._fmt_int(dfo.get("total_mappings", 0))}</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">Reuse Ratio</div>'
            f'<div class="kpi-value">{dfo.get("reuse_ratio", 0.0):.1f}x</div></div>'
            f'</div>'
        )

        return (
            '<div class="section"><h2>1. Data Flow Overview</h2>'
            f'{kpis}'
            '<div style="text-align:center;margin:20px 0">'
            '<div style="background:#1a5276;color:#fff;padding:16px;border-radius:8px;'
            'display:inline-block;font-size:18px;font-weight:bold;margin-bottom:16px">'
            'CBAM Data Hub</div></div>'
            f'<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px">'
            f'{nodes}</div></div>'
        )

    def _html_regulation_mapping(
        self, data: Dict[str, Any], key: str, num: str, title: str
    ) -> str:
        """Build generic HTML regulation mapping section."""
        mapping = data.get(key, {})
        fields = mapping.get("mapped_fields", [])
        total = mapping.get("total_mapped", 0)
        coverage = mapping.get("coverage_pct", 0.0)

        coverage_color = "#2ecc71" if coverage >= 80 else "#f39c12" if coverage >= 50 else "#e74c3c"

        summary = (
            f'<div style="display:flex;gap:24px;margin-bottom:16px">'
            f'<div><strong>Mapped Fields:</strong> {total}</div>'
            f'<div><strong>Coverage:</strong> '
            f'<span style="color:{coverage_color};font-weight:bold">{coverage:.1f}%</span></div>'
            f'</div>'
            f'<div class="progress-bar">'
            f'<div class="progress-fill" style="width:{min(coverage, 100):.0f}%;'
            f'background:{coverage_color}"></div></div>'
        )

        rows_html = ""
        status_colors = {"mapped": "#2ecc71", "pending": "#f39c12", "conflict": "#e74c3c"}

        for f in fields:
            status = f.get("status", "mapped")
            color = status_colors.get(status, "#95a5a6")
            rows_html += (
                f'<tr><td>{f.get("cbam_field", "")}</td>'
                f'<td>{f.get("target_field", "")}</td>'
                f'<td>{f.get("target_section", "")}</td>'
                f'<td>{f.get("mapping_type", "direct")}</td>'
                f'<td style="color:{color};font-weight:bold">{status.upper()}</td>'
                f'<td>{f.get("notes", "") or "-"}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="6"><em>No field mappings defined</em></td></tr>'

        return (
            f'<div class="section"><h2>{num}. {title}</h2>'
            f'{summary}'
            '<table><thead><tr>'
            '<th>CBAM Field</th><th>Target Field</th><th>Section</th>'
            '<th>Type</th><th>Status</th><th>Notes</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_ets_mapping(self, data: Dict[str, Any]) -> str:
        """Build HTML EU ETS mapping section with phase-out schedule."""
        base = self._html_regulation_mapping(
            data, "ets_mapping", "6", "EU ETS Mapping (Free Allocation)"
        )

        schedule = data.get("ets_mapping", {}).get("phase_out_schedule", [])
        if not schedule:
            return base

        rows_html = ""
        for entry in schedule:
            pct = entry.get("pct", 0.0)
            color = "#2ecc71" if pct >= 80 else "#f39c12" if pct >= 40 else "#e74c3c"
            rows_html += (
                f'<tr><td>{entry.get("year", "")}</td>'
                f'<td class="num" style="color:{color}">{pct:.1f}%</td>'
                f'<td><div class="progress-bar">'
                f'<div class="progress-fill" style="width:{min(pct, 100):.0f}%;'
                f'background:{color}"></div></div></td></tr>'
            )

        phase_out = (
            '<div style="margin-top:16px"><h3>Free Allocation Phase-Out Schedule</h3>'
            '<table><thead><tr><th>Year</th><th>Free Allocation</th>'
            '<th>Level</th></tr></thead>'
            f'<tbody>{rows_html}</tbody></table></div>'
        )

        # Insert before closing div
        return base[:-6] + phase_out + '</div>'

    def _html_reuse_statistics(self, data: Dict[str, Any]) -> str:
        """Build HTML data reuse statistics section."""
        stats = data.get("reuse_statistics", [])

        rows_html = ""
        total_shared = 0
        total_unique = 0
        total_hours = 0.0

        for s in stats:
            shared = s.get("data_points_shared", 0)
            unique = s.get("data_points_unique", 0)
            hours = s.get("efficiency_gain_hours", 0.0)
            reuse = s.get("reuse_pct", 0.0)
            total_shared += shared
            total_unique += unique
            total_hours += hours

            reuse_color = "#2ecc71" if reuse >= 70 else "#f39c12" if reuse >= 40 else "#e74c3c"
            rows_html += (
                f'<tr><td>{s.get("regulation", "")}</td>'
                f'<td class="num">{self._fmt_int(shared)}</td>'
                f'<td class="num">{self._fmt_int(unique)}</td>'
                f'<td class="num" style="color:{reuse_color}">{reuse:.1f}%</td>'
                f'<td class="num">{hours:.1f}h</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="5"><em>No statistics available</em></td></tr>'
        else:
            total_reuse = (total_shared / max(total_shared + total_unique, 1)) * 100
            rows_html += (
                f'<tr style="font-weight:bold;background:#eef2f7">'
                f'<td>TOTAL</td>'
                f'<td class="num">{self._fmt_int(total_shared)}</td>'
                f'<td class="num">{self._fmt_int(total_unique)}</td>'
                f'<td class="num">{total_reuse:.1f}%</td>'
                f'<td class="num">{total_hours:.1f}h</td></tr>'
            )

        return (
            '<div class="section"><h2>8. Data Reuse Statistics</h2>'
            '<table><thead><tr>'
            '<th>Regulation</th><th>Shared Points</th><th>Unique Points</th>'
            '<th>Reuse (%)</th><th>Hours Saved</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_consistency_checks(self, data: Dict[str, Any]) -> str:
        """Build HTML consistency checks section."""
        checks = data.get("consistency_checks", [])

        status_colors = {"consistent": "#2ecc71", "warning": "#f39c12", "conflict": "#e74c3c"}
        severity_colors = {"info": "#3498db", "warning": "#f39c12", "error": "#e74c3c"}

        rows_html = ""
        for c in checks:
            status = c.get("status", "consistent")
            severity = c.get("severity", "info")
            s_color = status_colors.get(status, "#95a5a6")
            sv_color = severity_colors.get(severity, "#95a5a6")
            resolution = c.get("resolution", "") or "-"

            rows_html += (
                f'<tr><td>{c.get("check_id", "")}</td>'
                f'<td>{c.get("regulation_a", "")}</td>'
                f'<td>{c.get("regulation_b", "")}</td>'
                f'<td>{c.get("field_name", "")}</td>'
                f'<td>{c.get("value_a", "")}</td>'
                f'<td>{c.get("value_b", "")}</td>'
                f'<td style="color:{s_color};font-weight:bold">{status.upper()}</td>'
                f'<td style="color:{sv_color}">{severity.upper()}</td>'
                f'<td>{resolution}</td></tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="9"><em>No checks performed</em></td></tr>'

        conflicts = sum(1 for c in checks if c.get("status") == "conflict")
        warnings = sum(1 for c in checks if c.get("status") == "warning")
        consistent = sum(1 for c in checks if c.get("status") == "consistent")

        summary = (
            f'<div style="margin-top:12px;padding:12px;background:#f8f9fa;'
            f'border-radius:4px;display:flex;gap:24px">'
            f'<div style="color:#2ecc71"><strong>Consistent:</strong> {consistent}</div>'
            f'<div style="color:#f39c12"><strong>Warnings:</strong> {warnings}</div>'
            f'<div style="color:#e74c3c"><strong>Conflicts:</strong> {conflicts}</div>'
            f'</div>'
        )

        return (
            '<div class="section"><h2>9. Consistency Checks</h2>'
            '<table><thead><tr>'
            '<th>ID</th><th>Reg A</th><th>Reg B</th><th>Field</th>'
            '<th>Value A</th><th>Value B</th><th>Status</th>'
            '<th>Severity</th><th>Resolution</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table>'
            f'{summary}</div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON section builders
    # ------------------------------------------------------------------ #

    def _json_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON data flow overview."""
        dfo = data.get("data_flow_overview", {})
        return {
            "total_data_points": dfo.get("total_data_points", 0),
            "regulations_supported": dfo.get("regulations_supported", 6),
            "total_mappings": dfo.get("total_mappings", 0),
            "reuse_ratio": round(dfo.get("reuse_ratio", 0.0), 2),
            "last_sync_date": dfo.get("last_sync_date", ""),
        }

    def _json_mapping_section(self, data: Dict[str, Any], key: str) -> Dict[str, Any]:
        """Build JSON mapping section for any regulation."""
        mapping = data.get(key, {})
        return {
            "total_mapped": mapping.get("total_mapped", 0),
            "coverage_pct": round(mapping.get("coverage_pct", 0.0), 2),
            "mapped_fields": [
                {
                    "cbam_field": f.get("cbam_field", ""),
                    "target_field": f.get("target_field", ""),
                    "target_section": f.get("target_section", ""),
                    "mapping_type": f.get("mapping_type", "direct"),
                    "status": f.get("status", "mapped"),
                    "notes": f.get("notes", ""),
                }
                for f in mapping.get("mapped_fields", [])
            ],
        }

    def _json_ets_mapping(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON EU ETS mapping with phase-out schedule."""
        result = self._json_mapping_section(data, "ets_mapping")
        mapping = data.get("ets_mapping", {})
        result["benchmark_references"] = mapping.get("benchmark_references", [])
        result["phase_out_schedule"] = [
            {
                "year": e.get("year", ""),
                "pct": round(e.get("pct", 0.0), 1),
            }
            for e in mapping.get("phase_out_schedule", [])
        ]
        return result

    def _json_reuse_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON reuse statistics."""
        stats = data.get("reuse_statistics", [])
        total_shared = sum(s.get("data_points_shared", 0) for s in stats)
        total_unique = sum(s.get("data_points_unique", 0) for s in stats)
        total_hours = sum(s.get("efficiency_gain_hours", 0.0) for s in stats)

        return {
            "per_regulation": [
                {
                    "regulation": s.get("regulation", ""),
                    "data_points_shared": s.get("data_points_shared", 0),
                    "data_points_unique": s.get("data_points_unique", 0),
                    "reuse_pct": round(s.get("reuse_pct", 0.0), 2),
                    "efficiency_gain_hours": round(s.get("efficiency_gain_hours", 0.0), 2),
                }
                for s in stats
            ],
            "totals": {
                "total_shared": total_shared,
                "total_unique": total_unique,
                "overall_reuse_pct": round(
                    (total_shared / max(total_shared + total_unique, 1)) * 100, 2
                ),
                "total_hours_saved": round(total_hours, 2),
            },
        }

    def _json_consistency_checks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON consistency checks."""
        checks = data.get("consistency_checks", [])
        conflicts = sum(1 for c in checks if c.get("status") == "conflict")
        warnings = sum(1 for c in checks if c.get("status") == "warning")
        consistent = sum(1 for c in checks if c.get("status") == "consistent")

        return {
            "checks": [
                {
                    "check_id": c.get("check_id", ""),
                    "regulation_a": c.get("regulation_a", ""),
                    "regulation_b": c.get("regulation_b", ""),
                    "field_name": c.get("field_name", ""),
                    "value_a": c.get("value_a", ""),
                    "value_b": c.get("value_b", ""),
                    "status": c.get("status", "consistent"),
                    "severity": c.get("severity", "info"),
                    "resolution": c.get("resolution", ""),
                }
                for c in checks
            ],
            "summary": {
                "total_checks": len(checks),
                "consistent": consistent,
                "warnings": warnings,
                "conflicts": conflicts,
            },
        }

    # ------------------------------------------------------------------ #
    #  Helper methods
    # ------------------------------------------------------------------ #

    def _compute_provenance_hash(self, content: str) -> str:
        """Compute SHA-256 provenance hash for audit trail."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _fmt_int(self, value: Union[int, float, None]) -> str:
        """Format integer with thousand separators."""
        if value is None:
            return "0"
        return f"{int(value):,}"

    def _fmt_num(self, value: Union[int, float], decimals: int = 2) -> str:
        """Format number with thousand separators and fixed decimals."""
        return f"{value:,.{decimals}f}"

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        """Wrap HTML body in a complete document with inline CSS."""
        css = self._get_css()
        return (
            f'<!DOCTYPE html><html lang="en"><head>'
            f'<meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1.0">'
            f'<title>{title}</title>'
            f'<style>{css}</style>'
            f'</head><body>'
            f'{body}'
            f'<div class="provenance">'
            f'Report generated: {self.generated_at} | '
            f'Template: {self.TEMPLATE_NAME} v{self.VERSION} | '
            f'Pack: {self.PACK_ID} | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )

    def _get_css(self) -> str:
        """Return inline CSS for HTML reports."""
        return (
            "body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
            "margin:0;padding:20px;background:#f5f7fa;color:#2c3e50}"
            ".report-header{background:#1a5276;color:#fff;padding:24px;border-radius:8px;"
            "margin-bottom:24px}"
            ".report-header h1{margin:0 0 8px 0;font-size:24px}"
            ".meta-item{font-size:13px;opacity:0.8}"
            ".kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin-bottom:16px}"
            ".kpi-card{background:#fff;padding:20px;border-radius:8px;text-align:center;"
            "box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".kpi-label{font-size:13px;color:#7f8c8d;margin-bottom:4px}"
            ".kpi-value{font-size:28px;font-weight:700;color:#1a5276}"
            ".section{background:#fff;padding:20px;border-radius:8px;"
            "margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".section h2{margin:0 0 16px 0;font-size:18px;color:#1a5276;"
            "border-bottom:2px solid #eef2f7;padding-bottom:8px}"
            ".section h3{margin:16px 0 8px 0;font-size:15px;color:#2c3e50}"
            "table{width:100%;border-collapse:collapse;font-size:14px}"
            "th{background:#eef2f7;padding:10px 12px;text-align:left;font-weight:600}"
            "td{padding:8px 12px;border-bottom:1px solid #eef2f7}"
            ".num{text-align:right;font-variant-numeric:tabular-nums}"
            ".progress-bar{background:#ecf0f1;border-radius:4px;height:12px;"
            "overflow:hidden;margin:4px 0}"
            ".progress-fill{height:100%;border-radius:4px}"
            ".provenance{text-align:center;color:#95a5a6;font-size:12px;margin-top:24px}"
        )
