"""
DeduplicationSavingsReportTemplate - Data deduplication and effort savings report.

This module implements the DeduplicationSavingsReportTemplate for PACK-009
EU Climate Compliance Bundle. It renders reports quantifying the benefits
of cross-regulation data deduplication including fields deduplicated,
effort reduction, cost impact analysis, per-category breakdowns, and
before/after comparisons.

Example:
    >>> template = DeduplicationSavingsReportTemplate()
    >>> data = DeduplicationData(
    ...     groups=[...],
    ...     savings=SavingsMetrics(...),
    ...     before_count=500,
    ...     after_count=320,
    ... )
    >>> md = template.render(data, fmt="markdown")
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
#  Pydantic data models
# ---------------------------------------------------------------------------

class DeduplicationGroup(BaseModel):
    """A group of deduplicated fields across regulations."""

    group_id: str = Field(..., description="Unique group identifier")
    canonical_field: str = Field(..., description="Canonical/master field name")
    category: str = Field("", description="Field category e.g. GHG, Financial")
    source_fields: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of regulation code to original field name",
    )
    regulations: List[str] = Field(
        default_factory=list, description="Regulations sharing this field"
    )
    dedup_type: str = Field("exact", description="exact, normalized, derived")
    original_count: int = Field(1, ge=1, description="Number of original separate fields")
    deduplicated_to: int = Field(1, ge=1, description="Number of fields after dedup")
    effort_saved_hours: float = Field(0.0, ge=0.0, description="Hours saved per reporting cycle")
    cost_saved_eur: float = Field(0.0, ge=0.0, description="Cost saved per reporting cycle EUR")
    description: str = Field("", description="Description of the deduplication")


class SavingsMetrics(BaseModel):
    """Aggregate savings metrics from deduplication."""

    total_hours_saved: float = Field(0.0, ge=0.0, description="Total hours saved per cycle")
    total_cost_saved_eur: float = Field(0.0, ge=0.0, description="Total cost saved per cycle EUR")
    annual_hours_saved: float = Field(0.0, ge=0.0, description="Annualized hours saved")
    annual_cost_saved_eur: float = Field(0.0, ge=0.0, description="Annualized cost saved EUR")
    fte_equivalent: float = Field(0.0, ge=0.0, description="FTE equivalent of hours saved")
    data_quality_improvement_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Data quality improvement %"
    )
    error_reduction_pct: float = Field(
        0.0, ge=0.0, le=100.0, description="Error reduction %"
    )
    reporting_cycles_per_year: int = Field(1, ge=1, description="Number of reporting cycles/year")


class CategorySavings(BaseModel):
    """Savings breakdown for a specific field category."""

    category: str = Field(..., description="Category name")
    fields_before: int = Field(0, ge=0, description="Fields before dedup")
    fields_after: int = Field(0, ge=0, description="Fields after dedup")
    dedup_pct: float = Field(0.0, ge=0.0, le=100.0, description="Deduplication %")
    hours_saved: float = Field(0.0, ge=0.0, description="Hours saved")
    cost_saved_eur: float = Field(0.0, ge=0.0, description="Cost saved EUR")


class DeduplicationConfig(BaseModel):
    """Configuration for the deduplication savings template."""

    title: str = Field(
        "Deduplication Savings Report",
        description="Report title",
    )
    hourly_rate_eur: float = Field(75.0, description="Assumed hourly rate for cost calculations")
    show_group_details: bool = Field(True, description="Show individual dedup group details")


class DeduplicationData(BaseModel):
    """Input data for the deduplication savings report."""

    groups: List[DeduplicationGroup] = Field(
        default_factory=list, description="Deduplication groups"
    )
    savings: SavingsMetrics = Field(
        default_factory=SavingsMetrics, description="Aggregate savings metrics"
    )
    category_savings: List[CategorySavings] = Field(
        default_factory=list, description="Per-category savings breakdown"
    )
    before_count: int = Field(0, ge=0, description="Total fields before deduplication")
    after_count: int = Field(0, ge=0, description="Total fields after deduplication")
    regulations_covered: List[str] = Field(
        default_factory=list, description="Regulations covered by deduplication"
    )
    reporting_period: str = Field("", description="Reporting period label")
    organization_name: str = Field("", description="Organization name")

    @field_validator("groups")
    @classmethod
    def validate_groups_present(cls, v: List[DeduplicationGroup]) -> List[DeduplicationGroup]:
        """Ensure at least one group is provided."""
        if not v:
            raise ValueError("groups must contain at least one entry")
        return v


# ---------------------------------------------------------------------------
#  Template class
# ---------------------------------------------------------------------------

class DeduplicationSavingsReportTemplate:
    """
    Deduplication savings report template for cross-regulation data optimization.

    Generates reports showing the quantified benefits of deduplicating shared
    data fields across regulations, including hours saved, cost impact,
    per-category breakdowns, and before/after comparisons.

    Attributes:
        config: Template configuration.
        generated_at: ISO timestamp of report generation.
    """

    DEDUP_TYPE_LABELS = {
        "exact": {"label": "Exact Match", "color": "#2ecc71"},
        "normalized": {"label": "Normalized", "color": "#3498db"},
        "derived": {"label": "Derived", "color": "#9b59b6"},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize DeduplicationSavingsReportTemplate.

        Args:
            config: Optional configuration dictionary.
        """
        raw = config or {}
        self.config = DeduplicationConfig(**raw) if raw else DeduplicationConfig()
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def render(self, data: DeduplicationData, fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """
        Render the deduplication savings report in the specified format.

        Args:
            data: Validated DeduplicationData input.
            fmt: Output format - 'markdown', 'html', or 'json'.

        Returns:
            Rendered output.

        Raises:
            ValueError: If fmt is unsupported.
        """
        if fmt == "markdown":
            return self.render_markdown(data)
        elif fmt == "html":
            return self.render_html(data)
        elif fmt == "json":
            return self.render_json(data)
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Use 'markdown', 'html', or 'json'.")

    def render_markdown(self, data: DeduplicationData) -> str:
        """Render as Markdown string."""
        sections: List[str] = [
            self._md_header(data),
            self._md_headline_savings(data),
            self._md_before_after(data),
            self._md_category_breakdown(data),
            self._md_group_details(data),
            self._md_cost_impact(data),
            self._md_footer(),
        ]
        content = "\n\n".join(s for s in sections if s)
        provenance = self._generate_provenance_hash(content)
        content += f"\n\n<!-- provenance_hash: {provenance} -->"
        return content

    def render_html(self, data: DeduplicationData) -> str:
        """Render as self-contained HTML document."""
        sections: List[str] = [
            self._html_header(data),
            self._html_headline_savings(data),
            self._html_before_after(data),
            self._html_category_breakdown(data),
            self._html_group_details(data),
            self._html_cost_impact(data),
        ]
        body = "\n".join(s for s in sections if s)
        provenance = self._generate_provenance_hash(body)
        return self._wrap_html(body, provenance)

    def render_json(self, data: DeduplicationData) -> Dict[str, Any]:
        """Render as structured dictionary."""
        report: Dict[str, Any] = {
            "report_type": "deduplication_savings",
            "template_version": "1.0",
            "generated_at": self.generated_at,
            "organization": data.organization_name,
            "reporting_period": data.reporting_period,
            "headline_savings": self._json_headline(data),
            "before_after": self._json_before_after(data),
            "category_breakdown": self._json_categories(data),
            "dedup_groups": self._json_groups(data),
            "cost_impact": self._json_cost_impact(data),
        }
        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._generate_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Computation helpers
    # ------------------------------------------------------------------ #

    def _dedup_percentage(self, data: DeduplicationData) -> float:
        """Calculate overall deduplication percentage."""
        if data.before_count == 0:
            return 0.0
        reduced = data.before_count - data.after_count
        return (reduced / data.before_count) * 100.0

    def _fields_deduplicated(self, data: DeduplicationData) -> int:
        """Calculate total fields deduplicated."""
        return data.before_count - data.after_count

    # ------------------------------------------------------------------ #
    #  Markdown builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: DeduplicationData) -> str:
        """Build markdown header."""
        org = data.organization_name or "Organization"
        period = data.reporting_period or "Current Period"
        regs = ", ".join(data.regulations_covered) if data.regulations_covered else "All bundled"
        return (
            f"# {self.config.title}\n\n"
            f"**Organization:** {org}\n\n"
            f"**Reporting Period:** {period}\n\n"
            f"**Regulations:** {regs}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_headline_savings(self, data: DeduplicationData) -> str:
        """Build headline savings section."""
        s = data.savings
        dedup_pct = self._dedup_percentage(data)
        deduped = self._fields_deduplicated(data)
        return (
            "## Headline Savings\n\n"
            f"- **Fields Deduplicated:** {deduped} of {data.before_count} ({dedup_pct:.1f}%)\n"
            f"- **Hours Saved Per Cycle:** {s.total_hours_saved:,.1f}\n"
            f"- **Cost Saved Per Cycle:** EUR {s.total_cost_saved_eur:,.0f}\n"
            f"- **Annual Hours Saved:** {s.annual_hours_saved:,.1f}\n"
            f"- **Annual Cost Saved:** EUR {s.annual_cost_saved_eur:,.0f}\n"
            f"- **FTE Equivalent:** {s.fte_equivalent:.2f}\n"
            f"- **Data Quality Improvement:** {s.data_quality_improvement_pct:.1f}%\n"
            f"- **Error Reduction:** {s.error_reduction_pct:.1f}%"
        )

    def _md_before_after(self, data: DeduplicationData) -> str:
        """Build before/after comparison section."""
        dedup_pct = self._dedup_percentage(data)
        before_bar_len = 20
        after_bar_len = int((data.after_count / data.before_count) * 20) if data.before_count > 0 else 0
        before_bar = "#" * before_bar_len
        after_bar = "#" * after_bar_len + "-" * (20 - after_bar_len)
        return (
            "## Before / After Comparison\n\n"
            f"```\nBefore: [{before_bar}] {data.before_count} fields\n"
            f"After:  [{after_bar}] {data.after_count} fields\n"
            f"Reduction: {dedup_pct:.1f}%\n```"
        )

    def _md_category_breakdown(self, data: DeduplicationData) -> str:
        """Build per-category breakdown table."""
        if not data.category_savings:
            return ""
        header = (
            "## Category Breakdown\n\n"
            "| Category | Before | After | Dedup % | Hours Saved | Cost Saved (EUR) |\n"
            "|----------|--------|-------|---------|-------------|------------------|\n"
        )
        rows: List[str] = []
        for cs in sorted(data.category_savings, key=lambda c: c.dedup_pct, reverse=True):
            rows.append(
                f"| {cs.category} | {cs.fields_before} | {cs.fields_after} | "
                f"{cs.dedup_pct:.1f}% | {cs.hours_saved:,.1f} | {cs.cost_saved_eur:,.0f} |"
            )
        total_before = sum(cs.fields_before for cs in data.category_savings)
        total_after = sum(cs.fields_after for cs in data.category_savings)
        total_hours = sum(cs.hours_saved for cs in data.category_savings)
        total_cost = sum(cs.cost_saved_eur for cs in data.category_savings)
        total_pct = ((total_before - total_after) / total_before * 100) if total_before > 0 else 0.0
        rows.append(
            f"| **Total** | **{total_before}** | **{total_after}** | "
            f"**{total_pct:.1f}%** | **{total_hours:,.1f}** | **{total_cost:,.0f}** |"
        )
        return header + "\n".join(rows)

    def _md_group_details(self, data: DeduplicationData) -> str:
        """Build individual deduplication group details."""
        if not self.config.show_group_details or not data.groups:
            return ""
        header = (
            "## Deduplication Groups\n\n"
            "| Group | Canonical Field | Category | Regulations | "
            "Type | Original | Deduped | Hours Saved |\n"
            "|-------|----------------|----------|-------------|"
            "------|----------|---------|-------------|\n"
        )
        rows: List[str] = []
        for g in sorted(data.groups, key=lambda x: x.effort_saved_hours, reverse=True):
            regs = ", ".join(g.regulations)
            type_label = self.DEDUP_TYPE_LABELS.get(g.dedup_type, {"label": g.dedup_type})["label"]
            rows.append(
                f"| {g.group_id} | {g.canonical_field} | {g.category} | "
                f"{regs} | {type_label} | {g.original_count} | "
                f"{g.deduplicated_to} | {g.effort_saved_hours:,.1f} |"
            )
        return header + "\n".join(rows)

    def _md_cost_impact(self, data: DeduplicationData) -> str:
        """Build cost impact analysis section."""
        s = data.savings
        hourly = self.config.hourly_rate_eur
        return (
            "## Cost Impact Analysis\n\n"
            f"**Assumed Hourly Rate:** EUR {hourly:,.0f}\n\n"
            f"| Metric | Per Cycle | Annual ({s.reporting_cycles_per_year}x) |\n"
            f"|--------|-----------|--------|\n"
            f"| Hours Saved | {s.total_hours_saved:,.1f} | {s.annual_hours_saved:,.1f} |\n"
            f"| Cost Saved | EUR {s.total_cost_saved_eur:,.0f} | EUR {s.annual_cost_saved_eur:,.0f} |\n"
            f"| FTE Equivalent | --- | {s.fte_equivalent:.2f} |\n"
            f"| Quality Improvement | {s.data_quality_improvement_pct:.1f}% | --- |\n"
            f"| Error Reduction | {s.error_reduction_pct:.1f}% | --- |"
        )

    def _md_footer(self) -> str:
        """Build markdown footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            "*Template: DeduplicationSavingsReportTemplate v1.0 | PACK-009*"
        )

    # ------------------------------------------------------------------ #
    #  HTML builders
    # ------------------------------------------------------------------ #

    def _html_header(self, data: DeduplicationData) -> str:
        """Build HTML header."""
        org = data.organization_name or "Organization"
        period = data.reporting_period or "Current Period"
        return (
            '<div class="report-header">'
            f'<h1>{self.config.title}</h1>'
            '<div class="header-meta">'
            f'<div class="meta-item">Organization: {org}</div>'
            f'<div class="meta-item">Period: {period}</div>'
            f'<div class="meta-item">Generated: {self.generated_at}</div>'
            '</div></div>'
        )

    def _html_headline_savings(self, data: DeduplicationData) -> str:
        """Build HTML headline savings cards."""
        s = data.savings
        dedup_pct = self._dedup_percentage(data)
        deduped = self._fields_deduplicated(data)
        return (
            '<div class="section"><h2>Headline Savings</h2>'
            '<div class="stat-grid">'
            f'<div class="stat-card highlight"><span class="stat-val">{deduped}</span>'
            f'<span class="stat-lbl">Fields Deduplicated ({dedup_pct:.0f}%)</span></div>'
            f'<div class="stat-card highlight"><span class="stat-val">'
            f'{s.annual_hours_saved:,.0f}h</span>'
            f'<span class="stat-lbl">Annual Hours Saved</span></div>'
            f'<div class="stat-card highlight"><span class="stat-val">'
            f'EUR {s.annual_cost_saved_eur:,.0f}</span>'
            f'<span class="stat-lbl">Annual Cost Saved</span></div>'
            f'<div class="stat-card"><span class="stat-val">{s.fte_equivalent:.1f}</span>'
            f'<span class="stat-lbl">FTE Equivalent</span></div>'
            f'<div class="stat-card"><span class="stat-val">'
            f'{s.data_quality_improvement_pct:.0f}%</span>'
            f'<span class="stat-lbl">Quality Improvement</span></div>'
            f'<div class="stat-card"><span class="stat-val">'
            f'{s.error_reduction_pct:.0f}%</span>'
            f'<span class="stat-lbl">Error Reduction</span></div>'
            '</div></div>'
        )

    def _html_before_after(self, data: DeduplicationData) -> str:
        """Build HTML before/after comparison with visual bars."""
        dedup_pct = self._dedup_percentage(data)
        after_pct = (data.after_count / data.before_count * 100) if data.before_count > 0 else 0
        return (
            '<div class="section"><h2>Before / After Comparison</h2>'
            '<div class="ba-comparison">'
            f'<div class="ba-row"><span class="ba-label">Before</span>'
            f'<div class="ba-bar"><div class="ba-fill" style="width:100%;background:#e74c3c">'
            f'</div></div><span class="ba-val">{data.before_count} fields</span></div>'
            f'<div class="ba-row"><span class="ba-label">After</span>'
            f'<div class="ba-bar"><div class="ba-fill" style="width:{after_pct:.0f}%;background:#2ecc71">'
            f'</div></div><span class="ba-val">{data.after_count} fields</span></div>'
            f'<div class="ba-summary">Reduction: <strong>{dedup_pct:.1f}%</strong></div>'
            '</div></div>'
        )

    def _html_category_breakdown(self, data: DeduplicationData) -> str:
        """Build HTML category breakdown table."""
        if not data.category_savings:
            return ""
        rows = ""
        for cs in sorted(data.category_savings, key=lambda c: c.dedup_pct, reverse=True):
            bar_color = "#2ecc71" if cs.dedup_pct >= 50 else "#f39c12" if cs.dedup_pct >= 25 else "#e74c3c"
            rows += (
                f'<tr><td>{cs.category}</td>'
                f'<td class="num">{cs.fields_before}</td>'
                f'<td class="num">{cs.fields_after}</td>'
                f'<td><div class="progress-bar"><div class="progress-fill" '
                f'style="width:{cs.dedup_pct:.0f}%;background:{bar_color}"></div></div>'
                f'{cs.dedup_pct:.1f}%</td>'
                f'<td class="num">{cs.hours_saved:,.1f}</td>'
                f'<td class="num">EUR {cs.cost_saved_eur:,.0f}</td></tr>'
            )
        return (
            '<div class="section"><h2>Category Breakdown</h2>'
            '<table><thead><tr>'
            '<th>Category</th><th>Before</th><th>After</th>'
            '<th>Dedup %</th><th>Hours Saved</th><th>Cost Saved</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _html_group_details(self, data: DeduplicationData) -> str:
        """Build HTML group details table."""
        if not self.config.show_group_details or not data.groups:
            return ""
        rows = ""
        for g in sorted(data.groups, key=lambda x: x.effort_saved_hours, reverse=True):
            regs = ", ".join(g.regulations)
            info = self.DEDUP_TYPE_LABELS.get(g.dedup_type, {"label": g.dedup_type, "color": "#95a5a6"})
            rows += (
                f'<tr><td>{g.group_id}</td>'
                f'<td>{g.canonical_field}</td>'
                f'<td>{g.category}</td>'
                f'<td>{regs}</td>'
                f'<td><span class="type-badge" style="background:{info["color"]}">'
                f'{info["label"]}</span></td>'
                f'<td class="num">{g.original_count}</td>'
                f'<td class="num">{g.deduplicated_to}</td>'
                f'<td class="num">{g.effort_saved_hours:,.1f}</td></tr>'
            )
        return (
            '<div class="section"><h2>Deduplication Groups</h2>'
            '<table><thead><tr>'
            '<th>Group</th><th>Field</th><th>Category</th>'
            '<th>Regulations</th><th>Type</th><th>Original</th>'
            '<th>Deduped</th><th>Hours Saved</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _html_cost_impact(self, data: DeduplicationData) -> str:
        """Build HTML cost impact section."""
        s = data.savings
        hourly = self.config.hourly_rate_eur
        return (
            '<div class="section"><h2>Cost Impact Analysis</h2>'
            f'<p>Assumed hourly rate: EUR {hourly:,.0f}</p>'
            '<table><thead><tr>'
            f'<th>Metric</th><th>Per Cycle</th><th>Annual ({s.reporting_cycles_per_year}x)</th>'
            '</tr></thead><tbody>'
            f'<tr><td>Hours Saved</td><td class="num">{s.total_hours_saved:,.1f}</td>'
            f'<td class="num">{s.annual_hours_saved:,.1f}</td></tr>'
            f'<tr><td>Cost Saved</td><td class="num">EUR {s.total_cost_saved_eur:,.0f}</td>'
            f'<td class="num">EUR {s.annual_cost_saved_eur:,.0f}</td></tr>'
            f'<tr><td>FTE Equivalent</td><td class="num">---</td>'
            f'<td class="num">{s.fte_equivalent:.2f}</td></tr>'
            '</tbody></table></div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON builders
    # ------------------------------------------------------------------ #

    def _json_headline(self, data: DeduplicationData) -> Dict[str, Any]:
        """Build JSON headline savings."""
        s = data.savings
        return {
            "fields_deduplicated": self._fields_deduplicated(data),
            "dedup_percentage": round(self._dedup_percentage(data), 1),
            "total_hours_saved": round(s.total_hours_saved, 1),
            "total_cost_saved_eur": round(s.total_cost_saved_eur, 2),
            "annual_hours_saved": round(s.annual_hours_saved, 1),
            "annual_cost_saved_eur": round(s.annual_cost_saved_eur, 2),
            "fte_equivalent": round(s.fte_equivalent, 2),
            "data_quality_improvement_pct": round(s.data_quality_improvement_pct, 1),
            "error_reduction_pct": round(s.error_reduction_pct, 1),
        }

    def _json_before_after(self, data: DeduplicationData) -> Dict[str, Any]:
        """Build JSON before/after comparison."""
        return {
            "before_count": data.before_count,
            "after_count": data.after_count,
            "fields_reduced": self._fields_deduplicated(data),
            "reduction_pct": round(self._dedup_percentage(data), 1),
        }

    def _json_categories(self, data: DeduplicationData) -> List[Dict[str, Any]]:
        """Build JSON category breakdown."""
        return [
            {
                "category": cs.category,
                "fields_before": cs.fields_before,
                "fields_after": cs.fields_after,
                "dedup_pct": round(cs.dedup_pct, 1),
                "hours_saved": round(cs.hours_saved, 1),
                "cost_saved_eur": round(cs.cost_saved_eur, 2),
            }
            for cs in data.category_savings
        ]

    def _json_groups(self, data: DeduplicationData) -> List[Dict[str, Any]]:
        """Build JSON dedup groups."""
        return [
            {
                "group_id": g.group_id,
                "canonical_field": g.canonical_field,
                "category": g.category,
                "regulations": g.regulations,
                "dedup_type": g.dedup_type,
                "original_count": g.original_count,
                "deduplicated_to": g.deduplicated_to,
                "effort_saved_hours": round(g.effort_saved_hours, 1),
                "cost_saved_eur": round(g.cost_saved_eur, 2),
            }
            for g in data.groups
        ]

    def _json_cost_impact(self, data: DeduplicationData) -> Dict[str, Any]:
        """Build JSON cost impact analysis."""
        s = data.savings
        return {
            "hourly_rate_eur": self.config.hourly_rate_eur,
            "per_cycle_hours": round(s.total_hours_saved, 1),
            "per_cycle_cost_eur": round(s.total_cost_saved_eur, 2),
            "annual_hours": round(s.annual_hours_saved, 1),
            "annual_cost_eur": round(s.annual_cost_saved_eur, 2),
            "fte_equivalent": round(s.fte_equivalent, 2),
            "reporting_cycles_per_year": s.reporting_cycles_per_year,
        }

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _generate_provenance_hash(self, content: str) -> str:
        """Generate SHA-256 provenance hash for audit trail."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _wrap_html(self, body: str, provenance_hash: str) -> str:
        """Wrap HTML body in a complete document with inline CSS."""
        css = (
            "body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
            "margin:0;padding:20px;background:#f5f7fa;color:#2c3e50}"
            ".report-header{background:#1a5276;color:#fff;padding:24px;border-radius:8px;"
            "margin-bottom:24px}"
            ".report-header h1{margin:0 0 12px 0;font-size:24px}"
            ".header-meta{display:flex;flex-wrap:wrap;gap:12px;font-size:14px}"
            ".meta-item{background:rgba(255,255,255,0.15);padding:4px 12px;border-radius:4px}"
            ".section{background:#fff;padding:20px;border-radius:8px;"
            "margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".section h2{margin:0 0 16px 0;font-size:18px;color:#1a5276;"
            "border-bottom:2px solid #eef2f7;padding-bottom:8px}"
            "table{width:100%;border-collapse:collapse;font-size:14px}"
            "th{background:#eef2f7;padding:10px 12px;text-align:left;font-weight:600}"
            "td{padding:8px 12px;border-bottom:1px solid #eef2f7}"
            ".num{text-align:right;font-variant-numeric:tabular-nums}"
            ".stat-grid{display:flex;flex-wrap:wrap;gap:12px}"
            ".stat-card{background:#f8f9fa;padding:16px;border-radius:8px;text-align:center;"
            "min-width:120px;flex:1}"
            ".stat-card.highlight{background:#e8f8f5;border:2px solid #2ecc71}"
            ".stat-val{display:block;font-size:24px;font-weight:700;color:#1a5276}"
            ".stat-lbl{display:block;font-size:11px;color:#7f8c8d;margin-top:4px}"
            ".type-badge{display:inline-block;padding:2px 8px;border-radius:4px;"
            "color:#fff;font-size:11px;font-weight:bold}"
            ".ba-comparison{max-width:600px}"
            ".ba-row{display:flex;align-items:center;gap:12px;margin-bottom:8px}"
            ".ba-label{width:60px;font-weight:600;font-size:14px}"
            ".ba-bar{flex:1;height:24px;background:#ecf0f1;border-radius:4px;overflow:hidden}"
            ".ba-fill{height:100%;border-radius:4px}"
            ".ba-val{width:100px;font-size:14px;text-align:right}"
            ".ba-summary{text-align:center;font-size:16px;margin-top:12px}"
            ".progress-bar{background:#ecf0f1;border-radius:4px;height:12px;"
            "overflow:hidden;display:inline-block;width:60%}"
            ".progress-fill{height:100%;border-radius:4px}"
            ".note{color:#7f8c8d;font-style:italic}"
            ".provenance{text-align:center;color:#95a5a6;font-size:12px;margin-top:24px}"
        )
        return (
            f'<!DOCTYPE html><html lang="en"><head>'
            f'<meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1.0">'
            f'<title>{self.config.title}</title>'
            f'<style>{css}</style>'
            f'</head><body>'
            f'{body}'
            f'<div class="provenance">'
            f'Report generated: {self.generated_at} | '
            f'Template: DeduplicationSavingsReportTemplate v1.0 | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )
