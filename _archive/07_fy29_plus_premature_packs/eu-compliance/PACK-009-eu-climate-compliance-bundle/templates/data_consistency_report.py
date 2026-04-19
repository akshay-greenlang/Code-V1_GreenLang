"""
DataConsistencyReportTemplate - Cross-regulation data consistency analysis.

This module implements the DataConsistencyReportTemplate for PACK-009
EU Climate Compliance Bundle. It renders consistency matrices showing
agreement and conflicts across shared fields, per-field comparison results
with values from each regulation, conflict severity and resolution status,
and overall consistency scoring.

Example:
    >>> template = DataConsistencyReportTemplate()
    >>> data = ConsistencyData(
    ...     checks=[...],
    ...     conflicts=[...],
    ...     resolutions=[...],
    ...     score=87.5,
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

class ConsistencyCheck(BaseModel):
    """A single consistency check comparing a field across regulations."""

    check_id: str = Field(..., description="Unique check identifier")
    field_name: str = Field(..., description="Canonical field name being checked")
    field_category: str = Field("", description="Field category e.g. GHG, Financial")
    regulations_compared: List[str] = Field(
        default_factory=list, description="Regulation codes compared"
    )
    values: Dict[str, Any] = Field(
        default_factory=dict, description="Map of regulation code to reported value"
    )
    units: Dict[str, str] = Field(
        default_factory=dict, description="Map of regulation code to value unit"
    )
    status: str = Field("consistent", description="consistent, inconsistent, partial")
    variance_pct: float = Field(0.0, description="Variance percentage across values")
    threshold_pct: float = Field(5.0, description="Acceptable variance threshold %")
    notes: str = Field("", description="Additional notes")


class ConsistencyConflict(BaseModel):
    """A detected conflict between regulation data values."""

    conflict_id: str = Field(..., description="Conflict identifier")
    check_id: str = Field(..., description="Related check ID")
    field_name: str = Field(..., description="Conflicting field name")
    regulation_a: str = Field(..., description="First regulation")
    regulation_b: str = Field(..., description="Second regulation")
    value_a: Any = Field(None, description="Value from regulation A")
    value_b: Any = Field(None, description="Value from regulation B")
    severity: str = Field("medium", description="high, medium, low")
    conflict_type: str = Field("value_mismatch", description="value_mismatch, unit_mismatch, missing")
    description: str = Field("", description="Conflict description")


class ConsistencyResolution(BaseModel):
    """Resolution record for a consistency conflict."""

    resolution_id: str = Field(..., description="Resolution identifier")
    conflict_id: str = Field(..., description="Related conflict ID")
    resolution_type: str = Field("manual", description="auto, manual, deferred")
    resolved_value: Any = Field(None, description="Resolved canonical value")
    resolution_notes: str = Field("", description="Explanation of resolution")
    resolved_by: str = Field("", description="Person or system that resolved")
    resolved_at: str = Field("", description="ISO timestamp of resolution")
    status: str = Field("resolved", description="resolved, pending, deferred")


class ConsistencyConfig(BaseModel):
    """Configuration for the consistency report template."""

    title: str = Field(
        "Data Consistency Report",
        description="Report title",
    )
    default_threshold_pct: float = Field(5.0, description="Default variance threshold %")
    show_consistent_fields: bool = Field(True, description="Whether to show consistent fields")
    regulations: List[str] = Field(
        default_factory=lambda: ["CSRD", "CBAM", "EU_TAXONOMY", "SFDR"],
        description="Regulation codes",
    )


class ConsistencyData(BaseModel):
    """Input data for the data consistency report."""

    checks: List[ConsistencyCheck] = Field(
        default_factory=list, description="List of consistency checks"
    )
    conflicts: List[ConsistencyConflict] = Field(
        default_factory=list, description="Detected conflicts"
    )
    resolutions: List[ConsistencyResolution] = Field(
        default_factory=list, description="Conflict resolutions"
    )
    score: float = Field(0.0, ge=0.0, le=100.0, description="Overall consistency score 0-100")
    total_fields_checked: int = Field(0, ge=0, description="Total fields checked")
    consistent_fields: int = Field(0, ge=0, description="Number of consistent fields")
    inconsistent_fields: int = Field(0, ge=0, description="Number of inconsistent fields")
    auto_resolved_count: int = Field(0, ge=0, description="Auto-resolved conflicts")
    manual_required_count: int = Field(0, ge=0, description="Conflicts requiring manual resolution")
    reporting_period: str = Field("", description="Reporting period label")
    organization_name: str = Field("", description="Organization name")

    @field_validator("checks")
    @classmethod
    def validate_checks_present(cls, v: List[ConsistencyCheck]) -> List[ConsistencyCheck]:
        """Ensure at least one check is provided."""
        if not v:
            raise ValueError("checks must contain at least one entry")
        return v


# ---------------------------------------------------------------------------
#  Template class
# ---------------------------------------------------------------------------

class DataConsistencyReportTemplate:
    """
    Data consistency report template for cross-regulation field comparison.

    Generates consistency matrices, per-field comparison results, conflict
    analysis with severity, resolution tracking, and overall consistency
    scoring across bundled regulations.

    Attributes:
        config: Template configuration.
        generated_at: ISO timestamp of report generation.
    """

    CONSISTENCY_COLORS = {
        "consistent": {"hex": "#2ecc71", "label": "CONSISTENT", "md": "[OK]"},
        "inconsistent": {"hex": "#e74c3c", "label": "INCONSISTENT", "md": "[XX]"},
        "partial": {"hex": "#f39c12", "label": "PARTIAL", "md": "[??]"},
    }

    SEVERITY_COLORS = {
        "high": "#e74c3c",
        "medium": "#f39c12",
        "low": "#27ae60",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize DataConsistencyReportTemplate.

        Args:
            config: Optional configuration dictionary.
        """
        raw = config or {}
        self.config = ConsistencyConfig(**raw) if raw else ConsistencyConfig()
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def render(self, data: ConsistencyData, fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """
        Render the consistency report in the specified format.

        Args:
            data: Validated ConsistencyData input.
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

    def render_markdown(self, data: ConsistencyData) -> str:
        """Render as Markdown string."""
        sections: List[str] = [
            self._md_header(data),
            self._md_score_summary(data),
            self._md_consistency_matrix(data),
            self._md_field_comparison(data),
            self._md_conflict_details(data),
            self._md_resolution_summary(data),
            self._md_footer(),
        ]
        content = "\n\n".join(s for s in sections if s)
        provenance = self._generate_provenance_hash(content)
        content += f"\n\n<!-- provenance_hash: {provenance} -->"
        return content

    def render_html(self, data: ConsistencyData) -> str:
        """Render as self-contained HTML document."""
        sections: List[str] = [
            self._html_header(data),
            self._html_score_summary(data),
            self._html_consistency_matrix(data),
            self._html_field_comparison(data),
            self._html_conflict_details(data),
            self._html_resolution_summary(data),
        ]
        body = "\n".join(s for s in sections if s)
        provenance = self._generate_provenance_hash(body)
        return self._wrap_html(body, provenance)

    def render_json(self, data: ConsistencyData) -> Dict[str, Any]:
        """Render as structured dictionary."""
        report: Dict[str, Any] = {
            "report_type": "data_consistency",
            "template_version": "1.0",
            "generated_at": self.generated_at,
            "organization": data.organization_name,
            "reporting_period": data.reporting_period,
            "score_summary": self._json_score_summary(data),
            "consistency_checks": self._json_checks(data),
            "conflicts": self._json_conflicts(data),
            "resolutions": self._json_resolutions(data),
        }
        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._generate_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Computation helpers
    # ------------------------------------------------------------------ #

    def _build_resolution_lookup(self, data: ConsistencyData) -> Dict[str, ConsistencyResolution]:
        """Build lookup of conflict_id to resolution."""
        return {r.conflict_id: r for r in data.resolutions}

    def _conflict_lookup_by_check(self, data: ConsistencyData) -> Dict[str, List[ConsistencyConflict]]:
        """Group conflicts by check_id."""
        lookup: Dict[str, List[ConsistencyConflict]] = {}
        for c in data.conflicts:
            lookup.setdefault(c.check_id, []).append(c)
        return lookup

    # ------------------------------------------------------------------ #
    #  Markdown builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: ConsistencyData) -> str:
        """Build markdown header."""
        org = data.organization_name or "Organization"
        period = data.reporting_period or "Current Period"
        return (
            f"# {self.config.title}\n\n"
            f"**Organization:** {org}\n\n"
            f"**Reporting Period:** {period}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_score_summary(self, data: ConsistencyData) -> str:
        """Build overall consistency score section."""
        score = data.score
        filled = int(score / 5)
        empty = 20 - filled
        gauge = "[" + "#" * filled + "-" * empty + "]"
        total = data.total_fields_checked or len(data.checks)
        return (
            "## Overall Consistency Score\n\n"
            f"```\n{gauge} {score:.1f}/100\n```\n\n"
            f"- **Total Fields Checked:** {total}\n"
            f"- **Consistent:** {data.consistent_fields}\n"
            f"- **Inconsistent:** {data.inconsistent_fields}\n"
            f"- **Conflicts Found:** {len(data.conflicts)}\n"
            f"- **Auto-Resolved:** {data.auto_resolved_count}\n"
            f"- **Manual Required:** {data.manual_required_count}"
        )

    def _md_consistency_matrix(self, data: ConsistencyData) -> str:
        """Build regulation pair consistency matrix."""
        regs = self.config.regulations
        pair_stats: Dict[str, Dict[str, int]] = {}
        for check in data.checks:
            for i, reg_a in enumerate(check.regulations_compared):
                for reg_b in check.regulations_compared[i + 1:]:
                    key = f"{reg_a}:{reg_b}"
                    if key not in pair_stats:
                        pair_stats[key] = {"total": 0, "consistent": 0}
                    pair_stats[key]["total"] += 1
                    if check.status == "consistent":
                        pair_stats[key]["consistent"] += 1
        header = "## Consistency Matrix\n\n"
        header += "| | " + " | ".join(regs) + " |\n"
        header += "|---" + "|---" * len(regs) + "|\n"
        rows: List[str] = []
        for reg_a in regs:
            cells: List[str] = [f"**{reg_a}**"]
            for reg_b in regs:
                if reg_a == reg_b:
                    cells.append("---")
                else:
                    key1 = f"{reg_a}:{reg_b}"
                    key2 = f"{reg_b}:{reg_a}"
                    stats = pair_stats.get(key1) or pair_stats.get(key2)
                    if stats and stats["total"] > 0:
                        pct = (stats["consistent"] / stats["total"]) * 100
                        cells.append(f"{pct:.0f}%")
                    else:
                        cells.append("N/A")
            rows.append("| " + " | ".join(cells) + " |")
        return header + "\n".join(rows)

    def _md_field_comparison(self, data: ConsistencyData) -> str:
        """Build per-field comparison table."""
        checks = data.checks
        if not self.config.show_consistent_fields:
            checks = [c for c in checks if c.status != "consistent"]
        if not checks:
            return "## Field Comparison\n\n*All fields are consistent.*"
        regs = self.config.regulations
        header = "## Field Comparison\n\n"
        header += "| Field | Category | " + " | ".join(regs) + " | Status | Variance |\n"
        header += "|-------|----------|" + "|-------" * len(regs) + "|--------|----------|\n"
        rows: List[str] = []
        for check in sorted(checks, key=lambda c: c.status != "inconsistent"):
            cells: List[str] = []
            for reg in regs:
                val = check.values.get(reg)
                unit = check.units.get(reg, "")
                if val is not None:
                    cells.append(f"{self._format_value(val)}{(' ' + unit) if unit else ''}")
                else:
                    cells.append("---")
            status_info = self.CONSISTENCY_COLORS.get(check.status, self.CONSISTENCY_COLORS["partial"])
            rows.append(
                f"| {check.field_name} | {check.field_category} | "
                + " | ".join(cells)
                + f" | {status_info['md']} {status_info['label']} | "
                f"{check.variance_pct:.1f}% |"
            )
        return header + "\n".join(rows)

    def _md_conflict_details(self, data: ConsistencyData) -> str:
        """Build conflict details section."""
        if not data.conflicts:
            return "## Conflicts\n\n*No conflicts detected.*"
        res_lookup = self._build_resolution_lookup(data)
        header = (
            "## Conflicts\n\n"
            "| ID | Field | Reg A | Value A | Reg B | Value B | Severity | Type | Resolved |\n"
            "|----|-------|-------|---------|-------|---------|----------|------|----------|\n"
        )
        rows: List[str] = []
        for c in sorted(data.conflicts, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.severity, 3)):
            res = res_lookup.get(c.conflict_id)
            resolved = "Yes" if (res and res.status == "resolved") else "No"
            rows.append(
                f"| {c.conflict_id} | {c.field_name} | "
                f"{c.regulation_a} | {self._format_value(c.value_a)} | "
                f"{c.regulation_b} | {self._format_value(c.value_b)} | "
                f"{c.severity.upper()} | {c.conflict_type.replace('_', ' ').title()} | "
                f"{resolved} |"
            )
        return header + "\n".join(rows)

    def _md_resolution_summary(self, data: ConsistencyData) -> str:
        """Build resolution summary section."""
        if not data.resolutions:
            return ""
        auto = sum(1 for r in data.resolutions if r.resolution_type == "auto")
        manual = sum(1 for r in data.resolutions if r.resolution_type == "manual")
        deferred = sum(1 for r in data.resolutions if r.resolution_type == "deferred")
        resolved = sum(1 for r in data.resolutions if r.status == "resolved")
        pending = sum(1 for r in data.resolutions if r.status == "pending")
        return (
            "## Resolution Summary\n\n"
            f"- **Total Resolutions:** {len(data.resolutions)}\n"
            f"- **Auto-Resolved:** {auto}\n"
            f"- **Manual:** {manual}\n"
            f"- **Deferred:** {deferred}\n"
            f"- **Status - Resolved:** {resolved} | **Pending:** {pending}"
        )

    def _md_footer(self) -> str:
        """Build markdown footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            "*Template: DataConsistencyReportTemplate v1.0 | PACK-009*"
        )

    # ------------------------------------------------------------------ #
    #  HTML builders
    # ------------------------------------------------------------------ #

    def _html_header(self, data: ConsistencyData) -> str:
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

    def _html_score_summary(self, data: ConsistencyData) -> str:
        """Build HTML score summary."""
        score = data.score
        color = "#2ecc71" if score >= 80 else "#f39c12" if score >= 50 else "#e74c3c"
        total = data.total_fields_checked or len(data.checks)
        cards = (
            f'<div class="score-gauge">'
            f'<div class="gauge-circle" style="border-color:{color}">'
            f'<span class="gauge-value">{score:.0f}</span>'
            f'<span class="gauge-label">out of 100</span></div></div>'
            f'<div class="stat-grid">'
            f'<div class="stat-card"><span class="stat-val">{total}</span>'
            f'<span class="stat-lbl">Total Checked</span></div>'
            f'<div class="stat-card"><span class="stat-val">{data.consistent_fields}</span>'
            f'<span class="stat-lbl">Consistent</span></div>'
            f'<div class="stat-card" style="border-top:3px solid #e74c3c">'
            f'<span class="stat-val">{data.inconsistent_fields}</span>'
            f'<span class="stat-lbl">Inconsistent</span></div>'
            f'<div class="stat-card"><span class="stat-val">{data.auto_resolved_count}</span>'
            f'<span class="stat-lbl">Auto-Resolved</span></div>'
            f'<div class="stat-card"><span class="stat-val">{data.manual_required_count}</span>'
            f'<span class="stat-lbl">Manual Required</span></div>'
            f'</div>'
        )
        return f'<div class="section"><h2>Overall Consistency Score</h2>{cards}</div>'

    def _html_consistency_matrix(self, data: ConsistencyData) -> str:
        """Build HTML consistency matrix."""
        regs = self.config.regulations
        pair_stats: Dict[str, Dict[str, int]] = {}
        for check in data.checks:
            for i, reg_a in enumerate(check.regulations_compared):
                for reg_b in check.regulations_compared[i + 1:]:
                    key = f"{reg_a}:{reg_b}"
                    if key not in pair_stats:
                        pair_stats[key] = {"total": 0, "consistent": 0}
                    pair_stats[key]["total"] += 1
                    if check.status == "consistent":
                        pair_stats[key]["consistent"] += 1
        header_cells = "".join(f"<th>{r}</th>" for r in regs)
        rows = ""
        for reg_a in regs:
            cells = f"<td><strong>{reg_a}</strong></td>"
            for reg_b in regs:
                if reg_a == reg_b:
                    cells += '<td class="matrix-diag">---</td>'
                else:
                    key1 = f"{reg_a}:{reg_b}"
                    key2 = f"{reg_b}:{reg_a}"
                    stats = pair_stats.get(key1) or pair_stats.get(key2)
                    if stats and stats["total"] > 0:
                        pct = (stats["consistent"] / stats["total"]) * 100
                        bg = "#2ecc71" if pct >= 80 else "#f39c12" if pct >= 50 else "#e74c3c"
                        cells += f'<td class="num" style="background:{bg};color:#fff">{pct:.0f}%</td>'
                    else:
                        cells += '<td class="num">N/A</td>'
            rows += f"<tr>{cells}</tr>"
        return (
            '<div class="section"><h2>Consistency Matrix</h2>'
            f'<table><thead><tr><th></th>{header_cells}</tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
        )

    def _html_field_comparison(self, data: ConsistencyData) -> str:
        """Build HTML field comparison table."""
        checks = data.checks
        if not self.config.show_consistent_fields:
            checks = [c for c in checks if c.status != "consistent"]
        if not checks:
            return (
                '<div class="section"><h2>Field Comparison</h2>'
                '<p class="note">All fields are consistent.</p></div>'
            )
        regs = self.config.regulations
        header_cells = "".join(f"<th>{r}</th>" for r in regs)
        rows = ""
        for check in sorted(checks, key=lambda c: c.status != "inconsistent"):
            cells = ""
            for reg in regs:
                val = check.values.get(reg)
                unit = check.units.get(reg, "")
                if val is not None:
                    display = f"{self._format_value(val)}{(' ' + unit) if unit else ''}"
                    cells += f'<td class="num">{display}</td>'
                else:
                    cells += '<td class="num" style="color:#bdc3c7">---</td>'
            info = self.CONSISTENCY_COLORS.get(check.status, self.CONSISTENCY_COLORS["partial"])
            rows += (
                f'<tr><td>{check.field_name}</td><td>{check.field_category}</td>'
                f'{cells}'
                f'<td><span class="status-badge" style="background:{info["hex"]}">'
                f'{info["label"]}</span></td>'
                f'<td class="num">{check.variance_pct:.1f}%</td></tr>'
            )
        return (
            '<div class="section"><h2>Field Comparison</h2>'
            f'<table><thead><tr><th>Field</th><th>Category</th>{header_cells}'
            f'<th>Status</th><th>Variance</th></tr></thead>'
            f'<tbody>{rows}</tbody></table></div>'
        )

    def _html_conflict_details(self, data: ConsistencyData) -> str:
        """Build HTML conflict details table."""
        if not data.conflicts:
            return (
                '<div class="section"><h2>Conflicts</h2>'
                '<p class="note">No conflicts detected.</p></div>'
            )
        res_lookup = self._build_resolution_lookup(data)
        rows = ""
        for c in sorted(data.conflicts, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.severity, 3)):
            color = self.SEVERITY_COLORS.get(c.severity, "#95a5a6")
            res = res_lookup.get(c.conflict_id)
            resolved = res and res.status == "resolved"
            res_icon = '<span style="color:#2ecc71">Yes</span>' if resolved else '<span style="color:#e74c3c">No</span>'
            rows += (
                f'<tr><td>{c.conflict_id}</td><td>{c.field_name}</td>'
                f'<td>{c.regulation_a}</td><td class="num">{self._format_value(c.value_a)}</td>'
                f'<td>{c.regulation_b}</td><td class="num">{self._format_value(c.value_b)}</td>'
                f'<td><span class="sev-badge" style="background:{color}">'
                f'{c.severity.upper()}</span></td>'
                f'<td>{c.conflict_type.replace("_", " ").title()}</td>'
                f'<td>{res_icon}</td></tr>'
            )
        return (
            '<div class="section"><h2>Conflicts</h2>'
            '<table><thead><tr>'
            '<th>ID</th><th>Field</th><th>Reg A</th><th>Value A</th>'
            '<th>Reg B</th><th>Value B</th><th>Severity</th><th>Type</th><th>Resolved</th>'
            f'</tr></thead><tbody>{rows}</tbody></table></div>'
        )

    def _html_resolution_summary(self, data: ConsistencyData) -> str:
        """Build HTML resolution summary."""
        if not data.resolutions:
            return ""
        auto = sum(1 for r in data.resolutions if r.resolution_type == "auto")
        manual = sum(1 for r in data.resolutions if r.resolution_type == "manual")
        deferred = sum(1 for r in data.resolutions if r.resolution_type == "deferred")
        resolved = sum(1 for r in data.resolutions if r.status == "resolved")
        pending = sum(1 for r in data.resolutions if r.status == "pending")
        total = len(data.resolutions) or 1
        auto_pct = (auto / total) * 100
        manual_pct = (manual / total) * 100
        cards = (
            f'<div class="stat-grid">'
            f'<div class="stat-card"><span class="stat-val">{len(data.resolutions)}</span>'
            f'<span class="stat-lbl">Total</span></div>'
            f'<div class="stat-card"><span class="stat-val">{auto}</span>'
            f'<span class="stat-lbl">Auto ({auto_pct:.0f}%)</span></div>'
            f'<div class="stat-card"><span class="stat-val">{manual}</span>'
            f'<span class="stat-lbl">Manual ({manual_pct:.0f}%)</span></div>'
            f'<div class="stat-card"><span class="stat-val">{deferred}</span>'
            f'<span class="stat-lbl">Deferred</span></div>'
            f'<div class="stat-card"><span class="stat-val">{resolved}</span>'
            f'<span class="stat-lbl">Resolved</span></div>'
            f'<div class="stat-card"><span class="stat-val">{pending}</span>'
            f'<span class="stat-lbl">Pending</span></div>'
            f'</div>'
        )
        return f'<div class="section"><h2>Resolution Summary</h2>{cards}</div>'

    # ------------------------------------------------------------------ #
    #  JSON builders
    # ------------------------------------------------------------------ #

    def _json_score_summary(self, data: ConsistencyData) -> Dict[str, Any]:
        """Build JSON score summary."""
        return {
            "score": round(data.score, 1),
            "total_fields_checked": data.total_fields_checked or len(data.checks),
            "consistent_fields": data.consistent_fields,
            "inconsistent_fields": data.inconsistent_fields,
            "conflicts_count": len(data.conflicts),
            "auto_resolved_count": data.auto_resolved_count,
            "manual_required_count": data.manual_required_count,
        }

    def _json_checks(self, data: ConsistencyData) -> List[Dict[str, Any]]:
        """Build JSON consistency checks."""
        return [
            {
                "check_id": c.check_id,
                "field_name": c.field_name,
                "field_category": c.field_category,
                "regulations_compared": c.regulations_compared,
                "values": c.values,
                "units": c.units,
                "status": c.status,
                "variance_pct": round(c.variance_pct, 2),
                "threshold_pct": c.threshold_pct,
            }
            for c in data.checks
        ]

    def _json_conflicts(self, data: ConsistencyData) -> List[Dict[str, Any]]:
        """Build JSON conflicts."""
        res_lookup = self._build_resolution_lookup(data)
        return [
            {
                "conflict_id": c.conflict_id,
                "check_id": c.check_id,
                "field_name": c.field_name,
                "regulation_a": c.regulation_a,
                "regulation_b": c.regulation_b,
                "value_a": c.value_a,
                "value_b": c.value_b,
                "severity": c.severity,
                "conflict_type": c.conflict_type,
                "resolved": (res_lookup.get(c.conflict_id, ConsistencyResolution(
                    resolution_id="", conflict_id="")).status == "resolved"),
            }
            for c in data.conflicts
        ]

    def _json_resolutions(self, data: ConsistencyData) -> List[Dict[str, Any]]:
        """Build JSON resolutions."""
        return [
            {
                "resolution_id": r.resolution_id,
                "conflict_id": r.conflict_id,
                "resolution_type": r.resolution_type,
                "resolved_value": r.resolved_value,
                "status": r.status,
                "resolved_by": r.resolved_by,
                "resolved_at": r.resolved_at,
            }
            for r in data.resolutions
        ]

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:,.2f}"
        if isinstance(value, int):
            return f"{value:,}"
        return str(value)

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
            ".stat-grid{display:flex;flex-wrap:wrap;gap:12px;margin-top:16px}"
            ".stat-card{background:#f8f9fa;padding:16px;border-radius:8px;text-align:center;"
            "min-width:100px;flex:1}"
            ".stat-val{display:block;font-size:24px;font-weight:700;color:#1a5276}"
            ".stat-lbl{display:block;font-size:11px;color:#7f8c8d;margin-top:4px}"
            ".score-gauge{text-align:center;margin:24px 0}"
            ".gauge-circle{display:inline-flex;flex-direction:column;align-items:center;"
            "justify-content:center;width:120px;height:120px;border-radius:50%;"
            "border:6px solid}"
            ".gauge-value{font-size:36px;font-weight:700;line-height:1}"
            ".gauge-label{font-size:12px;color:#7f8c8d}"
            ".status-badge,.sev-badge{display:inline-block;padding:2px 8px;border-radius:4px;"
            "color:#fff;font-size:11px;font-weight:bold}"
            ".matrix-diag{background:#ecf0f1;text-align:center;color:#95a5a6}"
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
            f'Template: DataConsistencyReportTemplate v1.0 | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )
