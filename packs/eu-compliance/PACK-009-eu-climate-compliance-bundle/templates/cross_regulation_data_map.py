"""
CrossRegulationDataMapTemplate - Visual mapping of shared data fields across regulations.

This module implements the CrossRegulationDataMapTemplate for PACK-009
EU Climate Compliance Bundle. It renders a matrix showing which data fields
appear in which regulations, coverage percentages per regulation pair,
field categories (GHG, Supply Chain, Financial, Climate Risk), and
mapping confidence levels.

Example:
    >>> template = CrossRegulationDataMapTemplate()
    >>> data = DataMapData(
    ...     field_mappings=[...],
    ...     coverage_stats=[...],
    ...     categories=["GHG", "Supply Chain"],
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

class FieldMapping(BaseModel):
    """A single data field and its presence across regulations."""

    field_name: str = Field(..., description="Canonical field name")
    field_id: str = Field("", description="Unique field identifier")
    category: str = Field(..., description="Field category: GHG, Supply Chain, Financial, Climate Risk, etc.")
    data_type: str = Field("string", description="Data type: string, numeric, date, boolean")
    description: str = Field("", description="Human-readable field description")
    regulations: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of regulation code to mapping type: exact, approximate, derived, absent",
    )
    confidence: str = Field("exact", description="Mapping confidence: exact, approximate, derived")
    source_field_names: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of regulation code to the field name used in that regulation",
    )
    notes: str = Field("", description="Additional mapping notes")


class CoverageStat(BaseModel):
    """Pairwise coverage statistics between two regulations."""

    regulation_a: str = Field(..., description="First regulation code")
    regulation_b: str = Field(..., description="Second regulation code")
    shared_fields: int = Field(0, ge=0, description="Number of shared fields")
    total_fields_a: int = Field(0, ge=0, description="Total fields in regulation A")
    total_fields_b: int = Field(0, ge=0, description="Total fields in regulation B")
    coverage_pct_a: float = Field(0.0, ge=0.0, le=100.0, description="% of A fields shared with B")
    coverage_pct_b: float = Field(0.0, ge=0.0, le=100.0, description="% of B fields shared with A")
    exact_matches: int = Field(0, ge=0, description="Fields with exact mapping")
    approximate_matches: int = Field(0, ge=0, description="Fields with approximate mapping")
    derived_matches: int = Field(0, ge=0, description="Fields with derived mapping")


class DataMapConfig(BaseModel):
    """Configuration for the cross-regulation data map template."""

    title: str = Field(
        "Cross-Regulation Data Field Map",
        description="Report title",
    )
    show_absent_fields: bool = Field(True, description="Show fields absent from some regulations")
    highlight_exact_only: bool = Field(False, description="Only highlight exact matches")
    regulations: List[str] = Field(
        default_factory=lambda: ["CSRD", "CBAM", "EU_TAXONOMY", "SFDR"],
        description="Regulation codes to include in the map",
    )


class DataMapData(BaseModel):
    """Input data for the cross-regulation data map."""

    field_mappings: List[FieldMapping] = Field(
        default_factory=list, description="List of field mappings across regulations"
    )
    coverage_stats: List[CoverageStat] = Field(
        default_factory=list, description="Pairwise coverage statistics"
    )
    categories: List[str] = Field(
        default_factory=lambda: ["GHG", "Supply Chain", "Financial", "Climate Risk", "Governance", "Social"],
        description="Field categories to display",
    )
    total_unique_fields: int = Field(0, ge=0, description="Total unique fields across all regulations")
    reporting_period: str = Field("", description="Reporting period label")
    organization_name: str = Field("", description="Reporting organization")

    @field_validator("field_mappings")
    @classmethod
    def validate_mappings_present(cls, v: List[FieldMapping]) -> List[FieldMapping]:
        """Ensure at least one field mapping is provided."""
        if not v:
            raise ValueError("field_mappings must contain at least one entry")
        return v


# ---------------------------------------------------------------------------
#  Template class
# ---------------------------------------------------------------------------

class CrossRegulationDataMapTemplate:
    """
    Cross-regulation data field map template.

    Generates visual mappings showing which data fields appear across which
    EU climate regulations, with coverage matrices, confidence levels,
    and category breakdowns.

    Attributes:
        config: Template configuration.
        generated_at: ISO timestamp of report generation.
    """

    CONFIDENCE_SYMBOLS = {
        "exact": {"md": "[=]", "label": "Exact", "color": "#2ecc71"},
        "approximate": {"md": "[~]", "label": "Approx", "color": "#f39c12"},
        "derived": {"md": "[d]", "label": "Derived", "color": "#3498db"},
        "absent": {"md": "[ ]", "label": "Absent", "color": "#e74c3c"},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize CrossRegulationDataMapTemplate.

        Args:
            config: Optional configuration dictionary.
        """
        raw = config or {}
        self.config = DataMapConfig(**raw) if raw else DataMapConfig()
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def render(self, data: DataMapData, fmt: str = "markdown") -> Union[str, Dict[str, Any]]:
        """
        Render the data map in the specified format.

        Args:
            data: Validated DataMapData input.
            fmt: Output format - 'markdown', 'html', or 'json'.

        Returns:
            Rendered output as string or dict.

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

    def render_markdown(self, data: DataMapData) -> str:
        """Render as Markdown string."""
        sections: List[str] = [
            self._md_header(data),
            self._md_summary_stats(data),
            self._md_coverage_matrix(data),
            self._md_field_mapping_table(data),
            self._md_category_breakdown(data),
            self._md_confidence_legend(),
            self._md_footer(),
        ]
        content = "\n\n".join(s for s in sections if s)
        provenance = self._generate_provenance_hash(content)
        content += f"\n\n<!-- provenance_hash: {provenance} -->"
        return content

    def render_html(self, data: DataMapData) -> str:
        """Render as self-contained HTML document."""
        sections: List[str] = [
            self._html_header(data),
            self._html_summary_stats(data),
            self._html_coverage_matrix(data),
            self._html_field_mapping_table(data),
            self._html_category_breakdown(data),
            self._html_confidence_legend(),
        ]
        body = "\n".join(s for s in sections if s)
        provenance = self._generate_provenance_hash(body)
        return self._wrap_html(body, provenance)

    def render_json(self, data: DataMapData) -> Dict[str, Any]:
        """Render as structured dictionary."""
        report: Dict[str, Any] = {
            "report_type": "cross_regulation_data_map",
            "template_version": "1.0",
            "generated_at": self.generated_at,
            "organization": data.organization_name,
            "reporting_period": data.reporting_period,
            "summary": self._json_summary(data),
            "coverage_matrix": self._json_coverage_matrix(data),
            "field_mappings": self._json_field_mappings(data),
            "category_breakdown": self._json_category_breakdown(data),
        }
        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._generate_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Computation helpers
    # ------------------------------------------------------------------ #

    def _compute_category_stats(self, data: DataMapData) -> Dict[str, Dict[str, int]]:
        """Compute per-category field counts and coverage stats."""
        stats: Dict[str, Dict[str, int]] = {}
        for cat in data.categories:
            cat_fields = [f for f in data.field_mappings if f.category == cat]
            exact = sum(1 for f in cat_fields if f.confidence == "exact")
            approx = sum(1 for f in cat_fields if f.confidence == "approximate")
            derived = sum(1 for f in cat_fields if f.confidence == "derived")
            stats[cat] = {
                "total": len(cat_fields),
                "exact": exact,
                "approximate": approx,
                "derived": derived,
            }
        return stats

    def _build_coverage_lookup(self, data: DataMapData) -> Dict[str, CoverageStat]:
        """Build a lookup dict for coverage stats by regulation pair key."""
        lookup: Dict[str, CoverageStat] = {}
        for cs in data.coverage_stats:
            key_ab = f"{cs.regulation_a}:{cs.regulation_b}"
            key_ba = f"{cs.regulation_b}:{cs.regulation_a}"
            lookup[key_ab] = cs
            lookup[key_ba] = cs
        return lookup

    # ------------------------------------------------------------------ #
    #  Markdown builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: DataMapData) -> str:
        """Build markdown header."""
        org = data.organization_name or "Organization"
        period = data.reporting_period or "Current Period"
        return (
            f"# {self.config.title}\n\n"
            f"**Organization:** {org}\n\n"
            f"**Reporting Period:** {period}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_summary_stats(self, data: DataMapData) -> str:
        """Build summary statistics section."""
        total = data.total_unique_fields or len(data.field_mappings)
        regs = self.config.regulations
        exact = sum(1 for f in data.field_mappings if f.confidence == "exact")
        approx = sum(1 for f in data.field_mappings if f.confidence == "approximate")
        derived = sum(1 for f in data.field_mappings if f.confidence == "derived")
        multi_reg = sum(
            1 for f in data.field_mappings
            if sum(1 for v in f.regulations.values() if v != "absent") >= 2
        )
        return (
            "## Summary\n\n"
            f"- **Total Unique Fields:** {total}\n"
            f"- **Regulations Mapped:** {len(regs)} ({', '.join(regs)})\n"
            f"- **Fields in Multiple Regulations:** {multi_reg}\n"
            f"- **Exact Matches:** {exact}\n"
            f"- **Approximate Matches:** {approx}\n"
            f"- **Derived Matches:** {derived}\n"
            f"- **Categories:** {len(data.categories)}"
        )

    def _md_coverage_matrix(self, data: DataMapData) -> str:
        """Build pairwise coverage matrix table."""
        if not data.coverage_stats:
            return "## Coverage Matrix\n\n*No coverage statistics available.*"
        regs = self.config.regulations
        lookup = self._build_coverage_lookup(data)
        header_row = "| | " + " | ".join(regs) + " |\n"
        sep_row = "|---" + "|---" * len(regs) + "|\n"
        rows: List[str] = []
        for reg_a in regs:
            cells: List[str] = [f"**{reg_a}**"]
            for reg_b in regs:
                if reg_a == reg_b:
                    cells.append("---")
                else:
                    key = f"{reg_a}:{reg_b}"
                    cs = lookup.get(key)
                    if cs:
                        pct = cs.coverage_pct_a if cs.regulation_a == reg_a else cs.coverage_pct_b
                        cells.append(f"{pct:.0f}%")
                    else:
                        cells.append("N/A")
            rows.append("| " + " | ".join(cells) + " |")
        return (
            "## Coverage Matrix\n\n"
            "Percentage of fields in row regulation also present in column regulation.\n\n"
            + header_row + sep_row + "\n".join(rows)
        )

    def _md_field_mapping_table(self, data: DataMapData) -> str:
        """Build the main field mapping matrix table."""
        regs = self.config.regulations
        header = "## Field Mapping Matrix\n\n"
        header += "| Field | Category | " + " | ".join(regs) + " | Confidence |\n"
        header += "|-------|----------|" + "|-------" * len(regs) + "|------------|\n"
        rows: List[str] = []
        sorted_fields = sorted(data.field_mappings, key=lambda f: (f.category, f.field_name))
        for fm in sorted_fields:
            if not self.config.show_absent_fields:
                present_count = sum(1 for v in fm.regulations.values() if v != "absent")
                if present_count == 0:
                    continue
            cells: List[str] = []
            for reg in regs:
                mapping_type = fm.regulations.get(reg, "absent")
                symbol = self.CONFIDENCE_SYMBOLS.get(mapping_type, self.CONFIDENCE_SYMBOLS["absent"])
                cells.append(symbol["md"])
            conf_sym = self.CONFIDENCE_SYMBOLS.get(fm.confidence, self.CONFIDENCE_SYMBOLS["exact"])
            rows.append(
                f"| {fm.field_name} | {fm.category} | "
                + " | ".join(cells)
                + f" | {conf_sym['md']} {conf_sym['label']} |"
            )
        return header + "\n".join(rows)

    def _md_category_breakdown(self, data: DataMapData) -> str:
        """Build per-category breakdown section."""
        stats = self._compute_category_stats(data)
        header = (
            "## Category Breakdown\n\n"
            "| Category | Total Fields | Exact | Approximate | Derived |\n"
            "|----------|-------------|-------|-------------|----------|\n"
        )
        rows: List[str] = []
        for cat in data.categories:
            s = stats.get(cat, {"total": 0, "exact": 0, "approximate": 0, "derived": 0})
            rows.append(
                f"| {cat} | {s['total']} | {s['exact']} | {s['approximate']} | {s['derived']} |"
            )
        total_all = sum(s.get("total", 0) for s in stats.values())
        exact_all = sum(s.get("exact", 0) for s in stats.values())
        approx_all = sum(s.get("approximate", 0) for s in stats.values())
        derived_all = sum(s.get("derived", 0) for s in stats.values())
        rows.append(
            f"| **Total** | **{total_all}** | **{exact_all}** | "
            f"**{approx_all}** | **{derived_all}** |"
        )
        return header + "\n".join(rows)

    def _md_confidence_legend(self) -> str:
        """Build confidence level legend."""
        lines: List[str] = ["## Legend\n"]
        for key, info in self.CONFIDENCE_SYMBOLS.items():
            lines.append(f"- {info['md']} **{info['label']}**: {self._confidence_description(key)}")
        return "\n".join(lines)

    def _md_footer(self) -> str:
        """Build markdown footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            "*Template: CrossRegulationDataMapTemplate v1.0 | PACK-009*"
        )

    # ------------------------------------------------------------------ #
    #  HTML builders
    # ------------------------------------------------------------------ #

    def _html_header(self, data: DataMapData) -> str:
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

    def _html_summary_stats(self, data: DataMapData) -> str:
        """Build HTML summary stats cards."""
        total = data.total_unique_fields or len(data.field_mappings)
        exact = sum(1 for f in data.field_mappings if f.confidence == "exact")
        approx = sum(1 for f in data.field_mappings if f.confidence == "approximate")
        derived = sum(1 for f in data.field_mappings if f.confidence == "derived")
        multi_reg = sum(
            1 for f in data.field_mappings
            if sum(1 for v in f.regulations.values() if v != "absent") >= 2
        )
        cards = (
            f'<div class="stat-card"><span class="stat-val">{total}</span>'
            f'<span class="stat-lbl">Total Fields</span></div>'
            f'<div class="stat-card"><span class="stat-val">{multi_reg}</span>'
            f'<span class="stat-lbl">Multi-Reg Fields</span></div>'
            f'<div class="stat-card"><span class="stat-val">{exact}</span>'
            f'<span class="stat-lbl">Exact</span></div>'
            f'<div class="stat-card"><span class="stat-val">{approx}</span>'
            f'<span class="stat-lbl">Approximate</span></div>'
            f'<div class="stat-card"><span class="stat-val">{derived}</span>'
            f'<span class="stat-lbl">Derived</span></div>'
        )
        return f'<div class="section"><h2>Summary</h2><div class="stat-grid">{cards}</div></div>'

    def _html_coverage_matrix(self, data: DataMapData) -> str:
        """Build HTML coverage matrix."""
        if not data.coverage_stats:
            return (
                '<div class="section"><h2>Coverage Matrix</h2>'
                '<p class="note">No coverage statistics available.</p></div>'
            )
        regs = self.config.regulations
        lookup = self._build_coverage_lookup(data)
        header_cells = "".join(f"<th>{r}</th>" for r in regs)
        rows_html = ""
        for reg_a in regs:
            cells = f"<td><strong>{reg_a}</strong></td>"
            for reg_b in regs:
                if reg_a == reg_b:
                    cells += '<td class="matrix-diag">---</td>'
                else:
                    key = f"{reg_a}:{reg_b}"
                    cs = lookup.get(key)
                    if cs:
                        pct = cs.coverage_pct_a if cs.regulation_a == reg_a else cs.coverage_pct_b
                        bg = self._pct_color(pct)
                        cells += f'<td class="num" style="background:{bg};color:#fff">{pct:.0f}%</td>'
                    else:
                        cells += '<td class="num">N/A</td>'
            rows_html += f"<tr>{cells}</tr>"
        return (
            '<div class="section"><h2>Coverage Matrix</h2>'
            '<p>Percentage of fields in row regulation also present in column regulation.</p>'
            f'<table><thead><tr><th></th>{header_cells}</tr></thead>'
            f'<tbody>{rows_html}</tbody></table></div>'
        )

    def _html_field_mapping_table(self, data: DataMapData) -> str:
        """Build HTML field mapping matrix table."""
        regs = self.config.regulations
        header_cells = "".join(f"<th>{r}</th>" for r in regs)
        rows_html = ""
        sorted_fields = sorted(data.field_mappings, key=lambda f: (f.category, f.field_name))
        for fm in sorted_fields:
            if not self.config.show_absent_fields:
                present_count = sum(1 for v in fm.regulations.values() if v != "absent")
                if present_count == 0:
                    continue
            cells = ""
            for reg in regs:
                mapping_type = fm.regulations.get(reg, "absent")
                info = self.CONFIDENCE_SYMBOLS.get(mapping_type, self.CONFIDENCE_SYMBOLS["absent"])
                cells += (
                    f'<td style="text-align:center">'
                    f'<span class="map-badge" style="background:{info["color"]}">'
                    f'{info["label"]}</span></td>'
                )
            conf_info = self.CONFIDENCE_SYMBOLS.get(fm.confidence, self.CONFIDENCE_SYMBOLS["exact"])
            rows_html += (
                f'<tr><td>{fm.field_name}</td><td>{fm.category}</td>'
                f'{cells}'
                f'<td><span class="conf-badge" style="background:{conf_info["color"]}">'
                f'{conf_info["label"]}</span></td></tr>'
            )
        return (
            '<div class="section"><h2>Field Mapping Matrix</h2>'
            f'<table><thead><tr><th>Field</th><th>Category</th>{header_cells}'
            f'<th>Confidence</th></tr></thead>'
            f'<tbody>{rows_html}</tbody></table></div>'
        )

    def _html_category_breakdown(self, data: DataMapData) -> str:
        """Build HTML category breakdown."""
        stats = self._compute_category_stats(data)
        rows_html = ""
        for cat in data.categories:
            s = stats.get(cat, {"total": 0, "exact": 0, "approximate": 0, "derived": 0})
            total = s["total"] or 1
            exact_pct = (s["exact"] / total) * 100
            rows_html += (
                f'<tr><td>{cat}</td>'
                f'<td class="num">{s["total"]}</td>'
                f'<td class="num">{s["exact"]}</td>'
                f'<td class="num">{s["approximate"]}</td>'
                f'<td class="num">{s["derived"]}</td>'
                f'<td><div class="progress-bar"><div class="progress-fill" '
                f'style="width:{exact_pct:.0f}%;background:#2ecc71"></div></div></td>'
                f'</tr>'
            )
        return (
            '<div class="section"><h2>Category Breakdown</h2>'
            '<table><thead><tr>'
            '<th>Category</th><th>Total</th><th>Exact</th>'
            '<th>Approx</th><th>Derived</th><th>Exact %</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_confidence_legend(self) -> str:
        """Build HTML legend section."""
        items = ""
        for key, info in self.CONFIDENCE_SYMBOLS.items():
            items += (
                f'<div class="legend-item">'
                f'<span class="map-badge" style="background:{info["color"]}">'
                f'{info["label"]}</span> '
                f'{self._confidence_description(key)}</div>'
            )
        return f'<div class="section"><h2>Legend</h2>{items}</div>'

    # ------------------------------------------------------------------ #
    #  JSON builders
    # ------------------------------------------------------------------ #

    def _json_summary(self, data: DataMapData) -> Dict[str, Any]:
        """Build JSON summary section."""
        total = data.total_unique_fields or len(data.field_mappings)
        exact = sum(1 for f in data.field_mappings if f.confidence == "exact")
        approx = sum(1 for f in data.field_mappings if f.confidence == "approximate")
        derived = sum(1 for f in data.field_mappings if f.confidence == "derived")
        multi_reg = sum(
            1 for f in data.field_mappings
            if sum(1 for v in f.regulations.values() if v != "absent") >= 2
        )
        return {
            "total_unique_fields": total,
            "regulations_mapped": self.config.regulations,
            "multi_regulation_fields": multi_reg,
            "exact_matches": exact,
            "approximate_matches": approx,
            "derived_matches": derived,
            "categories": data.categories,
        }

    def _json_coverage_matrix(self, data: DataMapData) -> List[Dict[str, Any]]:
        """Build JSON coverage matrix."""
        return [
            {
                "regulation_a": cs.regulation_a,
                "regulation_b": cs.regulation_b,
                "shared_fields": cs.shared_fields,
                "total_fields_a": cs.total_fields_a,
                "total_fields_b": cs.total_fields_b,
                "coverage_pct_a": round(cs.coverage_pct_a, 1),
                "coverage_pct_b": round(cs.coverage_pct_b, 1),
                "exact_matches": cs.exact_matches,
                "approximate_matches": cs.approximate_matches,
                "derived_matches": cs.derived_matches,
            }
            for cs in data.coverage_stats
        ]

    def _json_field_mappings(self, data: DataMapData) -> List[Dict[str, Any]]:
        """Build JSON field mappings."""
        return [
            {
                "field_name": fm.field_name,
                "field_id": fm.field_id,
                "category": fm.category,
                "data_type": fm.data_type,
                "description": fm.description,
                "regulations": fm.regulations,
                "confidence": fm.confidence,
                "source_field_names": fm.source_field_names,
                "notes": fm.notes,
            }
            for fm in sorted(data.field_mappings, key=lambda f: (f.category, f.field_name))
        ]

    def _json_category_breakdown(self, data: DataMapData) -> Dict[str, Dict[str, int]]:
        """Build JSON category breakdown."""
        return self._compute_category_stats(data)

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _confidence_description(self, key: str) -> str:
        """Return a human-readable description for a confidence level."""
        descriptions = {
            "exact": "Field maps directly with identical semantics and units.",
            "approximate": "Field maps with similar semantics; minor transformation needed.",
            "derived": "Field value must be calculated or derived from other fields.",
            "absent": "Field is not required or present in this regulation.",
        }
        return descriptions.get(key, "Unknown confidence level.")

    def _pct_color(self, pct: float) -> str:
        """Return a hex color based on coverage percentage."""
        if pct >= 75:
            return "#2ecc71"
        elif pct >= 50:
            return "#f39c12"
        elif pct >= 25:
            return "#e67e22"
        return "#e74c3c"

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
            ".stat-val{display:block;font-size:28px;font-weight:700;color:#1a5276}"
            ".stat-lbl{display:block;font-size:12px;color:#7f8c8d;margin-top:4px}"
            ".map-badge,.conf-badge{display:inline-block;padding:2px 8px;border-radius:4px;"
            "color:#fff;font-size:11px;font-weight:bold}"
            ".matrix-diag{background:#ecf0f1;text-align:center;color:#95a5a6}"
            ".progress-bar{background:#ecf0f1;border-radius:4px;height:12px;overflow:hidden;"
            "width:100%}"
            ".progress-fill{height:100%;border-radius:4px}"
            ".legend-item{margin-bottom:8px;font-size:14px}"
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
            f'Template: CrossRegulationDataMapTemplate v1.0 | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )
