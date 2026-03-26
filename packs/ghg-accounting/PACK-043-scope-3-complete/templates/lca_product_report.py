# -*- coding: utf-8 -*-
"""
LCAProductReportTemplate - Product Carbon Footprint with BOM for PACK-043.

Generates a lifecycle assessment product report with bill-of-materials
breakdown, lifecycle stage waterfall (raw material, manufacturing,
distribution, use, end-of-life), product comparison table, sensitivity
analysis on key parameters, and circular economy metrics.

Sections:
    1. Product Summary
    2. Lifecycle Stage Waterfall
    3. Bill of Materials (BOM) Breakdown
    4. Product Comparison Table
    5. Sensitivity Analysis
    6. Circular Economy Metrics
    7. Improvement Opportunities

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, lifecycle teal #008B8B theme)
    - JSON (structured with chart-ready data)

Author: GreenLang Team
Version: 43.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "43.0.0"

# ---------------------------------------------------------------------------
# Lifecycle stage order
# ---------------------------------------------------------------------------

_LIFECYCLE_STAGES = [
    "raw_material",
    "manufacturing",
    "distribution",
    "use",
    "end_of_life",
]

_STAGE_LABELS = {
    "raw_material": "Raw Material Extraction",
    "manufacturing": "Manufacturing",
    "distribution": "Distribution & Logistics",
    "use": "Use Phase",
    "end_of_life": "End-of-Life Treatment",
}

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_num(value: Optional[float], decimals: int = 1) -> str:
    """Format numeric value with thousands separators."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.{decimals}f}K"
    return f"{value:,.{decimals}f}"


def _fmt_kgco2e(value: Optional[float]) -> str:
    """Format kgCO2e with appropriate scale."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.2f} tCO2e"
    return f"{value:,.2f} kgCO2e"


def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage."""
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def _pct_of(part: float, total: float) -> str:
    """Percentage of total, formatted."""
    if total == 0:
        return "0.0%"
    return f"{(part / total) * 100:.1f}%"


class LCAProductReportTemplate:
    """
    Product carbon footprint template with lifecycle analysis.

    Renders product-level lifecycle assessment reports with BOM
    breakdown, lifecycle stage waterfall, product comparisons,
    and sensitivity analysis. All outputs include SHA-256 provenance.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = LCAProductReportTemplate()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize LCAProductReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    def _total_pcf(self, data: Dict[str, Any]) -> float:
        """Calculate total product carbon footprint across lifecycle stages."""
        stages = data.get("lifecycle_stages", {})
        return sum(
            stages.get(stage, {}).get("emissions_kgco2e", 0.0)
            for stage in _LIFECYCLE_STAGES
        )

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render LCA product report as Markdown.

        Args:
            data: Validated product report data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_product_summary(data),
            self._md_lifecycle_waterfall(data),
            self._md_bom_breakdown(data),
            self._md_product_comparison(data),
            self._md_sensitivity_analysis(data),
            self._md_circular_economy(data),
            self._md_improvements(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render LCA product report as HTML.

        Args:
            data: Validated product report data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_product_summary(data),
            self._html_lifecycle_waterfall(data),
            self._html_bom_breakdown(data),
            self._html_product_comparison(data),
            self._html_sensitivity_analysis(data),
            self._html_circular_economy(data),
            self._html_improvements(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render LCA product report as JSON-serializable dict.

        Args:
            data: Validated product report data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        total = self._total_pcf(data)
        return {
            "template": "lca_product_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "product_name": self._get_val(data, "product_name", ""),
            "functional_unit": self._get_val(data, "functional_unit", ""),
            "total_pcf_kgco2e": total,
            "lifecycle_waterfall": self._json_waterfall(data, total),
            "bom_breakdown": data.get("bom_breakdown", []),
            "product_comparison": data.get("product_comparison", []),
            "sensitivity_analysis": data.get("sensitivity_analysis", []),
            "circular_economy_metrics": data.get("circular_economy", {}),
            "improvement_opportunities": data.get("improvements", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        product = self._get_val(data, "product_name", "Product")
        return (
            f"# Product Carbon Footprint - {product}\n\n"
            f"**Company:** {company} | "
            f"**Functional Unit:** {self._get_val(data, 'functional_unit', '-')} | "
            f"**Standard:** ISO 14067\n\n"
            "---"
        )

    def _md_product_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown product summary."""
        total = self._total_pcf(data)
        product = self._get_val(data, "product_name", "Product")
        fu = self._get_val(data, "functional_unit", "-")
        weight = data.get("product_weight_kg")
        lines = [
            "## 1. Product Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Product | {product} |",
            f"| Functional Unit | {fu} |",
            f"| **Total PCF** | **{_fmt_kgco2e(total)}** |",
        ]
        if weight is not None:
            intensity = total / weight if weight > 0 else 0
            lines.append(f"| Product Weight | {weight:.2f} kg |")
            lines.append(f"| Carbon Intensity | {intensity:.2f} kgCO2e/kg |")
        return "\n".join(lines)

    def _md_lifecycle_waterfall(self, data: Dict[str, Any]) -> str:
        """Render Markdown lifecycle stage waterfall."""
        stages = data.get("lifecycle_stages", {})
        if not stages:
            return "## 2. Lifecycle Stage Waterfall\n\nNo lifecycle data available."
        total = self._total_pcf(data)
        lines = [
            "## 2. Lifecycle Stage Waterfall",
            "",
            "| Stage | kgCO2e | % of Total |",
            "|-------|--------|-----------|",
        ]
        cumulative = 0.0
        for stage_key in _LIFECYCLE_STAGES:
            stage_data = stages.get(stage_key, {})
            em = stage_data.get("emissions_kgco2e", 0.0)
            cumulative += em
            label = _STAGE_LABELS.get(stage_key, stage_key)
            pct = _pct_of(em, total)
            lines.append(f"| {label} | {_fmt_kgco2e(em)} | {pct} |")
        lines.append(f"| **Total** | **{_fmt_kgco2e(total)}** | **100.0%** |")
        return "\n".join(lines)

    def _md_bom_breakdown(self, data: Dict[str, Any]) -> str:
        """Render Markdown BOM breakdown."""
        bom = data.get("bom_breakdown", [])
        if not bom:
            return "## 3. Bill of Materials Breakdown\n\nNo BOM data available."
        total = self._total_pcf(data)
        lines = [
            "## 3. Bill of Materials Breakdown",
            "",
            "| Material | Weight (kg) | EF (kgCO2e/kg) | kgCO2e | % of PCF |",
            "|----------|------------|----------------|--------|---------|",
        ]
        for item in bom:
            material = item.get("material_name", "-")
            weight = item.get("weight_kg")
            ef = item.get("emission_factor")
            em = item.get("emissions_kgco2e", 0.0)
            w_str = f"{weight:.3f}" if weight is not None else "-"
            ef_str = f"{ef:.4f}" if ef is not None else "-"
            pct = _pct_of(em, total)
            lines.append(
                f"| {material} | {w_str} | {ef_str} | {_fmt_kgco2e(em)} | {pct} |"
            )
        return "\n".join(lines)

    def _md_product_comparison(self, data: Dict[str, Any]) -> str:
        """Render Markdown product comparison table."""
        products = data.get("product_comparison", [])
        if not products:
            return "## 4. Product Comparison\n\nNo comparison data available."
        lines = [
            "## 4. Product Comparison",
            "",
            "| Product | Total PCF | Intensity | Best Stage | Worst Stage |",
            "|---------|----------|-----------|-----------|------------|",
        ]
        for p in products:
            name = p.get("product_name", "-")
            total = _fmt_kgco2e(p.get("total_pcf_kgco2e"))
            intensity = p.get("intensity")
            int_str = f"{intensity:.2f}" if intensity is not None else "-"
            best = p.get("best_stage", "-")
            worst = p.get("worst_stage", "-")
            lines.append(f"| {name} | {total} | {int_str} | {best} | {worst} |")
        return "\n".join(lines)

    def _md_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        """Render Markdown sensitivity analysis."""
        params = data.get("sensitivity_analysis", [])
        if not params:
            return "## 5. Sensitivity Analysis\n\nNo sensitivity data available."
        lines = [
            "## 5. Sensitivity Analysis",
            "",
            "| Parameter | Base Value | Low (-20%) | High (+20%) | Impact |",
            "|-----------|-----------|-----------|------------|--------|",
        ]
        for p in params:
            name = p.get("parameter_name", "-")
            base = p.get("base_value")
            low = p.get("low_result_kgco2e")
            high = p.get("high_result_kgco2e")
            impact = p.get("sensitivity_index")
            base_str = f"{base}" if base is not None else "-"
            low_str = _fmt_kgco2e(low) if low is not None else "-"
            high_str = _fmt_kgco2e(high) if high is not None else "-"
            imp_str = f"{impact:.2f}" if impact is not None else "-"
            lines.append(f"| {name} | {base_str} | {low_str} | {high_str} | {imp_str} |")
        return "\n".join(lines)

    def _md_circular_economy(self, data: Dict[str, Any]) -> str:
        """Render Markdown circular economy metrics."""
        ce = data.get("circular_economy", {})
        if not ce:
            return "## 6. Circular Economy Metrics\n\nNo circular economy data available."
        lines = [
            "## 6. Circular Economy Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        recycled = ce.get("recycled_content_pct")
        if recycled is not None:
            lines.append(f"| Recycled Content | {recycled:.1f}% |")
        recyclability = ce.get("recyclability_pct")
        if recyclability is not None:
            lines.append(f"| End-of-Life Recyclability | {recyclability:.1f}% |")
        circular_index = ce.get("circularity_index")
        if circular_index is not None:
            lines.append(f"| Circularity Index | {circular_index:.2f} |")
        avoided = ce.get("avoided_emissions_kgco2e")
        if avoided is not None:
            lines.append(f"| Avoided Emissions (Recycling) | {_fmt_kgco2e(avoided)} |")
        return "\n".join(lines)

    def _md_improvements(self, data: Dict[str, Any]) -> str:
        """Render Markdown improvement opportunities."""
        improvements = data.get("improvements", [])
        if not improvements:
            return "## 7. Improvement Opportunities\n\nNo improvement data available."
        lines = [
            "## 7. Improvement Opportunities",
            "",
            "| Opportunity | Reduction | Cost | Feasibility |",
            "|------------|----------|------|-------------|",
        ]
        for imp in improvements:
            opp = imp.get("description", "-")
            red = imp.get("reduction_kgco2e")
            red_str = _fmt_kgco2e(red) if red is not None else "-"
            cost = imp.get("cost_level", "-")
            feas = imp.get("feasibility", "-")
            lines.append(f"| {opp} | {red_str} | {cost} | {feas} |")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            "---\n\n"
            f"*Generated by GreenLang PACK-043 Scope 3 Complete v{_MODULE_VERSION} | {ts}*\n"
            f"*Provenance Hash: `{provenance}`*"
        )

    # ==================================================================
    # HTML SECTIONS
    # ==================================================================

    def _wrap_html(self, data: Dict[str, Any], body: str) -> str:
        """Wrap body in full HTML document with inline CSS."""
        company = self._get_val(data, "company_name", "Organization")
        product = self._get_val(data, "product_name", "Product")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Product Carbon Footprint - {product} ({company})</title>\n"
            "<style>\n"
            ":root{--primary:#008B8B;--primary-light:#20B2AA;--accent:#00CED1;"
            "--bg:#F0FDFA;--card-bg:#FFFFFF;--text:#1A1A2E;--text-muted:#6B7280;"
            "--border:#D1D5DB;--success:#059669;--warning:#F59E0B;--danger:#EF4444;}\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:2rem;"
            "background:var(--bg);color:var(--text);line-height:1.6;}\n"
            ".container{max-width:1100px;margin:0 auto;}\n"
            "h1{color:var(--primary);border-bottom:3px solid var(--primary);"
            "padding-bottom:0.5rem;}\n"
            "h2{color:var(--primary);margin-top:2rem;"
            "border-bottom:1px solid var(--border);padding-bottom:0.3rem;}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid var(--border);padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:var(--primary);color:#fff;font-weight:600;}\n"
            "tr:nth-child(even){background:#F0FDFA;}\n"
            ".total-row{font-weight:bold;background:#CCFBF1;}\n"
            ".section{margin-bottom:2rem;background:var(--card-bg);"
            "padding:1.5rem;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.08);}\n"
            ".waterfall-bar{display:inline-block;height:20px;border-radius:3px;"
            "background:var(--primary);margin-right:4px;}\n"
            ".metric-card{display:inline-block;background:var(--card-bg);border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:160px;"
            "border-top:3px solid var(--accent);box-shadow:0 1px 3px rgba(0,0,0,0.1);}\n"
            ".metric-value{font-size:1.6rem;font-weight:700;color:var(--primary);}\n"
            ".metric-label{font-size:0.85rem;color:var(--text-muted);}\n"
            ".provenance{font-size:0.8rem;color:var(--text-muted);font-family:monospace;}\n"
            "</style>\n</head>\n<body>\n<div class=\"container\">\n"
            f"{body}\n"
            "</div>\n</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        product = self._get_val(data, "product_name", "Product")
        fu = self._get_val(data, "functional_unit", "-")
        return (
            '<div class="section">\n'
            f"<h1>Product Carbon Footprint &mdash; {product}</h1>\n"
            f"<p><strong>Company:</strong> {company} | "
            f"<strong>Functional Unit:</strong> {fu} | "
            f"<strong>Standard:</strong> ISO 14067</p>\n"
            "<hr>\n</div>"
        )

    def _html_product_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML product summary with metric cards."""
        total = self._total_pcf(data)
        weight = data.get("product_weight_kg")
        cards = [("Total PCF", _fmt_kgco2e(total))]
        if weight is not None:
            intensity = total / weight if weight > 0 else 0
            cards.append(("Weight", f"{weight:.2f} kg"))
            cards.append(("Intensity", f"{intensity:.2f} kgCO2e/kg"))
        card_html = ""
        for label, val in cards:
            card_html += (
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>1. Product Summary</h2>\n"
            f"<div>{card_html}</div>\n</div>"
        )

    def _html_lifecycle_waterfall(self, data: Dict[str, Any]) -> str:
        """Render HTML lifecycle waterfall table with visual bars."""
        stages = data.get("lifecycle_stages", {})
        if not stages:
            return ""
        total = self._total_pcf(data)
        rows = ""
        for stage_key in _LIFECYCLE_STAGES:
            stage_data = stages.get(stage_key, {})
            em = stage_data.get("emissions_kgco2e", 0.0)
            label = _STAGE_LABELS.get(stage_key, stage_key)
            pct = (em / total * 100) if total > 0 else 0
            bar_width = max(2, min(200, int(pct * 2)))
            rows += (
                f"<tr><td>{label}</td><td>{_fmt_kgco2e(em)}</td>"
                f"<td>{pct:.1f}%</td>"
                f'<td><div class="waterfall-bar" style="width:{bar_width}px;"></div></td></tr>\n'
            )
        rows += (
            f'<tr class="total-row"><td>Total</td><td>{_fmt_kgco2e(total)}</td>'
            f"<td>100.0%</td><td></td></tr>\n"
        )
        return (
            '<div class="section">\n'
            "<h2>2. Lifecycle Stage Waterfall</h2>\n"
            "<table><thead><tr><th>Stage</th><th>kgCO2e</th>"
            "<th>%</th><th>Visual</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_bom_breakdown(self, data: Dict[str, Any]) -> str:
        """Render HTML BOM breakdown table."""
        bom = data.get("bom_breakdown", [])
        if not bom:
            return ""
        total = self._total_pcf(data)
        rows = ""
        for item in bom:
            material = item.get("material_name", "-")
            weight = item.get("weight_kg")
            ef = item.get("emission_factor")
            em = item.get("emissions_kgco2e", 0.0)
            w_str = f"{weight:.3f}" if weight is not None else "-"
            ef_str = f"{ef:.4f}" if ef is not None else "-"
            pct = _pct_of(em, total)
            rows += (
                f"<tr><td>{material}</td><td>{w_str}</td><td>{ef_str}</td>"
                f"<td>{_fmt_kgco2e(em)}</td><td>{pct}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>3. Bill of Materials Breakdown</h2>\n"
            "<table><thead><tr><th>Material</th><th>Weight (kg)</th>"
            "<th>EF (kgCO2e/kg)</th><th>kgCO2e</th><th>%</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_product_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML product comparison table."""
        products = data.get("product_comparison", [])
        if not products:
            return ""
        rows = ""
        for p in products:
            name = p.get("product_name", "-")
            total = _fmt_kgco2e(p.get("total_pcf_kgco2e"))
            intensity = p.get("intensity")
            int_str = f"{intensity:.2f}" if intensity is not None else "-"
            best = p.get("best_stage", "-")
            worst = p.get("worst_stage", "-")
            rows += (
                f"<tr><td>{name}</td><td>{total}</td><td>{int_str}</td>"
                f"<td>{best}</td><td>{worst}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>4. Product Comparison</h2>\n"
            "<table><thead><tr><th>Product</th><th>Total PCF</th>"
            "<th>Intensity</th><th>Best Stage</th><th>Worst Stage</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML sensitivity analysis."""
        params = data.get("sensitivity_analysis", [])
        if not params:
            return ""
        rows = ""
        for p in params:
            name = p.get("parameter_name", "-")
            base = p.get("base_value")
            low = p.get("low_result_kgco2e")
            high = p.get("high_result_kgco2e")
            impact = p.get("sensitivity_index")
            base_str = f"{base}" if base is not None else "-"
            low_str = _fmt_kgco2e(low) if low is not None else "-"
            high_str = _fmt_kgco2e(high) if high is not None else "-"
            imp_str = f"{impact:.2f}" if impact is not None else "-"
            rows += (
                f"<tr><td>{name}</td><td>{base_str}</td><td>{low_str}</td>"
                f"<td>{high_str}</td><td>{imp_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>5. Sensitivity Analysis</h2>\n"
            "<table><thead><tr><th>Parameter</th><th>Base Value</th>"
            "<th>Low (-20%)</th><th>High (+20%)</th><th>Sensitivity Index</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_circular_economy(self, data: Dict[str, Any]) -> str:
        """Render HTML circular economy metrics."""
        ce = data.get("circular_economy", {})
        if not ce:
            return ""
        rows = ""
        recycled = ce.get("recycled_content_pct")
        if recycled is not None:
            rows += f"<tr><td>Recycled Content</td><td>{recycled:.1f}%</td></tr>\n"
        recyclability = ce.get("recyclability_pct")
        if recyclability is not None:
            rows += f"<tr><td>End-of-Life Recyclability</td><td>{recyclability:.1f}%</td></tr>\n"
        ci = ce.get("circularity_index")
        if ci is not None:
            rows += f"<tr><td>Circularity Index</td><td>{ci:.2f}</td></tr>\n"
        avoided = ce.get("avoided_emissions_kgco2e")
        if avoided is not None:
            rows += f"<tr><td>Avoided Emissions</td><td>{_fmt_kgco2e(avoided)}</td></tr>\n"
        if not rows:
            return ""
        return (
            '<div class="section">\n'
            "<h2>6. Circular Economy Metrics</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_improvements(self, data: Dict[str, Any]) -> str:
        """Render HTML improvement opportunities."""
        improvements = data.get("improvements", [])
        if not improvements:
            return ""
        rows = ""
        for imp in improvements:
            opp = imp.get("description", "-")
            red = imp.get("reduction_kgco2e")
            red_str = _fmt_kgco2e(red) if red is not None else "-"
            cost = imp.get("cost_level", "-")
            feas = imp.get("feasibility", "-")
            rows += (
                f"<tr><td>{opp}</td><td>{red_str}</td>"
                f"<td>{cost}</td><td>{feas}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>7. Improvement Opportunities</h2>\n"
            "<table><thead><tr><th>Opportunity</th><th>Reduction</th>"
            "<th>Cost</th><th>Feasibility</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer with provenance hash."""
        provenance = self._compute_provenance(data)
        ts = self.generated_at.isoformat() if self.generated_at else "N/A"
        return (
            '<div class="section" style="font-size:0.85rem;color:#666;">\n<hr>\n'
            f"<p>Generated by GreenLang PACK-043 Scope 3 Complete v{_MODULE_VERSION} | {ts}</p>\n"
            f'<p class="provenance">Provenance Hash: {provenance}</p>\n</div>'
        )

    # ==================================================================
    # JSON HELPERS
    # ==================================================================

    def _json_waterfall(
        self, data: Dict[str, Any], total: float
    ) -> List[Dict[str, Any]]:
        """Build lifecycle waterfall chart data."""
        stages = data.get("lifecycle_stages", {})
        result = []
        cumulative = 0.0
        for stage_key in _LIFECYCLE_STAGES:
            stage_data = stages.get(stage_key, {})
            em = stage_data.get("emissions_kgco2e", 0.0)
            cumulative += em
            result.append({
                "stage": stage_key,
                "label": _STAGE_LABELS.get(stage_key, stage_key),
                "emissions_kgco2e": em,
                "cumulative_kgco2e": cumulative,
                "pct_of_total": (em / total * 100) if total > 0 else 0.0,
            })
        return result
