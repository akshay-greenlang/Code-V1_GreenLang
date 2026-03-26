# -*- coding: utf-8 -*-
"""
ScenarioReportTemplate - MACC Curve and What-If Scenario Analysis for PACK-043.

Generates a scenario planning report with MACC (Marginal Abatement Cost
Curve) data, what-if scenario results, cumulative reduction waterfall,
Paris alignment check (1.5C/WB2C), portfolio optimization results, and
budget-constrained top interventions for strategic decarbonization planning.

Sections:
    1. Scenario Overview
    2. MACC Curve Data
    3. What-If Scenario Results
    4. Cumulative Reduction Waterfall
    5. Paris Alignment Check
    6. Portfolio Optimization
    7. Budget-Constrained Top Interventions

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, scenario orange #D35400 theme)
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


def _fmt_num(value: Optional[float], decimals: int = 1) -> str:
    """Format numeric value with thousands separators."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.{decimals}f}K"
    return f"{value:,.{decimals}f}"


def _fmt_tco2e(value: Optional[float]) -> str:
    """Format tCO2e with scale suffix."""
    if value is None:
        return "N/A"
    return f"{_fmt_num(value)} tCO2e"


def _fmt_currency(value: Optional[float]) -> str:
    """Format currency value."""
    if value is None:
        return "N/A"
    return f"${_fmt_num(value)}"


def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage with sign."""
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


class ScenarioReportTemplate:
    """
    MACC curve and what-if scenario analysis template.

    Renders scenario planning reports with abatement cost curves,
    what-if analyses, Paris alignment checks, and budget-constrained
    intervention portfolios. All outputs include SHA-256 provenance.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = ScenarioReportTemplate()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ScenarioReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def _compute_provenance(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 provenance hash of input data."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_val(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Get value with config override support."""
        return self.config.get(key, data.get(key, default))

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render scenario report as Markdown.

        Args:
            data: Validated scenario report data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overview(data),
            self._md_macc_curve(data),
            self._md_whatif_scenarios(data),
            self._md_cumulative_waterfall(data),
            self._md_paris_alignment(data),
            self._md_portfolio_optimization(data),
            self._md_budget_interventions(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render scenario report as HTML.

        Args:
            data: Validated scenario report data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_overview(data),
            self._html_macc_curve(data),
            self._html_whatif_scenarios(data),
            self._html_cumulative_waterfall(data),
            self._html_paris_alignment(data),
            self._html_portfolio_optimization(data),
            self._html_budget_interventions(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render scenario report as JSON-serializable dict.

        Args:
            data: Validated scenario report data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "scenario_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "baseline_tco2e": data.get("baseline_tco2e"),
            "macc_curve": data.get("macc_curve", []),
            "whatif_scenarios": data.get("whatif_scenarios", []),
            "cumulative_waterfall": self._json_waterfall(data),
            "paris_alignment": data.get("paris_alignment", {}),
            "portfolio_optimization": data.get("portfolio_optimization", {}),
            "budget_interventions": data.get("budget_interventions", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Scope 3 Scenario Analysis - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Analysis Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        """Render Markdown scenario overview."""
        baseline = data.get("baseline_tco2e", 0.0)
        target = data.get("target_tco2e")
        budget = data.get("total_budget")
        scenarios_count = len(data.get("whatif_scenarios", []))
        interventions_count = len(data.get("macc_curve", []))
        lines = [
            "## 1. Scenario Overview",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Baseline Emissions | {_fmt_tco2e(baseline)} |",
        ]
        if target is not None:
            lines.append(f"| Target Emissions | {_fmt_tco2e(target)} |")
            gap = baseline - target
            lines.append(f"| Reduction Required | {_fmt_tco2e(gap)} |")
        if budget is not None:
            lines.append(f"| Available Budget | {_fmt_currency(budget)} |")
        lines.append(f"| Interventions Evaluated | {interventions_count} |")
        lines.append(f"| Scenarios Modelled | {scenarios_count} |")
        return "\n".join(lines)

    def _md_macc_curve(self, data: Dict[str, Any]) -> str:
        """Render Markdown MACC curve data table."""
        macc = data.get("macc_curve", [])
        if not macc:
            return "## 2. Marginal Abatement Cost Curve\n\nNo MACC data available."
        macc_sorted = sorted(macc, key=lambda x: x.get("cost_per_tco2e", 0))
        lines = [
            "## 2. Marginal Abatement Cost Curve (MACC)",
            "",
            "| Intervention | Reduction (tCO2e) | Cost ($/tCO2e) | Total Cost | Category |",
            "|-------------|-------------------|---------------|-----------|----------|",
        ]
        for item in macc_sorted:
            name = item.get("intervention_name", "-")
            red = _fmt_tco2e(item.get("reduction_tco2e"))
            cost = item.get("cost_per_tco2e")
            cost_str = f"${cost:,.0f}" if cost is not None else "-"
            total = _fmt_currency(item.get("total_cost"))
            cat = item.get("category", "-")
            lines.append(f"| {name} | {red} | {cost_str} | {total} | {cat} |")
        return "\n".join(lines)

    def _md_whatif_scenarios(self, data: Dict[str, Any]) -> str:
        """Render Markdown what-if scenario results."""
        scenarios = data.get("whatif_scenarios", [])
        if not scenarios:
            return "## 3. What-If Scenario Results\n\nNo scenario data available."
        lines = [
            "## 3. What-If Scenario Results",
            "",
            "| Scenario | Resulting tCO2e | Reduction % | Investment | NPV |",
            "|----------|----------------|------------|-----------|-----|",
        ]
        for sc in scenarios:
            name = sc.get("scenario_name", "-")
            result = _fmt_tco2e(sc.get("resulting_tco2e"))
            red_pct = sc.get("reduction_pct")
            red_str = f"{red_pct:.1f}%" if red_pct is not None else "-"
            invest = _fmt_currency(sc.get("total_investment"))
            npv = _fmt_currency(sc.get("npv"))
            lines.append(f"| {name} | {result} | {red_str} | {invest} | {npv} |")
        return "\n".join(lines)

    def _md_cumulative_waterfall(self, data: Dict[str, Any]) -> str:
        """Render Markdown cumulative reduction waterfall."""
        waterfall = data.get("cumulative_waterfall", [])
        if not waterfall:
            return "## 4. Cumulative Reduction Waterfall\n\nNo waterfall data available."
        lines = [
            "## 4. Cumulative Reduction Waterfall",
            "",
            "| Step | Intervention | Reduction | Cumulative | Remaining |",
            "|------|-------------|----------|-----------|-----------|",
        ]
        for i, step in enumerate(waterfall, 1):
            name = step.get("intervention_name", "-")
            red = _fmt_tco2e(step.get("reduction_tco2e"))
            cum = _fmt_tco2e(step.get("cumulative_reduction_tco2e"))
            rem = _fmt_tco2e(step.get("remaining_tco2e"))
            lines.append(f"| {i} | {name} | {red} | {cum} | {rem} |")
        return "\n".join(lines)

    def _md_paris_alignment(self, data: Dict[str, Any]) -> str:
        """Render Markdown Paris alignment check."""
        paris = data.get("paris_alignment", {})
        if not paris:
            return "## 5. Paris Alignment Check\n\nNo alignment data available."
        lines = [
            "## 5. Paris Alignment Check",
            "",
            "| Pathway | Required Reduction | Projected | Aligned |",
            "|---------|-------------------|-----------|---------|",
        ]
        pathways = paris.get("pathways", [])
        for pw in pathways:
            name = pw.get("pathway_name", "-")
            required = pw.get("required_reduction_pct")
            projected = pw.get("projected_reduction_pct")
            aligned = pw.get("is_aligned", False)
            req_str = f"{required:.1f}%" if required is not None else "-"
            proj_str = f"{projected:.1f}%" if projected is not None else "-"
            aligned_str = "Yes" if aligned else "No"
            lines.append(f"| {name} | {req_str} | {proj_str} | {aligned_str} |")
        gap_to_15 = paris.get("gap_to_1_5c_pct")
        if gap_to_15 is not None:
            lines.append(f"\n**Gap to 1.5C Pathway:** {gap_to_15:.1f} percentage points")
        return "\n".join(lines)

    def _md_portfolio_optimization(self, data: Dict[str, Any]) -> str:
        """Render Markdown portfolio optimization results."""
        portfolio = data.get("portfolio_optimization", {})
        if not portfolio:
            return "## 6. Portfolio Optimization\n\nNo optimization data available."
        total_red = portfolio.get("total_reduction_tco2e")
        total_cost = portfolio.get("total_cost")
        avg_cost = portfolio.get("average_cost_per_tco2e")
        selected = portfolio.get("selected_interventions", [])
        lines = [
            "## 6. Portfolio Optimization Results",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        if total_red is not None:
            lines.append(f"| Total Portfolio Reduction | {_fmt_tco2e(total_red)} |")
        if total_cost is not None:
            lines.append(f"| Total Portfolio Cost | {_fmt_currency(total_cost)} |")
        if avg_cost is not None:
            lines.append(f"| Average Cost per tCO2e | ${avg_cost:,.0f} |")
        if selected:
            lines.append("")
            lines.append("**Selected Interventions:**")
            for item in selected:
                lines.append(f"- {item.get('intervention_name', '-')}: "
                           f"{_fmt_tco2e(item.get('reduction_tco2e'))}")
        return "\n".join(lines)

    def _md_budget_interventions(self, data: Dict[str, Any]) -> str:
        """Render Markdown budget-constrained top interventions."""
        interventions = data.get("budget_interventions", [])
        if not interventions:
            return "## 7. Budget-Constrained Interventions\n\nNo budget analysis available."
        budget = data.get("total_budget")
        lines = [
            "## 7. Budget-Constrained Top Interventions",
            "",
        ]
        if budget is not None:
            lines.append(f"**Budget Constraint:** {_fmt_currency(budget)}")
            lines.append("")
        lines.extend([
            "| Priority | Intervention | Reduction | Cost | Cumulative Cost |",
            "|----------|-------------|----------|------|----------------|",
        ])
        cum_cost = 0.0
        for i, item in enumerate(interventions, 1):
            name = item.get("intervention_name", "-")
            red = _fmt_tco2e(item.get("reduction_tco2e"))
            cost = item.get("total_cost", 0.0)
            cum_cost += cost
            lines.append(
                f"| {i} | {name} | {red} | {_fmt_currency(cost)} | "
                f"{_fmt_currency(cum_cost)} |"
            )
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
        year = self._get_val(data, "reporting_year", "")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Scenario Analysis - {company} ({year})</title>\n"
            "<style>\n"
            ":root{--primary:#D35400;--primary-light:#E67E22;--accent:#F39C12;"
            "--bg:#FFF8F0;--card-bg:#FFFFFF;--text:#1A1A2E;--text-muted:#6B7280;"
            "--border:#D1D5DB;--success:#10B981;--warning:#F59E0B;--danger:#EF4444;}\n"
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
            "tr:nth-child(even){background:#FFF8F0;}\n"
            ".section{margin-bottom:2rem;background:var(--card-bg);"
            "padding:1.5rem;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.08);}\n"
            ".metric-card{display:inline-block;background:var(--card-bg);border-radius:8px;"
            "padding:1rem 1.5rem;margin:0.5rem;text-align:center;min-width:170px;"
            "border-top:3px solid var(--accent);box-shadow:0 1px 3px rgba(0,0,0,0.1);}\n"
            ".metric-value{font-size:1.6rem;font-weight:700;color:var(--primary);}\n"
            ".metric-label{font-size:0.85rem;color:var(--text-muted);}\n"
            ".aligned{color:var(--success);font-weight:700;}\n"
            ".not-aligned{color:var(--danger);font-weight:700;}\n"
            ".negative-cost{color:var(--success);}\n"
            ".provenance{font-size:0.8rem;color:var(--text-muted);font-family:monospace;}\n"
            "</style>\n</head>\n<body>\n<div class=\"container\">\n"
            f"{body}\n"
            "</div>\n</body>\n</html>"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            '<div class="section">\n'
            f"<h1>Scope 3 Scenario Analysis &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year} | "
            f"<strong>Pack:</strong> PACK-043 v{_MODULE_VERSION}</p>\n"
            "<hr>\n</div>"
        )

    def _html_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML scenario overview with cards."""
        baseline = data.get("baseline_tco2e", 0.0)
        target = data.get("target_tco2e")
        budget = data.get("total_budget")
        cards = [("Baseline", _fmt_tco2e(baseline))]
        if target is not None:
            cards.append(("Target", _fmt_tco2e(target)))
            gap = baseline - target
            cards.append(("Gap", _fmt_tco2e(gap)))
        if budget is not None:
            cards.append(("Budget", _fmt_currency(budget)))
        card_html = ""
        for label, val in cards:
            card_html += (
                f'<div class="metric-card">'
                f'<div class="metric-value">{val}</div>'
                f'<div class="metric-label">{label}</div></div>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>1. Scenario Overview</h2>\n"
            f"<div>{card_html}</div>\n</div>"
        )

    def _html_macc_curve(self, data: Dict[str, Any]) -> str:
        """Render HTML MACC curve table."""
        macc = data.get("macc_curve", [])
        if not macc:
            return ""
        macc_sorted = sorted(macc, key=lambda x: x.get("cost_per_tco2e", 0))
        rows = ""
        for item in macc_sorted:
            name = item.get("intervention_name", "-")
            red = _fmt_tco2e(item.get("reduction_tco2e"))
            cost = item.get("cost_per_tco2e")
            cost_str = f"${cost:,.0f}" if cost is not None else "-"
            cost_css = ' class="negative-cost"' if cost is not None and cost < 0 else ""
            total = _fmt_currency(item.get("total_cost"))
            cat = item.get("category", "-")
            rows += (
                f"<tr><td>{name}</td><td>{red}</td>"
                f"<td{cost_css}>{cost_str}</td><td>{total}</td><td>{cat}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>2. Marginal Abatement Cost Curve (MACC)</h2>\n"
            "<table><thead><tr><th>Intervention</th><th>Reduction</th>"
            "<th>$/tCO2e</th><th>Total Cost</th><th>Category</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_whatif_scenarios(self, data: Dict[str, Any]) -> str:
        """Render HTML what-if scenario results."""
        scenarios = data.get("whatif_scenarios", [])
        if not scenarios:
            return ""
        rows = ""
        for sc in scenarios:
            name = sc.get("scenario_name", "-")
            result = _fmt_tco2e(sc.get("resulting_tco2e"))
            red_pct = sc.get("reduction_pct")
            red_str = f"{red_pct:.1f}%" if red_pct is not None else "-"
            invest = _fmt_currency(sc.get("total_investment"))
            npv = _fmt_currency(sc.get("npv"))
            rows += (
                f"<tr><td>{name}</td><td>{result}</td><td>{red_str}</td>"
                f"<td>{invest}</td><td>{npv}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>3. What-If Scenario Results</h2>\n"
            "<table><thead><tr><th>Scenario</th><th>Result tCO2e</th>"
            "<th>Reduction %</th><th>Investment</th><th>NPV</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_cumulative_waterfall(self, data: Dict[str, Any]) -> str:
        """Render HTML cumulative reduction waterfall."""
        waterfall = data.get("cumulative_waterfall", [])
        if not waterfall:
            return ""
        rows = ""
        for i, step in enumerate(waterfall, 1):
            name = step.get("intervention_name", "-")
            red = _fmt_tco2e(step.get("reduction_tco2e"))
            cum = _fmt_tco2e(step.get("cumulative_reduction_tco2e"))
            rem = _fmt_tco2e(step.get("remaining_tco2e"))
            rows += (
                f"<tr><td>{i}</td><td>{name}</td><td>{red}</td>"
                f"<td>{cum}</td><td>{rem}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>4. Cumulative Reduction Waterfall</h2>\n"
            "<table><thead><tr><th>Step</th><th>Intervention</th>"
            "<th>Reduction</th><th>Cumulative</th><th>Remaining</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_paris_alignment(self, data: Dict[str, Any]) -> str:
        """Render HTML Paris alignment check."""
        paris = data.get("paris_alignment", {})
        pathways = paris.get("pathways", [])
        if not pathways:
            return ""
        rows = ""
        for pw in pathways:
            name = pw.get("pathway_name", "-")
            required = pw.get("required_reduction_pct")
            projected = pw.get("projected_reduction_pct")
            aligned = pw.get("is_aligned", False)
            req_str = f"{required:.1f}%" if required is not None else "-"
            proj_str = f"{projected:.1f}%" if projected is not None else "-"
            css = "aligned" if aligned else "not-aligned"
            aligned_str = "Yes" if aligned else "No"
            rows += (
                f"<tr><td>{name}</td><td>{req_str}</td><td>{proj_str}</td>"
                f'<td class="{css}">{aligned_str}</td></tr>\n'
            )
        gap_html = ""
        gap_to_15 = paris.get("gap_to_1_5c_pct")
        if gap_to_15 is not None:
            gap_html = f"<p><strong>Gap to 1.5C Pathway:</strong> {gap_to_15:.1f} pp</p>"
        return (
            '<div class="section">\n'
            "<h2>5. Paris Alignment Check</h2>\n"
            "<table><thead><tr><th>Pathway</th><th>Required</th>"
            "<th>Projected</th><th>Aligned</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n{gap_html}\n</div>"
        )

    def _html_portfolio_optimization(self, data: Dict[str, Any]) -> str:
        """Render HTML portfolio optimization."""
        portfolio = data.get("portfolio_optimization", {})
        if not portfolio:
            return ""
        total_red = portfolio.get("total_reduction_tco2e")
        total_cost = portfolio.get("total_cost")
        avg_cost = portfolio.get("average_cost_per_tco2e")
        rows = ""
        if total_red is not None:
            rows += f"<tr><td>Total Portfolio Reduction</td><td>{_fmt_tco2e(total_red)}</td></tr>\n"
        if total_cost is not None:
            rows += f"<tr><td>Total Portfolio Cost</td><td>{_fmt_currency(total_cost)}</td></tr>\n"
        if avg_cost is not None:
            rows += f"<tr><td>Average $/tCO2e</td><td>${avg_cost:,.0f}</td></tr>\n"
        selected = portfolio.get("selected_interventions", [])
        sel_html = ""
        if selected:
            sel_rows = ""
            for item in selected:
                sel_rows += (
                    f"<tr><td>{item.get('intervention_name', '-')}</td>"
                    f"<td>{_fmt_tco2e(item.get('reduction_tco2e'))}</td></tr>\n"
                )
            sel_html = (
                "<h3>Selected Interventions</h3>"
                "<table><thead><tr><th>Intervention</th><th>Reduction</th></tr></thead>"
                f"<tbody>{sel_rows}</tbody></table>"
            )
        return (
            '<div class="section">\n'
            "<h2>6. Portfolio Optimization</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n{sel_html}\n</div>"
        )

    def _html_budget_interventions(self, data: Dict[str, Any]) -> str:
        """Render HTML budget-constrained interventions."""
        interventions = data.get("budget_interventions", [])
        if not interventions:
            return ""
        budget = data.get("total_budget")
        budget_html = ""
        if budget is not None:
            budget_html = f"<p><strong>Budget Constraint:</strong> {_fmt_currency(budget)}</p>"
        rows = ""
        cum_cost = 0.0
        for i, item in enumerate(interventions, 1):
            name = item.get("intervention_name", "-")
            red = _fmt_tco2e(item.get("reduction_tco2e"))
            cost = item.get("total_cost", 0.0)
            cum_cost += cost
            rows += (
                f"<tr><td>{i}</td><td>{name}</td><td>{red}</td>"
                f"<td>{_fmt_currency(cost)}</td><td>{_fmt_currency(cum_cost)}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>7. Budget-Constrained Top Interventions</h2>\n"
            f"{budget_html}\n"
            "<table><thead><tr><th>#</th><th>Intervention</th><th>Reduction</th>"
            "<th>Cost</th><th>Cumulative</th></tr></thead>\n"
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

    def _json_waterfall(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build cumulative reduction waterfall chart data."""
        waterfall = data.get("cumulative_waterfall", [])
        return [
            {
                "step": i + 1,
                "intervention_name": step.get("intervention_name"),
                "reduction_tco2e": step.get("reduction_tco2e"),
                "cumulative_reduction_tco2e": step.get("cumulative_reduction_tco2e"),
                "remaining_tco2e": step.get("remaining_tco2e"),
            }
            for i, step in enumerate(waterfall)
        ]
