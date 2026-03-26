# -*- coding: utf-8 -*-
"""
ClimateRiskReportTemplate - TCFD-Aligned Climate Risk Report for PACK-043.

Generates a TCFD-aligned climate risk report with transition risk
(carbon pricing, stranded assets, market shift), physical risk
(supply chain disruption map), opportunities (low-carbon demand),
financial impact NPV table over 10/20/30yr horizons, and scenario
comparison (IEA NZE, NGFS).

Sections:
    1. Risk Overview
    2. Transition Risks
    3. Physical Risks
    4. Opportunities
    5. Financial Impact (NPV over 10/20/30yr)
    6. Scenario Comparison (IEA NZE, NGFS)
    7. Risk Mitigation Actions

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, risk amber #E67E22 theme)
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


def _fmt_currency(value: Optional[float]) -> str:
    """Format currency value with M/K suffix."""
    if value is None:
        return "N/A"
    return f"${_fmt_num(value)}"


def _risk_label(level: Optional[str]) -> str:
    """Normalize risk level."""
    if not level:
        return "Not Assessed"
    return level.capitalize()


def _risk_css(level: Optional[str]) -> str:
    """Map risk level to CSS class."""
    if not level:
        return ""
    low = level.lower()
    if low in ("high", "critical", "very high"):
        return "risk-high"
    if low in ("medium", "moderate"):
        return "risk-medium"
    return "risk-low"


class ClimateRiskReportTemplate:
    """
    TCFD-aligned climate risk report template.

    Renders climate risk reports with transition risk, physical risk,
    opportunity assessment, financial impact NPV tables, and climate
    scenario comparisons. All outputs include SHA-256 provenance.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = ClimateRiskReportTemplate()
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ClimateRiskReportTemplate."""
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
        """Render climate risk report as Markdown.

        Args:
            data: Validated climate risk data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_risk_overview(data),
            self._md_transition_risks(data),
            self._md_physical_risks(data),
            self._md_opportunities(data),
            self._md_financial_impact(data),
            self._md_scenario_comparison(data),
            self._md_mitigation_actions(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render climate risk report as HTML.

        Args:
            data: Validated climate risk data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_risk_overview(data),
            self._html_transition_risks(data),
            self._html_physical_risks(data),
            self._html_opportunities(data),
            self._html_financial_impact(data),
            self._html_scenario_comparison(data),
            self._html_mitigation_actions(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render climate risk report as JSON-serializable dict.

        Args:
            data: Validated climate risk data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "climate_risk_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "risk_overview": data.get("risk_overview", {}),
            "transition_risks": data.get("transition_risks", []),
            "physical_risks": data.get("physical_risks", []),
            "opportunities": data.get("opportunities", []),
            "financial_impact": data.get("financial_impact", {}),
            "scenario_comparison": data.get("scenario_comparison", []),
            "mitigation_actions": data.get("mitigation_actions", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Climate Risk Report (TCFD-Aligned) - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Framework:** TCFD Recommendations\n\n"
            "---"
        )

    def _md_risk_overview(self, data: Dict[str, Any]) -> str:
        """Render Markdown risk overview."""
        overview = data.get("risk_overview", {})
        if not overview:
            return "## 1. Risk Overview\n\nNo risk overview available."
        transition = _risk_label(overview.get("overall_transition_risk"))
        physical = _risk_label(overview.get("overall_physical_risk"))
        opportunity = _risk_label(overview.get("overall_opportunity_level"))
        total_financial = overview.get("total_financial_exposure_mln")
        lines = [
            "## 1. Risk Overview",
            "",
            "| Dimension | Level |",
            "|-----------|-------|",
            f"| Overall Transition Risk | {transition} |",
            f"| Overall Physical Risk | {physical} |",
            f"| Opportunity Level | {opportunity} |",
        ]
        if total_financial is not None:
            lines.append(f"| Total Financial Exposure | ${total_financial:.1f}M |")
        return "\n".join(lines)

    def _md_transition_risks(self, data: Dict[str, Any]) -> str:
        """Render Markdown transition risks."""
        risks = data.get("transition_risks", [])
        if not risks:
            return "## 2. Transition Risks\n\nNo transition risk data available."
        lines = [
            "## 2. Transition Risks",
            "",
            "| Risk Type | Description | Likelihood | Impact | Financial ($M) | Timeframe |",
            "|-----------|------------|-----------|--------|---------------|-----------|",
        ]
        for r in risks:
            rtype = r.get("risk_type", "-")
            desc = r.get("description", "-")
            likelihood = r.get("likelihood", "-")
            impact = r.get("impact", "-")
            financial = r.get("financial_impact_mln")
            fin_str = f"${financial:.1f}M" if financial is not None else "-"
            timeframe = r.get("timeframe", "-")
            lines.append(
                f"| {rtype} | {desc} | {likelihood} | {impact} | {fin_str} | {timeframe} |"
            )
        return "\n".join(lines)

    def _md_physical_risks(self, data: Dict[str, Any]) -> str:
        """Render Markdown physical risks."""
        risks = data.get("physical_risks", [])
        if not risks:
            return "## 3. Physical Risks\n\nNo physical risk data available."
        lines = [
            "## 3. Physical Risks",
            "",
            "| Risk Type | Region/Asset | Likelihood | Impact | Financial ($M) |",
            "|-----------|-------------|-----------|--------|---------------|",
        ]
        for r in risks:
            rtype = r.get("risk_type", "-")
            region = r.get("region", "-")
            likelihood = r.get("likelihood", "-")
            impact = r.get("impact", "-")
            financial = r.get("financial_impact_mln")
            fin_str = f"${financial:.1f}M" if financial is not None else "-"
            lines.append(
                f"| {rtype} | {region} | {likelihood} | {impact} | {fin_str} |"
            )
        return "\n".join(lines)

    def _md_opportunities(self, data: Dict[str, Any]) -> str:
        """Render Markdown opportunities."""
        opps = data.get("opportunities", [])
        if not opps:
            return "## 4. Opportunities\n\nNo opportunity data available."
        lines = [
            "## 4. Climate-Related Opportunities",
            "",
            "| Opportunity | Category | Potential Revenue ($M) | Timeframe | Confidence |",
            "|------------|---------|---------------------|-----------|-----------|",
        ]
        for o in opps:
            opp = o.get("description", "-")
            cat = o.get("category", "-")
            revenue = o.get("potential_revenue_mln")
            rev_str = f"${revenue:.1f}M" if revenue is not None else "-"
            timeframe = o.get("timeframe", "-")
            confidence = o.get("confidence", "-")
            lines.append(f"| {opp} | {cat} | {rev_str} | {timeframe} | {confidence} |")
        return "\n".join(lines)

    def _md_financial_impact(self, data: Dict[str, Any]) -> str:
        """Render Markdown financial impact NPV table."""
        fi = data.get("financial_impact", {})
        if not fi:
            return "## 5. Financial Impact\n\nNo financial impact data available."
        horizons = fi.get("horizons", [])
        lines = [
            "## 5. Financial Impact (NPV Analysis)",
            "",
        ]
        discount_rate = fi.get("discount_rate_pct")
        if discount_rate is not None:
            lines.append(f"**Discount Rate:** {discount_rate:.1f}%")
            lines.append("")
        if horizons:
            lines.extend([
                "| Horizon | Risk NPV ($M) | Opportunity NPV ($M) | Net NPV ($M) |",
                "|---------|-------------|--------------------|-----------| ",
            ])
            for h in horizons:
                period = h.get("period", "-")
                risk_npv = h.get("risk_npv_mln")
                opp_npv = h.get("opportunity_npv_mln")
                risk_str = f"${risk_npv:.1f}M" if risk_npv is not None else "-"
                opp_str = f"${opp_npv:.1f}M" if opp_npv is not None else "-"
                net = (opp_npv or 0) - (risk_npv or 0)
                net_str = f"${net:.1f}M"
                lines.append(f"| {period} | {risk_str} | {opp_str} | {net_str} |")
        return "\n".join(lines)

    def _md_scenario_comparison(self, data: Dict[str, Any]) -> str:
        """Render Markdown scenario comparison."""
        scenarios = data.get("scenario_comparison", [])
        if not scenarios:
            return "## 6. Scenario Comparison\n\nNo scenario data available."
        lines = [
            "## 6. Scenario Comparison",
            "",
            "| Scenario | Warming | Carbon Price ($/t) | Impact on Scope 3 | Financial Impact ($M) |",
            "|----------|---------|-------------------|-----------------|--------------------|",
        ]
        for sc in scenarios:
            name = sc.get("scenario_name", "-")
            warming = sc.get("warming_target", "-")
            carbon_price = sc.get("carbon_price_usd_per_t")
            cp_str = f"${carbon_price:,.0f}" if carbon_price is not None else "-"
            scope3_impact = sc.get("scope3_impact_pct")
            s3_str = _fmt_num(scope3_impact) + "%" if scope3_impact is not None else "-"
            financial = sc.get("financial_impact_mln")
            fin_str = f"${financial:.1f}M" if financial is not None else "-"
            lines.append(f"| {name} | {warming} | {cp_str} | {s3_str} | {fin_str} |")
        return "\n".join(lines)

    def _md_mitigation_actions(self, data: Dict[str, Any]) -> str:
        """Render Markdown mitigation actions."""
        actions = data.get("mitigation_actions", [])
        if not actions:
            return "## 7. Risk Mitigation Actions\n\nNo mitigation actions defined."
        lines = ["## 7. Risk Mitigation Actions", ""]
        for i, action in enumerate(actions, 1):
            title = action.get("title", "")
            desc = action.get("description", "")
            risk_addressed = action.get("risk_addressed", "-")
            priority = action.get("priority", "-")
            lines.append(f"### Action {i}: {title}")
            lines.append("")
            if desc:
                lines.append(desc)
                lines.append("")
            lines.append(f"- **Risk Addressed:** {risk_addressed}")
            lines.append(f"- **Priority:** {priority}")
            lines.append("")
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
            f"<title>Climate Risk Report - {company} ({year})</title>\n"
            "<style>\n"
            ":root{--primary:#E67E22;--primary-light:#F39C12;--accent:#F5B041;"
            "--bg:#FEF9E7;--card-bg:#FFFFFF;--text:#1A1A2E;--text-muted:#6B7280;"
            "--border:#D1D5DB;--success:#10B981;--warning:#F59E0B;--danger:#EF4444;}\n"
            "body{font-family:'Segoe UI',Arial,sans-serif;margin:0;padding:2rem;"
            "background:var(--bg);color:var(--text);line-height:1.6;}\n"
            ".container{max-width:1100px;margin:0 auto;}\n"
            "h1{color:var(--primary);border-bottom:3px solid var(--primary);"
            "padding-bottom:0.5rem;}\n"
            "h2{color:var(--primary);margin-top:2rem;"
            "border-bottom:1px solid var(--border);padding-bottom:0.3rem;}\n"
            "h3{color:var(--primary-light);}\n"
            "table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:0.9rem;}\n"
            "th,td{border:1px solid var(--border);padding:0.5rem 0.7rem;text-align:left;}\n"
            "th{background:var(--primary);color:#fff;font-weight:600;}\n"
            "tr:nth-child(even){background:#FEF9E7;}\n"
            ".section{margin-bottom:2rem;background:var(--card-bg);"
            "padding:1.5rem;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.08);}\n"
            ".risk-high{background:#FEE2E2;color:#991B1B;font-weight:700;}\n"
            ".risk-medium{background:#FEF3C7;color:#92400E;font-weight:700;}\n"
            ".risk-low{background:#D1FAE5;color:#065F46;font-weight:700;}\n"
            ".opportunity{color:var(--success);font-weight:700;}\n"
            ".action-card{background:#FEF9E7;border-left:4px solid var(--primary);"
            "padding:1rem;margin:0.5rem 0;border-radius:0 4px 4px 0;}\n"
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
            f"<h1>Climate Risk Report (TCFD) &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year} | "
            f"<strong>Framework:</strong> TCFD | "
            f"<strong>Pack:</strong> PACK-043 v{_MODULE_VERSION}</p>\n"
            "<hr>\n</div>"
        )

    def _html_risk_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML risk overview."""
        overview = data.get("risk_overview", {})
        if not overview:
            return ""
        transition = overview.get("overall_transition_risk", "")
        physical = overview.get("overall_physical_risk", "")
        rows = (
            f'<tr><td>Transition Risk</td><td class="{_risk_css(transition)}">'
            f"{_risk_label(transition)}</td></tr>\n"
            f'<tr><td>Physical Risk</td><td class="{_risk_css(physical)}">'
            f"{_risk_label(physical)}</td></tr>\n"
        )
        opportunity = overview.get("overall_opportunity_level")
        if opportunity:
            rows += f'<tr><td>Opportunity Level</td><td class="opportunity">{_risk_label(opportunity)}</td></tr>\n'
        total = overview.get("total_financial_exposure_mln")
        if total is not None:
            rows += f"<tr><td>Total Financial Exposure</td><td>${total:.1f}M</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>1. Risk Overview</h2>\n"
            "<table><thead><tr><th>Dimension</th><th>Level</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_transition_risks(self, data: Dict[str, Any]) -> str:
        """Render HTML transition risks."""
        risks = data.get("transition_risks", [])
        if not risks:
            return ""
        rows = ""
        for r in risks:
            rtype = r.get("risk_type", "-")
            desc = r.get("description", "-")
            likelihood = r.get("likelihood", "-")
            impact = r.get("impact", "-")
            financial = r.get("financial_impact_mln")
            fin_str = f"${financial:.1f}M" if financial is not None else "-"
            timeframe = r.get("timeframe", "-")
            imp_css = _risk_css(impact)
            rows += (
                f"<tr><td>{rtype}</td><td>{desc}</td><td>{likelihood}</td>"
                f'<td class="{imp_css}">{impact}</td><td>{fin_str}</td>'
                f"<td>{timeframe}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>2. Transition Risks</h2>\n"
            "<table><thead><tr><th>Type</th><th>Description</th>"
            "<th>Likelihood</th><th>Impact</th><th>Financial</th>"
            "<th>Timeframe</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_physical_risks(self, data: Dict[str, Any]) -> str:
        """Render HTML physical risks."""
        risks = data.get("physical_risks", [])
        if not risks:
            return ""
        rows = ""
        for r in risks:
            rtype = r.get("risk_type", "-")
            region = r.get("region", "-")
            likelihood = r.get("likelihood", "-")
            impact = r.get("impact", "-")
            financial = r.get("financial_impact_mln")
            fin_str = f"${financial:.1f}M" if financial is not None else "-"
            imp_css = _risk_css(impact)
            rows += (
                f"<tr><td>{rtype}</td><td>{region}</td><td>{likelihood}</td>"
                f'<td class="{imp_css}">{impact}</td><td>{fin_str}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>3. Physical Risks</h2>\n"
            "<table><thead><tr><th>Type</th><th>Region</th>"
            "<th>Likelihood</th><th>Impact</th><th>Financial</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_opportunities(self, data: Dict[str, Any]) -> str:
        """Render HTML opportunities."""
        opps = data.get("opportunities", [])
        if not opps:
            return ""
        rows = ""
        for o in opps:
            opp = o.get("description", "-")
            cat = o.get("category", "-")
            revenue = o.get("potential_revenue_mln")
            rev_str = f"${revenue:.1f}M" if revenue is not None else "-"
            timeframe = o.get("timeframe", "-")
            confidence = o.get("confidence", "-")
            rows += (
                f"<tr><td>{opp}</td><td>{cat}</td><td>{rev_str}</td>"
                f"<td>{timeframe}</td><td>{confidence}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>4. Climate-Related Opportunities</h2>\n"
            "<table><thead><tr><th>Opportunity</th><th>Category</th>"
            "<th>Revenue</th><th>Timeframe</th><th>Confidence</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_financial_impact(self, data: Dict[str, Any]) -> str:
        """Render HTML financial impact NPV table."""
        fi = data.get("financial_impact", {})
        horizons = fi.get("horizons", [])
        if not horizons:
            return ""
        discount_rate = fi.get("discount_rate_pct")
        dr_html = ""
        if discount_rate is not None:
            dr_html = f"<p><strong>Discount Rate:</strong> {discount_rate:.1f}%</p>"
        rows = ""
        for h in horizons:
            period = h.get("period", "-")
            risk_npv = h.get("risk_npv_mln")
            opp_npv = h.get("opportunity_npv_mln")
            risk_str = f"${risk_npv:.1f}M" if risk_npv is not None else "-"
            opp_str = f"${opp_npv:.1f}M" if opp_npv is not None else "-"
            net = (opp_npv or 0) - (risk_npv or 0)
            net_css = "opportunity" if net >= 0 else "risk-high"
            net_str = f"${net:.1f}M"
            rows += (
                f"<tr><td>{period}</td><td>{risk_str}</td><td>{opp_str}</td>"
                f'<td class="{net_css}">{net_str}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>5. Financial Impact (NPV)</h2>\n"
            f"{dr_html}\n"
            "<table><thead><tr><th>Horizon</th><th>Risk NPV</th>"
            "<th>Opportunity NPV</th><th>Net NPV</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_scenario_comparison(self, data: Dict[str, Any]) -> str:
        """Render HTML scenario comparison."""
        scenarios = data.get("scenario_comparison", [])
        if not scenarios:
            return ""
        rows = ""
        for sc in scenarios:
            name = sc.get("scenario_name", "-")
            warming = sc.get("warming_target", "-")
            carbon_price = sc.get("carbon_price_usd_per_t")
            cp_str = f"${carbon_price:,.0f}" if carbon_price is not None else "-"
            scope3_impact = sc.get("scope3_impact_pct")
            s3_str = f"{scope3_impact:.1f}%" if scope3_impact is not None else "-"
            financial = sc.get("financial_impact_mln")
            fin_str = f"${financial:.1f}M" if financial is not None else "-"
            rows += (
                f"<tr><td>{name}</td><td>{warming}</td><td>{cp_str}</td>"
                f"<td>{s3_str}</td><td>{fin_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>6. Scenario Comparison</h2>\n"
            "<table><thead><tr><th>Scenario</th><th>Warming</th>"
            "<th>Carbon Price ($/t)</th><th>Scope 3 Impact</th>"
            "<th>Financial Impact</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_mitigation_actions(self, data: Dict[str, Any]) -> str:
        """Render HTML mitigation actions."""
        actions = data.get("mitigation_actions", [])
        if not actions:
            return ""
        cards = ""
        for i, action in enumerate(actions, 1):
            title = action.get("title", "")
            desc = action.get("description", "")
            risk_addr = action.get("risk_addressed", "-")
            priority = action.get("priority", "-")
            cards += (
                f'<div class="action-card">\n'
                f"<h3>Action {i}: {title}</h3>\n"
                f"<p>{desc}</p>\n"
                f"<p><strong>Risk Addressed:</strong> {risk_addr} | "
                f"<strong>Priority:</strong> {priority}</p>\n</div>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>7. Risk Mitigation Actions</h2>\n"
            f"{cards}</div>"
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
