# -*- coding: utf-8 -*-
"""
SupplierProgrammeReportTemplate - Supplier Scorecard and Commitment Tracker for PACK-043.

Generates a supplier programme report with supplier scorecard summary,
commitment tracker (SBTi/RE100/CDP/net-zero), YoY reduction progress
per supplier tier (strategic/key/managed), programme ROI, and
transition risk heatmap for supply chain decarbonization management.

Sections:
    1. Programme Overview
    2. Supplier Scorecard Summary
    3. Commitment Tracker (SBTi/RE100/CDP/Net-Zero)
    4. Tier-Level Reduction Progress
    5. Programme ROI
    6. Transition Risk Heatmap
    7. Action Plan

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, partnership blue #2471A3 theme)
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


def _fmt_pct(value: Optional[float]) -> str:
    """Format percentage with sign."""
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}%"


def _fmt_currency(value: Optional[float]) -> str:
    """Format currency value."""
    if value is None:
        return "N/A"
    return f"${_fmt_num(value)}"


def _risk_css(level: Optional[str]) -> str:
    """Map risk level to CSS class name."""
    if not level:
        return "risk-unknown"
    low = level.lower()
    if low in ("high", "critical"):
        return "risk-high"
    if low in ("medium", "moderate"):
        return "risk-medium"
    return "risk-low"


class SupplierProgrammeReportTemplate:
    """
    Supplier programme scorecard and commitment tracker template.

    Renders supplier decarbonization programme reports with scorecards,
    commitment tracking, tier-level progress, ROI analysis, and
    transition risk heatmaps. All outputs include SHA-256 provenance.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = SupplierProgrammeReportTemplate()
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SupplierProgrammeReportTemplate."""
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
        """Render supplier programme report as Markdown.

        Args:
            data: Validated supplier programme data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_programme_overview(data),
            self._md_scorecard_summary(data),
            self._md_commitment_tracker(data),
            self._md_tier_progress(data),
            self._md_programme_roi(data),
            self._md_transition_risk(data),
            self._md_action_plan(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render supplier programme report as HTML.

        Args:
            data: Validated supplier programme data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_programme_overview(data),
            self._html_scorecard_summary(data),
            self._html_commitment_tracker(data),
            self._html_tier_progress(data),
            self._html_programme_roi(data),
            self._html_transition_risk(data),
            self._html_action_plan(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render supplier programme report as JSON-serializable dict.

        Args:
            data: Validated supplier programme data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "supplier_programme_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "programme_overview": data.get("programme_overview", {}),
            "scorecard_summary": data.get("scorecard_summary", []),
            "commitment_tracker": data.get("commitment_tracker", {}),
            "tier_progress": data.get("tier_progress", []),
            "programme_roi": data.get("programme_roi", {}),
            "transition_risk_heatmap": data.get("transition_risk_heatmap", []),
            "action_plan": data.get("action_plan", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Supplier Programme Report - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_programme_overview(self, data: Dict[str, Any]) -> str:
        """Render Markdown programme overview."""
        overview = data.get("programme_overview", {})
        if not overview:
            return "## 1. Programme Overview\n\nNo programme overview available."
        total = overview.get("total_suppliers", 0)
        engaged = overview.get("suppliers_engaged", 0)
        response = overview.get("response_rate_pct")
        coverage = overview.get("emission_coverage_pct")
        lines = [
            "## 1. Programme Overview",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Suppliers in Scope | {total} |",
            f"| Suppliers Engaged | {engaged} |",
        ]
        if response is not None:
            lines.append(f"| Response Rate | {response:.1f}% |")
        if coverage is not None:
            lines.append(f"| Scope 3 Emission Coverage | {coverage:.1f}% |")
        return "\n".join(lines)

    def _md_scorecard_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown supplier scorecard summary."""
        scorecards = data.get("scorecard_summary", [])
        if not scorecards:
            return "## 2. Supplier Scorecard Summary\n\nNo scorecard data available."
        lines = [
            "## 2. Supplier Scorecard Summary",
            "",
            "| Supplier | Tier | Score | Emissions (tCO2e) | YoY Change | Grade |",
            "|----------|------|-------|------------------|-----------|-------|",
        ]
        for sc in scorecards:
            name = sc.get("supplier_name", "-")
            tier = sc.get("tier", "-")
            score = sc.get("score")
            score_str = f"{score:.0f}/100" if score is not None else "-"
            em = _fmt_tco2e(sc.get("emissions_tco2e"))
            yoy = sc.get("yoy_change_pct")
            yoy_str = _fmt_pct(yoy) if yoy is not None else "-"
            grade = sc.get("grade", "-")
            lines.append(
                f"| {name} | {tier} | {score_str} | {em} | {yoy_str} | {grade} |"
            )
        return "\n".join(lines)

    def _md_commitment_tracker(self, data: Dict[str, Any]) -> str:
        """Render Markdown commitment tracker."""
        tracker = data.get("commitment_tracker", {})
        if not tracker:
            return "## 3. Commitment Tracker\n\nNo commitment data available."
        frameworks = ["sbti", "re100", "cdp", "net_zero"]
        labels = {"sbti": "SBTi", "re100": "RE100", "cdp": "CDP", "net_zero": "Net-Zero"}
        lines = [
            "## 3. Commitment Tracker",
            "",
            "| Framework | Committed | In Progress | Not Started | Total |",
            "|-----------|-----------|------------|------------|-------|",
        ]
        for fw in frameworks:
            fw_data = tracker.get(fw, {})
            committed = fw_data.get("committed", 0)
            in_progress = fw_data.get("in_progress", 0)
            not_started = fw_data.get("not_started", 0)
            total = committed + in_progress + not_started
            lines.append(
                f"| {labels.get(fw, fw)} | {committed} | {in_progress} | "
                f"{not_started} | {total} |"
            )
        return "\n".join(lines)

    def _md_tier_progress(self, data: Dict[str, Any]) -> str:
        """Render Markdown tier-level reduction progress."""
        tiers = data.get("tier_progress", [])
        if not tiers:
            return "## 4. Tier-Level Reduction Progress\n\nNo tier data available."
        lines = [
            "## 4. Tier-Level Reduction Progress",
            "",
            "| Tier | Suppliers | Emissions (tCO2e) | YoY Change | Reduction Target |",
            "|------|----------|------------------|-----------|-----------------|",
        ]
        for tier in tiers:
            name = tier.get("tier_name", "-")
            count = tier.get("supplier_count", 0)
            em = _fmt_tco2e(tier.get("total_emissions_tco2e"))
            yoy = tier.get("yoy_change_pct")
            yoy_str = _fmt_pct(yoy) if yoy is not None else "-"
            target = tier.get("reduction_target_pct")
            tgt_str = f"{target:.1f}%" if target is not None else "-"
            lines.append(f"| {name} | {count} | {em} | {yoy_str} | {tgt_str} |")
        return "\n".join(lines)

    def _md_programme_roi(self, data: Dict[str, Any]) -> str:
        """Render Markdown programme ROI."""
        roi = data.get("programme_roi", {})
        if not roi:
            return "## 5. Programme ROI\n\nNo ROI data available."
        lines = [
            "## 5. Programme ROI",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        investment = roi.get("total_investment")
        if investment is not None:
            lines.append(f"| Total Programme Investment | {_fmt_currency(investment)} |")
        reductions = roi.get("total_reductions_tco2e")
        if reductions is not None:
            lines.append(f"| Total Reductions Achieved | {_fmt_tco2e(reductions)} |")
        cost_per = roi.get("cost_per_tco2e")
        if cost_per is not None:
            lines.append(f"| Cost per tCO2e Reduced | ${cost_per:,.0f} |")
        roi_pct = roi.get("roi_pct")
        if roi_pct is not None:
            lines.append(f"| Programme ROI | {roi_pct:.0f}% |")
        return "\n".join(lines)

    def _md_transition_risk(self, data: Dict[str, Any]) -> str:
        """Render Markdown transition risk heatmap."""
        risks = data.get("transition_risk_heatmap", [])
        if not risks:
            return "## 6. Transition Risk Heatmap\n\nNo risk data available."
        lines = [
            "## 6. Transition Risk Heatmap",
            "",
            "| Supplier | Carbon Price Risk | Stranded Asset Risk | Market Risk | Overall |",
            "|----------|------------------|-------------------|------------|---------|",
        ]
        for r in risks:
            name = r.get("supplier_name", "-")
            carbon = r.get("carbon_price_risk", "-")
            stranded = r.get("stranded_asset_risk", "-")
            market = r.get("market_risk", "-")
            overall = r.get("overall_risk", "-")
            lines.append(
                f"| {name} | {carbon} | {stranded} | {market} | {overall} |"
            )
        return "\n".join(lines)

    def _md_action_plan(self, data: Dict[str, Any]) -> str:
        """Render Markdown action plan."""
        actions = data.get("action_plan", [])
        if not actions:
            return "## 7. Action Plan\n\nNo actions defined."
        lines = ["## 7. Action Plan", ""]
        for i, action in enumerate(actions, 1):
            title = action.get("title", "")
            desc = action.get("description", "")
            owner = action.get("owner", "-")
            deadline = action.get("deadline", "-")
            lines.append(f"### Action {i}: {title}")
            lines.append("")
            if desc:
                lines.append(desc)
                lines.append("")
            lines.append(f"- **Owner:** {owner}")
            lines.append(f"- **Deadline:** {deadline}")
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
            f"<title>Supplier Programme Report - {company} ({year})</title>\n"
            "<style>\n"
            ":root{--primary:#2471A3;--primary-light:#2E86C1;--accent:#5DADE2;"
            "--bg:#EBF5FB;--card-bg:#FFFFFF;--text:#1A1A2E;--text-muted:#6B7280;"
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
            "tr:nth-child(even){background:#EBF5FB;}\n"
            ".section{margin-bottom:2rem;background:var(--card-bg);"
            "padding:1.5rem;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.08);}\n"
            ".risk-high{background:#FEE2E2;color:#991B1B;font-weight:700;}\n"
            ".risk-medium{background:#FEF3C7;color:#92400E;font-weight:700;}\n"
            ".risk-low{background:#D1FAE5;color:#065F46;font-weight:700;}\n"
            ".risk-unknown{color:var(--text-muted);}\n"
            ".action-card{background:#EBF5FB;border-left:4px solid var(--primary);"
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
            f"<h1>Supplier Programme Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year} | "
            f"<strong>Pack:</strong> PACK-043 v{_MODULE_VERSION}</p>\n"
            "<hr>\n</div>"
        )

    def _html_programme_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML programme overview."""
        overview = data.get("programme_overview", {})
        if not overview:
            return ""
        total = overview.get("total_suppliers", 0)
        engaged = overview.get("suppliers_engaged", 0)
        response = overview.get("response_rate_pct")
        coverage = overview.get("emission_coverage_pct")
        rows = (
            f"<tr><td>Total Suppliers</td><td>{total}</td></tr>\n"
            f"<tr><td>Suppliers Engaged</td><td>{engaged}</td></tr>\n"
        )
        if response is not None:
            rows += f"<tr><td>Response Rate</td><td>{response:.1f}%</td></tr>\n"
        if coverage is not None:
            rows += f"<tr><td>Emission Coverage</td><td>{coverage:.1f}%</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>1. Programme Overview</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_scorecard_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML supplier scorecard table."""
        scorecards = data.get("scorecard_summary", [])
        if not scorecards:
            return ""
        rows = ""
        for sc in scorecards:
            name = sc.get("supplier_name", "-")
            tier = sc.get("tier", "-")
            score = sc.get("score")
            score_str = f"{score:.0f}/100" if score is not None else "-"
            em = _fmt_tco2e(sc.get("emissions_tco2e"))
            yoy = sc.get("yoy_change_pct")
            yoy_str = _fmt_pct(yoy) if yoy is not None else "-"
            grade = sc.get("grade", "-")
            rows += (
                f"<tr><td>{name}</td><td>{tier}</td><td>{score_str}</td>"
                f"<td>{em}</td><td>{yoy_str}</td><td>{grade}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>2. Supplier Scorecard Summary</h2>\n"
            "<table><thead><tr><th>Supplier</th><th>Tier</th><th>Score</th>"
            "<th>Emissions</th><th>YoY</th><th>Grade</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_commitment_tracker(self, data: Dict[str, Any]) -> str:
        """Render HTML commitment tracker."""
        tracker = data.get("commitment_tracker", {})
        if not tracker:
            return ""
        frameworks = ["sbti", "re100", "cdp", "net_zero"]
        labels = {"sbti": "SBTi", "re100": "RE100", "cdp": "CDP", "net_zero": "Net-Zero"}
        rows = ""
        for fw in frameworks:
            fw_data = tracker.get(fw, {})
            committed = fw_data.get("committed", 0)
            in_progress = fw_data.get("in_progress", 0)
            not_started = fw_data.get("not_started", 0)
            total = committed + in_progress + not_started
            rows += (
                f"<tr><td>{labels.get(fw, fw)}</td><td>{committed}</td>"
                f"<td>{in_progress}</td><td>{not_started}</td><td>{total}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>3. Commitment Tracker</h2>\n"
            "<table><thead><tr><th>Framework</th><th>Committed</th>"
            "<th>In Progress</th><th>Not Started</th><th>Total</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_tier_progress(self, data: Dict[str, Any]) -> str:
        """Render HTML tier-level progress."""
        tiers = data.get("tier_progress", [])
        if not tiers:
            return ""
        rows = ""
        for tier in tiers:
            name = tier.get("tier_name", "-")
            count = tier.get("supplier_count", 0)
            em = _fmt_tco2e(tier.get("total_emissions_tco2e"))
            yoy = tier.get("yoy_change_pct")
            yoy_str = _fmt_pct(yoy) if yoy is not None else "-"
            target = tier.get("reduction_target_pct")
            tgt_str = f"{target:.1f}%" if target is not None else "-"
            rows += (
                f"<tr><td>{name}</td><td>{count}</td><td>{em}</td>"
                f"<td>{yoy_str}</td><td>{tgt_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>4. Tier-Level Reduction Progress</h2>\n"
            "<table><thead><tr><th>Tier</th><th>Suppliers</th>"
            "<th>Emissions</th><th>YoY</th><th>Target</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_programme_roi(self, data: Dict[str, Any]) -> str:
        """Render HTML programme ROI."""
        roi = data.get("programme_roi", {})
        if not roi:
            return ""
        rows = ""
        investment = roi.get("total_investment")
        if investment is not None:
            rows += f"<tr><td>Total Investment</td><td>{_fmt_currency(investment)}</td></tr>\n"
        reductions = roi.get("total_reductions_tco2e")
        if reductions is not None:
            rows += f"<tr><td>Total Reductions</td><td>{_fmt_tco2e(reductions)}</td></tr>\n"
        cost_per = roi.get("cost_per_tco2e")
        if cost_per is not None:
            rows += f"<tr><td>Cost per tCO2e</td><td>${cost_per:,.0f}</td></tr>\n"
        roi_pct = roi.get("roi_pct")
        if roi_pct is not None:
            rows += f"<tr><td>Programme ROI</td><td>{roi_pct:.0f}%</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>5. Programme ROI</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_transition_risk(self, data: Dict[str, Any]) -> str:
        """Render HTML transition risk heatmap."""
        risks = data.get("transition_risk_heatmap", [])
        if not risks:
            return ""
        rows = ""
        for r in risks:
            name = r.get("supplier_name", "-")
            carbon = r.get("carbon_price_risk", "-")
            stranded = r.get("stranded_asset_risk", "-")
            market = r.get("market_risk", "-")
            overall = r.get("overall_risk", "-")
            rows += (
                f"<tr><td>{name}</td>"
                f'<td class="{_risk_css(carbon)}">{carbon}</td>'
                f'<td class="{_risk_css(stranded)}">{stranded}</td>'
                f'<td class="{_risk_css(market)}">{market}</td>'
                f'<td class="{_risk_css(overall)}">{overall}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>6. Transition Risk Heatmap</h2>\n"
            "<table><thead><tr><th>Supplier</th><th>Carbon Price</th>"
            "<th>Stranded Assets</th><th>Market</th><th>Overall</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_action_plan(self, data: Dict[str, Any]) -> str:
        """Render HTML action plan."""
        actions = data.get("action_plan", [])
        if not actions:
            return ""
        cards = ""
        for i, action in enumerate(actions, 1):
            title = action.get("title", "")
            desc = action.get("description", "")
            owner = action.get("owner", "-")
            deadline = action.get("deadline", "-")
            cards += (
                f'<div class="action-card">\n'
                f"<h3>Action {i}: {title}</h3>\n"
                f"<p>{desc}</p>\n"
                f"<p><strong>Owner:</strong> {owner} | "
                f"<strong>Deadline:</strong> {deadline}</p>\n</div>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>7. Action Plan</h2>\n"
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
