# -*- coding: utf-8 -*-
"""
MaturityReportTemplate - Maturity Tier Heatmap and Upgrade Roadmap for PACK-043.

Generates a maturity assessment report with current vs target tier per
category heatmap, upgrade roadmap timeline, ROI analysis per upgrade,
uncertainty reduction forecast, and budget allocation chart. Tracks
progression from spend-based to supplier-specific methodology approaches.

Sections:
    1. Overall Maturity Summary
    2. Category-Level Heatmap (current vs target tier)
    3. Upgrade Roadmap Timeline
    4. ROI Analysis per Upgrade
    5. Uncertainty Reduction Forecast
    6. Budget Allocation
    7. Implementation Priorities

Output Formats:
    - Markdown (with provenance hash)
    - HTML (self-contained with inline CSS, growth green #1B8A4A theme)
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
# Formatting helpers
# ---------------------------------------------------------------------------

_TIER_LABELS = {
    1: "Spend-Based",
    2: "Average-Data",
    3: "Hybrid",
    4: "Supplier-Specific",
    5: "Verified Primary",
}

_TIER_COLORS = {
    1: "#EF4444",
    2: "#F59E0B",
    3: "#3B82F6",
    4: "#10B981",
    5: "#059669",
}


def _fmt_num(value: Optional[float], decimals: int = 1) -> str:
    """Format numeric value with thousands separators."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.{decimals}f}K"
    return f"{value:,.{decimals}f}"


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


def _tier_label(tier: Optional[int]) -> str:
    """Get tier label from tier number."""
    if tier is None:
        return "N/A"
    return _TIER_LABELS.get(tier, f"Tier {tier}")


class MaturityReportTemplate:
    """
    Maturity tier heatmap and upgrade roadmap template.

    Renders category-level maturity assessment with tier progression,
    upgrade ROI analysis, uncertainty reduction forecasts, and budget
    allocation recommendations. All outputs include SHA-256 provenance.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render operation.

    Example:
        >>> template = MaturityReportTemplate()
        >>> md = template.render_markdown(data)
        >>> html = template.render_html(data)
        >>> result = template.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MaturityReportTemplate."""
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

    # ==================================================================
    # PUBLIC RENDER METHODS
    # ==================================================================

    def render_markdown(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render maturity report as Markdown.

        Args:
            data: Validated maturity report data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Complete Markdown string with provenance hash.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overall_summary(data),
            self._md_category_heatmap(data),
            self._md_upgrade_roadmap(data),
            self._md_roi_analysis(data),
            self._md_uncertainty_forecast(data),
            self._md_budget_allocation(data),
            self._md_implementation_priorities(data),
            self._md_footer(data),
        ]
        return "\n\n".join(s for s in sections if s)

    def render_html(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Render maturity report as HTML.

        Args:
            data: Validated maturity report data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Self-contained HTML document string with inline CSS.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        body_parts: List[str] = [
            self._html_header(data),
            self._html_overall_summary(data),
            self._html_category_heatmap(data),
            self._html_upgrade_roadmap(data),
            self._html_roi_analysis(data),
            self._html_uncertainty_forecast(data),
            self._html_budget_allocation(data),
            self._html_implementation_priorities(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        return self._wrap_html(data, body)

    def render_json(
        self, data: Dict[str, Any], config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Render maturity report as JSON-serializable dict.

        Args:
            data: Validated maturity report data dict.
            config: Optional per-render configuration overrides.

        Returns:
            Structured dictionary for JSON serialization.
        """
        if config:
            self.config.update(config)
        self.generated_at = datetime.utcnow()
        provenance = self._compute_provenance(data)
        return {
            "template": "maturity_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "provenance_hash": provenance,
            "company_name": self._get_val(data, "company_name", ""),
            "reporting_year": self._get_val(data, "reporting_year"),
            "overall_summary": self._json_overall(data),
            "category_heatmap": self._json_heatmap(data),
            "upgrade_roadmap": data.get("upgrade_roadmap", []),
            "roi_analysis": data.get("roi_analysis", []),
            "uncertainty_forecast": data.get("uncertainty_forecast", []),
            "budget_allocation": data.get("budget_allocation", {}),
            "implementation_priorities": data.get("implementation_priorities", []),
        }

    # ==================================================================
    # MARKDOWN SECTIONS
    # ==================================================================

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            f"# Scope 3 Maturity Report - {company}\n\n"
            f"**Reporting Year:** {year} | "
            f"**Report Date:** {self._get_val(data, 'report_date', datetime.utcnow().strftime('%Y-%m-%d'))}\n\n"
            "---"
        )

    def _md_overall_summary(self, data: Dict[str, Any]) -> str:
        """Render Markdown overall maturity summary."""
        summary = data.get("overall_summary", {})
        avg_tier = summary.get("average_tier")
        target_tier = summary.get("target_tier")
        cats_at_target = summary.get("categories_at_target", 0)
        total_cats = summary.get("total_categories", 15)
        lines = [
            "## 1. Overall Maturity Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        if avg_tier is not None:
            lines.append(
                f"| Average Maturity Tier | {avg_tier:.1f} ({_tier_label(round(avg_tier))}) |"
            )
        if target_tier is not None:
            lines.append(f"| Target Tier | {_tier_label(target_tier)} |")
        lines.append(f"| Categories at Target | {cats_at_target} / {total_cats} |")
        gap = summary.get("maturity_gap")
        if gap is not None:
            lines.append(f"| Maturity Gap | {gap:.1f} tiers |")
        return "\n".join(lines)

    def _md_category_heatmap(self, data: Dict[str, Any]) -> str:
        """Render Markdown category maturity heatmap table."""
        categories = data.get("category_maturity", [])
        if not categories:
            return "## 2. Category Maturity Heatmap\n\nNo category data available."
        lines = [
            "## 2. Category Maturity Heatmap",
            "",
            "| Category | Current Tier | Target Tier | Gap | Status |",
            "|----------|-------------|------------|-----|--------|",
        ]
        for cat in categories:
            name = cat.get("category_name", "Unknown")
            num = cat.get("category_number", "?")
            cur = cat.get("current_tier", 0)
            tgt = cat.get("target_tier", 0)
            gap = tgt - cur
            status = "At Target" if gap <= 0 else f"{gap} tier(s) behind"
            lines.append(
                f"| Cat {num} - {name} | {_tier_label(cur)} | "
                f"{_tier_label(tgt)} | {gap} | {status} |"
            )
        return "\n".join(lines)

    def _md_upgrade_roadmap(self, data: Dict[str, Any]) -> str:
        """Render Markdown upgrade roadmap timeline."""
        roadmap = data.get("upgrade_roadmap", [])
        if not roadmap:
            return "## 3. Upgrade Roadmap\n\nNo roadmap data available."
        lines = [
            "## 3. Upgrade Roadmap Timeline",
            "",
            "| Phase | Category | From | To | Target Date | Effort |",
            "|-------|----------|------|----|-------------|--------|",
        ]
        for item in roadmap:
            phase = item.get("phase", "-")
            cat = item.get("category_name", "-")
            from_tier = _tier_label(item.get("from_tier"))
            to_tier = _tier_label(item.get("to_tier"))
            target_date = item.get("target_date", "-")
            effort = item.get("effort_level", "-")
            lines.append(
                f"| {phase} | {cat} | {from_tier} | {to_tier} | "
                f"{target_date} | {effort} |"
            )
        return "\n".join(lines)

    def _md_roi_analysis(self, data: Dict[str, Any]) -> str:
        """Render Markdown ROI analysis per upgrade."""
        roi_items = data.get("roi_analysis", [])
        if not roi_items:
            return "## 4. ROI Analysis\n\nNo ROI data available."
        lines = [
            "## 4. ROI Analysis per Upgrade",
            "",
            "| Category | Upgrade | Investment | Annual Savings | ROI | Payback |",
            "|----------|---------|-----------|---------------|-----|---------|",
        ]
        for item in roi_items:
            cat = item.get("category_name", "-")
            upgrade = item.get("upgrade_description", "-")
            invest = _fmt_currency(item.get("investment"))
            savings = _fmt_currency(item.get("annual_savings"))
            roi = item.get("roi_pct")
            roi_str = f"{roi:.0f}%" if roi is not None else "-"
            payback = item.get("payback_months")
            pb_str = f"{payback:.0f} mo" if payback is not None else "-"
            lines.append(
                f"| {cat} | {upgrade} | {invest} | {savings} | {roi_str} | {pb_str} |"
            )
        return "\n".join(lines)

    def _md_uncertainty_forecast(self, data: Dict[str, Any]) -> str:
        """Render Markdown uncertainty reduction forecast."""
        forecasts = data.get("uncertainty_forecast", [])
        if not forecasts:
            return "## 5. Uncertainty Reduction Forecast\n\nNo forecast data available."
        lines = [
            "## 5. Uncertainty Reduction Forecast",
            "",
            "| Category | Current +/- | Post-Upgrade +/- | Reduction |",
            "|----------|------------|------------------|-----------|",
        ]
        for item in forecasts:
            cat = item.get("category_name", "-")
            cur = item.get("current_uncertainty_pct")
            post = item.get("post_upgrade_uncertainty_pct")
            cur_str = f"{cur:.0f}%" if cur is not None else "-"
            post_str = f"{post:.0f}%" if post is not None else "-"
            red = ""
            if cur is not None and post is not None:
                red = f"{cur - post:.0f}pp"
            lines.append(f"| {cat} | {cur_str} | {post_str} | {red} |")
        return "\n".join(lines)

    def _md_budget_allocation(self, data: Dict[str, Any]) -> str:
        """Render Markdown budget allocation chart data."""
        budget = data.get("budget_allocation", {})
        if not budget:
            return "## 6. Budget Allocation\n\nNo budget data available."
        total = budget.get("total_budget")
        allocations = budget.get("allocations", [])
        lines = [
            "## 6. Budget Allocation",
            "",
        ]
        if total is not None:
            lines.append(f"**Total Budget:** {_fmt_currency(total)}")
            lines.append("")
        if allocations:
            lines.append("| Activity | Amount | % of Budget |")
            lines.append("|----------|--------|------------|")
            for alloc in allocations:
                activity = alloc.get("activity", "-")
                amount = _fmt_currency(alloc.get("amount"))
                pct = alloc.get("pct_of_budget")
                pct_str = f"{pct:.1f}%" if pct is not None else "-"
                lines.append(f"| {activity} | {amount} | {pct_str} |")
        return "\n".join(lines)

    def _md_implementation_priorities(self, data: Dict[str, Any]) -> str:
        """Render Markdown implementation priorities."""
        priorities = data.get("implementation_priorities", [])
        if not priorities:
            return "## 7. Implementation Priorities\n\nNo priorities defined."
        lines = [
            "## 7. Implementation Priorities",
            "",
        ]
        for i, pri in enumerate(priorities, 1):
            title = pri.get("title", "")
            desc = pri.get("description", "")
            impact = pri.get("impact", "-")
            effort = pri.get("effort", "-")
            lines.append(f"### Priority {i}: {title}")
            lines.append("")
            if desc:
                lines.append(desc)
                lines.append("")
            lines.append(f"- **Impact:** {impact}")
            lines.append(f"- **Effort:** {effort}")
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
        """Wrap body content in full HTML document with inline CSS."""
        company = self._get_val(data, "company_name", "Organization")
        year = self._get_val(data, "reporting_year", "")
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n<head>\n'
            '<meta charset="UTF-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f"<title>Maturity Report - {company} ({year})</title>\n"
            "<style>\n"
            ":root{--primary:#1B8A4A;--primary-light:#2EAD60;--accent:#10B981;"
            "--bg:#F0FDF4;--card-bg:#FFFFFF;--text:#1A1A2E;--text-muted:#6B7280;"
            "--border:#D1D5DB;--success:#059669;--warning:#F59E0B;--danger:#EF4444;}\n"
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
            "tr:nth-child(even){background:#F0FDF4;}\n"
            ".section{margin-bottom:2rem;background:var(--card-bg);"
            "padding:1.5rem;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.08);}\n"
            ".tier-1{background:#FEE2E2;color:#991B1B;}\n"
            ".tier-2{background:#FEF3C7;color:#92400E;}\n"
            ".tier-3{background:#DBEAFE;color:#1E3A8A;}\n"
            ".tier-4{background:#D1FAE5;color:#065F46;}\n"
            ".tier-5{background:#A7F3D0;color:#064E3B;}\n"
            ".at-target{color:var(--success);font-weight:700;}\n"
            ".behind{color:var(--warning);font-weight:700;}\n"
            ".priority-card{background:#F0FDF4;border-left:4px solid var(--primary);"
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
            f"<h1>Scope 3 Maturity Report &mdash; {company}</h1>\n"
            f"<p><strong>Reporting Year:</strong> {year} | "
            f"<strong>Pack:</strong> PACK-043 v{_MODULE_VERSION}</p>\n"
            "<hr>\n</div>"
        )

    def _html_overall_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML overall maturity summary."""
        summary = data.get("overall_summary", {})
        avg_tier = summary.get("average_tier")
        target_tier = summary.get("target_tier")
        cats_at_target = summary.get("categories_at_target", 0)
        total_cats = summary.get("total_categories", 15)
        rows = ""
        if avg_tier is not None:
            rows += (
                f"<tr><td>Average Maturity Tier</td>"
                f"<td>{avg_tier:.1f} ({_tier_label(round(avg_tier))})</td></tr>\n"
            )
        if target_tier is not None:
            rows += f"<tr><td>Target Tier</td><td>{_tier_label(target_tier)}</td></tr>\n"
        rows += f"<tr><td>Categories at Target</td><td>{cats_at_target} / {total_cats}</td></tr>\n"
        gap = summary.get("maturity_gap")
        if gap is not None:
            rows += f"<tr><td>Maturity Gap</td><td>{gap:.1f} tiers</td></tr>\n"
        return (
            '<div class="section">\n'
            "<h2>1. Overall Maturity Summary</h2>\n"
            "<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_category_heatmap(self, data: Dict[str, Any]) -> str:
        """Render HTML category heatmap with colored tier cells."""
        categories = data.get("category_maturity", [])
        if not categories:
            return ""
        rows = ""
        for cat in categories:
            name = cat.get("category_name", "Unknown")
            num = cat.get("category_number", "?")
            cur = cat.get("current_tier", 0)
            tgt = cat.get("target_tier", 0)
            gap = tgt - cur
            status_css = "at-target" if gap <= 0 else "behind"
            status_text = "At Target" if gap <= 0 else f"{gap} tier(s) behind"
            cur_css = f"tier-{cur}" if 1 <= cur <= 5 else ""
            tgt_css = f"tier-{tgt}" if 1 <= tgt <= 5 else ""
            rows += (
                f"<tr><td>Cat {num} - {name}</td>"
                f'<td class="{cur_css}">{_tier_label(cur)}</td>'
                f'<td class="{tgt_css}">{_tier_label(tgt)}</td>'
                f"<td>{gap}</td>"
                f'<td class="{status_css}">{status_text}</td></tr>\n'
            )
        return (
            '<div class="section">\n'
            "<h2>2. Category Maturity Heatmap</h2>\n"
            "<table><thead><tr><th>Category</th><th>Current Tier</th>"
            "<th>Target Tier</th><th>Gap</th><th>Status</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_upgrade_roadmap(self, data: Dict[str, Any]) -> str:
        """Render HTML upgrade roadmap timeline."""
        roadmap = data.get("upgrade_roadmap", [])
        if not roadmap:
            return ""
        rows = ""
        for item in roadmap:
            phase = item.get("phase", "-")
            cat = item.get("category_name", "-")
            from_tier = _tier_label(item.get("from_tier"))
            to_tier = _tier_label(item.get("to_tier"))
            target_date = item.get("target_date", "-")
            effort = item.get("effort_level", "-")
            rows += (
                f"<tr><td>{phase}</td><td>{cat}</td><td>{from_tier}</td>"
                f"<td>{to_tier}</td><td>{target_date}</td><td>{effort}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>3. Upgrade Roadmap Timeline</h2>\n"
            "<table><thead><tr><th>Phase</th><th>Category</th><th>From</th>"
            "<th>To</th><th>Target Date</th><th>Effort</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_roi_analysis(self, data: Dict[str, Any]) -> str:
        """Render HTML ROI analysis table."""
        roi_items = data.get("roi_analysis", [])
        if not roi_items:
            return ""
        rows = ""
        for item in roi_items:
            cat = item.get("category_name", "-")
            upgrade = item.get("upgrade_description", "-")
            invest = _fmt_currency(item.get("investment"))
            savings = _fmt_currency(item.get("annual_savings"))
            roi = item.get("roi_pct")
            roi_str = f"{roi:.0f}%" if roi is not None else "-"
            payback = item.get("payback_months")
            pb_str = f"{payback:.0f} mo" if payback is not None else "-"
            rows += (
                f"<tr><td>{cat}</td><td>{upgrade}</td><td>{invest}</td>"
                f"<td>{savings}</td><td>{roi_str}</td><td>{pb_str}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>4. ROI Analysis per Upgrade</h2>\n"
            "<table><thead><tr><th>Category</th><th>Upgrade</th><th>Investment</th>"
            "<th>Annual Savings</th><th>ROI</th><th>Payback</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_uncertainty_forecast(self, data: Dict[str, Any]) -> str:
        """Render HTML uncertainty reduction forecast."""
        forecasts = data.get("uncertainty_forecast", [])
        if not forecasts:
            return ""
        rows = ""
        for item in forecasts:
            cat = item.get("category_name", "-")
            cur = item.get("current_uncertainty_pct")
            post = item.get("post_upgrade_uncertainty_pct")
            cur_str = f"{cur:.0f}%" if cur is not None else "-"
            post_str = f"{post:.0f}%" if post is not None else "-"
            red = ""
            if cur is not None and post is not None:
                red = f"{cur - post:.0f}pp"
            rows += (
                f"<tr><td>{cat}</td><td>{cur_str}</td>"
                f"<td>{post_str}</td><td>{red}</td></tr>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>5. Uncertainty Reduction Forecast</h2>\n"
            "<table><thead><tr><th>Category</th><th>Current +/-</th>"
            "<th>Post-Upgrade +/-</th><th>Reduction</th></tr></thead>\n"
            f"<tbody>{rows}</tbody></table>\n</div>"
        )

    def _html_budget_allocation(self, data: Dict[str, Any]) -> str:
        """Render HTML budget allocation."""
        budget = data.get("budget_allocation", {})
        if not budget:
            return ""
        total = budget.get("total_budget")
        allocations = budget.get("allocations", [])
        total_html = ""
        if total is not None:
            total_html = f"<p><strong>Total Budget:</strong> {_fmt_currency(total)}</p>"
        rows = ""
        for alloc in allocations:
            activity = alloc.get("activity", "-")
            amount = _fmt_currency(alloc.get("amount"))
            pct = alloc.get("pct_of_budget")
            pct_str = f"{pct:.1f}%" if pct is not None else "-"
            rows += f"<tr><td>{activity}</td><td>{amount}</td><td>{pct_str}</td></tr>\n"
        table = ""
        if rows:
            table = (
                "<table><thead><tr><th>Activity</th><th>Amount</th>"
                f"<th>% of Budget</th></tr></thead><tbody>{rows}</tbody></table>"
            )
        return (
            '<div class="section">\n'
            "<h2>6. Budget Allocation</h2>\n"
            f"{total_html}\n{table}\n</div>"
        )

    def _html_implementation_priorities(self, data: Dict[str, Any]) -> str:
        """Render HTML implementation priorities."""
        priorities = data.get("implementation_priorities", [])
        if not priorities:
            return ""
        cards = ""
        for i, pri in enumerate(priorities, 1):
            title = pri.get("title", "")
            desc = pri.get("description", "")
            impact = pri.get("impact", "-")
            effort = pri.get("effort", "-")
            cards += (
                f'<div class="priority-card">\n'
                f"<h3>Priority {i}: {title}</h3>\n"
                f"<p>{desc}</p>\n"
                f"<p><strong>Impact:</strong> {impact} | "
                f"<strong>Effort:</strong> {effort}</p>\n</div>\n"
            )
        return (
            '<div class="section">\n'
            "<h2>7. Implementation Priorities</h2>\n"
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

    # ==================================================================
    # JSON HELPERS
    # ==================================================================

    def _json_overall(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build overall summary data."""
        summary = data.get("overall_summary", {})
        return {
            "average_tier": summary.get("average_tier"),
            "target_tier": summary.get("target_tier"),
            "categories_at_target": summary.get("categories_at_target", 0),
            "total_categories": summary.get("total_categories", 15),
            "maturity_gap": summary.get("maturity_gap"),
        }

    def _json_heatmap(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build category heatmap data."""
        categories = data.get("category_maturity", [])
        result = []
        for cat in categories:
            cur = cat.get("current_tier", 0)
            tgt = cat.get("target_tier", 0)
            result.append({
                "category_number": cat.get("category_number"),
                "category_name": cat.get("category_name"),
                "current_tier": cur,
                "current_tier_label": _tier_label(cur),
                "target_tier": tgt,
                "target_tier_label": _tier_label(tgt),
                "gap": tgt - cur,
                "at_target": cur >= tgt,
            })
        return result
