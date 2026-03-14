"""
EnterpriseDashboardTemplate - Multi-tenant overview dashboard for CSRD Enterprise Pack.

This module implements the enterprise dashboard template providing a comprehensive
multi-tenant overview with KPI cards, compliance heatmaps, emission trends,
alert feeds, tenant health scores, and quick-action panels.

Example:
    >>> template = EnterpriseDashboardTemplate()
    >>> data = {
    ...     "tenants": [...],
    ...     "kpis": {...},
    ...     "compliance_heatmap": [...],
    ...     "emission_trends": [...],
    ...     "alerts": [...],
    ... }
    >>> html = template.render_html(data)
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class EnterpriseDashboardTemplate:
    """
    Multi-tenant enterprise dashboard template.

    Provides a comprehensive overview across all tenants with compliance
    status, emission trends, health scores, and operational alerts.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    ESRS_STANDARDS = [
        "E1", "E2", "E3", "E4", "E5",
        "S1", "S2", "S3", "S4",
        "G1",
    ]

    RISK_LEVELS = ["critical", "high", "medium", "low"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EnterpriseDashboardTemplate.

        Args:
            config: Optional configuration dict with rendering preferences.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render dashboard as Markdown.

        Args:
            data: Dashboard data containing tenants, kpis, compliance_heatmap,
                  emission_trends, alerts, health_scores, and quick_actions.

        Returns:
            Complete Markdown string of the dashboard.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = []

        sections.append(self._render_md_header(data))
        sections.append(self._render_md_tenant_selector(data))
        sections.append(self._render_md_kpi_cards(data))
        sections.append(self._render_md_compliance_heatmap(data))
        sections.append(self._render_md_emission_trends(data))
        sections.append(self._render_md_alert_feed(data))
        sections.append(self._render_md_health_scores(data))
        sections.append(self._render_md_quick_actions(data))
        sections.append(self._render_md_footer(data))

        content = "\n\n".join(sections)
        provenance = self._generate_provenance_hash(content)
        sections.append(f"<!-- Provenance: {provenance} -->")
        return "\n\n".join(sections)

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render dashboard as self-contained HTML with inline CSS.

        Args:
            data: Dashboard data dict.

        Returns:
            Complete HTML string with inline styles.
        """
        self.generated_at = datetime.utcnow()
        css = self._build_css()
        body_parts: List[str] = []

        body_parts.append(self._render_html_header(data))
        body_parts.append(self._render_html_tenant_selector(data))
        body_parts.append(self._render_html_kpi_cards(data))
        body_parts.append(self._render_html_compliance_heatmap(data))
        body_parts.append(self._render_html_emission_trends(data))
        body_parts.append(self._render_html_alert_feed(data))
        body_parts.append(self._render_html_health_scores(data))
        body_parts.append(self._render_html_quick_actions(data))
        body_parts.append(self._render_html_footer(data))

        body_html = "\n".join(body_parts)
        provenance = self._generate_provenance_hash(body_html)

        html = (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n<head>\n"
            "<meta charset=\"UTF-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
            f"<title>Enterprise Dashboard</title>\n<style>\n{css}\n</style>\n"
            "</head>\n<body>\n"
            f"<div class=\"dashboard-container\">\n{body_html}\n</div>\n"
            f"<!-- Provenance: {provenance} -->\n"
            "</body>\n</html>"
        )
        return html

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render dashboard as structured JSON dict.

        Args:
            data: Dashboard data dict.

        Returns:
            Structured dict with all dashboard sections.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "enterprise_dashboard",
            "version": "1.0.0",
            "generated_at": self.generated_at.isoformat(),
            "header": self._build_json_header(data),
            "tenant_selector": self._build_json_tenant_selector(data),
            "kpi_cards": self._build_json_kpi_cards(data),
            "compliance_heatmap": self._build_json_compliance_heatmap(data),
            "emission_trends": self._build_json_emission_trends(data),
            "alert_feed": self._build_json_alert_feed(data),
            "health_scores": self._build_json_health_scores(data),
            "quick_actions": self._build_json_quick_actions(data),
        }
        provenance = self._generate_provenance_hash(json.dumps(result, default=str))
        result["provenance_hash"] = provenance
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _render_md_header(self, data: Dict[str, Any]) -> str:
        """Render Markdown header section."""
        title = data.get("title", "Enterprise CSRD Dashboard")
        org = data.get("organization", "")
        ts = self._format_date(self.generated_at)
        lines = [
            f"# {title}",
            f"**Organization:** {org}" if org else "",
            f"**Generated:** {ts}",
            "---",
        ]
        return "\n".join(line for line in lines if line)

    def _render_md_tenant_selector(self, data: Dict[str, Any]) -> str:
        """Render tenant selector as a Markdown table with search hint."""
        tenants: List[Dict[str, Any]] = data.get("tenants", [])
        if not tenants:
            return "## Tenant Selector\n\n_No tenants configured._"

        lines = [
            "## Tenant Selector",
            "",
            f"_Showing {len(tenants)} tenant(s). Filter by name or region._",
            "",
            "| # | Tenant | Region | Status | Users |",
            "|---|--------|--------|--------|-------|",
        ]
        for idx, t in enumerate(tenants, 1):
            name = t.get("name", "Unknown")
            region = t.get("region", "-")
            status = t.get("status", "active")
            users = t.get("active_users", 0)
            lines.append(f"| {idx} | {name} | {region} | {status} | {users} |")
        return "\n".join(lines)

    def _render_md_kpi_cards(self, data: Dict[str, Any]) -> str:
        """Render KPI summary cards in Markdown."""
        kpis = data.get("kpis", {})
        total_tenants = kpis.get("total_tenants", 0)
        active_users = kpis.get("active_users", 0)
        compliance_rate = kpis.get("compliance_rate", 0.0)
        emission_trend = kpis.get("emission_trend_pct", 0.0)
        trend_direction = "decrease" if emission_trend < 0 else "increase"

        lines = [
            "## Key Performance Indicators",
            "",
            f"| KPI | Value |",
            f"|-----|-------|",
            f"| Total Tenants | {self._format_number(total_tenants, 0)} |",
            f"| Active Users | {self._format_number(active_users, 0)} |",
            f"| Compliance Rate | {self._format_percentage(compliance_rate)} |",
            f"| Emission Trend (YoY) | {self._format_percentage(emission_trend)} ({trend_direction}) |",
        ]
        return "\n".join(lines)

    def _render_md_compliance_heatmap(self, data: Dict[str, Any]) -> str:
        """Render compliance heatmap as a Markdown table (entity x standard)."""
        heatmap: List[Dict[str, Any]] = data.get("compliance_heatmap", [])
        if not heatmap:
            return "## Compliance Heatmap\n\n_No compliance data available._"

        headers = ["Entity"] + self.ESRS_STANDARDS
        header_row = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"

        rows = [header_row, separator]
        for entity_row in heatmap:
            entity_name = entity_row.get("entity", "Unknown")
            scores = entity_row.get("scores", {})
            cells = [entity_name]
            for std in self.ESRS_STANDARDS:
                score = scores.get(std)
                if score is None:
                    cells.append("N/A")
                else:
                    cells.append(self._compliance_indicator(score))
            rows.append("| " + " | ".join(cells) + " |")

        return "## Compliance Heatmap\n\n" + "\n".join(rows)

    def _render_md_emission_trends(self, data: Dict[str, Any]) -> str:
        """Render emission trend data as a Markdown table."""
        trends: List[Dict[str, Any]] = data.get("emission_trends", [])
        if not trends:
            return "## Emission Trends\n\n_No trend data available._"

        lines = [
            "## Emission Trends (Scope 1/2/3 Stacked)",
            "",
            "| Period | Scope 1 (tCO2e) | Scope 2 (tCO2e) | Scope 3 (tCO2e) | Total (tCO2e) |",
            "|--------|-----------------|-----------------|-----------------|---------------|",
        ]
        for row in trends:
            period = row.get("period", "-")
            s1 = self._format_number(row.get("scope_1", 0))
            s2 = self._format_number(row.get("scope_2", 0))
            s3 = self._format_number(row.get("scope_3", 0))
            total = self._format_number(row.get("total", 0))
            lines.append(f"| {period} | {s1} | {s2} | {s3} | {total} |")

        return "\n".join(lines)

    def _render_md_alert_feed(self, data: Dict[str, Any]) -> str:
        """Render recent alerts as a Markdown list."""
        alerts: List[Dict[str, Any]] = data.get("alerts", [])
        if not alerts:
            return "## Alert Feed\n\n_No recent alerts._"

        lines = ["## Alert Feed", ""]
        for alert in alerts[:20]:
            severity = alert.get("severity", "info").upper()
            message = alert.get("message", "")
            timestamp = alert.get("timestamp", "")
            source = alert.get("source", "")
            lines.append(f"- **[{severity}]** {message} _{source}_ ({timestamp})")

        return "\n".join(lines)

    def _render_md_health_scores(self, data: Dict[str, Any]) -> str:
        """Render tenant health scores as a Markdown table."""
        health: List[Dict[str, Any]] = data.get("health_scores", [])
        if not health:
            return "## Tenant Health Scores\n\n_No health data available._"

        lines = [
            "## Tenant Health Scores",
            "",
            "| Tenant | Health Score | Data Quality | Timeliness | Completeness |",
            "|--------|-------------|-------------|------------|--------------|",
        ]
        for h in health:
            name = h.get("tenant", "Unknown")
            score = self._format_number(h.get("health_score", 0), 1)
            dq = self._format_percentage(h.get("data_quality", 0))
            tl = self._format_percentage(h.get("timeliness", 0))
            cp = self._format_percentage(h.get("completeness", 0))
            lines.append(f"| {name} | {score}/100 | {dq} | {tl} | {cp} |")

        return "\n".join(lines)

    def _render_md_quick_actions(self, data: Dict[str, Any]) -> str:
        """Render quick-action panel in Markdown."""
        actions: List[Dict[str, Any]] = data.get("quick_actions", [
            {"label": "Create Tenant", "action": "create_tenant"},
            {"label": "Run Report", "action": "run_report"},
            {"label": "View Alerts", "action": "view_alerts"},
        ])
        lines = ["## Quick Actions", ""]
        for act in actions:
            label = act.get("label", "Action")
            desc = act.get("description", "")
            lines.append(f"- **{label}**{': ' + desc if desc else ''}")
        return "\n".join(lines)

    def _render_md_footer(self, data: Dict[str, Any]) -> str:
        """Render Markdown footer with generation timestamp."""
        ts = self._format_date(self.generated_at)
        return f"---\n_Dashboard generated at {ts} | PACK-003 CSRD Enterprise_"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _build_css(self) -> str:
        """Build inline CSS for the dashboard."""
        return """
:root {
    --primary: #1a56db;
    --primary-light: #e1effe;
    --success: #057a55;
    --warning: #e3a008;
    --danger: #e02424;
    --info: #1c64f2;
    --bg: #f9fafb;
    --card-bg: #ffffff;
    --text: #1f2937;
    --text-muted: #6b7280;
    --border: #e5e7eb;
    --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: var(--font); background: var(--bg); color: var(--text); }
.dashboard-container { max-width: 1400px; margin: 0 auto; padding: 24px; }
.dashboard-header { background: var(--primary); color: #fff; padding: 24px 32px;
    border-radius: 12px; margin-bottom: 24px; }
.dashboard-header h1 { font-size: 28px; font-weight: 700; }
.dashboard-header .subtitle { opacity: 0.85; margin-top: 4px; }
.section { margin-bottom: 24px; }
.section-title { font-size: 20px; font-weight: 600; margin-bottom: 12px;
    padding-bottom: 8px; border-bottom: 2px solid var(--primary); }
.kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px; margin-bottom: 24px; }
.kpi-card { background: var(--card-bg); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px; text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.kpi-card .kpi-value { font-size: 32px; font-weight: 700; color: var(--primary); }
.kpi-card .kpi-label { font-size: 13px; color: var(--text-muted); margin-top: 4px; }
.kpi-card .kpi-trend { font-size: 12px; margin-top: 6px; }
.kpi-card .kpi-trend.positive { color: var(--success); }
.kpi-card .kpi-trend.negative { color: var(--danger); }
table { width: 100%; border-collapse: collapse; background: var(--card-bg);
    border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
th { background: var(--primary-light); color: var(--primary); font-weight: 600;
    padding: 10px 14px; text-align: left; font-size: 13px; }
td { padding: 10px 14px; border-bottom: 1px solid var(--border); font-size: 13px; }
tr:last-child td { border-bottom: none; }
.heatmap-cell { width: 48px; height: 28px; border-radius: 4px;
    display: inline-block; text-align: center; line-height: 28px;
    font-size: 11px; font-weight: 600; color: #fff; }
.heatmap-green { background: #057a55; }
.heatmap-yellow { background: #e3a008; color: #1f2937; }
.heatmap-red { background: #e02424; }
.heatmap-gray { background: #d1d5db; color: #6b7280; }
.alert-list { list-style: none; }
.alert-item { padding: 10px 14px; border-left: 4px solid var(--border);
    margin-bottom: 8px; background: var(--card-bg); border-radius: 0 6px 6px 0; }
.alert-item.critical { border-left-color: var(--danger); }
.alert-item.high { border-left-color: #f59e0b; }
.alert-item.medium { border-left-color: var(--warning); }
.alert-item.low { border-left-color: var(--info); }
.alert-item .alert-severity { font-weight: 700; font-size: 11px; text-transform: uppercase; }
.alert-item .alert-message { margin-top: 2px; }
.alert-item .alert-meta { font-size: 11px; color: var(--text-muted); margin-top: 4px; }
.health-gauge { display: inline-block; width: 60px; height: 60px;
    border-radius: 50%; border: 5px solid var(--border); position: relative;
    text-align: center; line-height: 50px; font-weight: 700; font-size: 14px; }
.health-gauge.excellent { border-color: var(--success); color: var(--success); }
.health-gauge.good { border-color: #10b981; color: #10b981; }
.health-gauge.fair { border-color: var(--warning); color: var(--warning); }
.health-gauge.poor { border-color: var(--danger); color: var(--danger); }
.actions-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; }
.action-btn { display: block; padding: 14px 20px; background: var(--primary);
    color: #fff; border-radius: 8px; text-align: center; text-decoration: none;
    font-weight: 600; font-size: 14px; cursor: pointer; border: none; }
.action-btn:hover { opacity: 0.9; }
.tenant-search { width: 100%; padding: 10px 14px; border: 1px solid var(--border);
    border-radius: 8px; font-size: 14px; margin-bottom: 12px; }
.footer { text-align: center; color: var(--text-muted); font-size: 12px;
    padding: 16px 0; margin-top: 24px; border-top: 1px solid var(--border); }
.stacked-bar { display: flex; height: 24px; border-radius: 4px; overflow: hidden; }
.stacked-bar .scope1 { background: #1a56db; }
.stacked-bar .scope2 { background: #057a55; }
.stacked-bar .scope3 { background: #e3a008; }
"""

    def _render_html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        title = data.get("title", "Enterprise CSRD Dashboard")
        org = data.get("organization", "")
        ts = self._format_date(self.generated_at)
        org_line = f"<div class=\"subtitle\">{org}</div>" if org else ""
        return (
            f"<div class=\"dashboard-header\">\n"
            f"  <h1>{self._escape_html(title)}</h1>\n"
            f"  {org_line}\n"
            f"  <div class=\"subtitle\">Generated: {ts}</div>\n"
            f"</div>"
        )

    def _render_html_tenant_selector(self, data: Dict[str, Any]) -> str:
        """Render HTML tenant selector with search input and table."""
        tenants: List[Dict[str, Any]] = data.get("tenants", [])
        rows = ""
        for idx, t in enumerate(tenants, 1):
            name = self._escape_html(t.get("name", "Unknown"))
            region = self._escape_html(t.get("region", "-"))
            status = t.get("status", "active")
            users = t.get("active_users", 0)
            status_color = "var(--success)" if status == "active" else "var(--text-muted)"
            rows += (
                f"<tr><td>{idx}</td><td><strong>{name}</strong></td>"
                f"<td>{region}</td>"
                f"<td style=\"color:{status_color};font-weight:600\">{status}</td>"
                f"<td>{users}</td></tr>\n"
            )
        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Tenant Selector</h2>\n"
            "  <input type=\"text\" class=\"tenant-search\" "
            "placeholder=\"Search tenants by name or region...\">\n"
            "  <table>\n"
            "    <thead><tr><th>#</th><th>Tenant</th><th>Region</th>"
            "<th>Status</th><th>Users</th></tr></thead>\n"
            f"    <tbody>{rows}</tbody>\n"
            "  </table>\n"
            "</div>"
        )

    def _render_html_kpi_cards(self, data: Dict[str, Any]) -> str:
        """Render HTML KPI cards grid."""
        kpis = data.get("kpis", {})
        cards_data = [
            {
                "label": "Total Tenants",
                "value": self._format_number(kpis.get("total_tenants", 0), 0),
                "trend": None,
            },
            {
                "label": "Active Users",
                "value": self._format_number(kpis.get("active_users", 0), 0),
                "trend": None,
            },
            {
                "label": "Compliance Rate",
                "value": self._format_percentage(kpis.get("compliance_rate", 0)),
                "trend": kpis.get("compliance_trend_pct"),
            },
            {
                "label": "Emission Trend (YoY)",
                "value": self._format_percentage(kpis.get("emission_trend_pct", 0)),
                "trend": kpis.get("emission_trend_pct"),
            },
        ]

        cards_html = ""
        for card in cards_data:
            trend_html = ""
            if card["trend"] is not None:
                cls = "positive" if card["trend"] < 0 else "negative"
                arrow = "v" if card["trend"] < 0 else "^"
                trend_html = (
                    f"<div class=\"kpi-trend {cls}\">"
                    f"{arrow} {self._format_percentage(abs(card['trend']))}"
                    f"</div>"
                )
            cards_html += (
                f"<div class=\"kpi-card\">\n"
                f"  <div class=\"kpi-value\">{card['value']}</div>\n"
                f"  <div class=\"kpi-label\">{card['label']}</div>\n"
                f"  {trend_html}\n"
                f"</div>\n"
            )
        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Key Performance Indicators</h2>\n"
            f"  <div class=\"kpi-grid\">{cards_html}</div>\n"
            "</div>"
        )

    def _render_html_compliance_heatmap(self, data: Dict[str, Any]) -> str:
        """Render HTML compliance heatmap table."""
        heatmap: List[Dict[str, Any]] = data.get("compliance_heatmap", [])
        if not heatmap:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Compliance Heatmap</h2>\n"
                "  <p>No compliance data available.</p>\n"
                "</div>"
            )

        header_cells = "<th>Entity</th>" + "".join(
            f"<th>{std}</th>" for std in self.ESRS_STANDARDS
        )
        rows = ""
        for entity_row in heatmap:
            entity_name = self._escape_html(entity_row.get("entity", "Unknown"))
            scores = entity_row.get("scores", {})
            cells = f"<td><strong>{entity_name}</strong></td>"
            for std in self.ESRS_STANDARDS:
                score = scores.get(std)
                cells += f"<td>{self._heatmap_cell_html(score)}</td>"
            rows += f"<tr>{cells}</tr>\n"

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Compliance Heatmap (ESRS Standards)</h2>\n"
            f"  <table><thead><tr>{header_cells}</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_emission_trends(self, data: Dict[str, Any]) -> str:
        """Render HTML emission trend table with stacked bars."""
        trends: List[Dict[str, Any]] = data.get("emission_trends", [])
        if not trends:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Emission Trends</h2>\n"
                "  <p>No trend data available.</p>\n"
                "</div>"
            )

        max_total = max((r.get("total", 1) for r in trends), default=1)

        rows = ""
        for row in trends:
            period = row.get("period", "-")
            s1 = row.get("scope_1", 0)
            s2 = row.get("scope_2", 0)
            s3 = row.get("scope_3", 0)
            total = row.get("total", 0)
            pct1 = (s1 / total * 100) if total else 0
            pct2 = (s2 / total * 100) if total else 0
            pct3 = (s3 / total * 100) if total else 0
            bar_width = (total / max_total * 100) if max_total else 0
            bar = (
                f"<div class=\"stacked-bar\" style=\"width:{bar_width:.0f}%\">"
                f"<div class=\"scope1\" style=\"width:{pct1:.1f}%\"></div>"
                f"<div class=\"scope2\" style=\"width:{pct2:.1f}%\"></div>"
                f"<div class=\"scope3\" style=\"width:{pct3:.1f}%\"></div>"
                f"</div>"
            )
            rows += (
                f"<tr><td>{period}</td><td>{self._format_number(s1)}</td>"
                f"<td>{self._format_number(s2)}</td>"
                f"<td>{self._format_number(s3)}</td>"
                f"<td>{self._format_number(total)}</td><td>{bar}</td></tr>\n"
            )

        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Emission Trends (Scope 1/2/3)</h2>\n"
            "  <table><thead><tr><th>Period</th><th>Scope 1</th><th>Scope 2</th>"
            "<th>Scope 3</th><th>Total</th><th>Distribution</th></tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_alert_feed(self, data: Dict[str, Any]) -> str:
        """Render HTML alert feed."""
        alerts: List[Dict[str, Any]] = data.get("alerts", [])
        if not alerts:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Alert Feed</h2>\n"
                "  <p>No recent alerts.</p>\n"
                "</div>"
            )

        items = ""
        for alert in alerts[:20]:
            severity = alert.get("severity", "info")
            message = self._escape_html(alert.get("message", ""))
            source = self._escape_html(alert.get("source", ""))
            timestamp = alert.get("timestamp", "")
            items += (
                f"<li class=\"alert-item {severity}\">\n"
                f"  <span class=\"alert-severity\">{severity}</span>\n"
                f"  <div class=\"alert-message\">{message}</div>\n"
                f"  <div class=\"alert-meta\">{source} | {timestamp}</div>\n"
                f"</li>\n"
            )
        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Alert Feed</h2>\n"
            f"  <ul class=\"alert-list\">{items}</ul>\n"
            "</div>"
        )

    def _render_html_health_scores(self, data: Dict[str, Any]) -> str:
        """Render HTML tenant health scores with gauge indicators."""
        health: List[Dict[str, Any]] = data.get("health_scores", [])
        if not health:
            return (
                "<div class=\"section\">\n"
                "  <h2 class=\"section-title\">Tenant Health Scores</h2>\n"
                "  <p>No health data available.</p>\n"
                "</div>"
            )

        rows = ""
        for h in health:
            name = self._escape_html(h.get("tenant", "Unknown"))
            score = h.get("health_score", 0)
            gauge_cls = self._health_gauge_class(score)
            dq = self._format_percentage(h.get("data_quality", 0))
            tl = self._format_percentage(h.get("timeliness", 0))
            cp = self._format_percentage(h.get("completeness", 0))
            rows += (
                f"<tr><td><strong>{name}</strong></td>"
                f"<td><div class=\"health-gauge {gauge_cls}\">"
                f"{self._format_number(score, 0)}</div></td>"
                f"<td>{dq}</td><td>{tl}</td><td>{cp}</td></tr>\n"
            )
        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Tenant Health Scores</h2>\n"
            "  <table><thead><tr><th>Tenant</th><th>Health</th>"
            "<th>Data Quality</th><th>Timeliness</th><th>Completeness</th>"
            "</tr></thead>\n"
            f"  <tbody>{rows}</tbody></table>\n"
            "</div>"
        )

    def _render_html_quick_actions(self, data: Dict[str, Any]) -> str:
        """Render HTML quick-action buttons."""
        actions: List[Dict[str, Any]] = data.get("quick_actions", [
            {"label": "Create Tenant", "action": "create_tenant"},
            {"label": "Run Report", "action": "run_report"},
            {"label": "View Alerts", "action": "view_alerts"},
        ])
        buttons = ""
        for act in actions:
            label = self._escape_html(act.get("label", "Action"))
            action = act.get("action", "#")
            buttons += (
                f"<button class=\"action-btn\" data-action=\"{action}\">"
                f"{label}</button>\n"
            )
        return (
            "<div class=\"section\">\n"
            "  <h2 class=\"section-title\">Quick Actions</h2>\n"
            f"  <div class=\"actions-grid\">{buttons}</div>\n"
            "</div>"
        )

    def _render_html_footer(self, data: Dict[str, Any]) -> str:
        """Render HTML footer."""
        ts = self._format_date(self.generated_at)
        return (
            f"<div class=\"footer\">"
            f"Dashboard generated at {ts} | PACK-003 CSRD Enterprise"
            f"</div>"
        )

    # ------------------------------------------------------------------
    # JSON section builders
    # ------------------------------------------------------------------

    def _build_json_header(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON header section."""
        return {
            "title": data.get("title", "Enterprise CSRD Dashboard"),
            "organization": data.get("organization", ""),
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }

    def _build_json_tenant_selector(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON tenant selector section."""
        tenants = data.get("tenants", [])
        return {
            "total_tenants": len(tenants),
            "tenants": [
                {
                    "name": t.get("name", "Unknown"),
                    "region": t.get("region", "-"),
                    "status": t.get("status", "active"),
                    "active_users": t.get("active_users", 0),
                }
                for t in tenants
            ],
        }

    def _build_json_kpi_cards(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON KPI cards section."""
        kpis = data.get("kpis", {})
        return {
            "total_tenants": kpis.get("total_tenants", 0),
            "active_users": kpis.get("active_users", 0),
            "compliance_rate": kpis.get("compliance_rate", 0.0),
            "emission_trend_pct": kpis.get("emission_trend_pct", 0.0),
        }

    def _build_json_compliance_heatmap(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build JSON compliance heatmap section."""
        heatmap = data.get("compliance_heatmap", [])
        return [
            {
                "entity": row.get("entity", "Unknown"),
                "scores": {
                    std: row.get("scores", {}).get(std)
                    for std in self.ESRS_STANDARDS
                },
            }
            for row in heatmap
        ]

    def _build_json_emission_trends(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build JSON emission trends section."""
        return data.get("emission_trends", [])

    def _build_json_alert_feed(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON alert feed section."""
        return data.get("alerts", [])[:20]

    def _build_json_health_scores(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build JSON health scores section."""
        return data.get("health_scores", [])

    def _build_json_quick_actions(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build JSON quick actions section."""
        return data.get("quick_actions", [
            {"label": "Create Tenant", "action": "create_tenant"},
            {"label": "Run Report", "action": "run_report"},
            {"label": "View Alerts", "action": "view_alerts"},
        ])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_provenance_hash(content: str) -> str:
        """Generate SHA-256 provenance hash of rendered content.

        Args:
            content: The rendered content string.

        Returns:
            Hexadecimal SHA-256 hash string.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _format_number(value: Union[int, float], decimals: int = 2) -> str:
        """Format a numeric value with thousands separator.

        Args:
            value: Numeric value to format.
            decimals: Number of decimal places.

        Returns:
            Formatted string.
        """
        if decimals == 0:
            return f"{int(value):,}"
        return f"{value:,.{decimals}f}"

    @staticmethod
    def _format_percentage(value: Union[int, float]) -> str:
        """Format a value as a percentage string.

        Args:
            value: Numeric value (already as percentage, e.g. 85.5 for 85.5%).

        Returns:
            Formatted percentage string.
        """
        return f"{value:.1f}%"

    @staticmethod
    def _format_date(dt: Optional[datetime]) -> str:
        """Format a datetime as ISO-style string.

        Args:
            dt: Datetime object to format.

        Returns:
            Formatted date string.
        """
        if dt is None:
            return "N/A"
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters.

        Args:
            text: Raw text string.

        Returns:
            HTML-safe string.
        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )

    @staticmethod
    def _compliance_indicator(score: float) -> str:
        """Return a text indicator for compliance score.

        Args:
            score: Compliance score 0-100.

        Returns:
            Text indicator with score.
        """
        if score >= 80:
            return f"PASS ({score:.0f})"
        elif score >= 50:
            return f"PARTIAL ({score:.0f})"
        else:
            return f"FAIL ({score:.0f})"

    @staticmethod
    def _heatmap_cell_html(score: Optional[float]) -> str:
        """Return an HTML heatmap cell for a compliance score.

        Args:
            score: Compliance score 0-100 or None.

        Returns:
            HTML span element with color-coded score.
        """
        if score is None:
            return "<span class=\"heatmap-cell heatmap-gray\">N/A</span>"
        if score >= 80:
            cls = "heatmap-green"
        elif score >= 50:
            cls = "heatmap-yellow"
        else:
            cls = "heatmap-red"
        return f"<span class=\"heatmap-cell {cls}\">{score:.0f}</span>"

    @staticmethod
    def _health_gauge_class(score: float) -> str:
        """Return CSS class for health gauge based on score.

        Args:
            score: Health score 0-100.

        Returns:
            CSS class name.
        """
        if score >= 90:
            return "excellent"
        elif score >= 70:
            return "good"
        elif score >= 50:
            return "fair"
        else:
            return "poor"
