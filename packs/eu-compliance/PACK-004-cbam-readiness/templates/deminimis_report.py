"""
DeMinimisReportTemplate - CBAM de minimis threshold tracking template.

This module implements the de minimis threshold tracking report for CBAM
compliance. Under CBAM, consignments below certain thresholds may be exempt.
This template tracks cumulative imports against the 50-tonne per sector group
threshold, alerts when approaching limits, and projects year-end status.

Example:
    >>> template = DeMinimisReportTemplate()
    >>> data = {"sector_groups": [...], "alert_status": {...}, ...}
    >>> html = template.render_html(data)
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


class DeMinimisReportTemplate:
    """
    CBAM de minimis threshold tracking report template.

    Generates formatted reports tracking cumulative imports against the
    50-tonne de minimis threshold per sector group, with progress bars,
    monthly trends, alerts, exemptions, projections, and year comparisons.

    Attributes:
        config: Optional configuration dictionary.
        generated_at: Timestamp of report generation.
    """

    DEFAULT_THRESHOLD_TONNES: float = 50.0

    ALERT_LEVELS: Dict[str, Dict[str, Any]] = {
        "safe": {"label": "Safe", "color": "#2ecc71", "min_pct": 0.0},
        "approaching": {"label": "Approaching", "color": "#f39c12", "min_pct": 70.0},
        "critical": {"label": "Critical", "color": "#e74c3c", "min_pct": 90.0},
        "exceeded": {"label": "Exceeded", "color": "#c0392b", "min_pct": 100.0},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize DeMinimisReportTemplate.

        Args:
            config: Optional configuration dictionary. Supported keys:
                - threshold_tonnes (float): Override default 50t threshold.
                - alert_pct_approaching (float): % threshold for approaching alert.
                - alert_pct_critical (float): % threshold for critical alert.
        """
        self.config = config or {}
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public render methods
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """
        Render the de minimis report as Markdown.

        Args:
            data: Report data dictionary containing:
                - sector_groups (list[dict]): sector, cumulative_tonnes, threshold
                - monthly_imports (list[dict]): month, sector, tonnes
                - alert_status (list[dict]): sector alerts
                - exemptions (list[dict]): exempt sectors with reason
                - projection (list[dict]): year-end projections
                - historical (dict): this_year vs last_year comparison

        Returns:
            Formatted Markdown string.
        """
        sections: List[str] = []

        sections.append(self._md_header())
        sections.append(self._md_threshold_status(data))
        sections.append(self._md_progress_bars(data))
        sections.append(self._md_monthly_trend(data))
        sections.append(self._md_alert_status(data))
        sections.append(self._md_exemptions(data))
        sections.append(self._md_projection(data))
        sections.append(self._md_historical_comparison(data))
        sections.append(self._md_provenance_footer())

        content = "\n\n".join(sections)
        provenance_hash = self._generate_provenance_hash(content)
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the de minimis report as self-contained HTML.

        Args:
            data: Report data dictionary (same schema as render_markdown).

        Returns:
            Complete HTML document string with inline CSS.
        """
        sections: List[str] = []

        sections.append(self._html_header())
        sections.append(self._html_threshold_status(data))
        sections.append(self._html_progress_bars(data))
        sections.append(self._html_monthly_trend(data))
        sections.append(self._html_alert_status(data))
        sections.append(self._html_exemptions(data))
        sections.append(self._html_projection(data))
        sections.append(self._html_historical_comparison(data))

        body = "\n".join(sections)
        provenance_hash = self._generate_provenance_hash(body)

        return self._wrap_html(
            title="CBAM De Minimis Threshold Report",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the de minimis report as a structured dict.

        Args:
            data: Report data dictionary (same schema as render_markdown).

        Returns:
            Dictionary with all report sections and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "cbam_deminimis_report",
            "generated_at": self.generated_at,
            "threshold_tonnes": self._get_threshold(),
            "threshold_status": self._json_threshold_status(data),
            "monthly_imports": self._json_monthly_trend(data),
            "alert_status": self._json_alert_status(data),
            "exemptions": self._json_exemptions(data),
            "projection": self._json_projection(data),
            "historical_comparison": self._json_historical_comparison(data),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._generate_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown section builders
    # ------------------------------------------------------------------ #

    def _md_header(self) -> str:
        """Build Markdown header."""
        threshold = self._get_threshold()
        return (
            "# CBAM De Minimis Threshold Report\n\n"
            f"**Threshold:** {self._format_number(threshold)} tonnes per sector group\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_threshold_status(self, data: Dict[str, Any]) -> str:
        """Build Markdown threshold status by sector group."""
        sectors: List[Dict[str, Any]] = data.get("sector_groups", [])
        threshold = self._get_threshold()

        header = (
            "## Threshold Status by Sector Group\n\n"
            "| Sector Group | Cumulative (tonnes) | Threshold (tonnes) | "
            "Usage (%) | Status |\n"
            "|--------------|--------------------|--------------------|"
            "----------|--------|\n"
        )

        rows: List[str] = []
        for sg in sectors:
            cumulative = sg.get("cumulative_tonnes", 0.0)
            sector_threshold = sg.get("threshold_tonnes", threshold)
            usage_pct = (cumulative / sector_threshold * 100) if sector_threshold > 0 else 0.0
            alert = self._get_alert_level(usage_pct)

            rows.append(
                f"| {sg.get('sector', '')} | "
                f"{self._format_number(cumulative)} | "
                f"{self._format_number(sector_threshold)} | "
                f"{self._format_percentage(usage_pct)} | "
                f"{alert} |"
            )

        return header + "\n".join(rows)

    def _md_progress_bars(self, data: Dict[str, Any]) -> str:
        """Build Markdown progress bars for each sector group."""
        sectors: List[Dict[str, Any]] = data.get("sector_groups", [])
        threshold = self._get_threshold()

        section = "## Usage Progress\n\n"

        for sg in sectors:
            cumulative = sg.get("cumulative_tonnes", 0.0)
            sector_threshold = sg.get("threshold_tonnes", threshold)
            usage_pct = (cumulative / sector_threshold * 100) if sector_threshold > 0 else 0.0
            bar_len = min(int(usage_pct / 5), 20)
            bar = "#" * bar_len + "-" * (20 - bar_len)

            section += (
                f"**{sg.get('sector', '')}:** [{bar}] "
                f"{self._format_percentage(usage_pct)} "
                f"({self._format_number(cumulative)}/{self._format_number(sector_threshold)} t)\n\n"
            )

        return section.rstrip()

    def _md_monthly_trend(self, data: Dict[str, Any]) -> str:
        """Build Markdown monthly import trend table."""
        monthly: List[Dict[str, Any]] = data.get("monthly_imports", [])

        if not monthly:
            return "## Monthly Import Trend\n\n*No monthly data available.*"

        # Get unique sectors
        sectors = sorted(set(m.get("sector", "") for m in monthly))

        header = "## Monthly Import Trend\n\n| Month |"
        for s in sectors:
            header += f" {s} (t) |"
        header += "\n|-------|"
        for _ in sectors:
            header += "------|"
        header += "\n"

        # Group by month
        months: Dict[str, Dict[str, float]] = {}
        for m in monthly:
            month = m.get("month", "")
            sector = m.get("sector", "")
            if month not in months:
                months[month] = {}
            months[month][sector] = m.get("tonnes", 0.0)

        rows: List[str] = []
        for month in sorted(months.keys()):
            row = f"| {month} |"
            for s in sectors:
                val = months[month].get(s, 0.0)
                row += f" {self._format_number(val)} |"
            rows.append(row)

        return header + "\n".join(rows)

    def _md_alert_status(self, data: Dict[str, Any]) -> str:
        """Build Markdown alert status section."""
        alerts: List[Dict[str, Any]] = data.get("alert_status", [])

        if not alerts:
            return "## Alert Status\n\n*No active alerts.*"

        section = "## Alert Status\n\n"
        for alert in alerts:
            level = alert.get("level", "safe")
            section += (
                f"- **{alert.get('sector', '')}**: {level.upper()} - "
                f"{alert.get('message', '')}\n"
            )

        return section.rstrip()

    def _md_exemptions(self, data: Dict[str, Any]) -> str:
        """Build Markdown exemption status section."""
        exemptions: List[Dict[str, Any]] = data.get("exemptions", [])

        if not exemptions:
            return "## Exemption Status\n\n*No sector exemptions currently active.*"

        header = (
            "## Exemption Status\n\n"
            "| Sector Group | Exemption Status | Reason |\n"
            "|--------------|------------------|--------|\n"
        )

        rows: List[str] = []
        for ex in exemptions:
            status = "EXEMPT" if ex.get("exempt", False) else "NOT EXEMPT"
            rows.append(
                f"| {ex.get('sector', '')} | {status} | {ex.get('reason', 'N/A')} |"
            )

        return header + "\n".join(rows)

    def _md_projection(self, data: Dict[str, Any]) -> str:
        """Build Markdown year-end projection section."""
        projections: List[Dict[str, Any]] = data.get("projection", [])
        threshold = self._get_threshold()

        if not projections:
            return "## Year-End Projection\n\n*No projection data available.*"

        header = (
            "## Year-End Projection\n\n"
            "| Sector Group | Current (tonnes) | Projected Year-End (tonnes) | "
            "Will Exceed? | Projected Usage (%) |\n"
            "|--------------|------------------|-----------------------------|"
            "--------------|---------------------|\n"
        )

        rows: List[str] = []
        for proj in projections:
            current = proj.get("current_tonnes", 0.0)
            projected = proj.get("projected_year_end_tonnes", 0.0)
            sector_threshold = proj.get("threshold_tonnes", threshold)
            will_exceed = projected > sector_threshold
            usage_pct = (projected / sector_threshold * 100) if sector_threshold > 0 else 0.0

            rows.append(
                f"| {proj.get('sector', '')} | "
                f"{self._format_number(current)} | "
                f"{self._format_number(projected)} | "
                f"{'YES' if will_exceed else 'No'} | "
                f"{self._format_percentage(usage_pct)} |"
            )

        return header + "\n".join(rows)

    def _md_historical_comparison(self, data: Dict[str, Any]) -> str:
        """Build Markdown historical comparison section."""
        historical: Dict[str, Any] = data.get("historical", {})

        current_year = historical.get("current_year", "N/A")
        previous_year = historical.get("previous_year", "N/A")
        comparisons: List[Dict[str, Any]] = historical.get("comparisons", [])

        if not comparisons:
            return "## Historical Comparison\n\n*No historical data available.*"

        header = (
            "## Historical Comparison\n\n"
            f"| Sector Group | {previous_year} (tonnes) | "
            f"{current_year} (tonnes) | Change |\n"
            f"|--------------|---------------------|"
            f"---------------------|--------|\n"
        )

        rows: List[str] = []
        for comp in comparisons:
            prev = comp.get("previous_tonnes", 0.0)
            curr = comp.get("current_tonnes", 0.0)
            change = curr - prev
            change_pct = (change / prev * 100) if prev > 0 else 0.0

            rows.append(
                f"| {comp.get('sector', '')} | "
                f"{self._format_number(prev)} | "
                f"{self._format_number(curr)} | "
                f"{'+'if change >= 0 else ''}{self._format_number(change)} "
                f"({'+'if change_pct >= 0 else ''}{self._format_percentage(change_pct)}) |"
            )

        return header + "\n".join(rows)

    def _md_provenance_footer(self) -> str:
        """Build Markdown provenance footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            f"*Template: DeMinimisReportTemplate v1.0*"
        )

    # ------------------------------------------------------------------ #
    #  HTML section builders
    # ------------------------------------------------------------------ #

    def _html_header(self) -> str:
        """Build HTML header."""
        threshold = self._get_threshold()
        return (
            '<div class="report-header">'
            '<h1>CBAM De Minimis Threshold Report</h1>'
            f'<div class="header-meta">'
            f'<div class="meta-item">Threshold: {self._format_number(threshold)} tonnes per sector</div>'
            f'<div class="meta-item">Generated: {self.generated_at}</div>'
            '</div></div>'
        )

    def _html_threshold_status(self, data: Dict[str, Any]) -> str:
        """Build HTML threshold status table."""
        sectors: List[Dict[str, Any]] = data.get("sector_groups", [])
        threshold = self._get_threshold()

        rows_html = ""
        for sg in sectors:
            cumulative = sg.get("cumulative_tonnes", 0.0)
            sector_threshold = sg.get("threshold_tonnes", threshold)
            usage_pct = (cumulative / sector_threshold * 100) if sector_threshold > 0 else 0.0
            alert = self._get_alert_level(usage_pct)
            color = self._get_alert_color(usage_pct)

            rows_html += (
                f'<tr><td>{sg.get("sector", "")}</td>'
                f'<td class="num">{self._format_number(cumulative)}</td>'
                f'<td class="num">{self._format_number(sector_threshold)}</td>'
                f'<td class="num">{self._format_percentage(usage_pct)}</td>'
                f'<td><span class="status-badge" style="background:{color}">'
                f'{alert}</span></td></tr>'
            )

        return (
            '<div class="section"><h2>Threshold Status by Sector Group</h2>'
            '<table><thead><tr>'
            '<th>Sector Group</th><th>Cumulative (t)</th>'
            '<th>Threshold (t)</th><th>Usage (%)</th><th>Status</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_progress_bars(self, data: Dict[str, Any]) -> str:
        """Build HTML progress bars for each sector group."""
        sectors: List[Dict[str, Any]] = data.get("sector_groups", [])
        threshold = self._get_threshold()

        bars_html = ""
        for sg in sectors:
            cumulative = sg.get("cumulative_tonnes", 0.0)
            sector_threshold = sg.get("threshold_tonnes", threshold)
            usage_pct = (cumulative / sector_threshold * 100) if sector_threshold > 0 else 0.0
            color = self._get_alert_color(usage_pct)
            width = min(usage_pct, 100)

            bars_html += (
                f'<div class="progress-item">'
                f'<div class="progress-label">{sg.get("sector", "")}</div>'
                f'<div class="progress-bar">'
                f'<div class="progress-fill" style="width:{width}%;background:{color}"></div>'
                f'</div>'
                f'<div class="progress-value">{self._format_percentage(usage_pct)}</div>'
                f'</div>'
            )

        return (
            f'<div class="section"><h2>Usage Progress</h2>'
            f'<div class="progress-container">{bars_html}</div></div>'
        )

    def _html_monthly_trend(self, data: Dict[str, Any]) -> str:
        """Build HTML monthly import trend table."""
        monthly: List[Dict[str, Any]] = data.get("monthly_imports", [])

        if not monthly:
            return (
                '<div class="section"><h2>Monthly Import Trend</h2>'
                '<p class="note">No monthly data available.</p></div>'
            )

        sectors = sorted(set(m.get("sector", "") for m in monthly))

        headers_html = "".join(f'<th>{s} (t)</th>' for s in sectors)

        months: Dict[str, Dict[str, float]] = {}
        for m in monthly:
            month = m.get("month", "")
            sector = m.get("sector", "")
            if month not in months:
                months[month] = {}
            months[month][sector] = m.get("tonnes", 0.0)

        rows_html = ""
        for month in sorted(months.keys()):
            cells = ""
            for s in sectors:
                val = months[month].get(s, 0.0)
                cells += f'<td class="num">{self._format_number(val)}</td>'
            rows_html += f'<tr><td>{month}</td>{cells}</tr>'

        return (
            '<div class="section"><h2>Monthly Import Trend</h2>'
            f'<table><thead><tr><th>Month</th>{headers_html}'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_alert_status(self, data: Dict[str, Any]) -> str:
        """Build HTML alert status section."""
        alerts: List[Dict[str, Any]] = data.get("alert_status", [])

        if not alerts:
            return (
                '<div class="section"><h2>Alert Status</h2>'
                '<p class="note">No active alerts.</p></div>'
            )

        items_html = ""
        for alert in alerts:
            level = alert.get("level", "safe")
            color = self.ALERT_LEVELS.get(level, {}).get("color", "#95a5a6")

            items_html += (
                f'<div class="alert-item" style="border-left:4px solid {color}">'
                f'<span class="status-badge" style="background:{color}">'
                f'{level.upper()}</span> '
                f'<strong>{alert.get("sector", "")}</strong>: '
                f'{alert.get("message", "")}</div>'
            )

        return f'<div class="section"><h2>Alert Status</h2>{items_html}</div>'

    def _html_exemptions(self, data: Dict[str, Any]) -> str:
        """Build HTML exemption status section."""
        exemptions: List[Dict[str, Any]] = data.get("exemptions", [])

        if not exemptions:
            return (
                '<div class="section"><h2>Exemption Status</h2>'
                '<p class="note">No sector exemptions currently active.</p></div>'
            )

        rows_html = ""
        for ex in exemptions:
            exempt = ex.get("exempt", False)
            color = "#2ecc71" if exempt else "#95a5a6"
            label = "EXEMPT" if exempt else "NOT EXEMPT"

            rows_html += (
                f'<tr><td>{ex.get("sector", "")}</td>'
                f'<td><span class="status-badge" style="background:{color}">'
                f'{label}</span></td>'
                f'<td>{ex.get("reason", "N/A")}</td></tr>'
            )

        return (
            '<div class="section"><h2>Exemption Status</h2>'
            '<table><thead><tr>'
            '<th>Sector Group</th><th>Status</th><th>Reason</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_projection(self, data: Dict[str, Any]) -> str:
        """Build HTML year-end projection section."""
        projections: List[Dict[str, Any]] = data.get("projection", [])
        threshold = self._get_threshold()

        if not projections:
            return (
                '<div class="section"><h2>Year-End Projection</h2>'
                '<p class="note">No projection data available.</p></div>'
            )

        rows_html = ""
        for proj in projections:
            current = proj.get("current_tonnes", 0.0)
            projected = proj.get("projected_year_end_tonnes", 0.0)
            sector_threshold = proj.get("threshold_tonnes", threshold)
            will_exceed = projected > sector_threshold
            color = "#e74c3c" if will_exceed else "#2ecc71"

            rows_html += (
                f'<tr><td>{proj.get("sector", "")}</td>'
                f'<td class="num">{self._format_number(current)}</td>'
                f'<td class="num">{self._format_number(projected)}</td>'
                f'<td style="color:{color};font-weight:bold">'
                f'{"YES" if will_exceed else "No"}</td></tr>'
            )

        return (
            '<div class="section"><h2>Year-End Projection</h2>'
            '<table><thead><tr>'
            '<th>Sector Group</th><th>Current (t)</th>'
            '<th>Projected Year-End (t)</th><th>Will Exceed?</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_historical_comparison(self, data: Dict[str, Any]) -> str:
        """Build HTML historical comparison."""
        historical: Dict[str, Any] = data.get("historical", {})
        comparisons: List[Dict[str, Any]] = historical.get("comparisons", [])

        if not comparisons:
            return (
                '<div class="section"><h2>Historical Comparison</h2>'
                '<p class="note">No historical data available.</p></div>'
            )

        current_year = historical.get("current_year", "N/A")
        previous_year = historical.get("previous_year", "N/A")

        rows_html = ""
        for comp in comparisons:
            prev = comp.get("previous_tonnes", 0.0)
            curr = comp.get("current_tonnes", 0.0)
            change = curr - prev
            color = "#e74c3c" if change > 0 else "#2ecc71" if change < 0 else "#95a5a6"

            rows_html += (
                f'<tr><td>{comp.get("sector", "")}</td>'
                f'<td class="num">{self._format_number(prev)}</td>'
                f'<td class="num">{self._format_number(curr)}</td>'
                f'<td class="num" style="color:{color}">'
                f'{"+" if change >= 0 else ""}{self._format_number(change)}</td></tr>'
            )

        return (
            '<div class="section"><h2>Historical Comparison</h2>'
            f'<table><thead><tr>'
            f'<th>Sector Group</th><th>{previous_year} (t)</th>'
            f'<th>{current_year} (t)</th><th>Change (t)</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON section builders
    # ------------------------------------------------------------------ #

    def _json_threshold_status(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON threshold status."""
        sectors: List[Dict[str, Any]] = data.get("sector_groups", [])
        threshold = self._get_threshold()

        return [
            {
                "sector": sg.get("sector", ""),
                "cumulative_tonnes": round(sg.get("cumulative_tonnes", 0.0), 2),
                "threshold_tonnes": round(sg.get("threshold_tonnes", threshold), 2),
                "usage_pct": round(
                    (sg.get("cumulative_tonnes", 0.0) /
                     sg.get("threshold_tonnes", threshold) * 100)
                    if sg.get("threshold_tonnes", threshold) > 0 else 0.0, 2
                ),
                "alert_level": self._get_alert_level(
                    (sg.get("cumulative_tonnes", 0.0) /
                     sg.get("threshold_tonnes", threshold) * 100)
                    if sg.get("threshold_tonnes", threshold) > 0 else 0.0
                ).lower(),
            }
            for sg in sectors
        ]

    def _json_monthly_trend(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON monthly trend."""
        return data.get("monthly_imports", [])

    def _json_alert_status(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON alert status."""
        return data.get("alert_status", [])

    def _json_exemptions(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON exemptions."""
        return data.get("exemptions", [])

    def _json_projection(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON year-end projections."""
        projections: List[Dict[str, Any]] = data.get("projection", [])
        threshold = self._get_threshold()

        return [
            {
                "sector": proj.get("sector", ""),
                "current_tonnes": round(proj.get("current_tonnes", 0.0), 2),
                "projected_year_end_tonnes": round(proj.get("projected_year_end_tonnes", 0.0), 2),
                "threshold_tonnes": round(proj.get("threshold_tonnes", threshold), 2),
                "will_exceed": proj.get("projected_year_end_tonnes", 0.0) > proj.get("threshold_tonnes", threshold),
            }
            for proj in projections
        ]

    def _json_historical_comparison(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON historical comparison."""
        return data.get("historical", {})

    # ------------------------------------------------------------------ #
    #  Helper methods
    # ------------------------------------------------------------------ #

    def _get_threshold(self) -> float:
        """Get the de minimis threshold from config or default."""
        return self.config.get("threshold_tonnes", self.DEFAULT_THRESHOLD_TONNES)

    def _get_alert_level(self, usage_pct: float) -> str:
        """Determine alert level label from usage percentage."""
        if usage_pct >= 100.0:
            return "Exceeded"
        elif usage_pct >= 90.0:
            return "Critical"
        elif usage_pct >= 70.0:
            return "Approaching"
        return "Safe"

    def _get_alert_color(self, usage_pct: float) -> str:
        """Get CSS color for alert level from usage percentage."""
        if usage_pct >= 100.0:
            return "#c0392b"
        elif usage_pct >= 90.0:
            return "#e74c3c"
        elif usage_pct >= 70.0:
            return "#f39c12"
        return "#2ecc71"

    def _generate_provenance_hash(self, content: str) -> str:
        """Generate SHA-256 provenance hash for audit trail."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _format_number(self, value: Union[int, float], decimals: int = 2) -> str:
        """Format a numeric value with thousand separators and fixed decimals."""
        return f"{value:,.{decimals}f}"

    def _format_percentage(self, value: Union[int, float]) -> str:
        """Format a percentage value."""
        return f"{value:.2f}%"

    def _format_date(self, dt: Union[datetime, str]) -> str:
        """Format a datetime to ISO date string."""
        if isinstance(dt, str):
            return dt[:10] if len(dt) >= 10 else dt
        return dt.strftime("%Y-%m-%d")

    def _format_currency(self, value: Union[int, float], currency: str = "EUR") -> str:
        """Format a currency value."""
        return f"{currency} {value:,.2f}"

    def _format_cn_code(self, code: str) -> str:
        """Format a CN code to standard XXXX.XX format."""
        clean = code.replace(".", "").replace(" ", "")
        if len(clean) >= 6:
            return f"{clean[:4]}.{clean[4:6]}"
        elif len(clean) == 4:
            return f"{clean}.00"
        return code

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        """Wrap HTML body in a complete document with inline CSS."""
        css = (
            "body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
            "margin:0;padding:20px;background:#f5f7fa;color:#2c3e50}"
            ".report-header{background:#1a5276;color:#fff;padding:24px;border-radius:8px;"
            "margin-bottom:24px}"
            ".report-header h1{margin:0 0 12px 0;font-size:24px}"
            ".header-meta{display:flex;flex-wrap:wrap;gap:16px;font-size:14px}"
            ".meta-item{background:rgba(255,255,255,0.1);padding:4px 12px;border-radius:4px}"
            ".section{background:#fff;padding:20px;border-radius:8px;"
            "margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".section h2{margin:0 0 16px 0;font-size:18px;color:#1a5276;"
            "border-bottom:2px solid #eef2f7;padding-bottom:8px}"
            "table{width:100%;border-collapse:collapse;font-size:14px}"
            "th{background:#eef2f7;padding:10px 12px;text-align:left;font-weight:600}"
            "td{padding:8px 12px;border-bottom:1px solid #eef2f7}"
            ".num{text-align:right;font-variant-numeric:tabular-nums}"
            ".status-badge{display:inline-block;padding:2px 8px;border-radius:4px;"
            "color:#fff;font-size:11px;font-weight:bold}"
            ".progress-container{margin:16px 0}"
            ".progress-item{display:flex;align-items:center;gap:12px;margin-bottom:8px}"
            ".progress-label{width:160px;font-size:14px}"
            ".progress-bar{flex:1;background:#ecf0f1;border-radius:4px;height:20px;"
            "overflow:hidden}"
            ".progress-fill{height:100%;border-radius:4px;transition:width 0.3s}"
            ".progress-value{width:60px;font-size:13px;text-align:right}"
            ".alert-item{padding:8px 12px;margin-bottom:8px;background:#f8f9fa;"
            "border-radius:4px}"
            ".note{color:#7f8c8d;font-style:italic}"
            ".provenance{text-align:center;color:#95a5a6;font-size:12px;margin-top:24px}"
        )

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
            f'Template: DeMinimisReportTemplate v1.0 | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )
