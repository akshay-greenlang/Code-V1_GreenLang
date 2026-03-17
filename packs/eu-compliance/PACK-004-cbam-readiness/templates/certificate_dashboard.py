"""
CertificateDashboardTemplate - CBAM certificate obligation dashboard template.

This module implements the certificate obligation dashboard for CBAM compliance.
It generates formatted dashboards with KPI cards, certificate obligation waterfall,
EU ETS price tracking, quarterly holding compliance, cost projections under
multiple scenarios, and purchase timeline tracking.

Example:
    >>> template = CertificateDashboardTemplate()
    >>> data = {"net_certificates_required": 500, "ets_price_eur": 85.0, ...}
    >>> html = template.render_html(data)
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


class CertificateDashboardTemplate:
    """
    CBAM certificate obligation dashboard template.

    Generates formatted dashboards showing certificate obligations,
    ETS price trends, cost scenarios, and purchase timelines.

    Attributes:
        config: Optional configuration dictionary.
        generated_at: Timestamp of report generation.

    Example:
        >>> template = CertificateDashboardTemplate()
        >>> result = template.render_json(data)
        >>> assert "provenance_hash" in result
    """

    SCENARIO_LABELS: Dict[str, str] = {
        "low": "Conservative",
        "mid": "Base Case",
        "high": "Aggressive",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize CertificateDashboardTemplate.

        Args:
            config: Optional configuration dictionary. Supported keys:
                - currency (str): Currency code (default: EUR).
                - scenario_colors (dict): Color codes per scenario.
        """
        self.config = config or {}
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public render methods
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """
        Render the certificate dashboard as Markdown.

        Args:
            data: Dashboard data dictionary containing:
                - net_certificates_required (float)
                - estimated_annual_cost_eur (float)
                - ets_price_current (float)
                - ets_price_trend (list[dict]): date, price
                - free_allocation_pct (float)
                - obligation_waterfall (dict): gross, free_alloc, carbon_deduction, net
                - quarterly_holding (list[dict]): quarter, required, actual
                - cost_scenarios (dict): low/mid/high with annual projections
                - purchase_timeline (list[dict]): date, planned, executed

        Returns:
            Formatted Markdown string.
        """
        sections: List[str] = []

        sections.append(self._md_kpi_cards(data))
        sections.append(self._md_obligation_waterfall(data))
        sections.append(self._md_ets_price_trend(data))
        sections.append(self._md_quarterly_holding(data))
        sections.append(self._md_cost_scenarios(data))
        sections.append(self._md_purchase_timeline(data))
        sections.append(self._md_provenance_footer())

        content = "\n\n".join(sections)
        provenance_hash = self._generate_provenance_hash(content)
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the certificate dashboard as self-contained HTML.

        Args:
            data: Dashboard data dictionary (same schema as render_markdown).

        Returns:
            Complete HTML document string with inline CSS.
        """
        sections: List[str] = []

        sections.append(self._html_kpi_cards(data))
        sections.append(self._html_obligation_waterfall(data))
        sections.append(self._html_ets_price_trend(data))
        sections.append(self._html_quarterly_holding(data))
        sections.append(self._html_cost_scenarios(data))
        sections.append(self._html_purchase_timeline(data))

        body = "\n".join(sections)
        provenance_hash = self._generate_provenance_hash(body)

        return self._wrap_html(
            title="CBAM Certificate Dashboard",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the certificate dashboard as a structured dict.

        Args:
            data: Dashboard data dictionary (same schema as render_markdown).

        Returns:
            Dictionary with all dashboard sections and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "cbam_certificate_dashboard",
            "generated_at": self.generated_at,
            "kpi_summary": self._json_kpi_cards(data),
            "obligation_waterfall": self._json_obligation_waterfall(data),
            "ets_price_trend": self._json_ets_price_trend(data),
            "quarterly_holding": self._json_quarterly_holding(data),
            "cost_scenarios": self._json_cost_scenarios(data),
            "purchase_timeline": self._json_purchase_timeline(data),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._generate_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown section builders
    # ------------------------------------------------------------------ #

    def _md_kpi_cards(self, data: Dict[str, Any]) -> str:
        """Build Markdown KPI cards section."""
        net_certs = data.get("net_certificates_required", 0.0)
        annual_cost = data.get("estimated_annual_cost_eur", 0.0)
        ets_price = data.get("ets_price_current", 0.0)
        free_alloc = data.get("free_allocation_pct", 0.0)
        currency = self.config.get("currency", "EUR")

        return (
            "# CBAM Certificate Dashboard\n\n"
            f"**Generated:** {self.generated_at}\n\n"
            "## Key Performance Indicators\n\n"
            f"| KPI | Value |\n"
            f"|-----|-------|\n"
            f"| Net Certificates Required | {self._format_number(net_certs)} |\n"
            f"| Estimated Annual Cost | {self._format_currency(annual_cost, currency)} |\n"
            f"| Current EU ETS Price | {self._format_currency(ets_price, currency)}/tCO2e |\n"
            f"| Free Allocation Rate | {self._format_percentage(free_alloc)} |"
        )

    def _md_obligation_waterfall(self, data: Dict[str, Any]) -> str:
        """Build Markdown obligation waterfall section."""
        waterfall: Dict[str, Any] = data.get("obligation_waterfall", {})

        gross = waterfall.get("gross", 0.0)
        free_alloc = waterfall.get("free_allocation_deduction", 0.0)
        carbon_ded = waterfall.get("carbon_price_deduction", 0.0)
        net = waterfall.get("net", 0.0)

        return (
            "## Certificate Obligation Waterfall\n\n"
            "```\n"
            f"Gross obligation:         {self._format_number(gross):>12}\n"
            f"(-) Free allocation:      {'-' + self._format_number(free_alloc):>12}\n"
            f"(-) Carbon price deduct:  {'-' + self._format_number(carbon_ded):>12}\n"
            f"                          {'=' * 12}\n"
            f"Net obligation:           {self._format_number(net):>12}\n"
            "```"
        )

    def _md_ets_price_trend(self, data: Dict[str, Any]) -> str:
        """Build Markdown ETS price trend table."""
        trend: List[Dict[str, Any]] = data.get("ets_price_trend", [])
        currency = self.config.get("currency", "EUR")

        header = (
            "## EU ETS Price Trend\n\n"
            "| Date | Price | Change |\n"
            "|------|-------|--------|\n"
        )

        rows: List[str] = []
        prev_price = 0.0
        for entry in trend:
            price = entry.get("price", 0.0)
            date_str = self._format_date(entry.get("date", ""))
            change = price - prev_price if prev_price > 0 else 0.0
            change_str = f"{'+'if change >= 0 else ''}{self._format_currency(change, currency)}" if prev_price > 0 else "-"
            rows.append(
                f"| {date_str} | "
                f"{self._format_currency(price, currency)} | "
                f"{change_str} |"
            )
            prev_price = price

        return header + "\n".join(rows)

    def _md_quarterly_holding(self, data: Dict[str, Any]) -> str:
        """Build Markdown quarterly holding compliance table."""
        holdings: List[Dict[str, Any]] = data.get("quarterly_holding", [])

        header = (
            "## Quarterly Holding Compliance\n\n"
            "| Quarter | Required | Actual | Surplus/Deficit | Status |\n"
            "|---------|----------|--------|-----------------|--------|\n"
        )

        rows: List[str] = []
        for h in holdings:
            required = h.get("required", 0.0)
            actual = h.get("actual", 0.0)
            diff = actual - required
            status = "COMPLIANT" if diff >= 0 else "SHORTFALL"

            rows.append(
                f"| {h.get('quarter', '')} | "
                f"{self._format_number(required)} | "
                f"{self._format_number(actual)} | "
                f"{'+'if diff >= 0 else ''}{self._format_number(diff)} | "
                f"{status} |"
            )

        return header + "\n".join(rows)

    def _md_cost_scenarios(self, data: Dict[str, Any]) -> str:
        """Build Markdown cost projection scenarios table."""
        scenarios: Dict[str, Any] = data.get("cost_scenarios", {})
        currency = self.config.get("currency", "EUR")

        section = "## Cost Projection Scenarios\n\n"

        for scenario_key in ["low", "mid", "high"]:
            scenario = scenarios.get(scenario_key, {})
            label = self.SCENARIO_LABELS.get(scenario_key, scenario_key)
            ets_price = scenario.get("ets_price_eur", 0.0)
            projections: List[Dict[str, Any]] = scenario.get("annual_projections", [])

            section += (
                f"### {label} Scenario (ETS @ {self._format_currency(ets_price, currency)}/tCO2e)\n\n"
                f"| Year | Net Certificates | Estimated Cost |\n"
                f"|------|------------------|----------------|\n"
            )

            for proj in projections:
                section += (
                    f"| {proj.get('year', '')} | "
                    f"{self._format_number(proj.get('net_certificates', 0.0))} | "
                    f"{self._format_currency(proj.get('estimated_cost_eur', 0.0), currency)} |\n"
                )

            section += "\n"

        return section.rstrip()

    def _md_purchase_timeline(self, data: Dict[str, Any]) -> str:
        """Build Markdown purchase timeline table."""
        timeline: List[Dict[str, Any]] = data.get("purchase_timeline", [])
        currency = self.config.get("currency", "EUR")

        header = (
            "## Purchase Timeline\n\n"
            "| Date | Planned (certs) | Executed (certs) | "
            "Price Paid | Status |\n"
            "|------|-----------------|------------------|"
            "-----------|--------|\n"
        )

        rows: List[str] = []
        for entry in timeline:
            planned = entry.get("planned_certificates", 0.0)
            executed = entry.get("executed_certificates", 0.0)
            price = entry.get("price_paid_eur", 0.0)
            status = "Completed" if executed >= planned else "Pending"

            rows.append(
                f"| {self._format_date(entry.get('date', ''))} | "
                f"{self._format_number(planned)} | "
                f"{self._format_number(executed)} | "
                f"{self._format_currency(price, currency)} | "
                f"{status} |"
            )

        return header + "\n".join(rows)

    def _md_provenance_footer(self) -> str:
        """Build Markdown provenance footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            f"*Template: CertificateDashboardTemplate v1.0*"
        )

    # ------------------------------------------------------------------ #
    #  HTML section builders
    # ------------------------------------------------------------------ #

    def _html_kpi_cards(self, data: Dict[str, Any]) -> str:
        """Build HTML KPI cards section."""
        net_certs = data.get("net_certificates_required", 0.0)
        annual_cost = data.get("estimated_annual_cost_eur", 0.0)
        ets_price = data.get("ets_price_current", 0.0)
        free_alloc = data.get("free_allocation_pct", 0.0)
        currency = self.config.get("currency", "EUR")

        return (
            '<div class="report-header">'
            '<h1>CBAM Certificate Dashboard</h1>'
            f'<div class="meta-item">Generated: {self.generated_at}</div>'
            '</div>'
            '<div class="kpi-grid">'
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Net Certificates Required</div>'
            f'<div class="kpi-value">{self._format_number(net_certs, 0)}</div>'
            f'<div class="kpi-unit">certificates</div></div>'
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Estimated Annual Cost</div>'
            f'<div class="kpi-value">{self._format_currency(annual_cost, currency)}</div>'
            f'<div class="kpi-unit">total</div></div>'
            f'<div class="kpi-card">'
            f'<div class="kpi-label">EU ETS Price</div>'
            f'<div class="kpi-value">{self._format_currency(ets_price, currency)}</div>'
            f'<div class="kpi-unit">per tCO2e</div></div>'
            f'<div class="kpi-card">'
            f'<div class="kpi-label">Free Allocation Rate</div>'
            f'<div class="kpi-value">{self._format_percentage(free_alloc)}</div>'
            f'<div class="kpi-unit">of benchmark</div></div>'
            '</div>'
        )

    def _html_obligation_waterfall(self, data: Dict[str, Any]) -> str:
        """Build HTML obligation waterfall chart."""
        waterfall: Dict[str, Any] = data.get("obligation_waterfall", {})

        gross = waterfall.get("gross", 0.0)
        free_alloc = waterfall.get("free_allocation_deduction", 0.0)
        carbon_ded = waterfall.get("carbon_price_deduction", 0.0)
        net = waterfall.get("net", 0.0)

        max_val = max(gross, 1)

        return (
            '<div class="section"><h2>Certificate Obligation Waterfall</h2>'
            '<div class="waterfall">'
            f'<div class="waterfall-item">'
            f'<div class="waterfall-label">Gross obligation</div>'
            f'<div class="waterfall-bar positive" '
            f'style="width:{gross / max_val * 100:.0f}%">'
            f'{self._format_number(gross)}</div></div>'
            f'<div class="waterfall-item">'
            f'<div class="waterfall-label">(-) Free allocation</div>'
            f'<div class="waterfall-bar negative" '
            f'style="width:{free_alloc / max_val * 100:.0f}%">'
            f'-{self._format_number(free_alloc)}</div></div>'
            f'<div class="waterfall-item">'
            f'<div class="waterfall-label">(-) Carbon price deduction</div>'
            f'<div class="waterfall-bar negative" '
            f'style="width:{carbon_ded / max_val * 100:.0f}%">'
            f'-{self._format_number(carbon_ded)}</div></div>'
            f'<div class="waterfall-item">'
            f'<div class="waterfall-label"><strong>Net obligation</strong></div>'
            f'<div class="waterfall-bar net" '
            f'style="width:{net / max_val * 100:.0f}%">'
            f'<strong>{self._format_number(net)}</strong></div></div>'
            '</div></div>'
        )

    def _html_ets_price_trend(self, data: Dict[str, Any]) -> str:
        """Build HTML ETS price trend table."""
        trend: List[Dict[str, Any]] = data.get("ets_price_trend", [])
        currency = self.config.get("currency", "EUR")

        rows_html = ""
        prev_price = 0.0

        for entry in trend:
            price = entry.get("price", 0.0)
            date_str = self._format_date(entry.get("date", ""))
            change = price - prev_price if prev_price > 0 else 0.0
            color = "#2ecc71" if change >= 0 else "#e74c3c"

            if prev_price > 0:
                change_html = (
                    f'<span style="color:{color}">'
                    f'{"+" if change >= 0 else ""}{self._format_currency(change, currency)}'
                    f'</span>'
                )
            else:
                change_html = "-"

            rows_html += (
                f'<tr><td>{date_str}</td>'
                f'<td class="num">{self._format_currency(price, currency)}</td>'
                f'<td class="num">{change_html}</td></tr>'
            )
            prev_price = price

        return (
            '<div class="section"><h2>EU ETS Price Trend</h2>'
            '<table><thead><tr>'
            '<th>Date</th><th>Price</th><th>Change</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_quarterly_holding(self, data: Dict[str, Any]) -> str:
        """Build HTML quarterly holding compliance table."""
        holdings: List[Dict[str, Any]] = data.get("quarterly_holding", [])

        rows_html = ""
        for h in holdings:
            required = h.get("required", 0.0)
            actual = h.get("actual", 0.0)
            diff = actual - required
            compliant = diff >= 0
            color = "#2ecc71" if compliant else "#e74c3c"
            status = "COMPLIANT" if compliant else "SHORTFALL"

            rows_html += (
                f'<tr><td>{h.get("quarter", "")}</td>'
                f'<td class="num">{self._format_number(required)}</td>'
                f'<td class="num">{self._format_number(actual)}</td>'
                f'<td class="num" style="color:{color}">'
                f'{"+" if diff >= 0 else ""}{self._format_number(diff)}</td>'
                f'<td style="color:{color};font-weight:bold">{status}</td></tr>'
            )

        return (
            '<div class="section"><h2>Quarterly Holding Compliance</h2>'
            '<table><thead><tr>'
            '<th>Quarter</th><th>Required</th><th>Actual</th>'
            '<th>Surplus/Deficit</th><th>Status</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_cost_scenarios(self, data: Dict[str, Any]) -> str:
        """Build HTML cost projection scenarios."""
        scenarios: Dict[str, Any] = data.get("cost_scenarios", {})
        currency = self.config.get("currency", "EUR")

        scenario_colors = self.config.get("scenario_colors", {
            "low": "#2ecc71",
            "mid": "#3498db",
            "high": "#e74c3c",
        })

        scenarios_html = ""
        for scenario_key in ["low", "mid", "high"]:
            scenario = scenarios.get(scenario_key, {})
            label = self.SCENARIO_LABELS.get(scenario_key, scenario_key)
            ets_price = scenario.get("ets_price_eur", 0.0)
            color = scenario_colors.get(scenario_key, "#95a5a6")
            projections: List[Dict[str, Any]] = scenario.get("annual_projections", [])

            rows_html = ""
            for proj in projections:
                rows_html += (
                    f'<tr><td>{proj.get("year", "")}</td>'
                    f'<td class="num">{self._format_number(proj.get("net_certificates", 0.0))}</td>'
                    f'<td class="num">'
                    f'{self._format_currency(proj.get("estimated_cost_eur", 0.0), currency)}'
                    f'</td></tr>'
                )

            scenarios_html += (
                f'<div class="scenario-card" style="border-left:4px solid {color}">'
                f'<h3>{label} Scenario</h3>'
                f'<p>ETS Price: {self._format_currency(ets_price, currency)}/tCO2e</p>'
                f'<table><thead><tr>'
                f'<th>Year</th><th>Net Certificates</th><th>Estimated Cost</th>'
                f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
            )

        return (
            '<div class="section"><h2>Cost Projection Scenarios</h2>'
            f'<div class="scenario-grid">{scenarios_html}</div></div>'
        )

    def _html_purchase_timeline(self, data: Dict[str, Any]) -> str:
        """Build HTML purchase timeline table."""
        timeline: List[Dict[str, Any]] = data.get("purchase_timeline", [])
        currency = self.config.get("currency", "EUR")

        rows_html = ""
        for entry in timeline:
            planned = entry.get("planned_certificates", 0.0)
            executed = entry.get("executed_certificates", 0.0)
            price = entry.get("price_paid_eur", 0.0)
            completed = executed >= planned
            status = "Completed" if completed else "Pending"
            color = "#2ecc71" if completed else "#f39c12"

            rows_html += (
                f'<tr><td>{self._format_date(entry.get("date", ""))}</td>'
                f'<td class="num">{self._format_number(planned)}</td>'
                f'<td class="num">{self._format_number(executed)}</td>'
                f'<td class="num">{self._format_currency(price, currency)}</td>'
                f'<td style="color:{color};font-weight:bold">{status}</td></tr>'
            )

        return (
            '<div class="section"><h2>Purchase Timeline</h2>'
            '<table><thead><tr>'
            '<th>Date</th><th>Planned (certs)</th><th>Executed (certs)</th>'
            '<th>Price Paid</th><th>Status</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON section builders
    # ------------------------------------------------------------------ #

    def _json_kpi_cards(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON KPI summary."""
        return {
            "net_certificates_required": round(data.get("net_certificates_required", 0.0), 2),
            "estimated_annual_cost_eur": round(data.get("estimated_annual_cost_eur", 0.0), 2),
            "ets_price_current_eur": round(data.get("ets_price_current", 0.0), 2),
            "free_allocation_pct": round(data.get("free_allocation_pct", 0.0), 2),
        }

    def _json_obligation_waterfall(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON obligation waterfall."""
        waterfall: Dict[str, Any] = data.get("obligation_waterfall", {})
        return {
            "gross": round(waterfall.get("gross", 0.0), 2),
            "free_allocation_deduction": round(waterfall.get("free_allocation_deduction", 0.0), 2),
            "carbon_price_deduction": round(waterfall.get("carbon_price_deduction", 0.0), 2),
            "net": round(waterfall.get("net", 0.0), 2),
        }

    def _json_ets_price_trend(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON ETS price trend."""
        trend: List[Dict[str, Any]] = data.get("ets_price_trend", [])
        return [
            {
                "date": self._format_date(entry.get("date", "")),
                "price_eur": round(entry.get("price", 0.0), 2),
            }
            for entry in trend
        ]

    def _json_quarterly_holding(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON quarterly holding compliance."""
        holdings: List[Dict[str, Any]] = data.get("quarterly_holding", [])

        result: List[Dict[str, Any]] = []
        for h in holdings:
            required = h.get("required", 0.0)
            actual = h.get("actual", 0.0)
            diff = actual - required

            result.append({
                "quarter": h.get("quarter", ""),
                "required": round(required, 2),
                "actual": round(actual, 2),
                "surplus_deficit": round(diff, 2),
                "compliant": diff >= 0,
            })

        return result

    def _json_cost_scenarios(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON cost projection scenarios."""
        scenarios: Dict[str, Any] = data.get("cost_scenarios", {})

        result: Dict[str, Any] = {}
        for scenario_key in ["low", "mid", "high"]:
            scenario = scenarios.get(scenario_key, {})
            result[scenario_key] = {
                "label": self.SCENARIO_LABELS.get(scenario_key, scenario_key),
                "ets_price_eur": round(scenario.get("ets_price_eur", 0.0), 2),
                "annual_projections": [
                    {
                        "year": proj.get("year", ""),
                        "net_certificates": round(proj.get("net_certificates", 0.0), 2),
                        "estimated_cost_eur": round(proj.get("estimated_cost_eur", 0.0), 2),
                    }
                    for proj in scenario.get("annual_projections", [])
                ],
            }

        return result

    def _json_purchase_timeline(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON purchase timeline."""
        timeline: List[Dict[str, Any]] = data.get("purchase_timeline", [])
        return [
            {
                "date": self._format_date(entry.get("date", "")),
                "planned_certificates": round(entry.get("planned_certificates", 0.0), 2),
                "executed_certificates": round(entry.get("executed_certificates", 0.0), 2),
                "price_paid_eur": round(entry.get("price_paid_eur", 0.0), 2),
                "completed": entry.get("executed_certificates", 0.0) >= entry.get("planned_certificates", 0.0),
            }
            for entry in timeline
        ]

    # ------------------------------------------------------------------ #
    #  Helper methods
    # ------------------------------------------------------------------ #

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
            return dt[:10]
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

    def _html_progress_bar(self, pct: float, color: str) -> str:
        """Generate an inline HTML progress bar."""
        width = max(0, min(100, pct))
        return (
            f'<div class="progress-bar">'
            f'<div class="progress-fill" '
            f'style="width:{width}%;background:{color}"></div>'
            f'</div>'
        )

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        """Wrap HTML body in a complete document with inline CSS."""
        css = (
            "body{font-family:'Segoe UI',Tahoma,Geneva,Verdana,sans-serif;"
            "margin:0;padding:20px;background:#f5f7fa;color:#2c3e50}"
            ".report-header{background:#1a5276;color:#fff;padding:24px;border-radius:8px;"
            "margin-bottom:24px}"
            ".report-header h1{margin:0 0 8px 0;font-size:24px}"
            ".meta-item{font-size:13px;opacity:0.8}"
            ".kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));"
            "gap:16px;margin-bottom:24px}"
            ".kpi-card{background:#fff;padding:20px;border-radius:8px;text-align:center;"
            "box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".kpi-label{font-size:13px;color:#7f8c8d;margin-bottom:4px}"
            ".kpi-value{font-size:28px;font-weight:700;color:#1a5276}"
            ".kpi-unit{font-size:12px;color:#95a5a6;margin-top:4px}"
            ".section{background:#fff;padding:20px;border-radius:8px;"
            "margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".section h2{margin:0 0 16px 0;font-size:18px;color:#1a5276;"
            "border-bottom:2px solid #eef2f7;padding-bottom:8px}"
            "table{width:100%;border-collapse:collapse;font-size:14px}"
            "th{background:#eef2f7;padding:10px 12px;text-align:left;font-weight:600}"
            "td{padding:8px 12px;border-bottom:1px solid #eef2f7}"
            ".num{text-align:right;font-variant-numeric:tabular-nums}"
            ".waterfall{margin:16px 0}"
            ".waterfall-item{display:flex;align-items:center;margin-bottom:8px}"
            ".waterfall-label{width:240px;font-size:14px}"
            ".waterfall-bar{padding:6px 12px;border-radius:4px;color:#fff;"
            "font-size:13px;min-width:60px;text-align:right}"
            ".waterfall-bar.positive{background:#2ecc71}"
            ".waterfall-bar.negative{background:#e74c3c}"
            ".waterfall-bar.net{background:#1a5276}"
            ".scenario-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));"
            "gap:16px}"
            ".scenario-card{background:#f8f9fa;padding:16px;border-radius:8px}"
            ".scenario-card h3{margin:0 0 8px 0;font-size:16px;color:#2c3e50}"
            ".scenario-card p{margin:0 0 12px 0;font-size:13px;color:#7f8c8d}"
            ".progress-bar{background:#ecf0f1;border-radius:4px;height:12px;"
            "overflow:hidden;margin:4px 0}"
            ".progress-fill{height:100%;border-radius:4px}"
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
            f'Template: CertificateDashboardTemplate v1.0 | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )
