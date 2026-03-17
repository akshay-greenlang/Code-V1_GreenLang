"""
CostProjectionTemplate - CBAM certificate cost forecast template.

This module implements the certificate cost projection template for CBAM
compliance planning. It generates formatted reports with scenario comparisons,
annual cost forecasts, category breakdowns, free allocation impact analysis,
carbon deduction savings, sensitivity analysis, and budget planning summaries.

Example:
    >>> template = CostProjectionTemplate()
    >>> data = {"scenarios": {...}, "forecast_years": 4, ...}
    >>> html = template.render_html(data)
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


# Free allocation phase-out schedule
FREE_ALLOCATION_RATES: Dict[int, float] = {
    2026: 97.5, 2027: 95.0, 2028: 90.0, 2029: 77.5,
    2030: 51.5, 2031: 39.0, 2032: 26.5, 2033: 14.0, 2034: 0.0,
}


class CostProjectionTemplate:
    """
    CBAM certificate cost forecast template.

    Generates formatted cost projections with multiple ETS price scenarios,
    annual forecasts, category breakdowns, free allocation impact, carbon
    deduction savings, sensitivity analysis, and budget planning.

    Attributes:
        config: Optional configuration dictionary.
        generated_at: Timestamp of report generation.
    """

    SENSITIVITY_LEVELS: List[Dict[str, Any]] = [
        {"label": "-20%", "factor": 0.80},
        {"label": "-10%", "factor": 0.90},
        {"label": "Base", "factor": 1.00},
        {"label": "+10%", "factor": 1.10},
        {"label": "+20%", "factor": 1.20},
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize CostProjectionTemplate.

        Args:
            config: Optional configuration dictionary. Supported keys:
                - currency (str): Currency code (default: EUR).
                - forecast_years (int): Number of years to project (default: 3).
        """
        self.config = config or {}
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public render methods
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """
        Render the cost projection as Markdown.

        Args:
            data: Projection data dictionary containing:
                - scenarios (dict): low/mid/high ETS price scenarios
                - annual_forecast (list[dict]): year, emissions, cost by scenario
                - cost_by_category (list[dict]): category, emissions, cost
                - free_allocation_impact (dict): yearly impact data
                - carbon_deductions (list[dict]): country, amount, savings
                - sensitivity (dict): base cost and variant costs
                - budget_summary (dict): recommended budget, contingency

        Returns:
            Formatted Markdown string.
        """
        sections: List[str] = []

        sections.append(self._md_header())
        sections.append(self._md_scenario_comparison(data))
        sections.append(self._md_annual_forecast(data))
        sections.append(self._md_cost_by_category(data))
        sections.append(self._md_free_allocation_impact(data))
        sections.append(self._md_carbon_deductions(data))
        sections.append(self._md_sensitivity_analysis(data))
        sections.append(self._md_budget_summary(data))
        sections.append(self._md_provenance_footer())

        content = "\n\n".join(sections)
        provenance_hash = self._generate_provenance_hash(content)
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the cost projection as self-contained HTML.

        Args:
            data: Projection data dictionary (same schema as render_markdown).

        Returns:
            Complete HTML document string with inline CSS.
        """
        sections: List[str] = []

        sections.append(self._html_header())
        sections.append(self._html_scenario_comparison(data))
        sections.append(self._html_annual_forecast(data))
        sections.append(self._html_cost_by_category(data))
        sections.append(self._html_free_allocation_impact(data))
        sections.append(self._html_carbon_deductions(data))
        sections.append(self._html_sensitivity_analysis(data))
        sections.append(self._html_budget_summary(data))

        body = "\n".join(sections)
        provenance_hash = self._generate_provenance_hash(body)

        return self._wrap_html(
            title="CBAM Certificate Cost Forecast",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the cost projection as a structured dict.

        Args:
            data: Projection data dictionary (same schema as render_markdown).

        Returns:
            Dictionary with all projection sections and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "cbam_cost_projection",
            "generated_at": self.generated_at,
            "scenario_comparison": self._json_scenario_comparison(data),
            "annual_forecast": self._json_annual_forecast(data),
            "cost_by_category": self._json_cost_by_category(data),
            "free_allocation_impact": self._json_free_allocation_impact(data),
            "carbon_deductions": self._json_carbon_deductions(data),
            "sensitivity_analysis": self._json_sensitivity_analysis(data),
            "budget_summary": self._json_budget_summary(data),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._generate_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown section builders
    # ------------------------------------------------------------------ #

    def _md_header(self) -> str:
        """Build Markdown header."""
        return (
            "# CBAM Certificate Cost Forecast\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_scenario_comparison(self, data: Dict[str, Any]) -> str:
        """Build Markdown cost scenario comparison table."""
        scenarios: Dict[str, Any] = data.get("scenarios", {})
        currency = self.config.get("currency", "EUR")

        header = (
            "## Cost Scenario Comparison\n\n"
            "| Scenario | ETS Price (/tCO2e) | Net Certificates | "
            "Annual Cost | 3-Year Total |\n"
            "|----------|-------------------|------------------|"
            "------------|---------------|\n"
        )

        rows: List[str] = []
        for key in ["low", "mid", "high"]:
            s = scenarios.get(key, {})
            label = {"low": "Conservative", "mid": "Base Case", "high": "Aggressive"}.get(key, key)

            rows.append(
                f"| {label} | "
                f"{self._format_currency(s.get('ets_price_eur', 0.0), currency)} | "
                f"{self._format_number(s.get('net_certificates', 0.0))} | "
                f"{self._format_currency(s.get('annual_cost_eur', 0.0), currency)} | "
                f"{self._format_currency(s.get('three_year_total_eur', 0.0), currency)} |"
            )

        return header + "\n".join(rows)

    def _md_annual_forecast(self, data: Dict[str, Any]) -> str:
        """Build Markdown annual cost forecast table."""
        forecast: List[Dict[str, Any]] = data.get("annual_forecast", [])
        currency = self.config.get("currency", "EUR")

        header = (
            "## Annual Cost Forecast\n\n"
            "| Year | Net Emissions (tCO2e) | Low Scenario | "
            "Mid Scenario | High Scenario |\n"
            "|------|----------------------|--------------|"
            "--------------|----------------|\n"
        )

        rows: List[str] = []
        for entry in forecast:
            rows.append(
                f"| {entry.get('year', '')} | "
                f"{self._format_number(entry.get('net_emissions_tco2e', 0.0))} | "
                f"{self._format_currency(entry.get('cost_low_eur', 0.0), currency)} | "
                f"{self._format_currency(entry.get('cost_mid_eur', 0.0), currency)} | "
                f"{self._format_currency(entry.get('cost_high_eur', 0.0), currency)} |"
            )

        return header + "\n".join(rows)

    def _md_cost_by_category(self, data: Dict[str, Any]) -> str:
        """Build Markdown cost breakdown by goods category."""
        categories: List[Dict[str, Any]] = data.get("cost_by_category", [])
        currency = self.config.get("currency", "EUR")

        total_cost = sum(c.get("estimated_cost_eur", 0.0) for c in categories)

        header = (
            "## Cost Breakdown by Goods Category\n\n"
            "| Category | Emissions (tCO2e) | Net Certificates | "
            "Estimated Cost | Share (%) |\n"
            "|----------|-------------------|------------------|"
            "----------------|----------|\n"
        )

        rows: List[str] = []
        for c in sorted(categories, key=lambda x: x.get("estimated_cost_eur", 0.0), reverse=True):
            cost = c.get("estimated_cost_eur", 0.0)
            share = (cost / total_cost * 100) if total_cost > 0 else 0.0

            rows.append(
                f"| {c.get('category', '').capitalize()} | "
                f"{self._format_number(c.get('emissions_tco2e', 0.0))} | "
                f"{self._format_number(c.get('net_certificates', 0.0))} | "
                f"{self._format_currency(cost, currency)} | "
                f"{self._format_percentage(share)} |"
            )

        rows.append(
            f"| **TOTAL** | | | "
            f"**{self._format_currency(total_cost, currency)}** | **100.00%** |"
        )

        return header + "\n".join(rows)

    def _md_free_allocation_impact(self, data: Dict[str, Any]) -> str:
        """Build Markdown free allocation impact analysis."""
        impact: Dict[str, Any] = data.get("free_allocation_impact", {})
        currency = self.config.get("currency", "EUR")

        yearly: List[Dict[str, Any]] = impact.get("yearly", [])

        header = (
            "## Free Allocation Impact\n\n"
            f"As free allocation declines from {FREE_ALLOCATION_RATES.get(2026, 97.5)}% "
            f"to 0% by 2034, CBAM costs will increase proportionally.\n\n"
            "| Year | Free Alloc (%) | CBAM Coverage (%) | "
            "Additional Cost vs Current |\n"
            "|------|----------------|--------------------|-"
            "--------------------------|\n"
        )

        rows: List[str] = []
        for entry in yearly:
            year = entry.get("year", 0)
            fa_rate = entry.get("free_allocation_pct", FREE_ALLOCATION_RATES.get(year, 0.0))
            cbam_pct = 100.0 - fa_rate
            additional = entry.get("additional_cost_eur", 0.0)

            rows.append(
                f"| {year} | "
                f"{self._format_percentage(fa_rate)} | "
                f"{self._format_percentage(cbam_pct)} | "
                f"{self._format_currency(additional, currency)} |"
            )

        return header + "\n".join(rows)

    def _md_carbon_deductions(self, data: Dict[str, Any]) -> str:
        """Build Markdown carbon price deduction savings."""
        deductions: List[Dict[str, Any]] = data.get("carbon_deductions", [])
        currency = self.config.get("currency", "EUR")

        if not deductions:
            return (
                "## Carbon Price Deduction Savings\n\n"
                "*No third-country carbon pricing deductions applicable.*"
            )

        total_savings = sum(d.get("savings_eur", 0.0) for d in deductions)

        header = (
            "## Carbon Price Deduction Savings\n\n"
            f"**Total Savings:** {self._format_currency(total_savings, currency)}\n\n"
            "| Country | Carbon Price (/tCO2e) | Applicable Emissions (tCO2e) | "
            "Deduction (certificates) | Savings |\n"
            "|---------|----------------------|-----------------------------|"
            "--------------------------|--------|\n"
        )

        rows: List[str] = []
        for d in deductions:
            rows.append(
                f"| {d.get('country', '')} | "
                f"{self._format_currency(d.get('carbon_price_eur', 0.0), currency)} | "
                f"{self._format_number(d.get('applicable_emissions', 0.0))} | "
                f"{self._format_number(d.get('deduction_certificates', 0.0))} | "
                f"{self._format_currency(d.get('savings_eur', 0.0), currency)} |"
            )

        return header + "\n".join(rows)

    def _md_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        """Build Markdown sensitivity analysis section."""
        sensitivity: Dict[str, Any] = data.get("sensitivity", {})
        currency = self.config.get("currency", "EUR")

        base_cost = sensitivity.get("base_cost_eur", 0.0)
        base_emissions = sensitivity.get("base_emissions_tco2e", 0.0)

        header = (
            "## Sensitivity Analysis\n\n"
            f"**Base Emissions:** {self._format_number(base_emissions)} tCO2e\n\n"
            f"**Base Cost:** {self._format_currency(base_cost, currency)}\n\n"
            "| Emission Change | Adjusted Emissions (tCO2e) | Adjusted Cost | "
            "Cost Delta |\n"
            "|-----------------|---------------------------|---------------|"
            "-----------|\n"
        )

        rows: List[str] = []
        for level in self.SENSITIVITY_LEVELS:
            adjusted_emissions = base_emissions * level["factor"]
            adjusted_cost = base_cost * level["factor"]
            delta = adjusted_cost - base_cost

            rows.append(
                f"| {level['label']} | "
                f"{self._format_number(adjusted_emissions)} | "
                f"{self._format_currency(adjusted_cost, currency)} | "
                f"{'+'if delta >= 0 else ''}{self._format_currency(delta, currency)} |"
            )

        return header + "\n".join(rows)

    def _md_budget_summary(self, data: Dict[str, Any]) -> str:
        """Build Markdown budget planning summary."""
        budget: Dict[str, Any] = data.get("budget_summary", {})
        currency = self.config.get("currency", "EUR")

        recommended = budget.get("recommended_budget_eur", 0.0)
        contingency_pct = budget.get("contingency_pct", 15.0)
        contingency_amount = recommended * (contingency_pct / 100.0)
        total_with_contingency = recommended + contingency_amount

        return (
            "## Budget Planning Summary\n\n"
            f"| Component | Amount |\n"
            f"|-----------|--------|\n"
            f"| Base budget (mid scenario) | {self._format_currency(recommended, currency)} |\n"
            f"| Contingency ({self._format_percentage(contingency_pct)}) | "
            f"{self._format_currency(contingency_amount, currency)} |\n"
            f"| **Total recommended budget** | "
            f"**{self._format_currency(total_with_contingency, currency)}** |\n\n"
            f"> Contingency covers ETS price volatility, emission estimate variance, "
            f"and free allocation changes.\n\n"
            f"**Key assumptions:**\n\n"
            f"- ETS price scenario: {budget.get('ets_scenario', 'mid')}\n"
            f"- Emission trend: {budget.get('emission_trend', 'flat')}\n"
            f"- Free allocation: per regulatory schedule"
        )

    def _md_provenance_footer(self) -> str:
        """Build Markdown provenance footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            f"*Template: CostProjectionTemplate v1.0*"
        )

    # ------------------------------------------------------------------ #
    #  HTML section builders
    # ------------------------------------------------------------------ #

    def _html_header(self) -> str:
        """Build HTML header."""
        return (
            '<div class="report-header">'
            '<h1>CBAM Certificate Cost Forecast</h1>'
            f'<div class="meta-item">Generated: {self.generated_at}</div>'
            '</div>'
        )

    def _html_scenario_comparison(self, data: Dict[str, Any]) -> str:
        """Build HTML cost scenario comparison."""
        scenarios: Dict[str, Any] = data.get("scenarios", {})
        currency = self.config.get("currency", "EUR")

        colors = {"low": "#2ecc71", "mid": "#3498db", "high": "#e74c3c"}
        labels = {"low": "Conservative", "mid": "Base Case", "high": "Aggressive"}

        cards_html = ""
        for key in ["low", "mid", "high"]:
            s = scenarios.get(key, {})
            color = colors.get(key, "#95a5a6")
            label = labels.get(key, key)

            cards_html += (
                f'<div class="scenario-card" style="border-top:4px solid {color}">'
                f'<h3>{label}</h3>'
                f'<div class="kpi-value">{self._format_currency(s.get("annual_cost_eur", 0.0), currency)}</div>'
                f'<div class="kpi-label">Annual Cost</div>'
                f'<div class="scenario-detail">'
                f'ETS: {self._format_currency(s.get("ets_price_eur", 0.0), currency)}/tCO2e<br>'
                f'Certificates: {self._format_number(s.get("net_certificates", 0.0))}<br>'
                f'3-Year: {self._format_currency(s.get("three_year_total_eur", 0.0), currency)}'
                f'</div></div>'
            )

        return (
            '<div class="section"><h2>Cost Scenario Comparison</h2>'
            f'<div class="scenario-grid">{cards_html}</div></div>'
        )

    def _html_annual_forecast(self, data: Dict[str, Any]) -> str:
        """Build HTML annual cost forecast."""
        forecast: List[Dict[str, Any]] = data.get("annual_forecast", [])
        currency = self.config.get("currency", "EUR")

        rows_html = ""
        for entry in forecast:
            rows_html += (
                f'<tr><td>{entry.get("year", "")}</td>'
                f'<td class="num">{self._format_number(entry.get("net_emissions_tco2e", 0.0))}</td>'
                f'<td class="num" style="color:#2ecc71">'
                f'{self._format_currency(entry.get("cost_low_eur", 0.0), currency)}</td>'
                f'<td class="num" style="color:#3498db">'
                f'{self._format_currency(entry.get("cost_mid_eur", 0.0), currency)}</td>'
                f'<td class="num" style="color:#e74c3c">'
                f'{self._format_currency(entry.get("cost_high_eur", 0.0), currency)}</td></tr>'
            )

        return (
            '<div class="section"><h2>Annual Cost Forecast</h2>'
            '<table><thead><tr>'
            '<th>Year</th><th>Net Emissions (tCO2e)</th>'
            '<th style="color:#2ecc71">Low</th>'
            '<th style="color:#3498db">Mid</th>'
            '<th style="color:#e74c3c">High</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_cost_by_category(self, data: Dict[str, Any]) -> str:
        """Build HTML cost breakdown by goods category."""
        categories: List[Dict[str, Any]] = data.get("cost_by_category", [])
        currency = self.config.get("currency", "EUR")

        total_cost = sum(c.get("estimated_cost_eur", 0.0) for c in categories)
        max_cost = max((c.get("estimated_cost_eur", 0.0) for c in categories), default=1)

        rows_html = ""
        for c in sorted(categories, key=lambda x: x.get("estimated_cost_eur", 0.0), reverse=True):
            cost = c.get("estimated_cost_eur", 0.0)
            share = (cost / total_cost * 100) if total_cost > 0 else 0.0
            bar_width = (cost / max_cost * 100) if max_cost > 0 else 0

            rows_html += (
                f'<tr><td>{c.get("category", "").capitalize()}</td>'
                f'<td class="num">{self._format_number(c.get("emissions_tco2e", 0.0))}</td>'
                f'<td class="num">{self._format_number(c.get("net_certificates", 0.0))}</td>'
                f'<td class="num">{self._format_currency(cost, currency)}</td>'
                f'<td><div class="bar-chart">'
                f'<div class="bar-fill" style="width:{bar_width:.0f}%"></div>'
                f'<span>{self._format_percentage(share)}</span>'
                f'</div></td></tr>'
            )

        return (
            '<div class="section"><h2>Cost Breakdown by Goods Category</h2>'
            '<table><thead><tr>'
            '<th>Category</th><th>Emissions (tCO2e)</th>'
            '<th>Net Certificates</th><th>Estimated Cost</th><th>Share</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_free_allocation_impact(self, data: Dict[str, Any]) -> str:
        """Build HTML free allocation impact analysis."""
        impact: Dict[str, Any] = data.get("free_allocation_impact", {})
        currency = self.config.get("currency", "EUR")
        yearly: List[Dict[str, Any]] = impact.get("yearly", [])

        rows_html = ""
        for entry in yearly:
            year = entry.get("year", 0)
            fa_rate = entry.get("free_allocation_pct", FREE_ALLOCATION_RATES.get(year, 0.0))
            cbam_pct = 100.0 - fa_rate
            additional = entry.get("additional_cost_eur", 0.0)

            rows_html += (
                f'<tr><td>{year}</td>'
                f'<td class="num">{self._format_percentage(fa_rate)}</td>'
                f'<td class="num">{self._format_percentage(cbam_pct)}</td>'
                f'<td class="num">{self._format_currency(additional, currency)}</td></tr>'
            )

        return (
            '<div class="section"><h2>Free Allocation Impact</h2>'
            '<p>As free allocation declines, CBAM certificate costs increase proportionally.</p>'
            '<table><thead><tr>'
            '<th>Year</th><th>Free Allocation (%)</th>'
            '<th>CBAM Coverage (%)</th><th>Additional Cost vs Current</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_carbon_deductions(self, data: Dict[str, Any]) -> str:
        """Build HTML carbon price deduction savings."""
        deductions: List[Dict[str, Any]] = data.get("carbon_deductions", [])
        currency = self.config.get("currency", "EUR")

        if not deductions:
            return (
                '<div class="section"><h2>Carbon Price Deduction Savings</h2>'
                '<p class="note">No third-country carbon pricing deductions applicable.</p></div>'
            )

        total_savings = sum(d.get("savings_eur", 0.0) for d in deductions)

        rows_html = ""
        for d in deductions:
            rows_html += (
                f'<tr><td>{d.get("country", "")}</td>'
                f'<td class="num">{self._format_currency(d.get("carbon_price_eur", 0.0), currency)}</td>'
                f'<td class="num">{self._format_number(d.get("applicable_emissions", 0.0))}</td>'
                f'<td class="num">{self._format_number(d.get("deduction_certificates", 0.0))}</td>'
                f'<td class="num">{self._format_currency(d.get("savings_eur", 0.0), currency)}</td></tr>'
            )

        return (
            '<div class="section"><h2>Carbon Price Deduction Savings</h2>'
            f'<div class="kpi-card" style="margin-bottom:16px;text-align:center">'
            f'<div class="kpi-label">Total Savings</div>'
            f'<div class="kpi-value" style="color:#2ecc71">'
            f'{self._format_currency(total_savings, currency)}</div></div>'
            '<table><thead><tr>'
            '<th>Country</th><th>Carbon Price</th><th>Applicable Emissions</th>'
            '<th>Deduction (certs)</th><th>Savings</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_sensitivity_analysis(self, data: Dict[str, Any]) -> str:
        """Build HTML sensitivity analysis."""
        sensitivity: Dict[str, Any] = data.get("sensitivity", {})
        currency = self.config.get("currency", "EUR")

        base_cost = sensitivity.get("base_cost_eur", 0.0)
        base_emissions = sensitivity.get("base_emissions_tco2e", 0.0)

        rows_html = ""
        for level in self.SENSITIVITY_LEVELS:
            adjusted_emissions = base_emissions * level["factor"]
            adjusted_cost = base_cost * level["factor"]
            delta = adjusted_cost - base_cost
            is_base = level["factor"] == 1.0
            color = "#2c3e50" if is_base else ("#2ecc71" if delta < 0 else "#e74c3c")
            row_class = ' class="highlight-row"' if is_base else ""

            rows_html += (
                f'<tr{row_class}>'
                f'<td>{level["label"]}</td>'
                f'<td class="num">{self._format_number(adjusted_emissions)}</td>'
                f'<td class="num">{self._format_currency(adjusted_cost, currency)}</td>'
                f'<td class="num" style="color:{color}">'
                f'{"+" if delta >= 0 else ""}{self._format_currency(delta, currency)}</td></tr>'
            )

        return (
            '<div class="section"><h2>Sensitivity Analysis</h2>'
            f'<p>Impact of emission changes on total cost '
            f'(base: {self._format_number(base_emissions)} tCO2e, '
            f'{self._format_currency(base_cost, currency)})</p>'
            '<table><thead><tr>'
            '<th>Emission Change</th><th>Adjusted Emissions</th>'
            '<th>Adjusted Cost</th><th>Cost Delta</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_budget_summary(self, data: Dict[str, Any]) -> str:
        """Build HTML budget planning summary."""
        budget: Dict[str, Any] = data.get("budget_summary", {})
        currency = self.config.get("currency", "EUR")

        recommended = budget.get("recommended_budget_eur", 0.0)
        contingency_pct = budget.get("contingency_pct", 15.0)
        contingency_amount = recommended * (contingency_pct / 100.0)
        total = recommended + contingency_amount

        return (
            '<div class="section"><h2>Budget Planning Summary</h2>'
            '<div class="budget-grid">'
            f'<div class="budget-card">'
            f'<div class="kpi-label">Base Budget</div>'
            f'<div class="kpi-value">{self._format_currency(recommended, currency)}</div></div>'
            f'<div class="budget-card">'
            f'<div class="kpi-label">Contingency ({self._format_percentage(contingency_pct)})</div>'
            f'<div class="kpi-value">{self._format_currency(contingency_amount, currency)}</div></div>'
            f'<div class="budget-card highlight">'
            f'<div class="kpi-label">Total Recommended</div>'
            f'<div class="kpi-value">{self._format_currency(total, currency)}</div></div>'
            '</div>'
            '<div class="assumptions">'
            '<strong>Key assumptions:</strong>'
            f'<ul><li>ETS price scenario: {budget.get("ets_scenario", "mid")}</li>'
            f'<li>Emission trend: {budget.get("emission_trend", "flat")}</li>'
            f'<li>Free allocation: per regulatory schedule</li></ul>'
            '</div></div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON section builders
    # ------------------------------------------------------------------ #

    def _json_scenario_comparison(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON scenario comparison."""
        scenarios: Dict[str, Any] = data.get("scenarios", {})
        result: Dict[str, Any] = {}

        for key in ["low", "mid", "high"]:
            s = scenarios.get(key, {})
            result[key] = {
                "ets_price_eur": round(s.get("ets_price_eur", 0.0), 2),
                "net_certificates": round(s.get("net_certificates", 0.0), 2),
                "annual_cost_eur": round(s.get("annual_cost_eur", 0.0), 2),
                "three_year_total_eur": round(s.get("three_year_total_eur", 0.0), 2),
            }

        return result

    def _json_annual_forecast(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON annual forecast."""
        forecast: List[Dict[str, Any]] = data.get("annual_forecast", [])
        return [
            {
                "year": e.get("year", 0),
                "net_emissions_tco2e": round(e.get("net_emissions_tco2e", 0.0), 2),
                "cost_low_eur": round(e.get("cost_low_eur", 0.0), 2),
                "cost_mid_eur": round(e.get("cost_mid_eur", 0.0), 2),
                "cost_high_eur": round(e.get("cost_high_eur", 0.0), 2),
            }
            for e in forecast
        ]

    def _json_cost_by_category(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON cost by category."""
        categories: List[Dict[str, Any]] = data.get("cost_by_category", [])
        total = sum(c.get("estimated_cost_eur", 0.0) for c in categories)

        return [
            {
                "category": c.get("category", ""),
                "emissions_tco2e": round(c.get("emissions_tco2e", 0.0), 2),
                "net_certificates": round(c.get("net_certificates", 0.0), 2),
                "estimated_cost_eur": round(c.get("estimated_cost_eur", 0.0), 2),
                "share_pct": round(
                    (c.get("estimated_cost_eur", 0.0) / total * 100) if total > 0 else 0.0, 2
                ),
            }
            for c in categories
        ]

    def _json_free_allocation_impact(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON free allocation impact."""
        impact: Dict[str, Any] = data.get("free_allocation_impact", {})
        yearly: List[Dict[str, Any]] = impact.get("yearly", [])

        return {
            "yearly": [
                {
                    "year": e.get("year", 0),
                    "free_allocation_pct": round(
                        e.get("free_allocation_pct",
                              FREE_ALLOCATION_RATES.get(e.get("year", 0), 0.0)), 2
                    ),
                    "cbam_coverage_pct": round(
                        100.0 - e.get("free_allocation_pct",
                                      FREE_ALLOCATION_RATES.get(e.get("year", 0), 0.0)), 2
                    ),
                    "additional_cost_eur": round(e.get("additional_cost_eur", 0.0), 2),
                }
                for e in yearly
            ],
        }

    def _json_carbon_deductions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON carbon deductions."""
        deductions: List[Dict[str, Any]] = data.get("carbon_deductions", [])
        total = sum(d.get("savings_eur", 0.0) for d in deductions)

        return {
            "total_savings_eur": round(total, 2),
            "deductions": [
                {
                    "country": d.get("country", ""),
                    "carbon_price_eur": round(d.get("carbon_price_eur", 0.0), 2),
                    "applicable_emissions_tco2e": round(d.get("applicable_emissions", 0.0), 2),
                    "deduction_certificates": round(d.get("deduction_certificates", 0.0), 2),
                    "savings_eur": round(d.get("savings_eur", 0.0), 2),
                }
                for d in deductions
            ],
        }

    def _json_sensitivity_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON sensitivity analysis."""
        sensitivity: Dict[str, Any] = data.get("sensitivity", {})
        base_cost = sensitivity.get("base_cost_eur", 0.0)
        base_emissions = sensitivity.get("base_emissions_tco2e", 0.0)

        variants = []
        for level in self.SENSITIVITY_LEVELS:
            adjusted_cost = base_cost * level["factor"]
            variants.append({
                "label": level["label"],
                "factor": level["factor"],
                "adjusted_emissions_tco2e": round(base_emissions * level["factor"], 2),
                "adjusted_cost_eur": round(adjusted_cost, 2),
                "cost_delta_eur": round(adjusted_cost - base_cost, 2),
            })

        return {
            "base_emissions_tco2e": round(base_emissions, 2),
            "base_cost_eur": round(base_cost, 2),
            "variants": variants,
        }

    def _json_budget_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON budget summary."""
        budget: Dict[str, Any] = data.get("budget_summary", {})
        recommended = budget.get("recommended_budget_eur", 0.0)
        contingency_pct = budget.get("contingency_pct", 15.0)
        contingency_amount = recommended * (contingency_pct / 100.0)

        return {
            "recommended_budget_eur": round(recommended, 2),
            "contingency_pct": round(contingency_pct, 2),
            "contingency_amount_eur": round(contingency_amount, 2),
            "total_with_contingency_eur": round(recommended + contingency_amount, 2),
            "ets_scenario": budget.get("ets_scenario", "mid"),
            "emission_trend": budget.get("emission_trend", "flat"),
        }

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
            ".report-header h1{margin:0 0 8px 0;font-size:24px}"
            ".meta-item{font-size:13px;opacity:0.8}"
            ".section{background:#fff;padding:20px;border-radius:8px;"
            "margin-bottom:16px;box-shadow:0 1px 3px rgba(0,0,0,0.1)}"
            ".section h2{margin:0 0 16px 0;font-size:18px;color:#1a5276;"
            "border-bottom:2px solid #eef2f7;padding-bottom:8px}"
            "table{width:100%;border-collapse:collapse;font-size:14px}"
            "th{background:#eef2f7;padding:10px 12px;text-align:left;font-weight:600}"
            "td{padding:8px 12px;border-bottom:1px solid #eef2f7}"
            ".num{text-align:right;font-variant-numeric:tabular-nums}"
            ".highlight-row{background:#fff3cd}"
            ".scenario-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}"
            ".scenario-card{background:#f8f9fa;padding:20px;border-radius:8px;text-align:center}"
            ".scenario-card h3{margin:0 0 8px 0;font-size:16px}"
            ".scenario-detail{font-size:13px;color:#7f8c8d;margin-top:8px}"
            ".kpi-card{background:#f8f9fa;padding:16px;border-radius:8px}"
            ".kpi-label{font-size:13px;color:#7f8c8d;margin-bottom:4px}"
            ".kpi-value{font-size:24px;font-weight:700;color:#1a5276}"
            ".bar-chart{display:flex;align-items:center;gap:8px}"
            ".bar-fill{height:12px;background:#3498db;border-radius:4px}"
            ".bar-chart span{font-size:12px;white-space:nowrap}"
            ".budget-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;"
            "margin-bottom:16px}"
            ".budget-card{background:#f8f9fa;padding:16px;border-radius:8px;text-align:center}"
            ".budget-card.highlight{background:#1a5276;color:#fff}"
            ".budget-card.highlight .kpi-label{color:#b0c4de}"
            ".budget-card.highlight .kpi-value{color:#fff}"
            ".assumptions{background:#f8f9fa;padding:12px 16px;border-radius:8px;font-size:14px}"
            ".assumptions ul{margin:8px 0;padding-left:20px}"
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
            f'Template: CostProjectionTemplate v1.0 | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )
