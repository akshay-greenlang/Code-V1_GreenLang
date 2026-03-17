"""
AnnualDeclarationTemplate - CBAM annual declaration report template.

This module implements the annual CBAM declaration template for the definitive
period. It generates formatted reports containing annual import summaries,
certificate obligations, cost summaries, quarterly breakdowns, supplier data
coverage, and free allocation phase-out schedules.

Example:
    >>> template = AnnualDeclarationTemplate()
    >>> data = {"year": 2027, "importer_eori": "DE123456789000001", ...}
    >>> markdown = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


# Free allocation phase-out schedule (CBAM Regulation Article 31)
FREE_ALLOCATION_SCHEDULE: Dict[int, float] = {
    2026: 97.5,
    2027: 95.0,
    2028: 90.0,
    2029: 77.5,
    2030: 51.5,
    2031: 39.0,
    2032: 26.5,
    2033: 14.0,
    2034: 0.0,
}


class AnnualDeclarationTemplate:
    """
    CBAM annual declaration report template.

    Generates formatted annual declaration reports for CBAM importers during
    the definitive period. Includes import summaries, certificate obligations,
    cost summaries, quarterly breakdowns, and free allocation phase-out tracking.

    Attributes:
        config: Optional configuration dictionary for template customization.
        generated_at: Timestamp of report generation.

    Example:
        >>> template = AnnualDeclarationTemplate()
        >>> result = template.render_json({"year": 2027, ...})
        >>> assert "provenance_hash" in result
    """

    GOODS_CATEGORIES: List[str] = [
        "cement",
        "steel",
        "aluminium",
        "fertilizers",
        "electricity",
        "hydrogen",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize AnnualDeclarationTemplate.

        Args:
            config: Optional configuration dictionary. Supported keys:
                - decimal_places (int): Number of decimal places for numbers.
                - currency (str): Currency code for cost formatting.
                - ets_price_eur (float): Current EU ETS price for estimates.
        """
        self.config = config or {}
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public render methods
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """
        Render the annual declaration as Markdown.

        Args:
            data: Report data dictionary containing:
                - year (int): Reporting year
                - importer_eori (str): EORI number
                - authorized_declarant (dict): name, role, contact
                - imports_by_category (list[dict]): total imports per category
                - total_embedded_emissions (float): total tCO2e
                - yoy_comparison (dict): year-over-year changes
                - certificate_obligation (dict): gross, deductions, net
                - cost_summary (dict): estimated costs
                - quarterly_breakdown (list[dict]): Q1-Q4 data
                - supplier_data_coverage (dict): actual vs default percentages
                - free_allocation (dict): current year rate and phase-out

        Returns:
            Formatted Markdown string with provenance footer.
        """
        sections: List[str] = []

        sections.append(self._md_header(data))
        sections.append(self._md_annual_summary(data))
        sections.append(self._md_yoy_comparison(data))
        sections.append(self._md_certificate_obligation(data))
        sections.append(self._md_cost_summary(data))
        sections.append(self._md_quarterly_breakdown(data))
        sections.append(self._md_supplier_data_coverage(data))
        sections.append(self._md_free_allocation_schedule(data))
        sections.append(self._md_provenance_footer(data))

        content = "\n\n".join(sections)
        provenance_hash = self._generate_provenance_hash(content)
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the annual declaration as self-contained HTML.

        Args:
            data: Report data dictionary (same schema as render_markdown).

        Returns:
            Complete HTML document string with inline CSS.
        """
        sections: List[str] = []

        sections.append(self._html_header(data))
        sections.append(self._html_annual_summary(data))
        sections.append(self._html_yoy_comparison(data))
        sections.append(self._html_certificate_obligation(data))
        sections.append(self._html_cost_summary(data))
        sections.append(self._html_quarterly_breakdown(data))
        sections.append(self._html_supplier_data_coverage(data))
        sections.append(self._html_free_allocation_schedule(data))

        body = "\n".join(sections)
        provenance_hash = self._generate_provenance_hash(body)

        return self._wrap_html(
            title=f"CBAM Annual Declaration - {data.get('year', '')}",
            body=body,
            provenance_hash=provenance_hash,
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the annual declaration as a structured JSON-compatible dict.

        Args:
            data: Report data dictionary (same schema as render_markdown).

        Returns:
            Dictionary with all report sections, metadata, and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "cbam_annual_declaration",
            "generated_at": self.generated_at,
            "year": data.get("year", 0),
            "importer_eori": data.get("importer_eori", ""),
            "authorized_declarant": data.get("authorized_declarant", {}),
            "annual_summary": self._json_annual_summary(data),
            "yoy_comparison": self._json_yoy_comparison(data),
            "certificate_obligation": self._json_certificate_obligation(data),
            "cost_summary": self._json_cost_summary(data),
            "quarterly_breakdown": self._json_quarterly_breakdown(data),
            "supplier_data_coverage": self._json_supplier_data_coverage(data),
            "free_allocation": self._json_free_allocation_schedule(data),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._generate_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown section builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Build Markdown header section."""
        year = data.get("year", "N/A")
        eori = data.get("importer_eori", "N/A")
        declarant = data.get("authorized_declarant", {})

        return (
            f"# CBAM Annual Declaration - {year}\n\n"
            f"**Importer EORI:** {eori}\n\n"
            f"**Authorized Declarant:**\n\n"
            f"- Name: {declarant.get('name', 'N/A')}\n"
            f"- Role: {declarant.get('role', 'N/A')}\n"
            f"- Contact: {declarant.get('contact', 'N/A')}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_annual_summary(self, data: Dict[str, Any]) -> str:
        """Build Markdown annual import summary table."""
        imports: List[Dict[str, Any]] = data.get("imports_by_category", [])
        total_emissions = data.get("total_embedded_emissions", 0.0)

        header = (
            "## Annual Import Summary\n\n"
            "| Category | Quantity (tonnes) | Embedded Emissions (tCO2e) | "
            "Share (%) |\n"
            "|----------|-------------------|---------------------------|"
            "---------|\n"
        )

        rows: List[str] = []
        total_qty = 0.0
        computed_total = 0.0

        for imp in imports:
            cat = imp.get("category", "").capitalize()
            qty = imp.get("quantity_tonnes", 0.0)
            emissions = imp.get("embedded_emissions_tco2e", 0.0)
            share = (emissions / total_emissions * 100) if total_emissions > 0 else 0.0
            total_qty += qty
            computed_total += emissions

            rows.append(
                f"| {cat} | {self._format_number(qty)} | "
                f"{self._format_number(emissions)} | "
                f"{self._format_percentage(share)} |"
            )

        rows.append(
            f"| **TOTAL** | **{self._format_number(total_qty)}** | "
            f"**{self._format_number(computed_total)}** | **100.00%** |"
        )

        return header + "\n".join(rows)

    def _md_yoy_comparison(self, data: Dict[str, Any]) -> str:
        """Build Markdown year-over-year comparison section."""
        yoy: Dict[str, Any] = data.get("yoy_comparison", {})

        current_year = data.get("year", "N/A")
        prev_year = yoy.get("previous_year", "N/A")
        current_emissions = yoy.get("current_emissions", 0.0)
        previous_emissions = yoy.get("previous_emissions", 0.0)
        change_pct = yoy.get("change_pct", 0.0)
        change_abs = yoy.get("change_absolute", 0.0)

        direction = "increase" if change_pct > 0 else "decrease" if change_pct < 0 else "no change"

        return (
            "## Year-over-Year Comparison\n\n"
            f"| Metric | {prev_year} | {current_year} | Change |\n"
            f"|--------|------|------|--------|\n"
            f"| Total Emissions (tCO2e) | "
            f"{self._format_number(previous_emissions)} | "
            f"{self._format_number(current_emissions)} | "
            f"{self._format_number(change_abs)} ({'+' if change_pct > 0 else ''}"
            f"{self._format_percentage(change_pct)}) |\n\n"
            f"**Trend:** {self._format_percentage(abs(change_pct))} {direction} "
            f"compared to {prev_year}."
        )

    def _md_certificate_obligation(self, data: Dict[str, Any]) -> str:
        """Build Markdown certificate obligation section."""
        cert: Dict[str, Any] = data.get("certificate_obligation", {})

        gross = cert.get("gross_certificates", 0.0)
        free_alloc = cert.get("free_allocation_deduction", 0.0)
        carbon_price = cert.get("carbon_price_deduction", 0.0)
        net = cert.get("net_obligation", 0.0)

        return (
            "## Certificate Obligation\n\n"
            "| Component | Certificates |\n"
            "|-----------|-------------|\n"
            f"| Gross obligation | {self._format_number(gross)} |\n"
            f"| (-) Free allocation deduction | -{self._format_number(free_alloc)} |\n"
            f"| (-) Carbon price deduction | -{self._format_number(carbon_price)} |\n"
            f"| **Net obligation** | **{self._format_number(net)}** |\n\n"
            f"> One CBAM certificate = 1 tCO2e. Net certificates must be "
            f"surrendered by 31 May of the following year."
        )

    def _md_cost_summary(self, data: Dict[str, Any]) -> str:
        """Build Markdown cost summary section."""
        cost: Dict[str, Any] = data.get("cost_summary", {})

        ets_price = cost.get("ets_price_eur", 0.0)
        total_cost = cost.get("estimated_total_cost_eur", 0.0)
        cost_by_category: List[Dict[str, Any]] = cost.get("by_category", [])

        currency = self.config.get("currency", "EUR")

        header = (
            "## Cost Summary\n\n"
            f"**EU ETS Price (reference):** {self._format_currency(ets_price, currency)}/tCO2e\n\n"
            f"**Estimated Total Cost:** {self._format_currency(total_cost, currency)}\n\n"
            "| Category | Net Certificates | Estimated Cost |\n"
            "|----------|------------------|----------------|\n"
        )

        rows: List[str] = []
        for cat_cost in cost_by_category:
            rows.append(
                f"| {cat_cost.get('category', '').capitalize()} | "
                f"{self._format_number(cat_cost.get('net_certificates', 0.0))} | "
                f"{self._format_currency(cat_cost.get('estimated_cost_eur', 0.0), currency)} |"
            )

        return header + "\n".join(rows)

    def _md_quarterly_breakdown(self, data: Dict[str, Any]) -> str:
        """Build Markdown quarterly breakdown section."""
        quarters: List[Dict[str, Any]] = data.get("quarterly_breakdown", [])

        header = (
            "## Quarterly Breakdown\n\n"
            "| Quarter | Quantity (tonnes) | Embedded Emissions (tCO2e) | "
            "Avg Intensity (tCO2e/t) |\n"
            "|---------|-------------------|---------------------------|"
            "------------------------|\n"
        )

        rows: List[str] = []
        for q in quarters:
            qty = q.get("quantity_tonnes", 0.0)
            emissions = q.get("embedded_emissions_tco2e", 0.0)
            intensity = emissions / qty if qty > 0 else 0.0

            rows.append(
                f"| {q.get('quarter', '')} | "
                f"{self._format_number(qty)} | "
                f"{self._format_number(emissions)} | "
                f"{self._format_number(intensity, 4)} |"
            )

        return header + "\n".join(rows)

    def _md_supplier_data_coverage(self, data: Dict[str, Any]) -> str:
        """Build Markdown supplier data coverage section."""
        coverage: Dict[str, Any] = data.get("supplier_data_coverage", {})

        pct_actual = coverage.get("pct_actual", 0.0)
        pct_default = coverage.get("pct_default", 0.0)
        total_suppliers = coverage.get("total_suppliers", 0)
        suppliers_with_actual = coverage.get("suppliers_with_actual_data", 0)
        suppliers_with_default = coverage.get("suppliers_with_default_only", 0)

        return (
            "## Supplier Data Coverage\n\n"
            f"**Total Suppliers:** {total_suppliers}\n\n"
            f"- Suppliers providing actual emission data: "
            f"{suppliers_with_actual} ({self._format_percentage(pct_actual)})\n"
            f"- Suppliers using default values: "
            f"{suppliers_with_default} ({self._format_percentage(pct_default)})\n\n"
            f"> **Target:** 100% actual emission data from all installations by "
            f"end of transitional period."
        )

    def _md_free_allocation_schedule(self, data: Dict[str, Any]) -> str:
        """Build Markdown free allocation phase-out schedule."""
        fa: Dict[str, Any] = data.get("free_allocation", {})
        current_year = data.get("year", 2027)
        current_rate = fa.get("current_rate_pct", FREE_ALLOCATION_SCHEDULE.get(current_year, 0.0))

        header = (
            "## Free Allocation Phase-Out Schedule\n\n"
            f"**Current Year ({current_year}) Rate:** "
            f"{self._format_percentage(current_rate)}\n\n"
            "| Year | Free Allocation Rate (%) | CBAM Coverage (%) |\n"
            "|------|--------------------------|--------------------|\n"
        )

        rows: List[str] = []
        for year, rate in sorted(FREE_ALLOCATION_SCHEDULE.items()):
            cbam_coverage = 100.0 - rate
            marker = " **<-- current**" if year == current_year else ""
            rows.append(
                f"| {year} | {self._format_percentage(rate)} | "
                f"{self._format_percentage(cbam_coverage)} |{marker}"
            )

        return header + "\n".join(rows)

    def _md_provenance_footer(self, data: Dict[str, Any]) -> str:
        """Build Markdown provenance footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            f"*Template: AnnualDeclarationTemplate v1.0*\n\n"
            f"*Reporting year: {data.get('year', 'N/A')}*"
        )

    # ------------------------------------------------------------------ #
    #  HTML section builders
    # ------------------------------------------------------------------ #

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Build HTML header section."""
        year = data.get("year", "N/A")
        eori = data.get("importer_eori", "N/A")
        declarant = data.get("authorized_declarant", {})

        return (
            '<div class="report-header">'
            f'<h1>CBAM Annual Declaration - {year}</h1>'
            '<div class="header-meta">'
            f'<div class="meta-item"><strong>Importer EORI:</strong> {eori}</div>'
            f'<div class="meta-item"><strong>Declarant:</strong> '
            f'{declarant.get("name", "N/A")} ({declarant.get("role", "N/A")})</div>'
            f'<div class="meta-item"><strong>Generated:</strong> {self.generated_at}</div>'
            '</div></div>'
        )

    def _html_annual_summary(self, data: Dict[str, Any]) -> str:
        """Build HTML annual import summary table."""
        imports: List[Dict[str, Any]] = data.get("imports_by_category", [])
        total_emissions = data.get("total_embedded_emissions", 0.0)

        rows_html = ""
        total_qty = 0.0
        computed_total = 0.0

        for imp in imports:
            qty = imp.get("quantity_tonnes", 0.0)
            emissions = imp.get("embedded_emissions_tco2e", 0.0)
            share = (emissions / total_emissions * 100) if total_emissions > 0 else 0.0
            total_qty += qty
            computed_total += emissions

            rows_html += (
                f'<tr><td>{imp.get("category", "").capitalize()}</td>'
                f'<td class="num">{self._format_number(qty)}</td>'
                f'<td class="num">{self._format_number(emissions)}</td>'
                f'<td class="num">{self._format_percentage(share)}</td></tr>'
            )

        rows_html += (
            f'<tr class="total-row"><td><strong>TOTAL</strong></td>'
            f'<td class="num"><strong>{self._format_number(total_qty)}</strong></td>'
            f'<td class="num"><strong>{self._format_number(computed_total)}</strong></td>'
            f'<td class="num"><strong>100.00%</strong></td></tr>'
        )

        return (
            '<div class="section"><h2>Annual Import Summary</h2>'
            '<table><thead><tr>'
            '<th>Category</th><th>Quantity (tonnes)</th>'
            '<th>Embedded Emissions (tCO2e)</th><th>Share (%)</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_yoy_comparison(self, data: Dict[str, Any]) -> str:
        """Build HTML year-over-year comparison."""
        yoy: Dict[str, Any] = data.get("yoy_comparison", {})
        current_year = data.get("year", "N/A")
        prev_year = yoy.get("previous_year", "N/A")
        current_em = yoy.get("current_emissions", 0.0)
        previous_em = yoy.get("previous_emissions", 0.0)
        change_pct = yoy.get("change_pct", 0.0)

        color = "#e74c3c" if change_pct > 0 else "#2ecc71" if change_pct < 0 else "#95a5a6"
        arrow = "^" if change_pct > 0 else "v" if change_pct < 0 else "="

        return (
            '<div class="section"><h2>Year-over-Year Comparison</h2>'
            '<div class="kpi-grid">'
            f'<div class="kpi-card"><div class="kpi-label">{prev_year} Emissions</div>'
            f'<div class="kpi-value">{self._format_number(previous_em)} tCO2e</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">{current_year} Emissions</div>'
            f'<div class="kpi-value">{self._format_number(current_em)} tCO2e</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">Change</div>'
            f'<div class="kpi-value" style="color:{color}">'
            f'{arrow} {self._format_percentage(abs(change_pct))}</div></div>'
            '</div></div>'
        )

    def _html_certificate_obligation(self, data: Dict[str, Any]) -> str:
        """Build HTML certificate obligation waterfall."""
        cert: Dict[str, Any] = data.get("certificate_obligation", {})

        gross = cert.get("gross_certificates", 0.0)
        free_alloc = cert.get("free_allocation_deduction", 0.0)
        carbon_price = cert.get("carbon_price_deduction", 0.0)
        net = cert.get("net_obligation", 0.0)

        max_val = max(gross, 1)  # Avoid division by zero

        return (
            '<div class="section"><h2>Certificate Obligation</h2>'
            '<div class="waterfall">'
            f'<div class="waterfall-item">'
            f'<div class="waterfall-label">Gross obligation</div>'
            f'<div class="waterfall-bar positive" style="width:{gross / max_val * 100:.0f}%">'
            f'{self._format_number(gross)}</div></div>'
            f'<div class="waterfall-item">'
            f'<div class="waterfall-label">(-) Free allocation</div>'
            f'<div class="waterfall-bar negative" style="width:{free_alloc / max_val * 100:.0f}%">'
            f'-{self._format_number(free_alloc)}</div></div>'
            f'<div class="waterfall-item">'
            f'<div class="waterfall-label">(-) Carbon price deduction</div>'
            f'<div class="waterfall-bar negative" style="width:{carbon_price / max_val * 100:.0f}%">'
            f'-{self._format_number(carbon_price)}</div></div>'
            f'<div class="waterfall-item">'
            f'<div class="waterfall-label"><strong>Net obligation</strong></div>'
            f'<div class="waterfall-bar net" style="width:{net / max_val * 100:.0f}%">'
            f'<strong>{self._format_number(net)}</strong></div></div>'
            '</div>'
            '<p class="note">One CBAM certificate = 1 tCO2e. '
            'Net certificates must be surrendered by 31 May of the following year.</p>'
            '</div>'
        )

    def _html_cost_summary(self, data: Dict[str, Any]) -> str:
        """Build HTML cost summary section."""
        cost: Dict[str, Any] = data.get("cost_summary", {})
        currency = self.config.get("currency", "EUR")

        ets_price = cost.get("ets_price_eur", 0.0)
        total_cost = cost.get("estimated_total_cost_eur", 0.0)
        cost_by_category: List[Dict[str, Any]] = cost.get("by_category", [])

        rows_html = ""
        for cat_cost in cost_by_category:
            rows_html += (
                f'<tr><td>{cat_cost.get("category", "").capitalize()}</td>'
                f'<td class="num">{self._format_number(cat_cost.get("net_certificates", 0.0))}</td>'
                f'<td class="num">{self._format_currency(cat_cost.get("estimated_cost_eur", 0.0), currency)}</td></tr>'
            )

        return (
            '<div class="section"><h2>Cost Summary</h2>'
            '<div class="kpi-grid">'
            f'<div class="kpi-card"><div class="kpi-label">EU ETS Reference Price</div>'
            f'<div class="kpi-value">{self._format_currency(ets_price, currency)}/tCO2e</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">Estimated Total Cost</div>'
            f'<div class="kpi-value">{self._format_currency(total_cost, currency)}</div></div>'
            '</div>'
            '<table><thead><tr>'
            '<th>Category</th><th>Net Certificates</th><th>Estimated Cost</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_quarterly_breakdown(self, data: Dict[str, Any]) -> str:
        """Build HTML quarterly breakdown section."""
        quarters: List[Dict[str, Any]] = data.get("quarterly_breakdown", [])

        rows_html = ""
        for q in quarters:
            qty = q.get("quantity_tonnes", 0.0)
            emissions = q.get("embedded_emissions_tco2e", 0.0)
            intensity = emissions / qty if qty > 0 else 0.0

            rows_html += (
                f'<tr><td>{q.get("quarter", "")}</td>'
                f'<td class="num">{self._format_number(qty)}</td>'
                f'<td class="num">{self._format_number(emissions)}</td>'
                f'<td class="num">{self._format_number(intensity, 4)}</td></tr>'
            )

        return (
            '<div class="section"><h2>Quarterly Breakdown</h2>'
            '<table><thead><tr>'
            '<th>Quarter</th><th>Quantity (tonnes)</th>'
            '<th>Embedded Emissions (tCO2e)</th><th>Avg Intensity (tCO2e/t)</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    def _html_supplier_data_coverage(self, data: Dict[str, Any]) -> str:
        """Build HTML supplier data coverage section."""
        coverage: Dict[str, Any] = data.get("supplier_data_coverage", {})

        pct_actual = coverage.get("pct_actual", 0.0)
        pct_default = coverage.get("pct_default", 0.0)
        total = coverage.get("total_suppliers", 0)
        with_actual = coverage.get("suppliers_with_actual_data", 0)

        return (
            '<div class="section"><h2>Supplier Data Coverage</h2>'
            '<div class="kpi-grid">'
            f'<div class="kpi-card"><div class="kpi-label">Total Suppliers</div>'
            f'<div class="kpi-value">{total}</div></div>'
            f'<div class="kpi-card"><div class="kpi-label">Actual Data</div>'
            f'<div class="kpi-value">{with_actual} ({self._format_percentage(pct_actual)})</div></div>'
            '</div>'
            f'<div class="progress-container">'
            f'<div class="progress-label">Actual emissions data coverage</div>'
            f'{self._html_progress_bar(pct_actual, "#2ecc71")}'
            f'<div class="progress-label">Default values used</div>'
            f'{self._html_progress_bar(pct_default, "#e67e22")}'
            f'</div></div>'
        )

    def _html_free_allocation_schedule(self, data: Dict[str, Any]) -> str:
        """Build HTML free allocation phase-out schedule."""
        current_year = data.get("year", 2027)

        rows_html = ""
        for year, rate in sorted(FREE_ALLOCATION_SCHEDULE.items()):
            cbam_coverage = 100.0 - rate
            row_class = ' class="highlight-row"' if year == current_year else ""
            rows_html += (
                f'<tr{row_class}><td>{year}</td>'
                f'<td class="num">{self._format_percentage(rate)}</td>'
                f'<td class="num">{self._format_percentage(cbam_coverage)}</td></tr>'
            )

        return (
            '<div class="section"><h2>Free Allocation Phase-Out Schedule</h2>'
            '<table><thead><tr>'
            '<th>Year</th><th>Free Allocation Rate (%)</th><th>CBAM Coverage (%)</th>'
            f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON section builders
    # ------------------------------------------------------------------ #

    def _json_annual_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON annual import summary."""
        imports: List[Dict[str, Any]] = data.get("imports_by_category", [])
        total_emissions = data.get("total_embedded_emissions", 0.0)

        items = []
        total_qty = 0.0

        for imp in imports:
            qty = imp.get("quantity_tonnes", 0.0)
            emissions = imp.get("embedded_emissions_tco2e", 0.0)
            share = (emissions / total_emissions * 100) if total_emissions > 0 else 0.0
            total_qty += qty

            items.append({
                "category": imp.get("category", ""),
                "quantity_tonnes": round(qty, 2),
                "embedded_emissions_tco2e": round(emissions, 2),
                "share_pct": round(share, 2),
            })

        return {
            "items": items,
            "total_quantity_tonnes": round(total_qty, 2),
            "total_embedded_emissions_tco2e": round(total_emissions, 2),
        }

    def _json_yoy_comparison(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON year-over-year comparison."""
        yoy: Dict[str, Any] = data.get("yoy_comparison", {})
        return {
            "previous_year": yoy.get("previous_year", ""),
            "current_year": data.get("year", ""),
            "previous_emissions_tco2e": round(yoy.get("previous_emissions", 0.0), 2),
            "current_emissions_tco2e": round(yoy.get("current_emissions", 0.0), 2),
            "change_absolute_tco2e": round(yoy.get("change_absolute", 0.0), 2),
            "change_pct": round(yoy.get("change_pct", 0.0), 2),
        }

    def _json_certificate_obligation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON certificate obligation."""
        cert: Dict[str, Any] = data.get("certificate_obligation", {})
        return {
            "gross_certificates": round(cert.get("gross_certificates", 0.0), 2),
            "free_allocation_deduction": round(cert.get("free_allocation_deduction", 0.0), 2),
            "carbon_price_deduction": round(cert.get("carbon_price_deduction", 0.0), 2),
            "net_obligation": round(cert.get("net_obligation", 0.0), 2),
        }

    def _json_cost_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON cost summary."""
        cost: Dict[str, Any] = data.get("cost_summary", {})
        return {
            "ets_price_eur": round(cost.get("ets_price_eur", 0.0), 2),
            "estimated_total_cost_eur": round(cost.get("estimated_total_cost_eur", 0.0), 2),
            "by_category": cost.get("by_category", []),
        }

    def _json_quarterly_breakdown(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON quarterly breakdown."""
        quarters: List[Dict[str, Any]] = data.get("quarterly_breakdown", [])

        result: List[Dict[str, Any]] = []
        for q in quarters:
            qty = q.get("quantity_tonnes", 0.0)
            emissions = q.get("embedded_emissions_tco2e", 0.0)
            intensity = emissions / qty if qty > 0 else 0.0

            result.append({
                "quarter": q.get("quarter", ""),
                "quantity_tonnes": round(qty, 2),
                "embedded_emissions_tco2e": round(emissions, 2),
                "avg_intensity_tco2e_per_t": round(intensity, 4),
            })

        return result

    def _json_supplier_data_coverage(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON supplier data coverage."""
        coverage: Dict[str, Any] = data.get("supplier_data_coverage", {})
        return {
            "total_suppliers": coverage.get("total_suppliers", 0),
            "suppliers_with_actual_data": coverage.get("suppliers_with_actual_data", 0),
            "suppliers_with_default_only": coverage.get("suppliers_with_default_only", 0),
            "pct_actual": round(coverage.get("pct_actual", 0.0), 2),
            "pct_default": round(coverage.get("pct_default", 0.0), 2),
        }

    def _json_free_allocation_schedule(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON free allocation phase-out schedule."""
        current_year = data.get("year", 2027)
        fa: Dict[str, Any] = data.get("free_allocation", {})
        current_rate = fa.get("current_rate_pct", FREE_ALLOCATION_SCHEDULE.get(current_year, 0.0))

        schedule = []
        for year, rate in sorted(FREE_ALLOCATION_SCHEDULE.items()):
            schedule.append({
                "year": year,
                "free_allocation_rate_pct": rate,
                "cbam_coverage_pct": 100.0 - rate,
                "is_current": year == current_year,
            })

        return {
            "current_year": current_year,
            "current_rate_pct": current_rate,
            "schedule": schedule,
        }

    # ------------------------------------------------------------------ #
    #  Helper methods
    # ------------------------------------------------------------------ #

    def _generate_provenance_hash(self, content: str) -> str:
        """
        Generate SHA-256 provenance hash for audit trail.

        Args:
            content: String content to hash.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
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
            ".total-row{background:#f8f9fa;font-weight:600}"
            ".highlight-row{background:#fff3cd}"
            ".note{color:#7f8c8d;font-style:italic;font-size:13px}"
            ".kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin-bottom:16px}"
            ".kpi-card{background:#f8f9fa;padding:16px;border-radius:8px;text-align:center}"
            ".kpi-label{font-size:13px;color:#7f8c8d;margin-bottom:4px}"
            ".kpi-value{font-size:24px;font-weight:700;color:#1a5276}"
            ".waterfall{margin:16px 0}"
            ".waterfall-item{display:flex;align-items:center;margin-bottom:8px}"
            ".waterfall-label{width:220px;font-size:14px}"
            ".waterfall-bar{padding:6px 12px;border-radius:4px;color:#fff;font-size:13px;"
            "min-width:60px;text-align:right}"
            ".waterfall-bar.positive{background:#2ecc71}"
            ".waterfall-bar.negative{background:#e74c3c}"
            ".waterfall-bar.net{background:#1a5276}"
            ".progress-bar{background:#ecf0f1;border-radius:4px;height:12px;"
            "overflow:hidden;margin:4px 0}"
            ".progress-fill{height:100%;border-radius:4px}"
            ".progress-container{margin:16px 0}"
            ".progress-label{font-size:13px;color:#7f8c8d;margin-top:8px}"
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
            f'Template: AnnualDeclarationTemplate v1.0 | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )
