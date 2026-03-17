"""
QuarterlyReportTemplate - CBAM transitional quarterly report template.

This module implements the quarterly report template for CBAM transitional
period reporting. It generates formatted reports containing goods summaries,
emission breakdowns by category and country of origin, installation-level
details, calculation method summaries, and compliance status indicators.

Example:
    >>> template = QuarterlyReportTemplate()
    >>> data = {"reporting_period": "Q1 2026", "importer_eori": "DE123456789000001", ...}
    >>> markdown = template.render_markdown(data)
    >>> html = template.render_html(data)
    >>> json_out = template.render_json(data)
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union


class QuarterlyReportTemplate:
    """
    CBAM transitional quarterly report template.

    Generates formatted quarterly reports for CBAM importers during the
    transitional period. Includes goods summaries, emission breakdowns,
    installation-level details, and compliance status.

    Attributes:
        config: Optional configuration dictionary for template customization.
        generated_at: Timestamp of report generation.

    Example:
        >>> template = QuarterlyReportTemplate()
        >>> result = template.render_json({"reporting_period": "Q1 2026", ...})
        >>> assert "provenance_hash" in result
    """

    # CBAM goods categories as defined in Annex I
    GOODS_CATEGORIES: List[str] = [
        "cement",
        "steel",
        "aluminium",
        "fertilizers",
        "electricity",
        "hydrogen",
    ]

    # Calculation method types
    CALC_METHODS: List[str] = [
        "actual",
        "default",
        "country_default",
    ]

    # Data quality tiers
    QUALITY_TIERS: Dict[str, str] = {
        "excellent": "Verified actual data from installation",
        "good": "Actual data, not yet verified",
        "fair": "Estimated using country-specific defaults",
        "poor": "EU default values applied",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize QuarterlyReportTemplate.

        Args:
            config: Optional configuration dictionary. Supported keys:
                - decimal_places (int): Number of decimal places for numbers.
                - currency (str): Currency code for cost formatting.
                - include_installation_detail (bool): Whether to include
                  installation-level breakdowns.
        """
        self.config = config or {}
        self.generated_at: str = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------ #
    #  Public render methods
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """
        Render the quarterly report as Markdown.

        Args:
            data: Report data dictionary containing:
                - reporting_period (str): e.g. "Q1 2026"
                - importer_eori (str): EORI number
                - authorized_declarant (dict): name, role, contact
                - goods (list[dict]): goods line items
                - installations (list[dict]): installation-level records
                - calculation_methods (dict): method usage percentages
                - data_quality (dict): quality indicators
                - compliance (dict): compliance status and rules

        Returns:
            Formatted Markdown string with provenance footer.
        """
        sections: List[str] = []

        sections.append(self._md_header(data))
        sections.append(self._md_goods_summary(data))
        sections.append(self._md_category_breakdown(data))
        sections.append(self._md_country_breakdown(data))
        sections.append(self._md_installation_detail(data))
        sections.append(self._md_calculation_method_summary(data))
        sections.append(self._md_data_quality(data))
        sections.append(self._md_compliance_status(data))
        sections.append(self._md_provenance_footer(data))

        content = "\n\n".join(sections)
        provenance_hash = self._generate_provenance_hash(content)
        content += f"\n\n<!-- provenance_hash: {provenance_hash} -->"
        return content

    def render_html(self, data: Dict[str, Any]) -> str:
        """
        Render the quarterly report as self-contained HTML.

        The HTML includes inline CSS with no external dependencies,
        suitable for email embedding or standalone viewing.

        Args:
            data: Report data dictionary (same schema as render_markdown).

        Returns:
            Complete HTML document string.
        """
        sections: List[str] = []

        sections.append(self._html_header(data))
        sections.append(self._html_goods_summary(data))
        sections.append(self._html_category_breakdown(data))
        sections.append(self._html_country_breakdown(data))
        sections.append(self._html_installation_detail(data))
        sections.append(self._html_calculation_method_summary(data))
        sections.append(self._html_data_quality(data))
        sections.append(self._html_compliance_status(data))

        body = "\n".join(sections)
        provenance_hash = self._generate_provenance_hash(body)

        html = self._wrap_html(
            title=f"CBAM Quarterly Report - {data.get('reporting_period', '')}",
            body=body,
            provenance_hash=provenance_hash,
        )
        return html

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Render the quarterly report as a structured JSON-compatible dict.

        Args:
            data: Report data dictionary (same schema as render_markdown).

        Returns:
            Dictionary with all report sections, metadata, and provenance hash.
        """
        report: Dict[str, Any] = {
            "report_type": "cbam_quarterly_report",
            "generated_at": self.generated_at,
            "reporting_period": data.get("reporting_period", ""),
            "importer_eori": data.get("importer_eori", ""),
            "authorized_declarant": data.get("authorized_declarant", {}),
            "goods_summary": self._json_goods_summary(data),
            "category_breakdown": self._json_category_breakdown(data),
            "country_breakdown": self._json_country_breakdown(data),
            "installation_detail": self._json_installation_detail(data),
            "calculation_methods": self._json_calculation_method_summary(data),
            "data_quality": self._json_data_quality(data),
            "compliance_status": self._json_compliance_status(data),
        }

        content_str = json.dumps(report, sort_keys=True, default=str)
        report["provenance_hash"] = self._generate_provenance_hash(content_str)
        return report

    # ------------------------------------------------------------------ #
    #  Markdown section builders
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Build Markdown header section with reporting period and declarant."""
        period = data.get("reporting_period", "N/A")
        eori = data.get("importer_eori", "N/A")
        declarant = data.get("authorized_declarant", {})
        declarant_name = declarant.get("name", "N/A")
        declarant_role = declarant.get("role", "N/A")
        declarant_contact = declarant.get("contact", "N/A")

        return (
            f"# CBAM Quarterly Report\n\n"
            f"**Reporting Period:** {period}\n\n"
            f"**Importer EORI:** {eori}\n\n"
            f"**Authorized Declarant:**\n\n"
            f"- Name: {declarant_name}\n"
            f"- Role: {declarant_role}\n"
            f"- Contact: {declarant_contact}\n\n"
            f"**Generated:** {self.generated_at}"
        )

    def _md_goods_summary(self, data: Dict[str, Any]) -> str:
        """Build Markdown goods summary table."""
        goods: List[Dict[str, Any]] = data.get("goods", [])

        header = (
            "## Goods Summary\n\n"
            "| CN Code | Description | Country of Origin | "
            "Quantity (tonnes) | Embedded Emissions (tCO2e) | "
            "Emission Intensity (tCO2e/t) |\n"
            "|---------|-------------|-------------------|"
            "-------------------|---------------------------|"
            "-----------------------------|\n"
        )

        rows: List[str] = []
        total_qty = 0.0
        total_emissions = 0.0

        for g in goods:
            cn_code = self._format_cn_code(g.get("cn_code", ""))
            desc = g.get("description", "")
            country = g.get("country_of_origin", "")
            qty = g.get("quantity_tonnes", 0.0)
            emissions = g.get("embedded_emissions_tco2e", 0.0)
            intensity = emissions / qty if qty > 0 else 0.0

            total_qty += qty
            total_emissions += emissions

            rows.append(
                f"| {cn_code} | {desc} | {country} | "
                f"{self._format_number(qty)} | "
                f"{self._format_number(emissions)} | "
                f"{self._format_number(intensity, 4)} |"
            )

        total_intensity = total_emissions / total_qty if total_qty > 0 else 0.0
        rows.append(
            f"| **TOTAL** | | | "
            f"**{self._format_number(total_qty)}** | "
            f"**{self._format_number(total_emissions)}** | "
            f"**{self._format_number(total_intensity, 4)}** |"
        )

        return header + "\n".join(rows)

    def _md_category_breakdown(self, data: Dict[str, Any]) -> str:
        """Build Markdown breakdown by CBAM goods category."""
        goods: List[Dict[str, Any]] = data.get("goods", [])

        # Aggregate by category
        categories: Dict[str, Dict[str, float]] = {}
        for g in goods:
            cat = g.get("category", "other").lower()
            if cat not in categories:
                categories[cat] = {"quantity": 0.0, "emissions": 0.0}
            categories[cat]["quantity"] += g.get("quantity_tonnes", 0.0)
            categories[cat]["emissions"] += g.get("embedded_emissions_tco2e", 0.0)

        header = (
            "## Breakdown by Goods Category\n\n"
            "| Category | Quantity (tonnes) | Embedded Emissions (tCO2e) | "
            "Share of Total Emissions |\n"
            "|----------|-------------------|---------------------------|"
            "-------------------------|\n"
        )

        total_emissions = sum(c["emissions"] for c in categories.values())
        rows: List[str] = []

        for cat_name in self.GOODS_CATEGORIES:
            if cat_name in categories:
                cat = categories[cat_name]
                share = (cat["emissions"] / total_emissions * 100) if total_emissions > 0 else 0.0
                rows.append(
                    f"| {cat_name.capitalize()} | "
                    f"{self._format_number(cat['quantity'])} | "
                    f"{self._format_number(cat['emissions'])} | "
                    f"{self._format_percentage(share)} |"
                )

        # Include any categories not in the standard list
        for cat_name, cat in categories.items():
            if cat_name not in self.GOODS_CATEGORIES:
                share = (cat["emissions"] / total_emissions * 100) if total_emissions > 0 else 0.0
                rows.append(
                    f"| {cat_name.capitalize()} | "
                    f"{self._format_number(cat['quantity'])} | "
                    f"{self._format_number(cat['emissions'])} | "
                    f"{self._format_percentage(share)} |"
                )

        return header + "\n".join(rows)

    def _md_country_breakdown(self, data: Dict[str, Any]) -> str:
        """Build Markdown breakdown by country of origin."""
        goods: List[Dict[str, Any]] = data.get("goods", [])

        countries: Dict[str, Dict[str, float]] = {}
        for g in goods:
            country = g.get("country_of_origin", "Unknown")
            if country not in countries:
                countries[country] = {"quantity": 0.0, "emissions": 0.0}
            countries[country]["quantity"] += g.get("quantity_tonnes", 0.0)
            countries[country]["emissions"] += g.get("embedded_emissions_tco2e", 0.0)

        header = (
            "## Breakdown by Country of Origin\n\n"
            "| Country | Quantity (tonnes) | Embedded Emissions (tCO2e) | "
            "Avg Intensity (tCO2e/t) |\n"
            "|---------|-------------------|---------------------------|"
            "------------------------|\n"
        )

        rows: List[str] = []
        for country, vals in sorted(countries.items(), key=lambda x: x[1]["emissions"], reverse=True):
            intensity = vals["emissions"] / vals["quantity"] if vals["quantity"] > 0 else 0.0
            rows.append(
                f"| {country} | "
                f"{self._format_number(vals['quantity'])} | "
                f"{self._format_number(vals['emissions'])} | "
                f"{self._format_number(intensity, 4)} |"
            )

        return header + "\n".join(rows)

    def _md_installation_detail(self, data: Dict[str, Any]) -> str:
        """Build Markdown installation-level detail table."""
        include = self.config.get("include_installation_detail", True)
        if not include:
            return "## Installation-Level Detail\n\n*Installation detail omitted per configuration.*"

        installations: List[Dict[str, Any]] = data.get("installations", [])

        header = (
            "## Installation-Level Detail\n\n"
            "| Installation ID | Name | Country | Operator | "
            "CN Code | Specific Emissions (tCO2e/t) | Method |\n"
            "|-----------------|------|---------|----------|"
            "---------|------------------------------|--------|\n"
        )

        rows: List[str] = []
        for inst in installations:
            rows.append(
                f"| {inst.get('installation_id', '')} | "
                f"{inst.get('name', '')} | "
                f"{inst.get('country', '')} | "
                f"{inst.get('operator', '')} | "
                f"{self._format_cn_code(inst.get('cn_code', ''))} | "
                f"{self._format_number(inst.get('specific_emissions', 0.0), 4)} | "
                f"{inst.get('method', '')} |"
            )

        if not rows:
            return header + "| *No installation data available* | | | | | | |"

        return header + "\n".join(rows)

    def _md_calculation_method_summary(self, data: Dict[str, Any]) -> str:
        """Build Markdown calculation method summary."""
        methods: Dict[str, Any] = data.get("calculation_methods", {})

        pct_actual = methods.get("pct_actual", 0.0)
        pct_default = methods.get("pct_default", 0.0)
        pct_country_default = methods.get("pct_country_default", 0.0)

        section = (
            "## Calculation Method Summary\n\n"
            "| Method | Usage (%) | Description |\n"
            "|--------|-----------|-------------|\n"
            f"| Actual emissions | {self._format_percentage(pct_actual)} | "
            f"Based on verified installation data |\n"
            f"| EU default values | {self._format_percentage(pct_default)} | "
            f"EU-published default emission factors |\n"
            f"| Country-specific defaults | {self._format_percentage(pct_country_default)} | "
            f"Third-country published emission factors |\n"
        )

        total_pct = pct_actual + pct_default + pct_country_default
        if abs(total_pct - 100.0) > 0.1:
            section += (
                f"\n> **Warning:** Method percentages sum to "
                f"{self._format_percentage(total_pct)} (expected 100.00%)"
            )

        return section

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        """Build Markdown data quality indicators section."""
        quality: Dict[str, Any] = data.get("data_quality", {})

        overall_score = quality.get("overall_score", 0.0)
        tier = quality.get("tier", "unknown")
        completeness = quality.get("completeness_pct", 0.0)
        timeliness = quality.get("timeliness_pct", 0.0)
        accuracy = quality.get("accuracy_pct", 0.0)
        verification_coverage = quality.get("verification_coverage_pct", 0.0)

        indicators: List[Dict[str, Any]] = quality.get("indicators", [])

        section = (
            "## Data Quality Indicators\n\n"
            f"**Overall Data Quality Score:** {self._format_number(overall_score, 1)}/100 "
            f"({tier.capitalize()})\n\n"
            "| Indicator | Score |\n"
            "|-----------|-------|\n"
            f"| Completeness | {self._format_percentage(completeness)} |\n"
            f"| Timeliness | {self._format_percentage(timeliness)} |\n"
            f"| Accuracy | {self._format_percentage(accuracy)} |\n"
            f"| Verification Coverage | {self._format_percentage(verification_coverage)} |\n"
        )

        if indicators:
            section += "\n### Additional Indicators\n\n"
            for ind in indicators:
                section += (
                    f"- **{ind.get('name', '')}:** "
                    f"{ind.get('value', 'N/A')} - {ind.get('description', '')}\n"
                )

        return section

    def _md_compliance_status(self, data: Dict[str, Any]) -> str:
        """Build Markdown compliance status section."""
        compliance: Dict[str, Any] = data.get("compliance", {})

        status = compliance.get("status", "UNKNOWN")
        status_icon = "PASS" if status == "PASS" else "FAIL"
        rules: List[Dict[str, Any]] = compliance.get("rules", [])

        section = f"## Compliance Status: **{status_icon}**\n\n"

        if rules:
            section += (
                "| Rule ID | Rule Description | Status | Detail |\n"
                "|---------|------------------|--------|--------|\n"
            )
            for rule in rules:
                rule_status = rule.get("status", "UNKNOWN")
                section += (
                    f"| {rule.get('rule_id', '')} | "
                    f"{rule.get('description', '')} | "
                    f"{'PASS' if rule_status == 'PASS' else 'FAIL'} | "
                    f"{rule.get('detail', '')} |\n"
                )
        else:
            section += "*No compliance rules evaluated.*\n"

        return section

    def _md_provenance_footer(self, data: Dict[str, Any]) -> str:
        """Build Markdown provenance footer."""
        return (
            "---\n\n"
            f"*Report generated: {self.generated_at}*\n\n"
            f"*Template: QuarterlyReportTemplate v1.0*\n\n"
            f"*Reporting period: {data.get('reporting_period', 'N/A')}*"
        )

    # ------------------------------------------------------------------ #
    #  HTML section builders
    # ------------------------------------------------------------------ #

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Build HTML header section."""
        period = data.get("reporting_period", "N/A")
        eori = data.get("importer_eori", "N/A")
        declarant = data.get("authorized_declarant", {})

        return (
            '<div class="report-header">'
            f'<h1>CBAM Quarterly Report</h1>'
            f'<div class="header-meta">'
            f'<div class="meta-item"><strong>Reporting Period:</strong> {period}</div>'
            f'<div class="meta-item"><strong>Importer EORI:</strong> {eori}</div>'
            f'<div class="meta-item"><strong>Authorized Declarant:</strong> '
            f'{declarant.get("name", "N/A")} ({declarant.get("role", "N/A")})</div>'
            f'<div class="meta-item"><strong>Contact:</strong> '
            f'{declarant.get("contact", "N/A")}</div>'
            f'<div class="meta-item"><strong>Generated:</strong> {self.generated_at}</div>'
            f'</div></div>'
        )

    def _html_goods_summary(self, data: Dict[str, Any]) -> str:
        """Build HTML goods summary table."""
        goods: List[Dict[str, Any]] = data.get("goods", [])

        rows_html = ""
        total_qty = 0.0
        total_emissions = 0.0

        for g in goods:
            cn_code = self._format_cn_code(g.get("cn_code", ""))
            qty = g.get("quantity_tonnes", 0.0)
            emissions = g.get("embedded_emissions_tco2e", 0.0)
            intensity = emissions / qty if qty > 0 else 0.0
            total_qty += qty
            total_emissions += emissions

            rows_html += (
                f'<tr>'
                f'<td class="code">{cn_code}</td>'
                f'<td>{g.get("description", "")}</td>'
                f'<td>{g.get("country_of_origin", "")}</td>'
                f'<td class="num">{self._format_number(qty)}</td>'
                f'<td class="num">{self._format_number(emissions)}</td>'
                f'<td class="num">{self._format_number(intensity, 4)}</td>'
                f'</tr>'
            )

        total_intensity = total_emissions / total_qty if total_qty > 0 else 0.0
        rows_html += (
            f'<tr class="total-row">'
            f'<td><strong>TOTAL</strong></td><td></td><td></td>'
            f'<td class="num"><strong>{self._format_number(total_qty)}</strong></td>'
            f'<td class="num"><strong>{self._format_number(total_emissions)}</strong></td>'
            f'<td class="num"><strong>{self._format_number(total_intensity, 4)}</strong></td>'
            f'</tr>'
        )

        return (
            '<div class="section">'
            '<h2>Goods Summary</h2>'
            '<table>'
            '<thead><tr>'
            '<th>CN Code</th><th>Description</th><th>Country of Origin</th>'
            '<th>Quantity (tonnes)</th><th>Embedded Emissions (tCO2e)</th>'
            '<th>Emission Intensity (tCO2e/t)</th>'
            '</tr></thead>'
            f'<tbody>{rows_html}</tbody>'
            '</table></div>'
        )

    def _html_category_breakdown(self, data: Dict[str, Any]) -> str:
        """Build HTML breakdown by goods category."""
        goods: List[Dict[str, Any]] = data.get("goods", [])

        categories: Dict[str, Dict[str, float]] = {}
        for g in goods:
            cat = g.get("category", "other").lower()
            if cat not in categories:
                categories[cat] = {"quantity": 0.0, "emissions": 0.0}
            categories[cat]["quantity"] += g.get("quantity_tonnes", 0.0)
            categories[cat]["emissions"] += g.get("embedded_emissions_tco2e", 0.0)

        total_emissions = sum(c["emissions"] for c in categories.values())
        rows_html = ""

        all_cats = [c for c in self.GOODS_CATEGORIES if c in categories]
        all_cats += [c for c in categories if c not in self.GOODS_CATEGORIES]

        for cat_name in all_cats:
            cat = categories[cat_name]
            share = (cat["emissions"] / total_emissions * 100) if total_emissions > 0 else 0.0
            rows_html += (
                f'<tr>'
                f'<td>{cat_name.capitalize()}</td>'
                f'<td class="num">{self._format_number(cat["quantity"])}</td>'
                f'<td class="num">{self._format_number(cat["emissions"])}</td>'
                f'<td class="num">{self._format_percentage(share)}</td>'
                f'</tr>'
            )

        return (
            '<div class="section">'
            '<h2>Breakdown by Goods Category</h2>'
            '<table><thead><tr>'
            '<th>Category</th><th>Quantity (tonnes)</th>'
            '<th>Embedded Emissions (tCO2e)</th><th>Share of Total</th>'
            '</tr></thead>'
            f'<tbody>{rows_html}</tbody></table></div>'
        )

    def _html_country_breakdown(self, data: Dict[str, Any]) -> str:
        """Build HTML breakdown by country of origin."""
        goods: List[Dict[str, Any]] = data.get("goods", [])

        countries: Dict[str, Dict[str, float]] = {}
        for g in goods:
            country = g.get("country_of_origin", "Unknown")
            if country not in countries:
                countries[country] = {"quantity": 0.0, "emissions": 0.0}
            countries[country]["quantity"] += g.get("quantity_tonnes", 0.0)
            countries[country]["emissions"] += g.get("embedded_emissions_tco2e", 0.0)

        rows_html = ""
        for country, vals in sorted(countries.items(), key=lambda x: x[1]["emissions"], reverse=True):
            intensity = vals["emissions"] / vals["quantity"] if vals["quantity"] > 0 else 0.0
            rows_html += (
                f'<tr>'
                f'<td>{country}</td>'
                f'<td class="num">{self._format_number(vals["quantity"])}</td>'
                f'<td class="num">{self._format_number(vals["emissions"])}</td>'
                f'<td class="num">{self._format_number(intensity, 4)}</td>'
                f'</tr>'
            )

        return (
            '<div class="section">'
            '<h2>Breakdown by Country of Origin</h2>'
            '<table><thead><tr>'
            '<th>Country</th><th>Quantity (tonnes)</th>'
            '<th>Embedded Emissions (tCO2e)</th><th>Avg Intensity (tCO2e/t)</th>'
            '</tr></thead>'
            f'<tbody>{rows_html}</tbody></table></div>'
        )

    def _html_installation_detail(self, data: Dict[str, Any]) -> str:
        """Build HTML installation-level detail table."""
        include = self.config.get("include_installation_detail", True)
        if not include:
            return (
                '<div class="section">'
                '<h2>Installation-Level Detail</h2>'
                '<p class="note">Installation detail omitted per configuration.</p>'
                '</div>'
            )

        installations: List[Dict[str, Any]] = data.get("installations", [])
        rows_html = ""

        for inst in installations:
            rows_html += (
                f'<tr>'
                f'<td class="code">{inst.get("installation_id", "")}</td>'
                f'<td>{inst.get("name", "")}</td>'
                f'<td>{inst.get("country", "")}</td>'
                f'<td>{inst.get("operator", "")}</td>'
                f'<td class="code">{self._format_cn_code(inst.get("cn_code", ""))}</td>'
                f'<td class="num">{self._format_number(inst.get("specific_emissions", 0.0), 4)}</td>'
                f'<td>{inst.get("method", "")}</td>'
                f'</tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="7" class="note">No installation data available</td></tr>'

        return (
            '<div class="section">'
            '<h2>Installation-Level Detail</h2>'
            '<table><thead><tr>'
            '<th>Installation ID</th><th>Name</th><th>Country</th>'
            '<th>Operator</th><th>CN Code</th>'
            '<th>Specific Emissions (tCO2e/t)</th><th>Method</th>'
            '</tr></thead>'
            f'<tbody>{rows_html}</tbody></table></div>'
        )

    def _html_calculation_method_summary(self, data: Dict[str, Any]) -> str:
        """Build HTML calculation method summary."""
        methods: Dict[str, Any] = data.get("calculation_methods", {})

        pct_actual = methods.get("pct_actual", 0.0)
        pct_default = methods.get("pct_default", 0.0)
        pct_country_default = methods.get("pct_country_default", 0.0)

        bar_actual = self._html_progress_bar(pct_actual, "#2ecc71")
        bar_default = self._html_progress_bar(pct_default, "#e67e22")
        bar_country = self._html_progress_bar(pct_country_default, "#3498db")

        return (
            '<div class="section">'
            '<h2>Calculation Method Summary</h2>'
            '<div class="method-grid">'
            f'<div class="method-item"><strong>Actual emissions</strong>'
            f'{bar_actual}<span>{self._format_percentage(pct_actual)}</span></div>'
            f'<div class="method-item"><strong>EU default values</strong>'
            f'{bar_default}<span>{self._format_percentage(pct_default)}</span></div>'
            f'<div class="method-item"><strong>Country-specific defaults</strong>'
            f'{bar_country}<span>{self._format_percentage(pct_country_default)}</span></div>'
            '</div></div>'
        )

    def _html_data_quality(self, data: Dict[str, Any]) -> str:
        """Build HTML data quality indicators section."""
        quality: Dict[str, Any] = data.get("data_quality", {})

        overall_score = quality.get("overall_score", 0.0)
        tier = quality.get("tier", "unknown")
        completeness = quality.get("completeness_pct", 0.0)
        timeliness = quality.get("timeliness_pct", 0.0)
        accuracy = quality.get("accuracy_pct", 0.0)
        verification_coverage = quality.get("verification_coverage_pct", 0.0)

        tier_color = {
            "excellent": "#2ecc71",
            "good": "#27ae60",
            "fair": "#f39c12",
            "poor": "#e74c3c",
        }.get(tier.lower(), "#95a5a6")

        return (
            '<div class="section">'
            '<h2>Data Quality Indicators</h2>'
            f'<div class="quality-score" style="border-left: 4px solid {tier_color}">'
            f'<strong>Overall Score:</strong> {self._format_number(overall_score, 1)}/100 '
            f'<span class="tier-badge" style="background:{tier_color}">{tier.capitalize()}</span>'
            '</div>'
            '<table><thead><tr><th>Indicator</th><th>Score</th></tr></thead><tbody>'
            f'<tr><td>Completeness</td><td class="num">{self._format_percentage(completeness)}</td></tr>'
            f'<tr><td>Timeliness</td><td class="num">{self._format_percentage(timeliness)}</td></tr>'
            f'<tr><td>Accuracy</td><td class="num">{self._format_percentage(accuracy)}</td></tr>'
            f'<tr><td>Verification Coverage</td>'
            f'<td class="num">{self._format_percentage(verification_coverage)}</td></tr>'
            '</tbody></table></div>'
        )

    def _html_compliance_status(self, data: Dict[str, Any]) -> str:
        """Build HTML compliance status section."""
        compliance: Dict[str, Any] = data.get("compliance", {})

        status = compliance.get("status", "UNKNOWN")
        status_color = "#2ecc71" if status == "PASS" else "#e74c3c"
        rules: List[Dict[str, Any]] = compliance.get("rules", [])

        rows_html = ""
        for rule in rules:
            rule_status = rule.get("status", "UNKNOWN")
            color = "#2ecc71" if rule_status == "PASS" else "#e74c3c"
            rows_html += (
                f'<tr>'
                f'<td class="code">{rule.get("rule_id", "")}</td>'
                f'<td>{rule.get("description", "")}</td>'
                f'<td style="color:{color};font-weight:bold">'
                f'{"PASS" if rule_status == "PASS" else "FAIL"}</td>'
                f'<td>{rule.get("detail", "")}</td>'
                f'</tr>'
            )

        if not rows_html:
            rows_html = '<tr><td colspan="4" class="note">No compliance rules evaluated.</td></tr>'

        return (
            '<div class="section">'
            f'<h2>Compliance Status: '
            f'<span style="color:{status_color}">{"PASS" if status == "PASS" else "FAIL"}</span></h2>'
            '<table><thead><tr>'
            '<th>Rule ID</th><th>Description</th><th>Status</th><th>Detail</th>'
            '</tr></thead>'
            f'<tbody>{rows_html}</tbody></table></div>'
        )

    # ------------------------------------------------------------------ #
    #  JSON section builders
    # ------------------------------------------------------------------ #

    def _json_goods_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON goods summary."""
        goods: List[Dict[str, Any]] = data.get("goods", [])

        items = []
        total_qty = 0.0
        total_emissions = 0.0

        for g in goods:
            qty = g.get("quantity_tonnes", 0.0)
            emissions = g.get("embedded_emissions_tco2e", 0.0)
            intensity = emissions / qty if qty > 0 else 0.0
            total_qty += qty
            total_emissions += emissions

            items.append({
                "cn_code": self._format_cn_code(g.get("cn_code", "")),
                "description": g.get("description", ""),
                "country_of_origin": g.get("country_of_origin", ""),
                "quantity_tonnes": round(qty, 2),
                "embedded_emissions_tco2e": round(emissions, 2),
                "emission_intensity_tco2e_per_t": round(intensity, 4),
            })

        return {
            "items": items,
            "totals": {
                "quantity_tonnes": round(total_qty, 2),
                "embedded_emissions_tco2e": round(total_emissions, 2),
                "avg_emission_intensity": round(
                    total_emissions / total_qty if total_qty > 0 else 0.0, 4
                ),
            },
        }

    def _json_category_breakdown(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON breakdown by goods category."""
        goods: List[Dict[str, Any]] = data.get("goods", [])

        categories: Dict[str, Dict[str, float]] = {}
        for g in goods:
            cat = g.get("category", "other").lower()
            if cat not in categories:
                categories[cat] = {"quantity": 0.0, "emissions": 0.0}
            categories[cat]["quantity"] += g.get("quantity_tonnes", 0.0)
            categories[cat]["emissions"] += g.get("embedded_emissions_tco2e", 0.0)

        total_emissions = sum(c["emissions"] for c in categories.values())
        result: List[Dict[str, Any]] = []

        for cat_name, cat in sorted(categories.items(), key=lambda x: x[1]["emissions"], reverse=True):
            share = (cat["emissions"] / total_emissions * 100) if total_emissions > 0 else 0.0
            result.append({
                "category": cat_name,
                "quantity_tonnes": round(cat["quantity"], 2),
                "embedded_emissions_tco2e": round(cat["emissions"], 2),
                "share_pct": round(share, 2),
            })

        return result

    def _json_country_breakdown(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON breakdown by country of origin."""
        goods: List[Dict[str, Any]] = data.get("goods", [])

        countries: Dict[str, Dict[str, float]] = {}
        for g in goods:
            country = g.get("country_of_origin", "Unknown")
            if country not in countries:
                countries[country] = {"quantity": 0.0, "emissions": 0.0}
            countries[country]["quantity"] += g.get("quantity_tonnes", 0.0)
            countries[country]["emissions"] += g.get("embedded_emissions_tco2e", 0.0)

        result: List[Dict[str, Any]] = []
        for country, vals in sorted(countries.items(), key=lambda x: x[1]["emissions"], reverse=True):
            intensity = vals["emissions"] / vals["quantity"] if vals["quantity"] > 0 else 0.0
            result.append({
                "country": country,
                "quantity_tonnes": round(vals["quantity"], 2),
                "embedded_emissions_tco2e": round(vals["emissions"], 2),
                "avg_intensity_tco2e_per_t": round(intensity, 4),
            })

        return result

    def _json_installation_detail(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build JSON installation-level detail."""
        installations: List[Dict[str, Any]] = data.get("installations", [])

        return [
            {
                "installation_id": inst.get("installation_id", ""),
                "name": inst.get("name", ""),
                "country": inst.get("country", ""),
                "operator": inst.get("operator", ""),
                "cn_code": self._format_cn_code(inst.get("cn_code", "")),
                "specific_emissions_tco2e_per_t": round(inst.get("specific_emissions", 0.0), 4),
                "method": inst.get("method", ""),
            }
            for inst in installations
        ]

    def _json_calculation_method_summary(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Build JSON calculation method summary."""
        methods: Dict[str, Any] = data.get("calculation_methods", {})
        return {
            "pct_actual": round(methods.get("pct_actual", 0.0), 2),
            "pct_default": round(methods.get("pct_default", 0.0), 2),
            "pct_country_default": round(methods.get("pct_country_default", 0.0), 2),
        }

    def _json_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON data quality indicators."""
        quality: Dict[str, Any] = data.get("data_quality", {})
        return {
            "overall_score": round(quality.get("overall_score", 0.0), 1),
            "tier": quality.get("tier", "unknown"),
            "completeness_pct": round(quality.get("completeness_pct", 0.0), 2),
            "timeliness_pct": round(quality.get("timeliness_pct", 0.0), 2),
            "accuracy_pct": round(quality.get("accuracy_pct", 0.0), 2),
            "verification_coverage_pct": round(quality.get("verification_coverage_pct", 0.0), 2),
            "indicators": quality.get("indicators", []),
        }

    def _json_compliance_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON compliance status."""
        compliance: Dict[str, Any] = data.get("compliance", {})
        return {
            "status": compliance.get("status", "UNKNOWN"),
            "rules": compliance.get("rules", []),
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
        """
        Format a numeric value with thousand separators and fixed decimals.

        Args:
            value: Numeric value to format.
            decimals: Number of decimal places (default: 2).

        Returns:
            Formatted string, e.g. "1,234.56".
        """
        return f"{value:,.{decimals}f}"

    def _format_percentage(self, value: Union[int, float]) -> str:
        """
        Format a percentage value.

        Args:
            value: Percentage value (0-100).

        Returns:
            Formatted string, e.g. "85.50%".
        """
        return f"{value:.2f}%"

    def _format_date(self, dt: Union[datetime, str]) -> str:
        """
        Format a datetime to ISO date string.

        Args:
            dt: Datetime object or ISO string.

        Returns:
            Formatted date string (YYYY-MM-DD).
        """
        if isinstance(dt, str):
            return dt[:10]
        return dt.strftime("%Y-%m-%d")

    def _format_currency(self, value: Union[int, float], currency: str = "EUR") -> str:
        """
        Format a currency value.

        Args:
            value: Monetary value.
            currency: Currency code (default: EUR).

        Returns:
            Formatted string, e.g. "EUR 1,234.56".
        """
        return f"{currency} {value:,.2f}"

    def _format_cn_code(self, code: str) -> str:
        """
        Format a CN code to standard XXXX.XX format.

        Args:
            code: Raw CN code string (may or may not have dots).

        Returns:
            Formatted CN code, e.g. "7207.11".
        """
        clean = code.replace(".", "").replace(" ", "")
        if len(clean) >= 6:
            return f"{clean[:4]}.{clean[4:6]}"
        elif len(clean) == 4:
            return f"{clean}.00"
        return code

    def _html_progress_bar(self, pct: float, color: str) -> str:
        """
        Generate an inline HTML progress bar.

        Args:
            pct: Percentage value (0-100).
            color: CSS color for the bar fill.

        Returns:
            HTML string for the progress bar.
        """
        width = max(0, min(100, pct))
        return (
            f'<div class="progress-bar">'
            f'<div class="progress-fill" '
            f'style="width:{width}%;background:{color}"></div>'
            f'</div>'
        )

    def _wrap_html(self, title: str, body: str, provenance_hash: str) -> str:
        """
        Wrap HTML body content in a complete HTML document with inline CSS.

        Args:
            title: HTML page title.
            body: HTML body content.
            provenance_hash: SHA-256 hash for provenance tracking.

        Returns:
            Complete HTML document string.
        """
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
            ".code{font-family:monospace;font-size:13px}"
            ".total-row{background:#f8f9fa;font-weight:600}"
            ".note{color:#7f8c8d;font-style:italic}"
            ".progress-bar{background:#ecf0f1;border-radius:4px;height:12px;"
            "overflow:hidden;margin:4px 0}"
            ".progress-fill{height:100%;border-radius:4px;transition:width 0.3s}"
            ".method-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px}"
            ".method-item{background:#f8f9fa;padding:12px;border-radius:6px}"
            ".quality-score{padding:12px;margin-bottom:16px;background:#f8f9fa;border-radius:6px}"
            ".tier-badge{display:inline-block;padding:2px 8px;border-radius:4px;"
            "color:#fff;font-size:12px;margin-left:8px}"
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
            f'Template: QuarterlyReportTemplate v1.0 | '
            f'Provenance: {provenance_hash}'
            f'</div>'
            f'</body></html>'
        )
