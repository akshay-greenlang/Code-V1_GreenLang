# -*- coding: utf-8 -*-
"""
EnPIMethodologyTemplate - ISO 50006 EnPI/EnB Methodology for PACK-034.

Generates methodology documents for Energy Performance Indicators (EnPIs)
and Energy Baselines (EnBs) aligned with ISO 50006:2014. Covers EnPI
definitions, normalization approaches, baseline establishment with model
selection, statistical criteria (R2, CV(RMSE), p-value), data requirements,
measurement plans, reporting protocols, and review triggers.

Sections:
    1. Methodology Overview
    2. EnPI Definitions
    3. Baseline Establishment
    4. Data Requirements
    5. Measurement Plan
    6. Statistical Validation Criteria
    7. Reporting Protocol
    8. Review Triggers

Author: GreenLang Team
Version: 34.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EnPIMethodologyTemplate:
    """
    ISO 50006 EnPI/EnB methodology document template.

    Renders methodology documents for Energy Performance Indicators and
    Energy Baselines with statistical validation criteria, measurement
    plans, and reporting protocols across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EnPIMethodologyTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render EnPI methodology document as Markdown.

        Args:
            data: Methodology data including enpi_definitions,
                  baseline_models, data_requirements, and
                  validation_criteria.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_methodology_overview(data),
            self._md_enpi_definitions(data),
            self._md_baseline_establishment(data),
            self._md_data_requirements(data),
            self._md_measurement_plan(data),
            self._md_statistical_validation(data),
            self._md_reporting_protocol(data),
            self._md_review_triggers(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render EnPI methodology document as self-contained HTML.

        Args:
            data: Methodology data dict.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_methodology_overview(data),
            self._html_enpi_definitions(data),
            self._html_baseline_establishment(data),
            self._html_data_requirements(data),
            self._html_statistical_validation(data),
            self._html_reporting_protocol(data),
            self._html_review_triggers(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>EnPI Methodology - ISO 50006</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render EnPI methodology document as structured JSON.

        Args:
            data: Methodology data dict.

        Returns:
            Dict with structured methodology sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "enpi_methodology",
            "version": "34.0.0",
            "generated_at": self.generated_at.isoformat(),
            "methodology_overview": data.get("methodology_overview", {}),
            "enpi_definitions": data.get("enpi_definitions", []),
            "baseline_models": data.get("baseline_models", []),
            "data_requirements": data.get("data_requirements", []),
            "measurement_plan": data.get("measurement_plan", []),
            "validation_criteria": self._json_validation_criteria(data),
            "reporting_protocol": data.get("reporting_protocol", {}),
            "review_triggers": data.get("review_triggers", []),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with methodology metadata."""
        org = data.get("organization_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# EnPI/EnB Methodology Document\n\n"
            f"**Organization:** {org}  \n"
            f"**Document Date:** {data.get('document_date', '')}  \n"
            f"**Standard Reference:** ISO 50006:2014, ISO 50001:2018 (Clauses 6.4, 6.5)  \n"
            f"**Version:** {data.get('document_version', '1.0')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-034 EnPIMethodologyTemplate v34.0.0\n\n---"
        )

    def _md_methodology_overview(self, data: Dict[str, Any]) -> str:
        """Render methodology overview section."""
        overview = data.get("methodology_overview", {})
        approach = overview.get("approach", "Bottom-up facility-level analysis")
        scope = overview.get("scope", "All SEUs identified in the energy review")
        return (
            "## 1. Methodology Overview\n\n"
            "This document defines the methodology for establishing, calculating, "
            "and reporting Energy Performance Indicators (EnPIs) and Energy "
            "Baselines (EnBs) per ISO 50006:2014.\n\n"
            f"**Approach:** {approach}  \n"
            f"**Scope:** {scope}  \n"
            f"**Normalization Method:** {overview.get('normalization_method', 'Regression-based')}  \n"
            f"**Reporting Frequency:** {overview.get('reporting_frequency', 'Monthly')}  \n"
            f"**Baseline Period:** {overview.get('baseline_period', 'To be defined')}"
        )

    def _md_enpi_definitions(self, data: Dict[str, Any]) -> str:
        """Render EnPI definitions section."""
        definitions = data.get("enpi_definitions", [])
        if not definitions:
            return "## 2. EnPI Definitions\n\n_No EnPIs defined yet._"
        lines = [
            "## 2. EnPI Definitions\n",
            "### EnPI Types\n",
            "- **Measured Value EnPI:** Direct measurement of energy consumption or efficiency",
            "- **Ratio EnPI:** Energy consumption divided by a relevant variable (e.g., kWh/m2)",
            "- **Model-Based EnPI:** Statistical model predicting expected energy use",
            "- **CUSUM EnPI:** Cumulative Sum of differences between actual and expected\n",
            "### Defined EnPIs\n",
            "| # | EnPI Name | Type | Formula / Description | Unit | SEU | Normalization Variable |",
            "|---|-----------|------|--------------------|------|-----|----------------------|",
        ]
        for i, d in enumerate(definitions, 1):
            lines.append(
                f"| {i} | {d.get('name', '-')} "
                f"| {d.get('type', '-')} "
                f"| {d.get('formula', '-')} "
                f"| {d.get('unit', '-')} "
                f"| {d.get('seu', '-')} "
                f"| {d.get('normalization_variable', '-')} |"
            )
        return "\n".join(lines)

    def _md_baseline_establishment(self, data: Dict[str, Any]) -> str:
        """Render baseline establishment section."""
        models = data.get("baseline_models", [])
        lines = [
            "## 3. Baseline Establishment\n",
            "Energy baselines (EnBs) serve as the quantitative reference against "
            "which energy performance is compared (ISO 50001:2018 Clause 6.5).\n",
            "### Model Selection Criteria\n",
            "- Best-fit regression model (linear, multivariate, polynomial)",
            "- Minimum R-squared threshold for model acceptance",
            "- Statistical significance of all independent variables",
            "- Residuals normality and independence verification",
            "- Cross-validation to prevent overfitting\n",
        ]
        if models:
            lines.extend([
                "### Established Baseline Models\n",
                "| EnPI | Baseline Period | Model Type | Variables | R2 | CV(RMSE) (%) | Status |",
                "|------|----------------|-----------|-----------|-----|-------------|--------|",
            ])
            for m in models:
                variables = ", ".join(m.get("variables", []))
                lines.append(
                    f"| {m.get('enpi', '-')} "
                    f"| {m.get('baseline_period', '-')} "
                    f"| {m.get('model_type', '-')} "
                    f"| {variables} "
                    f"| {self._fmt(m.get('r_squared', 0), 4)} "
                    f"| {self._fmt(m.get('cv_rmse_pct', 0))}% "
                    f"| {m.get('status', '-')} |"
                )
        return "\n".join(lines)

    def _md_data_requirements(self, data: Dict[str, Any]) -> str:
        """Render data requirements section."""
        requirements = data.get("data_requirements", [])
        if not requirements:
            return "## 4. Data Requirements\n\n_Data requirements to be defined._"
        lines = [
            "## 4. Data Requirements\n",
            "| Data Point | Source | Frequency | Unit | Accuracy | Responsible |",
            "|-----------|--------|-----------|------|----------|-------------|",
        ]
        for r in requirements:
            lines.append(
                f"| {r.get('data_point', '-')} "
                f"| {r.get('source', '-')} "
                f"| {r.get('frequency', '-')} "
                f"| {r.get('unit', '-')} "
                f"| {r.get('accuracy', '-')} "
                f"| {r.get('responsible', '-')} |"
            )
        return "\n".join(lines)

    def _md_measurement_plan(self, data: Dict[str, Any]) -> str:
        """Render measurement plan section."""
        plan = data.get("measurement_plan", [])
        if not plan:
            return "## 5. Measurement Plan\n\n_Measurement plan to be defined._"
        lines = [
            "## 5. Measurement Plan\n",
            "The measurement plan ensures consistent, accurate data collection "
            "for all EnPIs and relevant variables.\n",
            "| Meter / Instrument | Location | Parameter | Range | Accuracy Class | "
            "Calibration Freq | Last Calibration |",
            "|-------------------|----------|-----------|-------|---------------|"
            "-----------------|-----------------|",
        ]
        for m in plan:
            lines.append(
                f"| {m.get('meter', '-')} "
                f"| {m.get('location', '-')} "
                f"| {m.get('parameter', '-')} "
                f"| {m.get('range', '-')} "
                f"| {m.get('accuracy_class', '-')} "
                f"| {m.get('calibration_frequency', '-')} "
                f"| {m.get('last_calibration', '-')} |"
            )
        return "\n".join(lines)

    def _md_statistical_validation(self, data: Dict[str, Any]) -> str:
        """Render statistical validation criteria section."""
        criteria = data.get("validation_criteria", {})
        r2_min = criteria.get("r_squared_min", 0.75)
        cv_rmse_max = criteria.get("cv_rmse_max_pct", 25.0)
        p_value_max = criteria.get("p_value_max", 0.05)
        return (
            "## 6. Statistical Validation Criteria\n\n"
            "All baseline models and EnPI calculations must satisfy the following "
            "statistical criteria per ISO 50006 guidance:\n\n"
            "| Criterion | Threshold | Description |\n"
            "|-----------|-----------|-------------|\n"
            f"| R-squared (R2) | >= {self._fmt(r2_min, 2)} | "
            "Model explains sufficient variance in energy use |\n"
            f"| CV(RMSE) | <= {self._fmt(cv_rmse_max, 1)}% | "
            "Coefficient of variation of root mean square error |\n"
            f"| p-value | <= {self._fmt(p_value_max, 3)} | "
            "All independent variables statistically significant |\n"
            f"| F-statistic | > {criteria.get('f_statistic_threshold', 'Critical value')} | "
            "Overall model significance |\n"
            f"| Durbin-Watson | {criteria.get('durbin_watson_range', '1.5 - 2.5')} | "
            "Residuals independence (no autocorrelation) |\n"
            f"| Residuals Normality | Shapiro-Wilk p > {self._fmt(criteria.get('normality_p_min', 0.05), 2)} | "
            "Residuals follow normal distribution |\n\n"
            "### Model Acceptance Flow\n\n"
            "1. Collect baseline period data (minimum 12 months)\n"
            "2. Identify relevant variables (energy drivers)\n"
            "3. Develop regression model (single or multivariate)\n"
            "4. Validate against statistical criteria above\n"
            "5. If criteria not met, iterate on model selection\n"
            "6. Document model parameters and acceptance rationale\n"
            "7. Obtain approval from energy team lead"
        )

    def _md_reporting_protocol(self, data: Dict[str, Any]) -> str:
        """Render reporting protocol section."""
        protocol = data.get("reporting_protocol", {})
        frequency = protocol.get("frequency", "Monthly")
        recipients = protocol.get("recipients", [])
        lines = [
            "## 7. Reporting Protocol\n",
            f"**Reporting Frequency:** {frequency}  ",
            f"**Report Format:** {protocol.get('format', 'Dashboard + Monthly Report')}  ",
            f"**Data Lag:** {protocol.get('data_lag', '5 business days after period end')}\n",
        ]
        if recipients:
            lines.append("### Recipients\n")
            lines.append("| Recipient | Role | Report Type | Frequency |")
            lines.append("|-----------|------|------------|-----------|")
            for r in recipients:
                lines.append(
                    f"| {r.get('name', '-')} "
                    f"| {r.get('role', '-')} "
                    f"| {r.get('report_type', '-')} "
                    f"| {r.get('frequency', '-')} |"
                )
        lines.extend([
            "\n### Report Contents\n",
            "- EnPI values for current period vs baseline",
            "- CUSUM chart showing cumulative savings/losses",
            "- Normalized consumption comparison",
            "- Data quality indicators",
            "- Commentary on significant deviations",
            "- Recommended actions for underperformance",
        ])
        return "\n".join(lines)

    def _md_review_triggers(self, data: Dict[str, Any]) -> str:
        """Render review triggers section."""
        triggers = data.get("review_triggers", [])
        if not triggers:
            triggers = [
                "Major process or equipment changes affecting SEUs",
                "Significant shift in relevant variables beyond model range",
                "EnPI showing sustained deviation from expected performance",
                "Change in energy sources or tariff structures",
                "Organizational changes (expansion, acquisition, divestiture)",
                "Baseline adjustment criteria met per ISO 50006",
                "Planned 3-year periodic review",
            ]
        lines = [
            "## 8. Review Triggers\n",
            "The EnPI methodology and baselines shall be reviewed when:\n",
        ]
        for i, trigger in enumerate(triggers, 1):
            lines.append(f"{i}. {trigger}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-034 ISO 50001 Energy Management System Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        org = data.get("organization_name", "Organization")
        return (
            f'<h1>EnPI/EnB Methodology Document</h1>\n'
            f'<p class="subtitle">Organization: {org} | '
            f'ISO 50006:2014 | ISO 50001:2018 Clauses 6.4, 6.5</p>'
        )

    def _html_methodology_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML methodology overview."""
        overview = data.get("methodology_overview", {})
        return (
            '<h2>1. Methodology Overview</h2>\n'
            '<div class="info-box">'
            '<p>This document defines the methodology for establishing, calculating, '
            'and reporting EnPIs and EnBs per ISO 50006:2014.</p>\n'
            f'<p><strong>Approach:</strong> {overview.get("approach", "Bottom-up")}</p>\n'
            f'<p><strong>Normalization:</strong> {overview.get("normalization_method", "Regression-based")}</p>'
            '</div>'
        )

    def _html_enpi_definitions(self, data: Dict[str, Any]) -> str:
        """Render HTML EnPI definitions."""
        definitions = data.get("enpi_definitions", [])
        rows = ""
        for d in definitions:
            rows += (
                f'<tr><td><strong>{d.get("name", "-")}</strong></td>'
                f'<td>{d.get("type", "-")}</td>'
                f'<td><code>{d.get("formula", "-")}</code></td>'
                f'<td>{d.get("unit", "-")}</td>'
                f'<td>{d.get("seu", "-")}</td></tr>\n'
            )
        return (
            '<h2>2. EnPI Definitions</h2>\n'
            '<table>\n<tr><th>EnPI Name</th><th>Type</th>'
            f'<th>Formula</th><th>Unit</th><th>SEU</th></tr>\n{rows}</table>'
        )

    def _html_baseline_establishment(self, data: Dict[str, Any]) -> str:
        """Render HTML baseline establishment."""
        models = data.get("baseline_models", [])
        rows = ""
        for m in models:
            r2 = m.get("r_squared", 0)
            cls = "status-improved" if r2 >= 0.75 else "status-declined"
            rows += (
                f'<tr><td>{m.get("enpi", "-")}</td>'
                f'<td>{m.get("baseline_period", "-")}</td>'
                f'<td>{m.get("model_type", "-")}</td>'
                f'<td class="{cls}">{self._fmt(r2, 4)}</td>'
                f'<td>{self._fmt(m.get("cv_rmse_pct", 0))}%</td></tr>\n'
            )
        return (
            '<h2>3. Baseline Establishment</h2>\n'
            '<table>\n<tr><th>EnPI</th><th>Period</th>'
            f'<th>Model</th><th>R2</th><th>CV(RMSE)</th></tr>\n{rows}</table>'
        )

    def _html_data_requirements(self, data: Dict[str, Any]) -> str:
        """Render HTML data requirements."""
        requirements = data.get("data_requirements", [])
        rows = ""
        for r in requirements:
            rows += (
                f'<tr><td>{r.get("data_point", "-")}</td>'
                f'<td>{r.get("source", "-")}</td>'
                f'<td>{r.get("frequency", "-")}</td>'
                f'<td>{r.get("unit", "-")}</td>'
                f'<td>{r.get("accuracy", "-")}</td></tr>\n'
            )
        return (
            '<h2>4. Data Requirements</h2>\n'
            '<table>\n<tr><th>Data Point</th><th>Source</th>'
            f'<th>Frequency</th><th>Unit</th><th>Accuracy</th></tr>\n{rows}</table>'
        )

    def _html_statistical_validation(self, data: Dict[str, Any]) -> str:
        """Render HTML statistical validation criteria."""
        criteria = data.get("validation_criteria", {})
        return (
            '<h2>6. Statistical Validation Criteria</h2>\n'
            '<table>\n'
            '<tr><th>Criterion</th><th>Threshold</th><th>Description</th></tr>\n'
            f'<tr><td>R-squared (R2)</td><td>&gt;= {self._fmt(criteria.get("r_squared_min", 0.75), 2)}</td>'
            '<td>Model explains sufficient variance</td></tr>\n'
            f'<tr><td>CV(RMSE)</td><td>&lt;= {self._fmt(criteria.get("cv_rmse_max_pct", 25.0), 1)}%</td>'
            '<td>Coefficient of variation of RMSE</td></tr>\n'
            f'<tr><td>p-value</td><td>&lt;= {self._fmt(criteria.get("p_value_max", 0.05), 3)}</td>'
            '<td>Variables statistically significant</td></tr>\n'
            '</table>'
        )

    def _html_reporting_protocol(self, data: Dict[str, Any]) -> str:
        """Render HTML reporting protocol."""
        protocol = data.get("reporting_protocol", {})
        return (
            '<h2>7. Reporting Protocol</h2>\n'
            f'<p><strong>Frequency:</strong> {protocol.get("frequency", "Monthly")}</p>\n'
            f'<p><strong>Format:</strong> {protocol.get("format", "Dashboard + Report")}</p>\n'
            '<h3>Report Contents</h3>\n'
            '<ul>\n'
            '<li>EnPI values vs baseline</li>\n'
            '<li>CUSUM chart</li>\n'
            '<li>Normalized consumption comparison</li>\n'
            '<li>Data quality indicators</li>\n'
            '<li>Deviation commentary</li>\n'
            '</ul>'
        )

    def _html_review_triggers(self, data: Dict[str, Any]) -> str:
        """Render HTML review triggers."""
        triggers = data.get("review_triggers", [
            "Major process/equipment changes",
            "Significant relevant variable shift",
            "Sustained EnPI deviation",
        ])
        items = "".join(f'<li>{t}</li>\n' for t in triggers)
        return f'<h2>8. Review Triggers</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_validation_criteria(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON validation criteria."""
        criteria = data.get("validation_criteria", {})
        return {
            "r_squared_min": criteria.get("r_squared_min", 0.75),
            "cv_rmse_max_pct": criteria.get("cv_rmse_max_pct", 25.0),
            "p_value_max": criteria.get("p_value_max", 0.05),
            "f_statistic_threshold": criteria.get("f_statistic_threshold", "Critical value"),
            "durbin_watson_range": criteria.get("durbin_watson_range", "1.5 - 2.5"),
            "normality_p_min": criteria.get("normality_p_min", 0.05),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Build inline CSS for HTML rendering."""
        return (
            "body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:20px;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;}"
            "h1{color:#0d6efd;border-bottom:3px solid #0d6efd;padding-bottom:10px;}"
            "h2{color:#198754;margin-top:30px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            "code{background:#e9ecef;padding:2px 6px;border-radius:3px;font-size:0.9em;}"
            ".info-box{background:#e3f2fd;border-left:4px solid #0d6efd;padding:15px 20px;margin:15px 0;border-radius:4px;}"
            ".status-improved{color:#198754;font-weight:600;}"
            ".status-declined{color:#dc3545;font-weight:600;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _format_currency(self, val: Any) -> str:
        """Format a currency value with comma separators.

        Args:
            val: Numeric value to format.

        Returns:
            Formatted currency string.
        """
        if isinstance(val, (int, float)):
            return f"EUR {val:,.2f}"
        return str(val)

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value with comma separators.

        Args:
            val: Value to format.
            decimals: Decimal places.

        Returns:
            Formatted string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _generate_provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
