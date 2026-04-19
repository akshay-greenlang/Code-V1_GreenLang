# -*- coding: utf-8 -*-
"""
MVPlanReportTemplate - M&V Plan Report for PACK-040.

Generates comprehensive Measurement & Verification plan reports covering
ECM description, IPMVP option selection rationale, measurement boundary
definition, baseline and reporting period specification, metering plan
details, and adjustment methodology documentation.

Sections:
    1. Plan Overview
    2. ECM Description
    3. IPMVP Option Selection
    4. Measurement Boundary
    5. Baseline Period
    6. Reporting Period
    7. Metering Plan
    8. Adjustment Methodology
    9. Risk Assessment
    10. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - IPMVP Core Concepts 2022
    - ASHRAE Guideline 14-2014
    - ISO 50015:2014 (M&V of energy performance)
    - FEMP M&V Guidelines 4.0

Author: GreenLang Team
Version: 40.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class MVPlanReportTemplate:
    """
    M&V plan report template.

    Renders comprehensive M&V plan reports showing ECM description,
    IPMVP option selection with rationale, measurement boundary
    definition, baseline and reporting period specification, metering
    plan details, and adjustment methodology across markdown, HTML,
    and JSON formats. All outputs include SHA-256 provenance hashing
    for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MVPlanReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render M&V plan report as Markdown.

        Args:
            data: M&V plan engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_plan_overview(data),
            self._md_ecm_description(data),
            self._md_ipmvp_option(data),
            self._md_measurement_boundary(data),
            self._md_baseline_period(data),
            self._md_reporting_period(data),
            self._md_metering_plan(data),
            self._md_adjustment_methodology(data),
            self._md_risk_assessment(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render M&V plan report as self-contained HTML.

        Args:
            data: M&V plan engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_plan_overview(data),
            self._html_ecm_description(data),
            self._html_ipmvp_option(data),
            self._html_measurement_boundary(data),
            self._html_baseline_period(data),
            self._html_reporting_period(data),
            self._html_metering_plan(data),
            self._html_adjustment_methodology(data),
            self._html_risk_assessment(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>M&amp;V Plan Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render M&V plan report as structured JSON.

        Args:
            data: M&V plan engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "mv_plan_report",
            "version": "40.0.0",
            "generated_at": self.generated_at.isoformat(),
            "plan_overview": self._json_plan_overview(data),
            "ecm_description": data.get("ecm_description", {}),
            "ipmvp_option": data.get("ipmvp_option", {}),
            "measurement_boundary": data.get("measurement_boundary", {}),
            "baseline_period": data.get("baseline_period", {}),
            "reporting_period": data.get("reporting_period", {}),
            "metering_plan": data.get("metering_plan", {}),
            "adjustment_methodology": data.get("adjustment_methodology", {}),
            "risk_assessment": data.get("risk_assessment", []),
            "recommendations": data.get("recommendations", []),
            "charts": self._json_charts(data),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with project metadata."""
        project = data.get("project_name", "M&V Project")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# M&V Plan Report\n\n"
            f"**Project:** {project}  \n"
            f"**Facility:** {data.get('facility_name', '-')}  \n"
            f"**IPMVP Option:** {data.get('ipmvp_option_label', '-')}  \n"
            f"**Plan Version:** {data.get('plan_version', '1.0')}  \n"
            f"**Prepared By:** {data.get('prepared_by', '-')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-040 MVPlanReportTemplate v40.0.0\n\n---"
        )

    def _md_plan_overview(self, data: Dict[str, Any]) -> str:
        """Render plan overview section."""
        overview = data.get("plan_overview", {})
        return (
            "## 1. Plan Overview\n\n"
            "| Item | Detail |\n|------|--------|\n"
            f"| Project ID | {overview.get('project_id', '-')} |\n"
            f"| ECM Count | {overview.get('ecm_count', 0)} |\n"
            f"| Total Estimated Savings | {self._format_energy(overview.get('estimated_savings_mwh', 0))} |\n"
            f"| Estimated Cost Savings | {self._format_currency(overview.get('estimated_cost_savings', 0))} |\n"
            f"| M&V Duration | {overview.get('mv_duration_months', 12)} months |\n"
            f"| Confidence Target | {self._fmt(overview.get('confidence_target_pct', 90))}% |\n"
            f"| Precision Target | {self._fmt(overview.get('precision_target_pct', 50))}% at 90% confidence |\n"
            f"| IPMVP Option | {overview.get('ipmvp_option', '-')} |\n"
            f"| Status | {overview.get('status', 'Draft')} |"
        )

    def _md_ecm_description(self, data: Dict[str, Any]) -> str:
        """Render ECM description section."""
        ecms = data.get("ecm_list", [])
        if not ecms:
            return "## 2. ECM Description\n\n_No ECM data available._"
        lines = [
            "## 2. ECM Description\n",
            "| # | ECM Name | Type | Estimated Savings (MWh) | Status | Interaction |",
            "|---|----------|------|----------------------:|--------|-------------|",
        ]
        for i, ecm in enumerate(ecms, 1):
            lines.append(
                f"| {i} | {ecm.get('name', '-')} "
                f"| {ecm.get('type', '-')} "
                f"| {self._fmt(ecm.get('estimated_savings_mwh', 0), 1)} "
                f"| {ecm.get('status', '-')} "
                f"| {ecm.get('interaction', 'None')} |"
            )
        desc = data.get("ecm_description", {})
        if desc.get("narrative"):
            lines.append(f"\n**Description:** {desc['narrative']}")
        if desc.get("interaction_effects"):
            lines.append(f"\n**Interaction Effects:** {desc['interaction_effects']}")
        return "\n".join(lines)

    def _md_ipmvp_option(self, data: Dict[str, Any]) -> str:
        """Render IPMVP option selection section."""
        option = data.get("ipmvp_option", {})
        if not option:
            return "## 3. IPMVP Option Selection\n\n_No IPMVP option data available._"
        return (
            "## 3. IPMVP Option Selection\n\n"
            "| Criterion | Detail |\n|-----------|--------|\n"
            f"| Selected Option | {option.get('option', '-')} |\n"
            f"| Option Name | {option.get('option_name', '-')} |\n"
            f"| Selection Rationale | {option.get('rationale', '-')} |\n"
            f"| ECM Isolation | {option.get('ecm_isolation', '-')} |\n"
            f"| Measurement Approach | {option.get('measurement_approach', '-')} |\n"
            f"| Expected Accuracy | {self._fmt(option.get('expected_accuracy_pct', 0))}% |\n"
            f"| Cost of M&V | {self._format_currency(option.get('mv_cost', 0))} |\n"
            f"| M&V Cost Ratio | {self._fmt(option.get('mv_cost_ratio_pct', 0))}% of savings |"
        )

    def _md_measurement_boundary(self, data: Dict[str, Any]) -> str:
        """Render measurement boundary section."""
        boundary = data.get("measurement_boundary", {})
        if not boundary:
            return "## 4. Measurement Boundary\n\n_No measurement boundary data available._"
        energy_streams = boundary.get("energy_streams", [])
        lines = [
            "## 4. Measurement Boundary\n",
            f"**Boundary Description:** {boundary.get('description', '-')}  \n"
            f"**Boundary Type:** {boundary.get('boundary_type', '-')}  \n"
            f"**Facility Area:** {self._fmt(boundary.get('facility_area_sqm', 0), 0)} m2  \n"
            f"**Affected Systems:** {', '.join(boundary.get('affected_systems', []))}  \n",
        ]
        if energy_streams:
            lines.append("### Energy Streams\n")
            lines.append("| Stream | Type | Metered | Direction | Unit |")
            lines.append("|--------|------|---------|-----------|------|")
            for es in energy_streams:
                lines.append(
                    f"| {es.get('name', '-')} "
                    f"| {es.get('type', '-')} "
                    f"| {'Yes' if es.get('metered') else 'No'} "
                    f"| {es.get('direction', '-')} "
                    f"| {es.get('unit', '-')} |"
                )
        static_factors = boundary.get("static_factors", [])
        if static_factors:
            lines.append("\n### Static Factors\n")
            lines.append("| Factor | Baseline Value | Current Value | Changed |")
            lines.append("|--------|-------------:|-------------:|---------|")
            for sf in static_factors:
                lines.append(
                    f"| {sf.get('name', '-')} "
                    f"| {self._fmt(sf.get('baseline_value', 0), 2)} "
                    f"| {self._fmt(sf.get('current_value', 0), 2)} "
                    f"| {'Yes' if sf.get('changed') else 'No'} |"
                )
        return "\n".join(lines)

    def _md_baseline_period(self, data: Dict[str, Any]) -> str:
        """Render baseline period section."""
        baseline = data.get("baseline_period", {})
        if not baseline:
            return "## 5. Baseline Period\n\n_No baseline period data available._"
        return (
            "## 5. Baseline Period\n\n"
            "| Parameter | Value |\n|-----------|-------|\n"
            f"| Start Date | {baseline.get('start_date', '-')} |\n"
            f"| End Date | {baseline.get('end_date', '-')} |\n"
            f"| Duration | {baseline.get('duration_months', 0)} months |\n"
            f"| Data Interval | {baseline.get('data_interval', '-')} |\n"
            f"| Data Points | {baseline.get('data_points', 0)} |\n"
            f"| Data Completeness | {self._fmt(baseline.get('data_completeness_pct', 0))}% |\n"
            f"| Baseline Energy | {self._format_energy(baseline.get('baseline_energy_mwh', 0))} |\n"
            f"| Baseline Cost | {self._format_currency(baseline.get('baseline_cost', 0))} |\n"
            f"| Representative | {baseline.get('representative', '-')} |"
        )

    def _md_reporting_period(self, data: Dict[str, Any]) -> str:
        """Render reporting period section."""
        reporting = data.get("reporting_period", {})
        if not reporting:
            return "## 6. Reporting Period\n\n_No reporting period data available._"
        return (
            "## 6. Reporting Period\n\n"
            "| Parameter | Value |\n|-----------|-------|\n"
            f"| Start Date | {reporting.get('start_date', '-')} |\n"
            f"| End Date | {reporting.get('end_date', '-')} |\n"
            f"| Duration | {reporting.get('duration_months', 0)} months |\n"
            f"| Data Interval | {reporting.get('data_interval', '-')} |\n"
            f"| Reporting Frequency | {reporting.get('reporting_frequency', '-')} |\n"
            f"| First Report Due | {reporting.get('first_report_due', '-')} |\n"
            f"| Post-Installation Verification | {reporting.get('post_install_verification', '-')} |"
        )

    def _md_metering_plan(self, data: Dict[str, Any]) -> str:
        """Render metering plan section."""
        metering = data.get("metering_plan", {})
        meters = metering.get("meters", [])
        if not meters:
            return "## 7. Metering Plan\n\n_No metering plan data available._"
        lines = [
            "## 7. Metering Plan\n",
            f"**Total Meters:** {metering.get('total_meters', len(meters))}  \n"
            f"**Sampling Approach:** {metering.get('sampling_approach', 'Census')}  \n"
            f"**Data Collection:** {metering.get('data_collection_method', '-')}  \n",
            "| Meter ID | Type | Location | Accuracy | Interval | Calibration Due |",
            "|----------|------|----------|----------|----------|-----------------|",
        ]
        for m in meters:
            lines.append(
                f"| {m.get('meter_id', '-')} "
                f"| {m.get('type', '-')} "
                f"| {m.get('location', '-')} "
                f"| {self._fmt(m.get('accuracy_pct', 0), 1)}% "
                f"| {m.get('interval', '-')} "
                f"| {m.get('calibration_due', '-')} |"
            )
        return "\n".join(lines)

    def _md_adjustment_methodology(self, data: Dict[str, Any]) -> str:
        """Render adjustment methodology section."""
        adj = data.get("adjustment_methodology", {})
        if not adj:
            return "## 8. Adjustment Methodology\n\n_No adjustment methodology data available._"
        routine = adj.get("routine_adjustments", [])
        non_routine = adj.get("non_routine_adjustments", [])
        lines = [
            "## 8. Adjustment Methodology\n",
            f"**Model Type:** {adj.get('model_type', '-')}  \n"
            f"**Independent Variables:** {', '.join(adj.get('independent_variables', []))}  \n"
            f"**Adjustment Approach:** {adj.get('approach', '-')}  \n",
        ]
        if routine:
            lines.append("### Routine Adjustments\n")
            lines.append("| Variable | Method | Frequency | Source |")
            lines.append("|----------|--------|-----------|--------|")
            for ra in routine:
                lines.append(
                    f"| {ra.get('variable', '-')} "
                    f"| {ra.get('method', '-')} "
                    f"| {ra.get('frequency', '-')} "
                    f"| {ra.get('source', '-')} |"
                )
        if non_routine:
            lines.append("\n### Non-Routine Adjustments\n")
            lines.append("| Event | Impact (MWh) | Method | Documentation |")
            lines.append("|-------|----------:|--------|---------------|")
            for nra in non_routine:
                lines.append(
                    f"| {nra.get('event', '-')} "
                    f"| {self._fmt(nra.get('impact_mwh', 0), 1)} "
                    f"| {nra.get('method', '-')} "
                    f"| {nra.get('documentation', '-')} |"
                )
        return "\n".join(lines)

    def _md_risk_assessment(self, data: Dict[str, Any]) -> str:
        """Render risk assessment section."""
        risks = data.get("risk_assessment", [])
        if not risks:
            return "## 9. Risk Assessment\n\n_No risk assessment data available._"
        lines = [
            "## 9. Risk Assessment\n",
            "| Risk | Likelihood | Impact | Mitigation | Owner |",
            "|------|-----------|--------|------------|-------|",
        ]
        for r in risks:
            lines.append(
                f"| {r.get('risk', '-')} "
                f"| {r.get('likelihood', '-')} "
                f"| {r.get('impact', '-')} "
                f"| {r.get('mitigation', '-')} "
                f"| {r.get('owner', '-')} |"
            )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Ensure baseline data completeness exceeds 90% before model fitting",
                "Verify meter calibration prior to baseline period commencement",
                "Document all non-routine adjustments with engineering calculations",
                "Review M&V plan at 6-month intervals and update as needed",
            ]
        lines = ["## 10. Recommendations\n"]
        for i, rec in enumerate(recs, 1):
            lines.append(f"{i}. {rec}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-040 M&V Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        project = data.get("project_name", "M&V Project")
        return (
            f'<h1>M&amp;V Plan Report</h1>\n'
            f'<p class="subtitle">Project: {project} | '
            f'Facility: {data.get("facility_name", "-")} | '
            f'IPMVP Option: {data.get("ipmvp_option_label", "-")}</p>'
        )

    def _html_plan_overview(self, data: Dict[str, Any]) -> str:
        """Render HTML plan overview cards."""
        o = data.get("plan_overview", {})
        return (
            '<h2>1. Plan Overview</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">ECM Count</span>'
            f'<span class="value">{o.get("ecm_count", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Est. Savings</span>'
            f'<span class="value">{self._fmt(o.get("estimated_savings_mwh", 0), 1)} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Cost Savings</span>'
            f'<span class="value">{self._format_currency(o.get("estimated_cost_savings", 0))}</span></div>\n'
            f'  <div class="card"><span class="label">Duration</span>'
            f'<span class="value">{o.get("mv_duration_months", 12)} months</span></div>\n'
            f'  <div class="card"><span class="label">IPMVP Option</span>'
            f'<span class="value">{o.get("ipmvp_option", "-")}</span></div>\n'
            '</div>'
        )

    def _html_ecm_description(self, data: Dict[str, Any]) -> str:
        """Render HTML ECM description table."""
        ecms = data.get("ecm_list", [])
        rows = ""
        for i, ecm in enumerate(ecms, 1):
            rows += (
                f'<tr><td>{i}</td><td>{ecm.get("name", "-")}</td>'
                f'<td>{ecm.get("type", "-")}</td>'
                f'<td>{self._fmt(ecm.get("estimated_savings_mwh", 0), 1)}</td>'
                f'<td>{ecm.get("status", "-")}</td>'
                f'<td>{ecm.get("interaction", "None")}</td></tr>\n'
            )
        return (
            '<h2>2. ECM Description</h2>\n'
            '<table>\n<tr><th>#</th><th>ECM Name</th><th>Type</th>'
            f'<th>Est. Savings (MWh)</th><th>Status</th><th>Interaction</th></tr>\n{rows}</table>'
        )

    def _html_ipmvp_option(self, data: Dict[str, Any]) -> str:
        """Render HTML IPMVP option table."""
        option = data.get("ipmvp_option", {})
        return (
            '<h2>3. IPMVP Option Selection</h2>\n'
            '<table>\n'
            f'<tr><th>Criterion</th><th>Detail</th></tr>\n'
            f'<tr><td>Selected Option</td><td>{option.get("option", "-")}</td></tr>\n'
            f'<tr><td>Option Name</td><td>{option.get("option_name", "-")}</td></tr>\n'
            f'<tr><td>Rationale</td><td>{option.get("rationale", "-")}</td></tr>\n'
            f'<tr><td>ECM Isolation</td><td>{option.get("ecm_isolation", "-")}</td></tr>\n'
            f'<tr><td>Measurement Approach</td><td>{option.get("measurement_approach", "-")}</td></tr>\n'
            f'<tr><td>Expected Accuracy</td><td>{self._fmt(option.get("expected_accuracy_pct", 0))}%</td></tr>\n'
            f'<tr><td>M&amp;V Cost</td><td>{self._format_currency(option.get("mv_cost", 0))}</td></tr>\n'
            '</table>'
        )

    def _html_measurement_boundary(self, data: Dict[str, Any]) -> str:
        """Render HTML measurement boundary section."""
        boundary = data.get("measurement_boundary", {})
        streams = boundary.get("energy_streams", [])
        rows = ""
        for es in streams:
            rows += (
                f'<tr><td>{es.get("name", "-")}</td>'
                f'<td>{es.get("type", "-")}</td>'
                f'<td>{"Yes" if es.get("metered") else "No"}</td>'
                f'<td>{es.get("direction", "-")}</td>'
                f'<td>{es.get("unit", "-")}</td></tr>\n'
            )
        return (
            '<h2>4. Measurement Boundary</h2>\n'
            f'<p>Boundary: {boundary.get("description", "-")} | '
            f'Type: {boundary.get("boundary_type", "-")} | '
            f'Area: {self._fmt(boundary.get("facility_area_sqm", 0), 0)} m&sup2;</p>\n'
            '<table>\n<tr><th>Stream</th><th>Type</th><th>Metered</th>'
            f'<th>Direction</th><th>Unit</th></tr>\n{rows}</table>'
        )

    def _html_baseline_period(self, data: Dict[str, Any]) -> str:
        """Render HTML baseline period table."""
        bl = data.get("baseline_period", {})
        return (
            '<h2>5. Baseline Period</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Start</span>'
            f'<span class="value">{bl.get("start_date", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">End</span>'
            f'<span class="value">{bl.get("end_date", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Data Points</span>'
            f'<span class="value">{bl.get("data_points", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Completeness</span>'
            f'<span class="value">{self._fmt(bl.get("data_completeness_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Energy</span>'
            f'<span class="value">{self._fmt(bl.get("baseline_energy_mwh", 0), 1)} MWh</span></div>\n'
            '</div>'
        )

    def _html_reporting_period(self, data: Dict[str, Any]) -> str:
        """Render HTML reporting period table."""
        rp = data.get("reporting_period", {})
        return (
            '<h2>6. Reporting Period</h2>\n'
            '<table>\n'
            f'<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Start Date</td><td>{rp.get("start_date", "-")}</td></tr>\n'
            f'<tr><td>End Date</td><td>{rp.get("end_date", "-")}</td></tr>\n'
            f'<tr><td>Duration</td><td>{rp.get("duration_months", 0)} months</td></tr>\n'
            f'<tr><td>Data Interval</td><td>{rp.get("data_interval", "-")}</td></tr>\n'
            f'<tr><td>Reporting Frequency</td><td>{rp.get("reporting_frequency", "-")}</td></tr>\n'
            f'<tr><td>First Report Due</td><td>{rp.get("first_report_due", "-")}</td></tr>\n'
            '</table>'
        )

    def _html_metering_plan(self, data: Dict[str, Any]) -> str:
        """Render HTML metering plan table."""
        metering = data.get("metering_plan", {})
        meters = metering.get("meters", [])
        rows = ""
        for m in meters:
            rows += (
                f'<tr><td>{m.get("meter_id", "-")}</td>'
                f'<td>{m.get("type", "-")}</td>'
                f'<td>{m.get("location", "-")}</td>'
                f'<td>{self._fmt(m.get("accuracy_pct", 0), 1)}%</td>'
                f'<td>{m.get("interval", "-")}</td>'
                f'<td>{m.get("calibration_due", "-")}</td></tr>\n'
            )
        return (
            '<h2>7. Metering Plan</h2>\n'
            '<table>\n<tr><th>Meter ID</th><th>Type</th><th>Location</th>'
            f'<th>Accuracy</th><th>Interval</th><th>Calibration Due</th></tr>\n{rows}</table>'
        )

    def _html_adjustment_methodology(self, data: Dict[str, Any]) -> str:
        """Render HTML adjustment methodology."""
        adj = data.get("adjustment_methodology", {})
        routine = adj.get("routine_adjustments", [])
        r_rows = ""
        for ra in routine:
            r_rows += (
                f'<tr><td>{ra.get("variable", "-")}</td>'
                f'<td>{ra.get("method", "-")}</td>'
                f'<td>{ra.get("frequency", "-")}</td>'
                f'<td>{ra.get("source", "-")}</td></tr>\n'
            )
        non_routine = adj.get("non_routine_adjustments", [])
        nr_rows = ""
        for nra in non_routine:
            nr_rows += (
                f'<tr><td>{nra.get("event", "-")}</td>'
                f'<td>{self._fmt(nra.get("impact_mwh", 0), 1)}</td>'
                f'<td>{nra.get("method", "-")}</td>'
                f'<td>{nra.get("documentation", "-")}</td></tr>\n'
            )
        return (
            '<h2>8. Adjustment Methodology</h2>\n'
            f'<p>Model: {adj.get("model_type", "-")} | '
            f'Variables: {", ".join(adj.get("independent_variables", []))}</p>\n'
            '<h3>Routine Adjustments</h3>\n'
            '<table>\n<tr><th>Variable</th><th>Method</th>'
            f'<th>Frequency</th><th>Source</th></tr>\n{r_rows}</table>\n'
            '<h3>Non-Routine Adjustments</h3>\n'
            '<table>\n<tr><th>Event</th><th>Impact (MWh)</th>'
            f'<th>Method</th><th>Documentation</th></tr>\n{nr_rows}</table>'
        )

    def _html_risk_assessment(self, data: Dict[str, Any]) -> str:
        """Render HTML risk assessment table."""
        risks = data.get("risk_assessment", [])
        rows = ""
        for r in risks:
            cls = "severity-high" if r.get("impact") == "High" else (
                "severity-medium" if r.get("impact") == "Medium" else "severity-low"
            )
            rows += (
                f'<tr><td>{r.get("risk", "-")}</td>'
                f'<td>{r.get("likelihood", "-")}</td>'
                f'<td class="{cls}">{r.get("impact", "-")}</td>'
                f'<td>{r.get("mitigation", "-")}</td>'
                f'<td>{r.get("owner", "-")}</td></tr>\n'
            )
        return (
            '<h2>9. Risk Assessment</h2>\n'
            '<table>\n<tr><th>Risk</th><th>Likelihood</th><th>Impact</th>'
            f'<th>Mitigation</th><th>Owner</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Ensure baseline data completeness exceeds 90% before model fitting",
            "Verify meter calibration prior to baseline period commencement",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>10. Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_plan_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON plan overview."""
        o = data.get("plan_overview", {})
        return {
            "project_id": o.get("project_id", ""),
            "ecm_count": o.get("ecm_count", 0),
            "estimated_savings_mwh": o.get("estimated_savings_mwh", 0),
            "estimated_cost_savings": o.get("estimated_cost_savings", 0),
            "mv_duration_months": o.get("mv_duration_months", 12),
            "confidence_target_pct": o.get("confidence_target_pct", 90),
            "precision_target_pct": o.get("precision_target_pct", 50),
            "ipmvp_option": o.get("ipmvp_option", ""),
            "status": o.get("status", "Draft"),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        ecms = data.get("ecm_list", [])
        streams = data.get("measurement_boundary", {}).get("energy_streams", [])
        risks = data.get("risk_assessment", [])
        return {
            "ecm_savings": {
                "type": "bar",
                "labels": [e.get("name", "") for e in ecms],
                "values": [e.get("estimated_savings_mwh", 0) for e in ecms],
            },
            "energy_streams": {
                "type": "pie",
                "labels": [s.get("name", "") for s in streams],
                "values": [1 for _ in streams],
            },
            "risk_matrix": {
                "type": "scatter",
                "items": [
                    {
                        "risk": r.get("risk", ""),
                        "likelihood": r.get("likelihood", ""),
                        "impact": r.get("impact", ""),
                    }
                    for r in risks
                ],
            },
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
            "h3{color:#495057;margin-top:20px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:150px;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".severity-high{color:#dc3545;font-weight:700;}"
            ".severity-medium{color:#fd7e14;font-weight:600;}"
            ".severity-low{color:#198754;font-weight:500;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _format_currency(self, val: Any) -> str:
        """Format a currency value with comma separators.

        Args:
            val: Numeric value to format.

        Returns:
            Formatted currency string (e.g., 'EUR 1,234.00').
        """
        if isinstance(val, (int, float)):
            return f"EUR {val:,.2f}"
        return str(val)

    def _format_energy(self, val: Any) -> str:
        """Format an energy value with units.

        Args:
            val: Energy value in MWh.

        Returns:
            Formatted energy string (e.g., '1,234.00 MWh').
        """
        if isinstance(val, (int, float)):
            return f"{val:,.2f} MWh"
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

    def _pct(self, part: float, whole: float) -> str:
        """Calculate and format a percentage.

        Args:
            part: Numerator value.
            whole: Denominator value.

        Returns:
            Formatted percentage string.
        """
        if whole == 0:
            return "0.0%"
        return f"{(part / whole) * 100:.1f}%"

    def _generate_provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash.

        Args:
            content: Content string to hash.

        Returns:
            SHA-256 hex digest.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
