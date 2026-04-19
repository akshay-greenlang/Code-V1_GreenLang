# -*- coding: utf-8 -*-
"""
MeteringPlanReportTemplate - Metering Plan Report for PACK-040.

Generates comprehensive metering plan reports covering meter inventory
with specifications, calibration schedule tracking, sampling protocol
details, data management procedures, communication infrastructure,
and quality assurance requirements.

Sections:
    1. Metering Plan Summary
    2. Meter Inventory
    3. Meter Specifications
    4. Calibration Schedule
    5. Sampling Protocol
    6. Data Collection
    7. Data Management
    8. Quality Assurance
    9. Communication Infrastructure
    10. Recommendations

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - IPMVP Core Concepts 2022 (metering requirements)
    - ASHRAE Guideline 14-2014 (data requirements)
    - ISO 50015:2014 (measurement plans)
    - IEC 61000 (power quality monitoring)

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


class MeteringPlanReportTemplate:
    """
    Metering plan report template.

    Renders comprehensive metering plan reports showing meter inventory
    with full specifications, calibration schedule with drift tracking,
    sampling protocol details, data collection and management procedures,
    communication infrastructure, and quality assurance requirements
    across markdown, HTML, and JSON formats. All outputs include SHA-256
    provenance hashing for audit trail integrity.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize MeteringPlanReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render metering plan report as Markdown.

        Args:
            data: Metering engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_plan_summary(data),
            self._md_meter_inventory(data),
            self._md_meter_specifications(data),
            self._md_calibration_schedule(data),
            self._md_sampling_protocol(data),
            self._md_data_collection(data),
            self._md_data_management(data),
            self._md_quality_assurance(data),
            self._md_communication_infra(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render metering plan report as self-contained HTML.

        Args:
            data: Metering engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_plan_summary(data),
            self._html_meter_inventory(data),
            self._html_meter_specifications(data),
            self._html_calibration_schedule(data),
            self._html_sampling_protocol(data),
            self._html_data_collection(data),
            self._html_data_management(data),
            self._html_quality_assurance(data),
            self._html_communication_infra(data),
            self._html_recommendations(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Metering Plan Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render metering plan report as structured JSON.

        Args:
            data: Metering engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "metering_plan_report",
            "version": "40.0.0",
            "generated_at": self.generated_at.isoformat(),
            "plan_summary": self._json_plan_summary(data),
            "meter_inventory": data.get("meter_inventory", []),
            "meter_specifications": data.get("meter_specifications", []),
            "calibration_schedule": data.get("calibration_schedule", []),
            "sampling_protocol": data.get("sampling_protocol", {}),
            "data_collection": data.get("data_collection", {}),
            "data_management": data.get("data_management", {}),
            "quality_assurance": data.get("quality_assurance", {}),
            "communication_infra": data.get("communication_infra", {}),
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
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Metering Plan Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Project:** {data.get('project_name', '-')}  \n"
            f"**Total Meters:** {data.get('total_meters', 0)}  \n"
            f"**Plan Version:** {data.get('plan_version', '1.0')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-040 MeteringPlanReportTemplate v40.0.0\n\n---"
        )

    def _md_plan_summary(self, data: Dict[str, Any]) -> str:
        """Render metering plan summary section."""
        s = data.get("plan_summary", {})
        return (
            "## 1. Metering Plan Summary\n\n"
            "| Item | Detail |\n|------|--------|\n"
            f"| Total Meters | {s.get('total_meters', 0)} |\n"
            f"| Permanent Meters | {s.get('permanent_meters', 0)} |\n"
            f"| Temporary Meters | {s.get('temporary_meters', 0)} |\n"
            f"| Utility Meters | {s.get('utility_meters', 0)} |\n"
            f"| Sub-meters | {s.get('sub_meters', 0)} |\n"
            f"| Sampling Required | {s.get('sampling_required', '-')} |\n"
            f"| Data Interval | {s.get('data_interval', '-')} |\n"
            f"| Data Collection | {s.get('data_collection_method', '-')} |\n"
            f"| Estimated Cost | {self._format_currency(s.get('estimated_cost', 0))} |"
        )

    def _md_meter_inventory(self, data: Dict[str, Any]) -> str:
        """Render meter inventory section."""
        meters = data.get("meter_inventory", [])
        if not meters:
            return "## 2. Meter Inventory\n\n_No meter inventory data available._"
        lines = [
            "## 2. Meter Inventory\n",
            "| Meter ID | Type | Location | Measurement | Status | Owner |",
            "|----------|------|----------|-------------|--------|-------|",
        ]
        for m in meters:
            lines.append(
                f"| {m.get('meter_id', '-')} "
                f"| {m.get('type', '-')} "
                f"| {m.get('location', '-')} "
                f"| {m.get('measurement', '-')} "
                f"| {m.get('status', '-')} "
                f"| {m.get('owner', '-')} |"
            )
        return "\n".join(lines)

    def _md_meter_specifications(self, data: Dict[str, Any]) -> str:
        """Render meter specifications section."""
        specs = data.get("meter_specifications", [])
        if not specs:
            return "## 3. Meter Specifications\n\n_No meter specification data available._"
        lines = [
            "## 3. Meter Specifications\n",
            "| Meter ID | Make/Model | Range | Accuracy (%) | Resolution | Protocol |",
            "|----------|-----------|-------|----------:|-----------|----------|",
        ]
        for s in specs:
            lines.append(
                f"| {s.get('meter_id', '-')} "
                f"| {s.get('make_model', '-')} "
                f"| {s.get('range', '-')} "
                f"| {self._fmt(s.get('accuracy_pct', 0), 2)} "
                f"| {s.get('resolution', '-')} "
                f"| {s.get('protocol', '-')} |"
            )
        return "\n".join(lines)

    def _md_calibration_schedule(self, data: Dict[str, Any]) -> str:
        """Render calibration schedule section."""
        schedule = data.get("calibration_schedule", [])
        if not schedule:
            return "## 4. Calibration Schedule\n\n_No calibration schedule data available._"
        lines = [
            "## 4. Calibration Schedule\n",
            "| Meter ID | Last Calibration | Next Due | Frequency | Lab | Status |",
            "|----------|-----------------|----------|-----------|-----|--------|",
        ]
        for cal in schedule:
            lines.append(
                f"| {cal.get('meter_id', '-')} "
                f"| {cal.get('last_calibration', '-')} "
                f"| {cal.get('next_due', '-')} "
                f"| {cal.get('frequency', '-')} "
                f"| {cal.get('lab', '-')} "
                f"| {cal.get('status', '-')} |"
            )
        return "\n".join(lines)

    def _md_sampling_protocol(self, data: Dict[str, Any]) -> str:
        """Render sampling protocol section."""
        sampling = data.get("sampling_protocol", {})
        if not sampling:
            return "## 5. Sampling Protocol\n\n_No sampling protocol data available._"
        return (
            "## 5. Sampling Protocol\n\n"
            "| Parameter | Value |\n|-----------|-------|\n"
            f"| Sampling Required | {sampling.get('required', '-')} |\n"
            f"| Population Size | {sampling.get('population_size', '-')} |\n"
            f"| Sample Size | {sampling.get('sample_size', '-')} |\n"
            f"| Sampling Method | {sampling.get('method', '-')} |\n"
            f"| Stratification | {sampling.get('stratification', '-')} |\n"
            f"| Confidence Level | {self._fmt(sampling.get('confidence_level_pct', 90))}% |\n"
            f"| Precision Target | {self._fmt(sampling.get('precision_target_pct', 0))}% |\n"
            f"| Rotation Schedule | {sampling.get('rotation_schedule', '-')} |\n"
            f"| Sample Selection | {sampling.get('selection_criteria', '-')} |"
        )

    def _md_data_collection(self, data: Dict[str, Any]) -> str:
        """Render data collection section."""
        dc = data.get("data_collection", {})
        if not dc:
            return "## 6. Data Collection\n\n_No data collection data available._"
        return (
            "## 6. Data Collection\n\n"
            "| Parameter | Value |\n|-----------|-------|\n"
            f"| Collection Method | {dc.get('method', '-')} |\n"
            f"| Data Interval | {dc.get('interval', '-')} |\n"
            f"| Storage Format | {dc.get('storage_format', '-')} |\n"
            f"| Transmission | {dc.get('transmission', '-')} |\n"
            f"| Redundancy | {dc.get('redundancy', '-')} |\n"
            f"| Gap Handling | {dc.get('gap_handling', '-')} |\n"
            f"| Responsible Party | {dc.get('responsible_party', '-')} |\n"
            f"| Verification | {dc.get('verification_method', '-')} |"
        )

    def _md_data_management(self, data: Dict[str, Any]) -> str:
        """Render data management section."""
        dm = data.get("data_management", {})
        if not dm:
            return "## 7. Data Management\n\n_No data management data available._"
        return (
            "## 7. Data Management\n\n"
            "| Parameter | Value |\n|-----------|-------|\n"
            f"| Storage System | {dm.get('storage_system', '-')} |\n"
            f"| Backup Frequency | {dm.get('backup_frequency', '-')} |\n"
            f"| Retention Period | {dm.get('retention_period', '-')} |\n"
            f"| Access Control | {dm.get('access_control', '-')} |\n"
            f"| Data Validation | {dm.get('data_validation', '-')} |\n"
            f"| Audit Trail | {dm.get('audit_trail', '-')} |\n"
            f"| Export Formats | {dm.get('export_formats', '-')} |\n"
            f"| Integration | {dm.get('integration', '-')} |"
        )

    def _md_quality_assurance(self, data: Dict[str, Any]) -> str:
        """Render quality assurance section."""
        qa = data.get("quality_assurance", {})
        if not qa:
            return "## 8. Quality Assurance\n\n_No quality assurance data available._"
        checks = qa.get("checks", [])
        lines = [
            "## 8. Quality Assurance\n",
            f"**QA Level:** {qa.get('qa_level', '-')}  \n"
            f"**Completeness Target:** {self._fmt(qa.get('completeness_target_pct', 90))}%  \n"
            f"**Accuracy Target:** {self._fmt(qa.get('accuracy_target_pct', 0))}%  \n",
        ]
        if checks:
            lines.append("| Check | Frequency | Method | Responsible |")
            lines.append("|-------|-----------|--------|-------------|")
            for chk in checks:
                lines.append(
                    f"| {chk.get('check', '-')} "
                    f"| {chk.get('frequency', '-')} "
                    f"| {chk.get('method', '-')} "
                    f"| {chk.get('responsible', '-')} |"
                )
        return "\n".join(lines)

    def _md_communication_infra(self, data: Dict[str, Any]) -> str:
        """Render communication infrastructure section."""
        comm = data.get("communication_infra", {})
        if not comm:
            return "## 9. Communication Infrastructure\n\n_No communication data available._"
        channels = comm.get("channels", [])
        lines = [
            "## 9. Communication Infrastructure\n",
            f"**Network Type:** {comm.get('network_type', '-')}  \n"
            f"**Gateway:** {comm.get('gateway', '-')}  \n"
            f"**Uptime Target:** {self._fmt(comm.get('uptime_target_pct', 99.5))}%  \n",
        ]
        if channels:
            lines.append("| Channel | Protocol | Meters | Bandwidth | Redundancy |")
            lines.append("|---------|----------|-------:|-----------|-----------|")
            for ch in channels:
                lines.append(
                    f"| {ch.get('name', '-')} "
                    f"| {ch.get('protocol', '-')} "
                    f"| {ch.get('meter_count', 0)} "
                    f"| {ch.get('bandwidth', '-')} "
                    f"| {ch.get('redundancy', '-')} |"
                )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations section."""
        recs = data.get("recommendations", [])
        if not recs:
            recs = [
                "Verify all meter calibration certificates before baseline starts",
                "Establish automated data gap alerting with 24-hour escalation",
                "Implement redundant communication paths for critical meters",
                "Schedule quarterly metering plan review meetings",
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
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Metering Plan Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Project: {data.get("project_name", "-")} | '
            f'Meters: {data.get("total_meters", 0)}</p>'
        )

    def _html_plan_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML plan summary cards."""
        s = data.get("plan_summary", {})
        return (
            '<h2>1. Metering Plan Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Total Meters</span>'
            f'<span class="value">{s.get("total_meters", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Permanent</span>'
            f'<span class="value">{s.get("permanent_meters", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Temporary</span>'
            f'<span class="value">{s.get("temporary_meters", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Data Interval</span>'
            f'<span class="value">{s.get("data_interval", "-")}</span></div>\n'
            f'  <div class="card"><span class="label">Est. Cost</span>'
            f'<span class="value">{self._format_currency(s.get("estimated_cost", 0))}</span></div>\n'
            '</div>'
        )

    def _html_meter_inventory(self, data: Dict[str, Any]) -> str:
        """Render HTML meter inventory table."""
        meters = data.get("meter_inventory", [])
        rows = ""
        for m in meters:
            rows += (
                f'<tr><td>{m.get("meter_id", "-")}</td>'
                f'<td>{m.get("type", "-")}</td>'
                f'<td>{m.get("location", "-")}</td>'
                f'<td>{m.get("measurement", "-")}</td>'
                f'<td>{m.get("status", "-")}</td>'
                f'<td>{m.get("owner", "-")}</td></tr>\n'
            )
        return (
            '<h2>2. Meter Inventory</h2>\n'
            '<table>\n<tr><th>Meter ID</th><th>Type</th><th>Location</th>'
            f'<th>Measurement</th><th>Status</th><th>Owner</th></tr>\n{rows}</table>'
        )

    def _html_meter_specifications(self, data: Dict[str, Any]) -> str:
        """Render HTML meter specifications table."""
        specs = data.get("meter_specifications", [])
        rows = ""
        for s in specs:
            rows += (
                f'<tr><td>{s.get("meter_id", "-")}</td>'
                f'<td>{s.get("make_model", "-")}</td>'
                f'<td>{s.get("range", "-")}</td>'
                f'<td>{self._fmt(s.get("accuracy_pct", 0), 2)}%</td>'
                f'<td>{s.get("resolution", "-")}</td>'
                f'<td>{s.get("protocol", "-")}</td></tr>\n'
            )
        return (
            '<h2>3. Meter Specifications</h2>\n'
            '<table>\n<tr><th>Meter ID</th><th>Make/Model</th><th>Range</th>'
            f'<th>Accuracy</th><th>Resolution</th><th>Protocol</th></tr>\n{rows}</table>'
        )

    def _html_calibration_schedule(self, data: Dict[str, Any]) -> str:
        """Render HTML calibration schedule table."""
        schedule = data.get("calibration_schedule", [])
        rows = ""
        for cal in schedule:
            cls = "severity-high" if cal.get("status") == "Overdue" else "severity-low"
            rows += (
                f'<tr><td>{cal.get("meter_id", "-")}</td>'
                f'<td>{cal.get("last_calibration", "-")}</td>'
                f'<td>{cal.get("next_due", "-")}</td>'
                f'<td>{cal.get("frequency", "-")}</td>'
                f'<td class="{cls}">{cal.get("status", "-")}</td></tr>\n'
            )
        return (
            '<h2>4. Calibration Schedule</h2>\n'
            '<table>\n<tr><th>Meter ID</th><th>Last Cal</th><th>Next Due</th>'
            f'<th>Frequency</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_sampling_protocol(self, data: Dict[str, Any]) -> str:
        """Render HTML sampling protocol table."""
        sampling = data.get("sampling_protocol", {})
        return (
            '<h2>5. Sampling Protocol</h2>\n'
            '<table>\n'
            f'<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Population Size</td><td>{sampling.get("population_size", "-")}</td></tr>\n'
            f'<tr><td>Sample Size</td><td>{sampling.get("sample_size", "-")}</td></tr>\n'
            f'<tr><td>Sampling Method</td><td>{sampling.get("method", "-")}</td></tr>\n'
            f'<tr><td>Confidence Level</td><td>{self._fmt(sampling.get("confidence_level_pct", 90))}%</td></tr>\n'
            f'<tr><td>Precision Target</td><td>{self._fmt(sampling.get("precision_target_pct", 0))}%</td></tr>\n'
            '</table>'
        )

    def _html_data_collection(self, data: Dict[str, Any]) -> str:
        """Render HTML data collection table."""
        dc = data.get("data_collection", {})
        return (
            '<h2>6. Data Collection</h2>\n'
            '<table>\n'
            f'<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Method</td><td>{dc.get("method", "-")}</td></tr>\n'
            f'<tr><td>Interval</td><td>{dc.get("interval", "-")}</td></tr>\n'
            f'<tr><td>Transmission</td><td>{dc.get("transmission", "-")}</td></tr>\n'
            f'<tr><td>Gap Handling</td><td>{dc.get("gap_handling", "-")}</td></tr>\n'
            f'<tr><td>Responsible</td><td>{dc.get("responsible_party", "-")}</td></tr>\n'
            '</table>'
        )

    def _html_data_management(self, data: Dict[str, Any]) -> str:
        """Render HTML data management table."""
        dm = data.get("data_management", {})
        return (
            '<h2>7. Data Management</h2>\n'
            '<table>\n'
            f'<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Storage System</td><td>{dm.get("storage_system", "-")}</td></tr>\n'
            f'<tr><td>Backup Frequency</td><td>{dm.get("backup_frequency", "-")}</td></tr>\n'
            f'<tr><td>Retention Period</td><td>{dm.get("retention_period", "-")}</td></tr>\n'
            f'<tr><td>Access Control</td><td>{dm.get("access_control", "-")}</td></tr>\n'
            f'<tr><td>Audit Trail</td><td>{dm.get("audit_trail", "-")}</td></tr>\n'
            '</table>'
        )

    def _html_quality_assurance(self, data: Dict[str, Any]) -> str:
        """Render HTML quality assurance table."""
        qa = data.get("quality_assurance", {})
        checks = qa.get("checks", [])
        rows = ""
        for chk in checks:
            rows += (
                f'<tr><td>{chk.get("check", "-")}</td>'
                f'<td>{chk.get("frequency", "-")}</td>'
                f'<td>{chk.get("method", "-")}</td>'
                f'<td>{chk.get("responsible", "-")}</td></tr>\n'
            )
        return (
            '<h2>8. Quality Assurance</h2>\n'
            f'<p>Target Completeness: {self._fmt(qa.get("completeness_target_pct", 90))}% | '
            f'QA Level: {qa.get("qa_level", "-")}</p>\n'
            '<table>\n<tr><th>Check</th><th>Frequency</th>'
            f'<th>Method</th><th>Responsible</th></tr>\n{rows}</table>'
        )

    def _html_communication_infra(self, data: Dict[str, Any]) -> str:
        """Render HTML communication infrastructure."""
        comm = data.get("communication_infra", {})
        channels = comm.get("channels", [])
        rows = ""
        for ch in channels:
            rows += (
                f'<tr><td>{ch.get("name", "-")}</td>'
                f'<td>{ch.get("protocol", "-")}</td>'
                f'<td>{ch.get("meter_count", 0)}</td>'
                f'<td>{ch.get("bandwidth", "-")}</td>'
                f'<td>{ch.get("redundancy", "-")}</td></tr>\n'
            )
        return (
            '<h2>9. Communication Infrastructure</h2>\n'
            f'<p>Network: {comm.get("network_type", "-")} | '
            f'Uptime Target: {self._fmt(comm.get("uptime_target_pct", 99.5))}%</p>\n'
            '<table>\n<tr><th>Channel</th><th>Protocol</th><th>Meters</th>'
            f'<th>Bandwidth</th><th>Redundancy</th></tr>\n{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML recommendations."""
        recs = data.get("recommendations", [
            "Verify all meter calibration certificates before baseline starts",
            "Establish automated data gap alerting with 24-hour escalation",
        ])
        items = "".join(f'<li>{r}</li>\n' for r in recs)
        return f'<h2>10. Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_plan_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON plan summary."""
        s = data.get("plan_summary", {})
        return {
            "total_meters": s.get("total_meters", 0),
            "permanent_meters": s.get("permanent_meters", 0),
            "temporary_meters": s.get("temporary_meters", 0),
            "data_interval": s.get("data_interval", ""),
            "estimated_cost": s.get("estimated_cost", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        meters = data.get("meter_inventory", [])
        schedule = data.get("calibration_schedule", [])
        meter_types: Dict[str, int] = {}
        for m in meters:
            t = m.get("type", "Other")
            meter_types[t] = meter_types.get(t, 0) + 1
        cal_statuses: Dict[str, int] = {}
        for cal in schedule:
            s = cal.get("status", "Unknown")
            cal_statuses[s] = cal_statuses.get(s, 0) + 1
        return {
            "meter_type_distribution": {
                "type": "pie",
                "labels": list(meter_types.keys()),
                "values": list(meter_types.values()),
            },
            "calibration_status": {
                "type": "donut",
                "labels": list(cal_statuses.keys()),
                "values": list(cal_statuses.values()),
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
