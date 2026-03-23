# -*- coding: utf-8 -*-
"""
DispatchPlanReportTemplate - DR event curtailment sequence for PACK-037.

Generates dispatch plan reports for demand response events detailing load
curtailment sequences with timing, kW reduction targets per load, DER
dispatch orders, pre-conditioning schedules, and communication protocols.

Sections:
    1. Event Summary
    2. Pre-Conditioning Schedule
    3. Curtailment Sequence
    4. DER Dispatch Plan
    5. Communication Protocol
    6. Contingency Actions

Output Formats:
    - Markdown (with provenance hash comment)
    - HTML (self-contained with inline CSS)
    - JSON (structured with chart data)

Regulatory References:
    - FERC Order 2222 (DER dispatch requirements)
    - NERC Reliability Standards (BAL-002, EOP)
    - ISO/RTO operating procedures

Author: GreenLang Team
Version: 37.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


class DispatchPlanReportTemplate:
    """
    DR event dispatch plan report template.

    Renders load curtailment sequences for demand response events with
    timing schedules, DER dispatch orders, pre-conditioning plans, and
    contingency actions across markdown, HTML, and JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize DispatchPlanReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render dispatch plan report as Markdown.

        Args:
            data: Dispatch planning engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_event_summary(data),
            self._md_pre_conditioning(data),
            self._md_curtailment_sequence(data),
            self._md_der_dispatch(data),
            self._md_communication_protocol(data),
            self._md_contingency_actions(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render dispatch plan report as self-contained HTML.

        Args:
            data: Dispatch planning engine result data.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_event_summary(data),
            self._html_pre_conditioning(data),
            self._html_curtailment_sequence(data),
            self._html_der_dispatch(data),
            self._html_communication_protocol(data),
            self._html_contingency_actions(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Dispatch Plan Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render dispatch plan report as structured JSON.

        Args:
            data: Dispatch planning engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "dispatch_plan_report",
            "version": "37.0.0",
            "generated_at": self.generated_at.isoformat(),
            "event_summary": self._json_event_summary(data),
            "pre_conditioning": data.get("pre_conditioning", []),
            "curtailment_sequence": data.get("curtailment_sequence", []),
            "der_dispatch": data.get("der_dispatch", []),
            "communication_protocol": data.get("communication_protocol", []),
            "contingency_actions": data.get("contingency_actions", []),
            "charts": self._json_charts(data),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# DR Event Dispatch Plan\n\n"
            f"**Facility:** {facility}  \n"
            f"**Event ID:** {data.get('event_id', '')}  \n"
            f"**Program:** {data.get('program_name', '')}  \n"
            f"**Event Date:** {data.get('event_date', '')}  \n"
            f"**Event Window:** {data.get('event_start', '')} - {data.get('event_end', '')}  \n"
            f"**Plan Generated:** {ts}  \n"
            f"**Template:** PACK-037 DispatchPlanReportTemplate v37.0.0\n\n---"
        )

    def _md_event_summary(self, data: Dict[str, Any]) -> str:
        """Render event summary section."""
        summary = data.get("event_summary", {})
        return (
            "## 1. Event Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Event Type | {summary.get('event_type', '-')} |\n"
            f"| Required Curtailment | {self._format_power(summary.get('required_curtailment_kw', 0))} |\n"
            f"| Planned Curtailment | {self._format_power(summary.get('planned_curtailment_kw', 0))} |\n"
            f"| Safety Margin | {self._fmt(summary.get('safety_margin_pct', 0))}% |\n"
            f"| Event Duration | {summary.get('duration_hours', 0)} hours |\n"
            f"| Notification Lead Time | {summary.get('notification_lead_minutes', 0)} min |\n"
            f"| Loads Involved | {summary.get('loads_involved', 0)} |\n"
            f"| DERs Dispatched | {summary.get('ders_dispatched', 0)} |\n"
            f"| Estimated Revenue | {self._format_currency(summary.get('estimated_revenue', 0))} |"
        )

    def _md_pre_conditioning(self, data: Dict[str, Any]) -> str:
        """Render pre-conditioning schedule section."""
        steps = data.get("pre_conditioning", [])
        if not steps:
            return "## 2. Pre-Conditioning Schedule\n\n_No pre-conditioning required._"
        lines = [
            "## 2. Pre-Conditioning Schedule\n",
            "| Time (offset) | Action | System | Expected kW Impact | Notes |",
            "|--------------|--------|--------|------------------:|-------|",
        ]
        for step in steps:
            lines.append(
                f"| {step.get('time_offset', '-')} "
                f"| {step.get('action', '-')} "
                f"| {step.get('system', '-')} "
                f"| {self._fmt(step.get('kw_impact', 0), 1)} "
                f"| {step.get('notes', '-')} |"
            )
        return "\n".join(lines)

    def _md_curtailment_sequence(self, data: Dict[str, Any]) -> str:
        """Render curtailment sequence table."""
        sequence = data.get("curtailment_sequence", [])
        if not sequence:
            return "## 3. Curtailment Sequence\n\n_No curtailment sequence defined._"
        lines = [
            "## 3. Curtailment Sequence\n",
            "| Step | Time | Load | Action | Reduction (kW) | Cumulative (kW) | Responsible |",
            "|-----:|------|------|--------|---------------:|----------------:|-------------|",
        ]
        cumulative = 0
        for i, step in enumerate(sequence, 1):
            reduction = step.get("reduction_kw", 0)
            cumulative += reduction
            lines.append(
                f"| {i} | {step.get('time', '-')} "
                f"| {step.get('load_name', '-')} "
                f"| {step.get('action', '-')} "
                f"| {self._fmt(reduction, 1)} "
                f"| {self._fmt(cumulative, 1)} "
                f"| {step.get('responsible', '-')} |"
            )
        return "\n".join(lines)

    def _md_der_dispatch(self, data: Dict[str, Any]) -> str:
        """Render DER dispatch plan section."""
        ders = data.get("der_dispatch", [])
        if not ders:
            return "## 4. DER Dispatch Plan\n\n_No DERs available for dispatch._"
        lines = [
            "## 4. DER Dispatch Plan\n",
            "| DER Asset | Type | Capacity (kW) | Dispatch Time | Duration | SOC/Status |",
            "|-----------|------|-------------:|--------------|----------|-----------|",
        ]
        for der in ders:
            lines.append(
                f"| {der.get('asset_name', '-')} "
                f"| {der.get('type', '-')} "
                f"| {self._fmt(der.get('capacity_kw', 0), 1)} "
                f"| {der.get('dispatch_time', '-')} "
                f"| {der.get('duration', '-')} "
                f"| {der.get('soc_status', '-')} |"
            )
        total_der_kw = sum(d.get("capacity_kw", 0) for d in ders)
        lines.append(
            f"| **TOTAL** | | **{self._fmt(total_der_kw, 1)}** | | | |"
        )
        return "\n".join(lines)

    def _md_communication_protocol(self, data: Dict[str, Any]) -> str:
        """Render communication protocol section."""
        protocol = data.get("communication_protocol", [])
        if not protocol:
            protocol = [
                {"timing": "T-60 min", "action": "Notify operations team", "channel": "Email + SMS"},
                {"timing": "T-30 min", "action": "Confirm load readiness", "channel": "BMS dashboard"},
                {"timing": "T-0", "action": "Execute curtailment sequence", "channel": "Automated"},
                {"timing": "T+end", "action": "Begin load restoration", "channel": "Automated"},
            ]
        lines = [
            "## 5. Communication Protocol\n",
            "| Timing | Action | Channel | Contact |",
            "|--------|--------|---------|---------|",
        ]
        for item in protocol:
            lines.append(
                f"| {item.get('timing', '-')} "
                f"| {item.get('action', '-')} "
                f"| {item.get('channel', '-')} "
                f"| {item.get('contact', '-')} |"
            )
        return "\n".join(lines)

    def _md_contingency_actions(self, data: Dict[str, Any]) -> str:
        """Render contingency actions section."""
        actions = data.get("contingency_actions", [])
        if not actions:
            actions = [
                {"trigger": "Primary load fails to curtail", "action": "Activate backup load sequence",
                 "time_limit": "5 min"},
                {"trigger": "Battery SOC below threshold", "action": "Shift to grid import reduction only",
                 "time_limit": "Immediate"},
            ]
        lines = [
            "## 6. Contingency Actions\n",
            "| Trigger Condition | Contingency Action | Time Limit |",
            "|-------------------|-------------------|-----------|",
        ]
        for a in actions:
            lines.append(
                f"| {a.get('trigger', '-')} "
                f"| {a.get('action', '-')} "
                f"| {a.get('time_limit', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-037 Demand Response Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>DR Event Dispatch Plan</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'Event: {data.get("event_id", "-")} | '
            f'Program: {data.get("program_name", "-")} | '
            f'Window: {data.get("event_start", "-")} - {data.get("event_end", "-")}</p>'
        )

    def _html_event_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML event summary cards."""
        s = data.get("event_summary", {})
        return (
            '<h2>Event Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Required</span>'
            f'<span class="value">{self._fmt(s.get("required_curtailment_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Planned</span>'
            f'<span class="value">{self._fmt(s.get("planned_curtailment_kw", 0), 0)} kW</span></div>\n'
            f'  <div class="card"><span class="label">Duration</span>'
            f'<span class="value">{s.get("duration_hours", 0)} hrs</span></div>\n'
            f'  <div class="card"><span class="label">Loads</span>'
            f'<span class="value">{s.get("loads_involved", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Est. Revenue</span>'
            f'<span class="value">{self._format_currency(s.get("estimated_revenue", 0))}</span></div>\n'
            '</div>'
        )

    def _html_pre_conditioning(self, data: Dict[str, Any]) -> str:
        """Render HTML pre-conditioning table."""
        steps = data.get("pre_conditioning", [])
        rows = ""
        for step in steps:
            rows += (
                f'<tr><td>{step.get("time_offset", "-")}</td>'
                f'<td>{step.get("action", "-")}</td>'
                f'<td>{step.get("system", "-")}</td>'
                f'<td>{self._fmt(step.get("kw_impact", 0), 1)}</td></tr>\n'
            )
        return (
            '<h2>Pre-Conditioning</h2>\n'
            '<table>\n<tr><th>Time</th><th>Action</th>'
            f'<th>System</th><th>kW Impact</th></tr>\n{rows}</table>'
        )

    def _html_curtailment_sequence(self, data: Dict[str, Any]) -> str:
        """Render HTML curtailment sequence table."""
        sequence = data.get("curtailment_sequence", [])
        rows = ""
        cumulative = 0
        for step in sequence:
            reduction = step.get("reduction_kw", 0)
            cumulative += reduction
            rows += (
                f'<tr><td>{step.get("time", "-")}</td>'
                f'<td>{step.get("load_name", "-")}</td>'
                f'<td>{step.get("action", "-")}</td>'
                f'<td>{self._fmt(reduction, 1)}</td>'
                f'<td>{self._fmt(cumulative, 1)}</td></tr>\n'
            )
        return (
            '<h2>Curtailment Sequence</h2>\n'
            '<table>\n<tr><th>Time</th><th>Load</th>'
            f'<th>Action</th><th>Reduction kW</th><th>Cumulative kW</th></tr>\n{rows}</table>'
        )

    def _html_der_dispatch(self, data: Dict[str, Any]) -> str:
        """Render HTML DER dispatch table."""
        ders = data.get("der_dispatch", [])
        rows = ""
        for der in ders:
            rows += (
                f'<tr><td>{der.get("asset_name", "-")}</td>'
                f'<td>{der.get("type", "-")}</td>'
                f'<td>{self._fmt(der.get("capacity_kw", 0), 1)}</td>'
                f'<td>{der.get("dispatch_time", "-")}</td>'
                f'<td>{der.get("soc_status", "-")}</td></tr>\n'
            )
        return (
            '<h2>DER Dispatch Plan</h2>\n'
            '<table>\n<tr><th>Asset</th><th>Type</th>'
            f'<th>Capacity kW</th><th>Dispatch Time</th><th>SOC/Status</th></tr>\n{rows}</table>'
        )

    def _html_communication_protocol(self, data: Dict[str, Any]) -> str:
        """Render HTML communication protocol."""
        protocol = data.get("communication_protocol", [])
        rows = ""
        for item in protocol:
            rows += (
                f'<tr><td>{item.get("timing", "-")}</td>'
                f'<td>{item.get("action", "-")}</td>'
                f'<td>{item.get("channel", "-")}</td></tr>\n'
            )
        return (
            '<h2>Communication Protocol</h2>\n'
            '<table>\n<tr><th>Timing</th><th>Action</th>'
            f'<th>Channel</th></tr>\n{rows}</table>'
        )

    def _html_contingency_actions(self, data: Dict[str, Any]) -> str:
        """Render HTML contingency actions."""
        actions = data.get("contingency_actions", [])
        rows = ""
        for a in actions:
            rows += (
                f'<tr><td>{a.get("trigger", "-")}</td>'
                f'<td>{a.get("action", "-")}</td>'
                f'<td>{a.get("time_limit", "-")}</td></tr>\n'
            )
        return (
            '<h2>Contingency Actions</h2>\n'
            '<table>\n<tr><th>Trigger</th><th>Action</th>'
            f'<th>Time Limit</th></tr>\n{rows}</table>'
        )

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_event_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON event summary."""
        s = data.get("event_summary", {})
        return {
            "event_type": s.get("event_type", ""),
            "required_curtailment_kw": s.get("required_curtailment_kw", 0),
            "planned_curtailment_kw": s.get("planned_curtailment_kw", 0),
            "safety_margin_pct": s.get("safety_margin_pct", 0),
            "duration_hours": s.get("duration_hours", 0),
            "notification_lead_minutes": s.get("notification_lead_minutes", 0),
            "loads_involved": s.get("loads_involved", 0),
            "ders_dispatched": s.get("ders_dispatched", 0),
            "estimated_revenue": s.get("estimated_revenue", 0),
        }

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        sequence = data.get("curtailment_sequence", [])
        ders = data.get("der_dispatch", [])
        cumulative = 0
        timeline_points = []
        for step in sequence:
            cumulative += step.get("reduction_kw", 0)
            timeline_points.append({
                "time": step.get("time", ""),
                "cumulative_kw": cumulative,
            })
        return {
            "curtailment_timeline": {
                "type": "line",
                "labels": [p["time"] for p in timeline_points],
                "values": [p["cumulative_kw"] for p in timeline_points],
            },
            "load_reduction_bar": {
                "type": "bar",
                "labels": [s.get("load_name", "") for s in sequence],
                "values": [s.get("reduction_kw", 0) for s in sequence],
            },
            "der_capacity_pie": {
                "type": "pie",
                "labels": [d.get("asset_name", "") for d in ders],
                "values": [d.get("capacity_kw", 0) for d in ders],
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
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;flex-wrap:wrap;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;min-width:150px;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
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

    def _format_power(self, val: Any) -> str:
        """Format a power value with units.

        Args:
            val: Power value in kW.

        Returns:
            Formatted power string.
        """
        if isinstance(val, (int, float)):
            return f"{val:,.1f} kW"
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
