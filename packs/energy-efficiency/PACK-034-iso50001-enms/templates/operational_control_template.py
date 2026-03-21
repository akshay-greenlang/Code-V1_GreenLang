# -*- coding: utf-8 -*-
"""
OperationalControlTemplate - ISO 50001 Clause 8.1 Operational Controls for PACK-034.

Generates operational control documents aligned with ISO 50001:2018 Clause 8.1.
Covers SEU operating criteria, setpoint schedules, monitoring parameters,
deviation response procedures, maintenance schedules, procurement requirements
for energy-efficient equipment, communication of controls, and training
requirements.

Sections:
    1. SEU Operating Criteria
    2. Setpoint Schedules
    3. Monitoring Parameters
    4. Deviation Response Procedures
    5. Maintenance Schedules
    6. Procurement Requirements
    7. Communication of Controls
    8. Training Requirements

Author: GreenLang Team
Version: 34.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OperationalControlTemplate:
    """
    ISO 50001 operational controls document template.

    Renders operational control documents aligned with ISO 50001:2018
    Clause 8.1, covering SEU criteria, setpoints, monitoring, deviation
    response, maintenance, and procurement across markdown, HTML, and
    JSON formats.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize OperationalControlTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render operational controls document as Markdown.

        Args:
            data: Operational control data including seus,
                  operating_criteria, monitoring_params,
                  response_procedures, and maintenance_schedule.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_seu_operating_criteria(data),
            self._md_setpoint_schedules(data),
            self._md_monitoring_parameters(data),
            self._md_deviation_response(data),
            self._md_maintenance_schedules(data),
            self._md_procurement_requirements(data),
            self._md_communication(data),
            self._md_training_requirements(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._generate_provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render operational controls document as self-contained HTML.

        Args:
            data: Operational control data dict.

        Returns:
            Complete HTML string with inline CSS and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_seu_operating_criteria(data),
            self._html_setpoint_schedules(data),
            self._html_monitoring_parameters(data),
            self._html_deviation_response(data),
            self._html_maintenance_schedules(data),
            self._html_procurement_requirements(data),
            self._html_training_requirements(data),
        ])
        prov = self._generate_provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Operational Controls - ISO 50001</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render operational controls document as structured JSON.

        Args:
            data: Operational control data dict.

        Returns:
            Dict with structured control sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "operational_control",
            "version": "34.0.0",
            "generated_at": self.generated_at.isoformat(),
            "seus": data.get("seus", []),
            "operating_criteria": data.get("operating_criteria", {}),
            "setpoint_schedules": data.get("setpoint_schedules", []),
            "monitoring_parameters": data.get("monitoring_params", []),
            "deviation_procedures": data.get("response_procedures", []),
            "maintenance_schedule": data.get("maintenance_schedule", []),
            "procurement_requirements": data.get("procurement_requirements", []),
            "communication": data.get("communication", []),
            "training_requirements": data.get("training_requirements", []),
        }
        prov = self._generate_provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header with operational control metadata."""
        facility = data.get("facility_name", "Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Operational Controls Document\n\n"
            f"**Facility:** {facility}  \n"
            f"**Document Date:** {data.get('document_date', '')}  \n"
            f"**ISO 50001:2018 Clause:** 8.1  \n"
            f"**Version:** {data.get('document_version', '1.0')}  \n"
            f"**Report Generated:** {ts}  \n"
            f"**Template:** PACK-034 OperationalControlTemplate v34.0.0\n\n---"
        )

    def _md_seu_operating_criteria(self, data: Dict[str, Any]) -> str:
        """Render SEU operating criteria section."""
        seus = data.get("seus", [])
        criteria = data.get("operating_criteria", {})
        if not seus:
            return "## 1. SEU Operating Criteria\n\n_No SEUs defined._"
        lines = [
            "## 1. SEU Operating Criteria\n",
            "Operating criteria for each Significant Energy Use to prevent "
            "significant deviation in energy performance (ISO 50001 Clause 8.1).\n",
            "| SEU | Operating Mode | Optimal Range | Critical Limit | Control Method | Responsible |",
            "|-----|--------------|--------------|----------------|---------------|-------------|",
        ]
        for seu in seus:
            seu_criteria = criteria.get(seu.get("name", ""), {})
            if not seu_criteria and seu.get("criteria"):
                seu_criteria = seu["criteria"]
            lines.append(
                f"| {seu.get('name', '-')} "
                f"| {seu_criteria.get('operating_mode', seu.get('operating_mode', '-'))} "
                f"| {seu_criteria.get('optimal_range', seu.get('optimal_range', '-'))} "
                f"| {seu_criteria.get('critical_limit', seu.get('critical_limit', '-'))} "
                f"| {seu_criteria.get('control_method', seu.get('control_method', '-'))} "
                f"| {seu_criteria.get('responsible', seu.get('responsible', '-'))} |"
            )
        return "\n".join(lines)

    def _md_setpoint_schedules(self, data: Dict[str, Any]) -> str:
        """Render setpoint schedules section."""
        schedules = data.get("setpoint_schedules", [])
        if not schedules:
            return "## 2. Setpoint Schedules\n\n_No setpoint schedules defined._"
        lines = [
            "## 2. Setpoint Schedules\n",
            "| Equipment / System | Parameter | Occupied Setpoint | "
            "Unoccupied Setpoint | Weekend/Holiday | Season |",
            "|-------------------|-----------|------------------|"
            "-------------------|----------------|--------|",
        ]
        for s in schedules:
            lines.append(
                f"| {s.get('equipment', '-')} "
                f"| {s.get('parameter', '-')} "
                f"| {s.get('occupied_setpoint', '-')} "
                f"| {s.get('unoccupied_setpoint', '-')} "
                f"| {s.get('weekend_holiday', '-')} "
                f"| {s.get('season', '-')} |"
            )
        return "\n".join(lines)

    def _md_monitoring_parameters(self, data: Dict[str, Any]) -> str:
        """Render monitoring parameters section."""
        params = data.get("monitoring_params", [])
        if not params:
            return "## 3. Monitoring Parameters\n\n_No monitoring parameters defined._"
        lines = [
            "## 3. Monitoring Parameters\n",
            "| Parameter | SEU | Measurement Point | Frequency | "
            "Normal Range | Alert Threshold | Alarm Threshold |",
            "|-----------|-----|-------------------|-----------|"
            "------------|----------------|----------------|",
        ]
        for p in params:
            lines.append(
                f"| {p.get('parameter', '-')} "
                f"| {p.get('seu', '-')} "
                f"| {p.get('measurement_point', '-')} "
                f"| {p.get('frequency', '-')} "
                f"| {p.get('normal_range', '-')} "
                f"| {p.get('alert_threshold', '-')} "
                f"| {p.get('alarm_threshold', '-')} |"
            )
        return "\n".join(lines)

    def _md_deviation_response(self, data: Dict[str, Any]) -> str:
        """Render deviation response procedures section."""
        procedures = data.get("response_procedures", [])
        if not procedures:
            procedures = [
                {
                    "level": "Alert",
                    "trigger": "Parameter exceeds alert threshold",
                    "response": "Notify operator, log event, investigate within 24 hours",
                    "escalation": "Energy Team Lead",
                    "documentation": "Event log entry",
                },
                {
                    "level": "Alarm",
                    "trigger": "Parameter exceeds alarm threshold",
                    "response": "Immediate investigation, corrective action within 4 hours",
                    "escalation": "Energy Manager",
                    "documentation": "Incident report + corrective action",
                },
                {
                    "level": "Critical",
                    "trigger": "Parameter exceeds critical limit or equipment failure",
                    "response": "Immediate shutdown/safe mode, emergency response",
                    "escalation": "Facility Manager + Top Management",
                    "documentation": "Incident report + root cause analysis + NC/CA",
                },
            ]
        lines = [
            "## 4. Deviation Response Procedures\n",
            "| Level | Trigger | Response Action | Escalation To | Documentation |",
            "|-------|---------|----------------|--------------|---------------|",
        ]
        for p in procedures:
            lines.append(
                f"| {p.get('level', '-')} "
                f"| {p.get('trigger', '-')} "
                f"| {p.get('response', '-')} "
                f"| {p.get('escalation', '-')} "
                f"| {p.get('documentation', '-')} |"
            )
        return "\n".join(lines)

    def _md_maintenance_schedules(self, data: Dict[str, Any]) -> str:
        """Render maintenance schedules section."""
        schedule = data.get("maintenance_schedule", [])
        if not schedule:
            return "## 5. Maintenance Schedules\n\n_No maintenance schedules defined._"
        lines = [
            "## 5. Maintenance Schedules\n",
            "Preventive maintenance to ensure SEUs operate at optimal efficiency.\n",
            "| Equipment | Task | Frequency | Last Completed | Next Due | Responsible | Impact on EnPI |",
            "|-----------|------|-----------|---------------|----------|-------------|---------------|",
        ]
        for m in schedule:
            lines.append(
                f"| {m.get('equipment', '-')} "
                f"| {m.get('task', '-')} "
                f"| {m.get('frequency', '-')} "
                f"| {m.get('last_completed', '-')} "
                f"| {m.get('next_due', '-')} "
                f"| {m.get('responsible', '-')} "
                f"| {m.get('enpi_impact', '-')} |"
            )
        return "\n".join(lines)

    def _md_procurement_requirements(self, data: Dict[str, Any]) -> str:
        """Render procurement requirements section."""
        requirements = data.get("procurement_requirements", [])
        if not requirements:
            requirements = [
                {"category": "Electric Motors", "requirement": "IE3 or higher efficiency class", "standard": "IEC 60034-30-1"},
                {"category": "Lighting", "requirement": "LED only, minimum efficacy 120 lm/W", "standard": "EU Energy Label A+"},
                {"category": "HVAC Equipment", "requirement": "Minimum SEER 16 / SCOP 4.0", "standard": "EN 14825"},
                {"category": "Compressed Air", "requirement": "Variable speed drive, ISO 1217 tested", "standard": "ISO 1217"},
                {"category": "Building Envelope", "requirement": "U-value per local building code + 20% margin", "standard": "EN ISO 13790"},
            ]
        lines = [
            "## 6. Procurement Requirements\n",
            "Energy performance shall be considered when procuring energy-using "
            "products, equipment, and services that have or can have a significant "
            "impact on energy performance (ISO 50001 Clause 8.3).\n",
            "| Category | Energy Efficiency Requirement | Standard/Reference |",
            "|----------|----------------------------|-------------------|",
        ]
        for r in requirements:
            lines.append(
                f"| {r.get('category', '-')} "
                f"| {r.get('requirement', '-')} "
                f"| {r.get('standard', '-')} |"
            )
        return "\n".join(lines)

    def _md_communication(self, data: Dict[str, Any]) -> str:
        """Render communication of controls section."""
        comms = data.get("communication", [])
        if not comms:
            comms = [
                {"audience": "Operators", "content": "Operating criteria, setpoints, deviation procedures", "method": "SOPs, display boards, shift briefings"},
                {"audience": "Maintenance Team", "content": "Maintenance schedules, energy impact awareness", "method": "CMMS notifications, monthly meetings"},
                {"audience": "Procurement", "content": "Energy efficiency procurement criteria", "method": "Procurement policy, supplier briefings"},
                {"audience": "Contractors", "content": "Relevant operational controls for work scope", "method": "Contract clauses, site induction"},
            ]
        lines = [
            "## 7. Communication of Controls\n",
            "| Audience | Content | Method |",
            "|----------|---------|--------|",
        ]
        for c in comms:
            lines.append(
                f"| {c.get('audience', '-')} "
                f"| {c.get('content', '-')} "
                f"| {c.get('method', '-')} |"
            )
        return "\n".join(lines)

    def _md_training_requirements(self, data: Dict[str, Any]) -> str:
        """Render training requirements section."""
        training = data.get("training_requirements", [])
        if not training:
            training = [
                {"role": "Operators", "topic": "SEU operating procedures and setpoints", "frequency": "Annual + on change", "method": "Classroom + hands-on"},
                {"role": "Maintenance Staff", "topic": "Energy-efficient maintenance practices", "frequency": "Annual", "method": "Workshop"},
                {"role": "Energy Team", "topic": "EnMS procedures, EnPI monitoring, internal audit", "frequency": "Bi-annual", "method": "Formal training"},
                {"role": "New Employees", "topic": "Energy policy awareness, basic EnMS overview", "frequency": "On induction", "method": "Induction program"},
            ]
        lines = [
            "## 8. Training Requirements\n",
            "| Role | Training Topic | Frequency | Method |",
            "|------|---------------|-----------|--------|",
        ]
        for t in training:
            lines.append(
                f"| {t.get('role', '-')} "
                f"| {t.get('topic', '-')} "
                f"| {t.get('frequency', '-')} "
                f"| {t.get('method', '-')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-034 ISO 50001 Energy Management System Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Facility")
        return (
            f'<h1>Operational Controls Document</h1>\n'
            f'<p class="subtitle">Facility: {facility} | '
            f'ISO 50001:2018 Clause 8.1</p>'
        )

    def _html_seu_operating_criteria(self, data: Dict[str, Any]) -> str:
        """Render HTML SEU operating criteria."""
        seus = data.get("seus", [])
        criteria = data.get("operating_criteria", {})
        rows = ""
        for seu in seus:
            seu_criteria = criteria.get(seu.get("name", ""), {})
            if not seu_criteria and seu.get("criteria"):
                seu_criteria = seu["criteria"]
            rows += (
                f'<tr><td><strong>{seu.get("name", "-")}</strong></td>'
                f'<td>{seu_criteria.get("optimal_range", seu.get("optimal_range", "-"))}</td>'
                f'<td>{seu_criteria.get("critical_limit", seu.get("critical_limit", "-"))}</td>'
                f'<td>{seu_criteria.get("control_method", seu.get("control_method", "-"))}</td></tr>\n'
            )
        return (
            '<h2>1. SEU Operating Criteria</h2>\n'
            '<table>\n<tr><th>SEU</th><th>Optimal Range</th>'
            f'<th>Critical Limit</th><th>Control Method</th></tr>\n{rows}</table>'
        )

    def _html_setpoint_schedules(self, data: Dict[str, Any]) -> str:
        """Render HTML setpoint schedules."""
        schedules = data.get("setpoint_schedules", [])
        rows = ""
        for s in schedules:
            rows += (
                f'<tr><td>{s.get("equipment", "-")}</td>'
                f'<td>{s.get("parameter", "-")}</td>'
                f'<td>{s.get("occupied_setpoint", "-")}</td>'
                f'<td>{s.get("unoccupied_setpoint", "-")}</td></tr>\n'
            )
        return (
            '<h2>2. Setpoint Schedules</h2>\n'
            '<table>\n<tr><th>Equipment</th><th>Parameter</th>'
            f'<th>Occupied</th><th>Unoccupied</th></tr>\n{rows}</table>'
        )

    def _html_monitoring_parameters(self, data: Dict[str, Any]) -> str:
        """Render HTML monitoring parameters."""
        params = data.get("monitoring_params", [])
        rows = ""
        for p in params:
            rows += (
                f'<tr><td>{p.get("parameter", "-")}</td>'
                f'<td>{p.get("seu", "-")}</td>'
                f'<td>{p.get("frequency", "-")}</td>'
                f'<td>{p.get("normal_range", "-")}</td>'
                f'<td>{p.get("alert_threshold", "-")}</td></tr>\n'
            )
        return (
            '<h2>3. Monitoring Parameters</h2>\n'
            '<table>\n<tr><th>Parameter</th><th>SEU</th>'
            f'<th>Frequency</th><th>Normal Range</th><th>Alert</th></tr>\n{rows}</table>'
        )

    def _html_deviation_response(self, data: Dict[str, Any]) -> str:
        """Render HTML deviation response procedures."""
        procedures = data.get("response_procedures", [])
        rows = ""
        for p in procedures:
            level = p.get("level", "Alert").lower()
            cls = f"level-{level}"
            rows += (
                f'<tr><td class="{cls}"><strong>{p.get("level", "-")}</strong></td>'
                f'<td>{p.get("trigger", "-")}</td>'
                f'<td>{p.get("response", "-")}</td>'
                f'<td>{p.get("escalation", "-")}</td></tr>\n'
            )
        return (
            '<h2>4. Deviation Response Procedures</h2>\n'
            '<table>\n<tr><th>Level</th><th>Trigger</th>'
            f'<th>Response</th><th>Escalation</th></tr>\n{rows}</table>'
        )

    def _html_maintenance_schedules(self, data: Dict[str, Any]) -> str:
        """Render HTML maintenance schedules."""
        schedule = data.get("maintenance_schedule", [])
        rows = ""
        for m in schedule:
            rows += (
                f'<tr><td>{m.get("equipment", "-")}</td>'
                f'<td>{m.get("task", "-")}</td>'
                f'<td>{m.get("frequency", "-")}</td>'
                f'<td>{m.get("next_due", "-")}</td>'
                f'<td>{m.get("responsible", "-")}</td></tr>\n'
            )
        return (
            '<h2>5. Maintenance Schedules</h2>\n'
            '<table>\n<tr><th>Equipment</th><th>Task</th>'
            f'<th>Frequency</th><th>Next Due</th><th>Responsible</th></tr>\n{rows}</table>'
        )

    def _html_procurement_requirements(self, data: Dict[str, Any]) -> str:
        """Render HTML procurement requirements."""
        requirements = data.get("procurement_requirements", [])
        rows = ""
        for r in requirements:
            rows += (
                f'<tr><td>{r.get("category", "-")}</td>'
                f'<td>{r.get("requirement", "-")}</td>'
                f'<td>{r.get("standard", "-")}</td></tr>\n'
            )
        return (
            '<h2>6. Procurement Requirements</h2>\n'
            '<table>\n<tr><th>Category</th><th>Requirement</th>'
            f'<th>Standard</th></tr>\n{rows}</table>'
        )

    def _html_training_requirements(self, data: Dict[str, Any]) -> str:
        """Render HTML training requirements."""
        training = data.get("training_requirements", [])
        rows = ""
        for t in training:
            rows += (
                f'<tr><td>{t.get("role", "-")}</td>'
                f'<td>{t.get("topic", "-")}</td>'
                f'<td>{t.get("frequency", "-")}</td>'
                f'<td>{t.get("method", "-")}</td></tr>\n'
            )
        return (
            '<h2>8. Training Requirements</h2>\n'
            '<table>\n<tr><th>Role</th><th>Topic</th>'
            f'<th>Frequency</th><th>Method</th></tr>\n{rows}</table>'
        )

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
            ".level-alert{color:#fd7e14;font-weight:600;}"
            ".level-alarm{color:#dc3545;font-weight:700;}"
            ".level-critical{color:#dc3545;font-weight:700;background:#f8d7da;}"
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
