# -*- coding: utf-8 -*-
"""
EquipmentEfficiencyReportTemplate - Equipment-level efficiency assessment for PACK-031.

Generates equipment-level efficiency assessment reports with motor inventory
and analysis, pump curve evaluation, compressor profiles, boiler stack
loss analysis, efficiency gap identification, and upgrade recommendations.
Follows IEC 60034-30-1 motor efficiency classes and Europump guidelines.

Sections:
    1. Executive Summary
    2. Motor Inventory & Analysis
    3. Pump System Assessment
    4. Compressor Profile Analysis
    5. Boiler Stack Loss Analysis
    6. Fan & Blower Assessment
    7. Efficiency Gap Summary
    8. Upgrade Recommendations

Author: GreenLang Team
Version: 31.0.0
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EquipmentEfficiencyReportTemplate:
    """
    Equipment-level efficiency assessment report template.

    Renders motor, pump, compressor, boiler, and fan efficiency data
    with gap analysis and upgrade recommendations across markdown,
    HTML, and JSON formats. References IEC 60034-30-1 motor efficiency
    classes (IE1 through IE5) and Europump guidelines.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    MOTOR_EFFICIENCY_CLASSES: Dict[str, str] = {
        "IE1": "Standard Efficiency",
        "IE2": "High Efficiency",
        "IE3": "Premium Efficiency",
        "IE4": "Super Premium Efficiency",
        "IE5": "Ultra Premium Efficiency",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EquipmentEfficiencyReportTemplate.

        Args:
            config: Optional configuration dict with overrides.
        """
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public render methods
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render equipment efficiency assessment report as Markdown.

        Args:
            data: Equipment engine result data.

        Returns:
            Complete Markdown string with provenance hash.
        """
        self.generated_at = datetime.utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_motor_inventory(data),
            self._md_pump_assessment(data),
            self._md_compressor_profiles(data),
            self._md_boiler_stack_loss(data),
            self._md_fan_assessment(data),
            self._md_efficiency_gaps(data),
            self._md_upgrade_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = self._provenance(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render equipment efficiency assessment report as HTML.

        Args:
            data: Equipment engine result data.

        Returns:
            Complete HTML string with inline CSS.
        """
        self.generated_at = datetime.utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_motor_inventory(data),
            self._html_pump_assessment(data),
            self._html_compressor_profiles(data),
            self._html_efficiency_gaps(data),
            self._html_upgrade_recommendations(data),
        ])
        prov = self._provenance(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Equipment Efficiency Assessment Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render equipment efficiency report as structured JSON.

        Args:
            data: Equipment engine result data.

        Returns:
            Dict with structured report sections and provenance hash.
        """
        self.generated_at = datetime.utcnow()
        result: Dict[str, Any] = {
            "template": "equipment_efficiency_report",
            "version": "31.0.0",
            "standards": ["IEC 60034-30-1", "Europump", "EN 16247-3"],
            "generated_at": self.generated_at.isoformat(),
            "executive_summary": self._json_executive_summary(data),
            "motor_inventory": data.get("motor_inventory", {}),
            "pump_assessment": data.get("pump_assessment", {}),
            "compressor_profiles": data.get("compressor_profiles", {}),
            "boiler_stack_loss": data.get("boiler_stack_loss", {}),
            "fan_assessment": data.get("fan_assessment", {}),
            "efficiency_gaps": data.get("efficiency_gaps", []),
            "upgrade_recommendations": data.get("upgrade_recommendations", []),
            "charts": self._json_charts(data),
        }
        prov = self._provenance(json.dumps(result, default=str))
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Markdown section renderers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        facility = data.get("facility_name", "Industrial Facility")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Equipment Efficiency Assessment Report\n\n"
            f"**Facility:** {facility}  \n"
            f"**Standards:** IEC 60034-30-1, Europump, EN 16247-3  \n"
            f"**Assessment Date:** {data.get('assessment_date', '-')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Template:** PACK-031 EquipmentEfficiencyReportTemplate v31.0.0\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary."""
        s = data.get("executive_summary", {})
        return (
            "## 1. Executive Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Total Equipment Assessed | {s.get('total_equipment', 0)} |\n"
            f"| Total Installed Capacity | {self._fmt(s.get('total_capacity_kw', 0))} kW |\n"
            f"| Equipment Energy Use | {self._fmt(s.get('equipment_energy_mwh', 0))} MWh/yr |\n"
            f"| Average Operating Efficiency | {self._fmt(s.get('avg_efficiency_pct', 0))}% |\n"
            f"| Efficiency Gap (vs Best Practice) | {self._fmt(s.get('efficiency_gap_pct', 0))}% |\n"
            f"| Savings Potential | {self._fmt(s.get('savings_potential_mwh', 0))} MWh/yr |\n"
            f"| Cost Savings | EUR {self._fmt(s.get('cost_savings_eur', 0))} /yr |\n"
            f"| Required Investment | EUR {self._fmt(s.get('investment_eur', 0))} |\n"
            f"| Motors Below IE3 | {s.get('motors_below_ie3', 0)} "
            f"of {s.get('total_motors', 0)} |"
        )

    def _md_motor_inventory(self, data: Dict[str, Any]) -> str:
        """Render motor inventory and analysis section."""
        motors = data.get("motor_inventory", {})
        motor_list = motors.get("motors", [])
        summary = motors.get("summary", {})
        lines = [
            "## 2. Motor Inventory & Analysis\n",
            f"**Total Motors:** {summary.get('total_motors', 0)}  ",
            f"**Total Motor Capacity:** {self._fmt(summary.get('total_capacity_kw', 0))} kW  ",
            f"**IE Class Distribution:** "
            f"IE1={summary.get('ie1_count', 0)}, "
            f"IE2={summary.get('ie2_count', 0)}, "
            f"IE3={summary.get('ie3_count', 0)}, "
            f"IE4+={summary.get('ie4_plus_count', 0)}  ",
            f"**Average Age:** {self._fmt(summary.get('avg_age_years', 0), 1)} years  ",
            f"**Upgrade Candidates:** {summary.get('upgrade_candidates', 0)}",
        ]
        if motor_list:
            lines.extend([
                "\n### Motor Details\n",
                "| # | ID | Application | Power (kW) | IE Class | "
                "Load (%) | Eff. (%) | Age (yr) | Action |",
                "|---|-----|-----------|-----------|---------|"
                "---------|---------|---------|--------|",
            ])
            for i, m in enumerate(motor_list[:30], 1):
                lines.append(
                    f"| {i} | {m.get('motor_id', '-')} "
                    f"| {m.get('application', '-')} "
                    f"| {self._fmt(m.get('rated_power_kw', 0))} "
                    f"| {m.get('ie_class', '-')} "
                    f"| {self._fmt(m.get('avg_load_pct', 0))}% "
                    f"| {self._fmt(m.get('measured_efficiency_pct', 0), 1)}% "
                    f"| {m.get('age_years', '-')} "
                    f"| {m.get('recommended_action', '-')} |"
                )
        return "\n".join(lines)

    def _md_pump_assessment(self, data: Dict[str, Any]) -> str:
        """Render pump system assessment section."""
        pumps = data.get("pump_assessment", {})
        pump_list = pumps.get("pumps", [])
        lines = [
            "## 3. Pump System Assessment\n",
            f"**Total Pumps Assessed:** {pumps.get('total_pumps', 0)}  ",
            f"**Total Pump Energy:** {self._fmt(pumps.get('total_energy_mwh', 0))} MWh/yr  ",
            f"**Avg System Efficiency:** {self._fmt(pumps.get('avg_system_efficiency_pct', 0))}%  ",
            f"**Oversized Pumps:** {pumps.get('oversized_count', 0)}",
        ]
        if pump_list:
            lines.extend([
                "\n### Pump Analysis\n",
                "| Pump | Type | Power (kW) | BEP (%) | Operating (%) | "
                "System Eff. (%) | VSD | Savings (MWh) |",
                "|------|------|-----------|---------|-------------|"
                "---------------|-----|-------------|",
            ])
            for p in pump_list:
                lines.append(
                    f"| {p.get('pump_id', '-')} "
                    f"| {p.get('type', '-')} "
                    f"| {self._fmt(p.get('power_kw', 0))} "
                    f"| {self._fmt(p.get('bep_pct', 0))}% "
                    f"| {self._fmt(p.get('operating_point_pct', 0))}% "
                    f"| {self._fmt(p.get('system_efficiency_pct', 0))}% "
                    f"| {'Yes' if p.get('has_vsd') else 'No'} "
                    f"| {self._fmt(p.get('savings_mwh', 0))} |"
                )
        return "\n".join(lines)

    def _md_compressor_profiles(self, data: Dict[str, Any]) -> str:
        """Render compressor profile analysis section."""
        comp = data.get("compressor_profiles", {})
        compressors = comp.get("compressors", [])
        lines = [
            "## 4. Compressor Profile Analysis\n",
            f"**Total Compressors:** {comp.get('total_compressors', 0)}  ",
            f"**Total Compressor Energy:** {self._fmt(comp.get('total_energy_mwh', 0))} MWh/yr  ",
            f"**Avg Isentropic Efficiency:** {self._fmt(comp.get('avg_isentropic_eff_pct', 0))}%",
        ]
        if compressors:
            lines.extend([
                "\n### Compressor Efficiency\n",
                "| Compressor | Type | Power (kW) | Isentropic Eff. (%) | "
                "Load Profile | Savings (MWh) |",
                "|-----------|------|-----------|-------------------|"
                "------------|-------------|",
            ])
            for c in compressors:
                lines.append(
                    f"| {c.get('compressor_id', '-')} "
                    f"| {c.get('type', '-')} "
                    f"| {self._fmt(c.get('power_kw', 0))} "
                    f"| {self._fmt(c.get('isentropic_efficiency_pct', 0))}% "
                    f"| {c.get('load_profile', '-')} "
                    f"| {self._fmt(c.get('savings_mwh', 0))} |"
                )
        return "\n".join(lines)

    def _md_boiler_stack_loss(self, data: Dict[str, Any]) -> str:
        """Render boiler stack loss analysis section."""
        stack = data.get("boiler_stack_loss", {})
        boilers = stack.get("boilers", [])
        lines = [
            "## 5. Boiler Stack Loss Analysis\n",
            f"**Boilers Assessed:** {stack.get('total_boilers', 0)}  ",
            f"**Average Stack Loss:** {self._fmt(stack.get('avg_stack_loss_pct', 0))}%  ",
            f"**Improvement Potential:** {self._fmt(stack.get('improvement_mwh', 0))} MWh/yr",
        ]
        if boilers:
            lines.extend([
                "\n| Boiler | Fuel | Stack Temp (C) | O2 (%) | Stack Loss (%) | "
                "Radiation Loss (%) | Efficiency (%) |",
                "|--------|------|--------------|--------|---------------|"
                "------------------|---------------|",
            ])
            for b in boilers:
                lines.append(
                    f"| {b.get('boiler_id', '-')} "
                    f"| {b.get('fuel_type', '-')} "
                    f"| {self._fmt(b.get('stack_temp_c', 0), 0)} "
                    f"| {self._fmt(b.get('o2_pct', 0), 1)} "
                    f"| {self._fmt(b.get('stack_loss_pct', 0), 1)} "
                    f"| {self._fmt(b.get('radiation_loss_pct', 0), 1)} "
                    f"| {self._fmt(b.get('efficiency_pct', 0), 1)} |"
                )
        return "\n".join(lines)

    def _md_fan_assessment(self, data: Dict[str, Any]) -> str:
        """Render fan and blower assessment section."""
        fans = data.get("fan_assessment", {})
        fan_list = fans.get("fans", [])
        lines = [
            "## 6. Fan & Blower Assessment\n",
            f"**Total Fans Assessed:** {fans.get('total_fans', 0)}  ",
            f"**Total Fan Energy:** {self._fmt(fans.get('total_energy_mwh', 0))} MWh/yr  ",
            f"**Avg System Efficiency:** {self._fmt(fans.get('avg_efficiency_pct', 0))}%",
        ]
        if fan_list:
            lines.extend([
                "\n| Fan | Application | Power (kW) | Flow (%) | Eff. (%) | "
                "Control Method | Savings (MWh) |",
                "|------|-----------|-----------|---------|---------|"
                "--------------|-------------|",
            ])
            for f in fan_list:
                lines.append(
                    f"| {f.get('fan_id', '-')} "
                    f"| {f.get('application', '-')} "
                    f"| {self._fmt(f.get('power_kw', 0))} "
                    f"| {self._fmt(f.get('flow_pct_design', 0))}% "
                    f"| {self._fmt(f.get('efficiency_pct', 0))}% "
                    f"| {f.get('control_method', '-')} "
                    f"| {self._fmt(f.get('savings_mwh', 0))} |"
                )
        return "\n".join(lines)

    def _md_efficiency_gaps(self, data: Dict[str, Any]) -> str:
        """Render efficiency gap summary."""
        gaps = data.get("efficiency_gaps", [])
        if not gaps:
            return "## 7. Efficiency Gap Summary\n\n_No significant gaps identified._"
        lines = [
            "## 7. Efficiency Gap Summary\n",
            "| Equipment | Current Eff. (%) | Best Practice (%) | Gap (%) | "
            "Energy Impact (MWh/yr) | Priority |",
            "|-----------|-----------------|-------------------|---------|"
            "----------------------|----------|",
        ]
        for g in gaps:
            lines.append(
                f"| {g.get('equipment', '-')} "
                f"| {self._fmt(g.get('current_efficiency_pct', 0))}% "
                f"| {self._fmt(g.get('best_practice_pct', 0))}% "
                f"| {self._fmt(g.get('gap_pct', 0))}% "
                f"| {self._fmt(g.get('energy_impact_mwh', 0))} "
                f"| {g.get('priority', '-')} |"
            )
        return "\n".join(lines)

    def _md_upgrade_recommendations(self, data: Dict[str, Any]) -> str:
        """Render upgrade recommendations."""
        recs = data.get("upgrade_recommendations", [])
        if not recs:
            return "## 8. Upgrade Recommendations\n\n_No upgrade recommendations._"
        lines = [
            "## 8. Upgrade Recommendations\n",
            "| # | Equipment | Action | Savings (MWh/yr) | Cost Savings (EUR/yr) | "
            "Investment (EUR) | Payback (yr) |",
            "|---|-----------|--------|-----------------|---------------------|"
            "-----------------|-------------|",
        ]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"| {i} | {r.get('equipment', '-')} "
                f"| {r.get('action', '-')} "
                f"| {self._fmt(r.get('energy_savings_mwh', 0))} "
                f"| {self._fmt(r.get('cost_savings_eur', 0))} "
                f"| {self._fmt(r.get('investment_eur', 0))} "
                f"| {self._fmt(r.get('payback_years', 0), 1)} |"
            )
        total_savings = sum(r.get("energy_savings_mwh", 0) for r in recs)
        total_cost = sum(r.get("cost_savings_eur", 0) for r in recs)
        total_inv = sum(r.get("investment_eur", 0) for r in recs)
        lines.append(
            f"| | **TOTAL** | | **{self._fmt(total_savings)}** "
            f"| **{self._fmt(total_cost)}** "
            f"| **{self._fmt(total_inv)}** | |"
        )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render report footer."""
        return "---\n\n*Generated by GreenLang PACK-031 Industrial Energy Audit Pack*"

    # ------------------------------------------------------------------
    # HTML section renderers
    # ------------------------------------------------------------------

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        facility = data.get("facility_name", "Industrial Facility")
        return (
            f'<h1>Equipment Efficiency Assessment Report</h1>\n'
            f'<p class="subtitle">Facility: {facility} | IEC 60034-30-1</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render HTML executive summary cards."""
        s = data.get("executive_summary", {})
        return (
            '<h2>Executive Summary</h2>\n'
            '<div class="summary-cards">\n'
            f'  <div class="card"><span class="label">Equipment</span>'
            f'<span class="value">{s.get("total_equipment", 0)}</span></div>\n'
            f'  <div class="card"><span class="label">Avg Efficiency</span>'
            f'<span class="value">{self._fmt(s.get("avg_efficiency_pct", 0))}%</span></div>\n'
            f'  <div class="card"><span class="label">Savings</span>'
            f'<span class="value">{self._fmt(s.get("savings_potential_mwh", 0))} MWh</span></div>\n'
            f'  <div class="card"><span class="label">Below IE3</span>'
            f'<span class="value">{s.get("motors_below_ie3", 0)}</span></div>\n'
            '</div>'
        )

    def _html_motor_inventory(self, data: Dict[str, Any]) -> str:
        """Render HTML motor inventory."""
        motors = data.get("motor_inventory", {}).get("motors", [])
        rows = ""
        for m in motors[:20]:
            rows += (
                f'<tr><td>{m.get("motor_id", "-")}</td>'
                f'<td>{self._fmt(m.get("rated_power_kw", 0))} kW</td>'
                f'<td>{m.get("ie_class", "-")}</td>'
                f'<td>{self._fmt(m.get("measured_efficiency_pct", 0), 1)}%</td>'
                f'<td>{m.get("recommended_action", "-")}</td></tr>\n'
            )
        return (
            '<h2>Motor Inventory</h2>\n<table>\n'
            '<tr><th>ID</th><th>Power</th><th>IE Class</th>'
            f'<th>Efficiency</th><th>Action</th></tr>\n{rows}</table>'
        )

    def _html_pump_assessment(self, data: Dict[str, Any]) -> str:
        """Render HTML pump assessment."""
        pumps = data.get("pump_assessment", {})
        return (
            '<h2>Pump Assessment</h2>\n'
            f'<p>Total: {pumps.get("total_pumps", 0)} | '
            f'Avg Efficiency: {self._fmt(pumps.get("avg_system_efficiency_pct", 0))}% | '
            f'Oversized: {pumps.get("oversized_count", 0)}</p>'
        )

    def _html_compressor_profiles(self, data: Dict[str, Any]) -> str:
        """Render HTML compressor profiles."""
        comp = data.get("compressor_profiles", {})
        return (
            '<h2>Compressor Profiles</h2>\n'
            f'<p>Total: {comp.get("total_compressors", 0)} | '
            f'Avg Isentropic Eff: {self._fmt(comp.get("avg_isentropic_eff_pct", 0))}%</p>'
        )

    def _html_efficiency_gaps(self, data: Dict[str, Any]) -> str:
        """Render HTML efficiency gaps."""
        gaps = data.get("efficiency_gaps", [])
        rows = ""
        for g in gaps:
            color = "#dc2626" if g.get("gap_pct", 0) > 15 else "#d97706"
            rows += (
                f'<tr><td>{g.get("equipment", "-")}</td>'
                f'<td>{self._fmt(g.get("current_efficiency_pct", 0))}%</td>'
                f'<td>{self._fmt(g.get("best_practice_pct", 0))}%</td>'
                f'<td style="color:{color};">{self._fmt(g.get("gap_pct", 0))}%</td>'
                f'<td>{g.get("priority", "-")}</td></tr>\n'
            )
        return (
            '<h2>Efficiency Gaps</h2>\n<table>\n'
            '<tr><th>Equipment</th><th>Current</th><th>Best Practice</th>'
            f'<th>Gap</th><th>Priority</th></tr>\n{rows}</table>'
        )

    def _html_upgrade_recommendations(self, data: Dict[str, Any]) -> str:
        """Render HTML upgrade recommendations."""
        recs = data.get("upgrade_recommendations", [])
        items = "".join(
            f'<li><strong>{r.get("equipment", "-")}</strong>: {r.get("action", "-")} - '
            f'Savings: {self._fmt(r.get("energy_savings_mwh", 0))} MWh/yr, '
            f'Payback: {self._fmt(r.get("payback_years", 0), 1)} yr</li>\n'
            for r in recs
        )
        return f'<h2>Upgrade Recommendations</h2>\n<ol>\n{items}</ol>'

    # ------------------------------------------------------------------
    # JSON builders
    # ------------------------------------------------------------------

    def _json_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build JSON executive summary."""
        return data.get("executive_summary", {})

    def _json_charts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build chart data payloads for visualization."""
        motors = data.get("motor_inventory", {}).get("motors", [])
        gaps = data.get("efficiency_gaps", [])
        ie_summary = data.get("motor_inventory", {}).get("summary", {})
        return {
            "motor_ie_distribution": {
                "type": "pie",
                "labels": ["IE1", "IE2", "IE3", "IE4+"],
                "values": [
                    ie_summary.get("ie1_count", 0),
                    ie_summary.get("ie2_count", 0),
                    ie_summary.get("ie3_count", 0),
                    ie_summary.get("ie4_plus_count", 0),
                ],
            },
            "motor_load_histogram": {
                "type": "histogram",
                "values": [m.get("avg_load_pct", 0) for m in motors],
                "bins": [0, 20, 40, 60, 80, 100],
            },
            "efficiency_gap_bar": {
                "type": "horizontal_bar",
                "labels": [g.get("equipment", "") for g in gaps],
                "series": {
                    "current": [g.get("current_efficiency_pct", 0) for g in gaps],
                    "best_practice": [g.get("best_practice_pct", 0) for g in gaps],
                },
            },
            "savings_by_category": {
                "type": "pie",
                "labels": ["Motors", "Pumps", "Compressors", "Boilers", "Fans"],
                "values": [
                    data.get("motor_inventory", {}).get("summary", {}).get("savings_mwh", 0),
                    data.get("pump_assessment", {}).get("total_savings_mwh", 0),
                    data.get("compressor_profiles", {}).get("total_savings_mwh", 0),
                    data.get("boiler_stack_loss", {}).get("improvement_mwh", 0),
                    data.get("fan_assessment", {}).get("total_savings_mwh", 0),
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
            "table{width:100%;border-collapse:collapse;margin:15px 0;}"
            "th,td{border:1px solid #dee2e6;padding:8px 12px;text-align:left;}"
            "th{background:#f8f9fa;font-weight:600;}"
            "tr:nth-child(even){background:#f9fafb;}"
            ".summary-cards{display:flex;gap:15px;margin:15px 0;}"
            ".card{background:#f8f9fa;border-radius:8px;padding:15px;flex:1;text-align:center;}"
            ".label{display:block;font-size:0.85em;color:#6c757d;}"
            ".value{display:block;font-size:1.4em;font-weight:700;color:#198754;}"
            ".subtitle{color:#6c757d;font-size:0.95em;}"
        )

    def _fmt(self, val: Any, decimals: int = 2) -> str:
        """Format a numeric value with comma separators."""
        if isinstance(val, (int, float)):
            return f"{val:,.{decimals}f}"
        return str(val)

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
