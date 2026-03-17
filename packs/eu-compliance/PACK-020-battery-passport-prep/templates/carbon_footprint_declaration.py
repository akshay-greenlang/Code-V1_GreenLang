# -*- coding: utf-8 -*-
"""
CarbonFootprintDeclarationTemplate - EU Battery Regulation Art 7 Carbon Footprint Declaration

Renders the mandatory carbon footprint declaration for industrial and EV batteries
per Article 7 of Regulation (EU) 2023/1542. Covers lifecycle carbon footprint
calculation, performance class assignment, methodology references (Commission
Delegated Regulation (EU) 2023/1791 and EN 17615 / ISO 14067), and threshold
compliance verification against the maximum lifecycle carbon footprint thresholds
that apply from 18 February 2025 (declaration) and 18 August 2028 (limits).

Sections:
    1. Executive Summary - Declaration overview and compliance verdict
    2. Lifecycle Breakdown - Per-stage emissions (raw material, manufacturing,
       distribution, end-of-life, recycled content credit)
    3. Performance Class - Labelled class (A-E) per Delegated Act thresholds
    4. Methodology Reference - Standards, data sources, allocation rules
    5. Threshold Compliance - Current vs. maximum allowed lifecycle footprint

Author: GreenLang Team
Version: 20.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

_SECTIONS: List[str] = [
    "executive_summary",
    "lifecycle_breakdown",
    "performance_class",
    "methodology_reference",
    "threshold_compliance",
]

# Lifecycle stages per Commission Delegated Regulation (EU) 2023/1791
_LIFECYCLE_STAGES: List[str] = [
    "raw_material_acquisition",
    "main_production",
    "distribution",
    "end_of_life_recycling",
]

# Performance class thresholds (kgCO2e/kWh) - illustrative ranges
_PERFORMANCE_CLASSES: List[Dict[str, Any]] = [
    {"class": "A", "max_kgco2e_per_kwh": 60.0, "label": "Best-in-class"},
    {"class": "B", "max_kgco2e_per_kwh": 80.0, "label": "Above average"},
    {"class": "C", "max_kgco2e_per_kwh": 100.0, "label": "Average"},
    {"class": "D", "max_kgco2e_per_kwh": 120.0, "label": "Below average"},
    {"class": "E", "max_kgco2e_per_kwh": None, "label": "Lowest performing"},
]


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CarbonFootprintDeclarationTemplate:
    """
    Carbon Footprint Declaration template per EU Battery Regulation Art 7.

    Generates the mandatory carbon footprint declaration for industrial,
    light means of transport (LMT), and electric vehicle (EV) batteries.
    The declaration covers the entire lifecycle carbon footprint expressed
    in kgCO2e per kWh of total energy provided over the expected service
    life, per-stage breakdowns, performance class label, and compliance
    against the maximum lifecycle carbon footprint thresholds.

    Regulatory References:
        - Regulation (EU) 2023/1542, Article 7
        - Commission Delegated Regulation (EU) 2023/1791
        - EN 17615 (Carbon footprint of batteries)
        - ISO 14067 (Carbon footprint of products)

    Example:
        >>> tpl = CarbonFootprintDeclarationTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize CarbonFootprintDeclarationTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "report_id": _new_uuid(),
            "generated_at": self.generated_at.isoformat(),
        }
        for section in _SECTIONS:
            result[section] = self.render_section(section, data)
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_section(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render a single section by name."""
        handler = getattr(self, f"_section_{name}", None)
        if handler is None:
            raise ValueError(f"Unknown section: {name}")
        return handler(data)

    def get_sections(self) -> List[str]:
        """Return list of available section names."""
        return list(_SECTIONS)

    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data and return errors/warnings."""
        errors: List[str] = []
        warnings: List[str] = []
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if not data.get("battery_model"):
            errors.append("battery_model is required")
        if "total_carbon_footprint_kgco2e_per_kwh" not in data:
            errors.append("total_carbon_footprint_kgco2e_per_kwh is required")
        if not data.get("battery_type"):
            warnings.append("battery_type not specified; will default to 'ev_battery'")
        if "lifecycle_stages" not in data:
            warnings.append("lifecycle_stages missing; per-stage breakdown unavailable")
        if not data.get("rated_capacity_kwh"):
            warnings.append("rated_capacity_kwh missing; some calculations limited")
        if not data.get("methodology"):
            warnings.append("methodology missing; will use default references")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render carbon footprint declaration as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_lifecycle_breakdown(data),
            self._md_performance_class(data),
            self._md_methodology(data),
            self._md_threshold_compliance(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render carbon footprint declaration as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_lifecycle_breakdown(data),
            self._html_performance_class(data),
            self._html_methodology(data),
            self._html_threshold_compliance(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Carbon Footprint Declaration - Art 7</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render carbon footprint declaration as JSON."""
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {
            "template": "carbon_footprint_declaration",
            "regulation_reference": "EU Battery Regulation 2023/1542, Art 7",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "entity_name": data.get("entity_name", ""),
            "battery_model": data.get("battery_model", ""),
            "battery_type": data.get("battery_type", "ev_battery"),
            "executive_summary": self._section_executive_summary(data),
            "lifecycle_breakdown": self._section_lifecycle_breakdown(data),
            "performance_class": self._section_performance_class(data),
            "methodology_reference": self._section_methodology_reference(data),
            "threshold_compliance": self._section_threshold_compliance(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build executive summary section."""
        total_cf = data.get("total_carbon_footprint_kgco2e_per_kwh", 0.0)
        threshold = data.get("max_threshold_kgco2e_per_kwh", 0.0)
        battery_type = data.get("battery_type", "ev_battery")
        perf_class = self._determine_performance_class(total_cf)
        compliant = total_cf <= threshold if threshold > 0 else None
        return {
            "title": "Executive Summary",
            "entity_name": data.get("entity_name", ""),
            "battery_model": data.get("battery_model", ""),
            "battery_type": battery_type,
            "rated_capacity_kwh": data.get("rated_capacity_kwh", 0.0),
            "total_carbon_footprint_kgco2e_per_kwh": round(total_cf, 2),
            "performance_class": perf_class,
            "max_threshold_kgco2e_per_kwh": threshold,
            "threshold_compliant": compliant,
            "declaration_date": data.get(
                "declaration_date", _utcnow().strftime("%Y-%m-%d")
            ),
            "compliance_verdict": self._verdict(compliant),
        }

    def _section_lifecycle_breakdown(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build lifecycle breakdown section with per-stage emissions."""
        stages = data.get("lifecycle_stages", {})
        total_cf = data.get("total_carbon_footprint_kgco2e_per_kwh", 0.0)
        recycled_credit = data.get("recycled_content_credit_kgco2e_per_kwh", 0.0)
        stage_details: List[Dict[str, Any]] = []
        gross_total = 0.0

        for stage_key in _LIFECYCLE_STAGES:
            stage_data = stages.get(stage_key, {})
            stage_value = stage_data.get("kgco2e_per_kwh", 0.0)
            gross_total += stage_value
            pct = round(stage_value / total_cf * 100, 1) if total_cf > 0 else 0.0
            stage_details.append({
                "stage": stage_key,
                "stage_label": self._stage_label(stage_key),
                "kgco2e_per_kwh": round(stage_value, 2),
                "percentage_of_total": pct,
                "data_quality": stage_data.get("data_quality", "measured"),
                "notes": stage_data.get("notes", ""),
            })

        return {
            "title": "Lifecycle Carbon Footprint Breakdown",
            "stages": stage_details,
            "gross_total_kgco2e_per_kwh": round(gross_total, 2),
            "recycled_content_credit_kgco2e_per_kwh": round(recycled_credit, 2),
            "net_total_kgco2e_per_kwh": round(total_cf, 2),
            "functional_unit": "kgCO2e per kWh of total energy provided",
        }

    def _section_performance_class(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build performance class section."""
        total_cf = data.get("total_carbon_footprint_kgco2e_per_kwh", 0.0)
        perf_class = self._determine_performance_class(total_cf)
        class_info = next(
            (c for c in _PERFORMANCE_CLASSES if c["class"] == perf_class),
            _PERFORMANCE_CLASSES[-1],
        )
        return {
            "title": "Carbon Footprint Performance Class",
            "assigned_class": perf_class,
            "class_label": class_info["label"],
            "carbon_footprint_kgco2e_per_kwh": round(total_cf, 2),
            "class_thresholds": [
                {
                    "class": c["class"],
                    "max_kgco2e_per_kwh": c["max_kgco2e_per_kwh"],
                    "label": c["label"],
                    "is_assigned": c["class"] == perf_class,
                }
                for c in _PERFORMANCE_CLASSES
            ],
            "labelling_required": True,
            "label_must_include_class": True,
        }

    def _section_methodology_reference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build methodology reference section."""
        methodology = data.get("methodology", {})
        return {
            "title": "Methodology Reference",
            "calculation_standard": methodology.get(
                "standard", "Commission Delegated Regulation (EU) 2023/1791"
            ),
            "lca_standard": methodology.get("lca_standard", "ISO 14067:2018"),
            "pcr_reference": methodology.get(
                "pcr_reference", "EN 17615 (Product category rules for batteries)"
            ),
            "system_boundary": methodology.get("system_boundary", "cradle-to-grave"),
            "functional_unit": methodology.get(
                "functional_unit",
                "1 kWh of total energy provided over expected service life",
            ),
            "allocation_rules": methodology.get("allocation_rules", [
                "Mass-based allocation for multi-output processes",
                "System expansion for recycling credits",
                "Economic allocation where physical not feasible",
            ]),
            "data_sources": methodology.get("data_sources", []),
            "gwp_timeframe": methodology.get("gwp_timeframe", "100-year (GWP100)"),
            "gwp_reference": methodology.get("gwp_reference", "IPCC AR6"),
            "verified_by": methodology.get("verified_by", ""),
            "verification_date": methodology.get("verification_date", ""),
            "notified_body": methodology.get("notified_body", ""),
        }

    def _section_threshold_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build threshold compliance section."""
        total_cf = data.get("total_carbon_footprint_kgco2e_per_kwh", 0.0)
        threshold = data.get("max_threshold_kgco2e_per_kwh", 0.0)
        battery_type = data.get("battery_type", "ev_battery")
        margin = round(threshold - total_cf, 2) if threshold > 0 else 0.0
        margin_pct = round(margin / threshold * 100, 1) if threshold > 0 else 0.0
        compliant = total_cf <= threshold if threshold > 0 else None

        return {
            "title": "Maximum Lifecycle Carbon Footprint Threshold Compliance",
            "battery_type": battery_type,
            "current_footprint_kgco2e_per_kwh": round(total_cf, 2),
            "max_threshold_kgco2e_per_kwh": threshold,
            "margin_kgco2e_per_kwh": margin,
            "margin_percentage": margin_pct,
            "is_compliant": compliant,
            "compliance_verdict": self._verdict(compliant),
            "applicable_from": self._threshold_date(battery_type),
            "regulatory_milestones": [
                {
                    "milestone": "Carbon footprint declaration mandatory",
                    "date": "2025-02-18",
                    "article": "Art 7(1)",
                },
                {
                    "milestone": "Performance class label mandatory",
                    "date": "2026-08-18",
                    "article": "Art 7(2)",
                },
                {
                    "milestone": "Maximum lifecycle CF threshold applies",
                    "date": "2028-08-18",
                    "article": "Art 7(3)",
                },
            ],
            "reduction_needed_kgco2e_per_kwh": max(0.0, round(-margin, 2)),
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Carbon Footprint Declaration\n"
            f"## EU Battery Regulation (EU) 2023/1542 - Article 7\n\n"
            f"**Manufacturer:** {data.get('entity_name', '')}  \n"
            f"**Battery Model:** {data.get('battery_model', '')}  \n"
            f"**Battery Type:** {data.get('battery_type', 'ev_battery')}  \n"
            f"**Rated Capacity:** {data.get('rated_capacity_kwh', 0.0)} kWh  \n"
            f"**Generated:** {ts}"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary as markdown."""
        sec = self._section_executive_summary(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Total Carbon Footprint | **{sec['total_carbon_footprint_kgco2e_per_kwh']:.2f} "
            f"kgCO2e/kWh** |\n"
            f"| Performance Class | **{sec['performance_class']}** |\n"
            f"| Max Threshold | {sec['max_threshold_kgco2e_per_kwh']:.2f} kgCO2e/kWh |\n"
            f"| Compliance | **{sec['compliance_verdict']}** |\n"
            f"| Declaration Date | {sec['declaration_date']} |"
        )

    def _md_lifecycle_breakdown(self, data: Dict[str, Any]) -> str:
        """Render lifecycle breakdown as markdown."""
        sec = self._section_lifecycle_breakdown(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Functional Unit:** {sec['functional_unit']}\n",
            "| Lifecycle Stage | kgCO2e/kWh | % of Total | Data Quality |",
            "|----------------|----------:|----------:|:------------|",
        ]
        for stage in sec["stages"]:
            lines.append(
                f"| {stage['stage_label']} | {stage['kgco2e_per_kwh']:.2f} | "
                f"{stage['percentage_of_total']:.1f}% | {stage['data_quality']} |"
            )
        lines.append(
            f"| **Gross Total** | **{sec['gross_total_kgco2e_per_kwh']:.2f}** | | |"
        )
        lines.append(
            f"| Recycled Content Credit | -{sec['recycled_content_credit_kgco2e_per_kwh']:.2f}"
            f" | | |"
        )
        lines.append(
            f"| **Net Total** | **{sec['net_total_kgco2e_per_kwh']:.2f}** | **100%** | |"
        )
        return "\n".join(lines)

    def _md_performance_class(self, data: Dict[str, Any]) -> str:
        """Render performance class as markdown."""
        sec = self._section_performance_class(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Assigned Class:** {sec['assigned_class']} - {sec['class_label']}  \n"
            f"**Carbon Footprint:** {sec['carbon_footprint_kgco2e_per_kwh']:.2f} kgCO2e/kWh\n",
            "| Class | Max kgCO2e/kWh | Label | Assigned |",
            "|:-----:|---------------:|-------|:--------:|",
        ]
        for c in sec["class_thresholds"]:
            max_val = f"{c['max_kgco2e_per_kwh']:.1f}" if c["max_kgco2e_per_kwh"] else "No limit"
            marker = ">>>" if c["is_assigned"] else ""
            lines.append(f"| {c['class']} | {max_val} | {c['label']} | {marker} |")
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        """Render methodology reference as markdown."""
        sec = self._section_methodology_reference(data)
        lines = [
            f"## {sec['title']}\n",
            f"- **Calculation Standard:** {sec['calculation_standard']}",
            f"- **LCA Standard:** {sec['lca_standard']}",
            f"- **PCR Reference:** {sec['pcr_reference']}",
            f"- **System Boundary:** {sec['system_boundary']}",
            f"- **Functional Unit:** {sec['functional_unit']}",
            f"- **GWP Timeframe:** {sec['gwp_timeframe']}",
            f"- **GWP Reference:** {sec['gwp_reference']}",
        ]
        if sec["allocation_rules"]:
            lines.append("\n**Allocation Rules:**")
            for rule in sec["allocation_rules"]:
                lines.append(f"  - {rule}")
        if sec["verified_by"]:
            lines.append(f"\n**Verified by:** {sec['verified_by']}")
        if sec["notified_body"]:
            lines.append(f"**Notified Body:** {sec['notified_body']}")
        return "\n".join(lines)

    def _md_threshold_compliance(self, data: Dict[str, Any]) -> str:
        """Render threshold compliance as markdown."""
        sec = self._section_threshold_compliance(data)
        lines = [
            f"## {sec['title']}\n",
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Current Footprint | {sec['current_footprint_kgco2e_per_kwh']:.2f} kgCO2e/kWh |\n"
            f"| Maximum Threshold | {sec['max_threshold_kgco2e_per_kwh']:.2f} kgCO2e/kWh |\n"
            f"| Margin | {sec['margin_kgco2e_per_kwh']:.2f} kgCO2e/kWh "
            f"({sec['margin_percentage']:.1f}%) |\n"
            f"| Compliance | **{sec['compliance_verdict']}** |\n",
        ]
        if sec["reduction_needed_kgco2e_per_kwh"] > 0:
            lines.append(
                f"**Reduction needed:** {sec['reduction_needed_kgco2e_per_kwh']:.2f} "
                f"kgCO2e/kWh to meet threshold.\n"
            )
        lines.append("### Regulatory Milestones\n")
        lines.append("| Milestone | Date | Article |")
        lines.append("|-----------|------|---------|")
        for ms in sec["regulatory_milestones"]:
            lines.append(f"| {ms['milestone']} | {ms['date']} | {ms['article']} |")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n*Declaration generated by PACK-020 Battery Passport Prep Pack on {ts}*\n"
            f"*Regulation: EU Battery Regulation (EU) 2023/1542, Article 7*"
        )

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:1000px;margin:auto}"
            "h1{color:#0d47a1;border-bottom:2px solid #0d47a1;padding-bottom:.3em}"
            "h2{color:#1565c0;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e3f2fd}"
            ".pass{color:#2e7d32;font-weight:bold}"
            ".fail{color:#c62828;font-weight:bold}"
            ".class-badge{display:inline-block;padding:4px 12px;border-radius:4px;"
            "font-weight:bold;font-size:1.2em;background:#e3f2fd;color:#0d47a1}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Carbon Footprint Declaration</h1>\n"
            f"<p>EU Battery Regulation (EU) 2023/1542 - Article 7</p>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> | "
            f"{data.get('battery_model', '')} | "
            f"{data.get('battery_type', 'ev_battery')}</p>"
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        """Render executive summary HTML."""
        sec = self._section_executive_summary(data)
        css_cls = "pass" if sec.get("threshold_compliant") else "fail"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total Carbon Footprint: <strong>"
            f"{sec['total_carbon_footprint_kgco2e_per_kwh']:.2f} kgCO2e/kWh</strong></p>\n"
            f"<p>Performance Class: <span class='class-badge'>{sec['performance_class']}"
            f"</span></p>\n"
            f"<p class='{css_cls}'>Compliance: {sec['compliance_verdict']}</p>"
        )

    def _html_lifecycle_breakdown(self, data: Dict[str, Any]) -> str:
        """Render lifecycle breakdown HTML."""
        sec = self._section_lifecycle_breakdown(data)
        rows = "".join(
            f"<tr><td>{s['stage_label']}</td><td>{s['kgco2e_per_kwh']:.2f}</td>"
            f"<td>{s['percentage_of_total']:.1f}%</td><td>{s['data_quality']}</td></tr>"
            for s in sec["stages"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Stage</th><th>kgCO2e/kWh</th><th>%</th>"
            f"<th>Data Quality</th></tr>{rows}"
            f"<tr><td><strong>Net Total</strong></td>"
            f"<td><strong>{sec['net_total_kgco2e_per_kwh']:.2f}</strong></td>"
            f"<td>100%</td><td></td></tr></table>"
        )

    def _html_performance_class(self, data: Dict[str, Any]) -> str:
        """Render performance class HTML."""
        sec = self._section_performance_class(data)
        rows = ""
        for c in sec["class_thresholds"]:
            style = ' style="background:#e3f2fd;font-weight:bold"' if c["is_assigned"] else ""
            max_val = f"{c['max_kgco2e_per_kwh']:.1f}" if c["max_kgco2e_per_kwh"] else "No limit"
            rows += f"<tr{style}><td>{c['class']}</td><td>{max_val}</td><td>{c['label']}</td></tr>"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Assigned: <span class='class-badge'>{sec['assigned_class']}</span> "
            f"- {sec['class_label']}</p>\n"
            f"<table><tr><th>Class</th><th>Max kgCO2e/kWh</th><th>Label</th></tr>"
            f"{rows}</table>"
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        """Render methodology HTML."""
        sec = self._section_methodology_reference(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<ul>"
            f"<li><strong>Calculation Standard:</strong> {sec['calculation_standard']}</li>"
            f"<li><strong>LCA Standard:</strong> {sec['lca_standard']}</li>"
            f"<li><strong>System Boundary:</strong> {sec['system_boundary']}</li>"
            f"<li><strong>Functional Unit:</strong> {sec['functional_unit']}</li>"
            f"<li><strong>GWP:</strong> {sec['gwp_timeframe']} ({sec['gwp_reference']})</li>"
            f"</ul>"
        )

    def _html_threshold_compliance(self, data: Dict[str, Any]) -> str:
        """Render threshold compliance HTML."""
        sec = self._section_threshold_compliance(data)
        css_cls = "pass" if sec.get("is_compliant") else "fail"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Parameter</th><th>Value</th></tr>"
            f"<tr><td>Current Footprint</td>"
            f"<td>{sec['current_footprint_kgco2e_per_kwh']:.2f} kgCO2e/kWh</td></tr>"
            f"<tr><td>Max Threshold</td>"
            f"<td>{sec['max_threshold_kgco2e_per_kwh']:.2f} kgCO2e/kWh</td></tr>"
            f"<tr><td>Margin</td><td>{sec['margin_kgco2e_per_kwh']:.2f} kgCO2e/kWh</td></tr>"
            f"<tr><td>Compliance</td>"
            f"<td class='{css_cls}'>{sec['compliance_verdict']}</td></tr></table>"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _determine_performance_class(self, kgco2e_per_kwh: float) -> str:
        """Determine the performance class from the carbon footprint value."""
        for cls in _PERFORMANCE_CLASSES:
            if cls["max_kgco2e_per_kwh"] is not None:
                if kgco2e_per_kwh <= cls["max_kgco2e_per_kwh"]:
                    return cls["class"]
        return "E"

    def _verdict(self, compliant: Optional[bool]) -> str:
        """Return human-readable compliance verdict."""
        if compliant is None:
            return "Threshold not yet applicable"
        return "COMPLIANT" if compliant else "NON-COMPLIANT"

    def _stage_label(self, stage_key: str) -> str:
        """Convert lifecycle stage key to readable label."""
        labels = {
            "raw_material_acquisition": "Raw Material Acquisition & Pre-processing",
            "main_production": "Main Product Manufacturing",
            "distribution": "Distribution & Transport",
            "end_of_life_recycling": "End-of-Life & Recycling",
        }
        return labels.get(stage_key, stage_key.replace("_", " ").title())

    def _threshold_date(self, battery_type: str) -> str:
        """Return the date when thresholds become applicable."""
        return "2028-08-18"
