# -*- coding: utf-8 -*-
"""
RecycledContentReportTemplate - EU Battery Regulation Art 8 Recycled Content Report

Renders the mandatory recycled content disclosure for industrial, LMT, and EV
batteries per Article 8 of Regulation (EU) 2023/1542. Covers per-material
recycled content percentages, target tracking against mandatory minimums
(cobalt 16%, lead 85%, lithium 6%, nickel 6% from August 2031; increased to
cobalt 26%, lithium 12%, nickel 15% from August 2036), phase compliance
timelines, and actionable recommendations.

Sections:
    1. Material Inventory - Active materials and total material mass
    2. Per-Material Recycled Content - Recycled % per material vs. targets
    3. Target Tracking - Progress toward 2031 and 2036 milestones
    4. Phase Compliance - Regulatory phase status and readiness
    5. Recommendations - Actions to close recycled content gaps

Author: GreenLang Team
Version: 20.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

_SECTIONS: List[str] = [
    "material_inventory",
    "per_material_content",
    "target_tracking",
    "phase_compliance",
    "recommendations",
]

# Mandatory recycled content targets per Art 8
_RECYCLED_TARGETS_2031: Dict[str, float] = {
    "cobalt": 16.0,
    "lead": 85.0,
    "lithium": 6.0,
    "nickel": 6.0,
}

_RECYCLED_TARGETS_2036: Dict[str, float] = {
    "cobalt": 26.0,
    "lead": 85.0,
    "lithium": 12.0,
    "nickel": 15.0,
}

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

class RecycledContentReportTemplate:
    """
    Recycled Content Report template per EU Battery Regulation Art 8.

    Generates the recycled content disclosure required for industrial, LMT,
    and EV batteries. Tracks cobalt, lead, lithium, and nickel recycled
    content percentages against mandatory targets for the 2031 and 2036
    milestones. Identifies gaps and recommends sourcing actions.

    Regulatory References:
        - Regulation (EU) 2023/1542, Article 8
        - Commission Implementing Regulation on recycled content methodology
        - Annex XIII, Part B (battery passport recycled content fields)

    Example:
        >>> tpl = RecycledContentReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RecycledContentReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = utcnow()
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
        if "materials" not in data:
            errors.append("materials list is required (cobalt, lead, lithium, nickel)")
        else:
            for mat in data["materials"]:
                if "name" not in mat:
                    errors.append("Each material must have a 'name'")
                if "total_mass_kg" not in mat:
                    warnings.append(f"total_mass_kg missing for material {mat.get('name', '?')}")
        if not data.get("battery_type"):
            warnings.append("battery_type not specified; defaulting to 'ev_battery'")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render recycled content report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_material_inventory(data),
            self._md_per_material_content(data),
            self._md_target_tracking(data),
            self._md_phase_compliance(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render recycled content report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_material_inventory(data),
            self._html_per_material_content(data),
            self._html_target_tracking(data),
            self._html_phase_compliance(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Recycled Content Report - Art 8</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render recycled content report as JSON."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "recycled_content_report",
            "regulation_reference": "EU Battery Regulation 2023/1542, Art 8",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "entity_name": data.get("entity_name", ""),
            "battery_model": data.get("battery_model", ""),
            "material_inventory": self._section_material_inventory(data),
            "per_material_content": self._section_per_material_content(data),
            "target_tracking": self._section_target_tracking(data),
            "phase_compliance": self._section_phase_compliance(data),
            "recommendations": self._section_recommendations(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_material_inventory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build material inventory section."""
        materials = data.get("materials", [])
        total_battery_mass = data.get("total_battery_mass_kg", 0.0)
        inventory: List[Dict[str, Any]] = []
        total_active_mass = 0.0

        for mat in materials:
            mass = mat.get("total_mass_kg", 0.0)
            total_active_mass += mass
            pct_of_battery = (
                round(mass / total_battery_mass * 100, 2) if total_battery_mass > 0 else 0.0
            )
            inventory.append({
                "name": mat.get("name", ""),
                "total_mass_kg": round(mass, 3),
                "percentage_of_battery_mass": pct_of_battery,
                "source_countries": mat.get("source_countries", []),
                "is_regulated_material": mat.get("name", "").lower() in _RECYCLED_TARGETS_2031,
            })

        return {
            "title": "Material Inventory",
            "total_battery_mass_kg": round(total_battery_mass, 3),
            "total_active_material_mass_kg": round(total_active_mass, 3),
            "material_count": len(inventory),
            "regulated_material_count": sum(
                1 for m in inventory if m["is_regulated_material"]
            ),
            "materials": inventory,
        }

    def _section_per_material_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build per-material recycled content section."""
        materials = data.get("materials", [])
        content_details: List[Dict[str, Any]] = []

        for mat in materials:
            mat_name = mat.get("name", "").lower()
            total_mass = mat.get("total_mass_kg", 0.0)
            recycled_mass = mat.get("recycled_mass_kg", 0.0)
            recycled_pct = (
                round(recycled_mass / total_mass * 100, 2) if total_mass > 0 else 0.0
            )
            target_2031 = _RECYCLED_TARGETS_2031.get(mat_name, 0.0)
            target_2036 = _RECYCLED_TARGETS_2036.get(mat_name, 0.0)
            gap_2031 = max(0.0, round(target_2031 - recycled_pct, 2))
            gap_2036 = max(0.0, round(target_2036 - recycled_pct, 2))

            content_details.append({
                "name": mat.get("name", ""),
                "total_mass_kg": round(total_mass, 3),
                "recycled_mass_kg": round(recycled_mass, 3),
                "recycled_content_pct": recycled_pct,
                "target_2031_pct": target_2031,
                "target_2036_pct": target_2036,
                "meets_2031_target": recycled_pct >= target_2031 if target_2031 > 0 else True,
                "meets_2036_target": recycled_pct >= target_2036 if target_2036 > 0 else True,
                "gap_to_2031_pct": gap_2031,
                "gap_to_2036_pct": gap_2036,
                "pre_consumer_pct": mat.get("pre_consumer_recycled_pct", 0.0),
                "post_consumer_pct": mat.get("post_consumer_recycled_pct", 0.0),
                "documentation_available": mat.get("documentation_available", False),
            })

        return {
            "title": "Per-Material Recycled Content",
            "materials": content_details,
            "all_meet_2031": all(
                m["meets_2031_target"] for m in content_details
            ),
            "all_meet_2036": all(
                m["meets_2036_target"] for m in content_details
            ),
        }

    def _section_target_tracking(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build target tracking section for 2031 and 2036 milestones."""
        content_sec = self._section_per_material_content(data)
        materials = content_sec["materials"]

        phase1_items: List[Dict[str, Any]] = []
        phase2_items: List[Dict[str, Any]] = []

        for mat in materials:
            if mat["target_2031_pct"] > 0:
                phase1_items.append({
                    "material": mat["name"],
                    "current_pct": mat["recycled_content_pct"],
                    "target_pct": mat["target_2031_pct"],
                    "gap_pct": mat["gap_to_2031_pct"],
                    "status": "MET" if mat["meets_2031_target"] else "GAP",
                })
            if mat["target_2036_pct"] > 0:
                phase2_items.append({
                    "material": mat["name"],
                    "current_pct": mat["recycled_content_pct"],
                    "target_pct": mat["target_2036_pct"],
                    "gap_pct": mat["gap_to_2036_pct"],
                    "status": "MET" if mat["meets_2036_target"] else "GAP",
                })

        return {
            "title": "Target Tracking",
            "phase_1_deadline": "2031-08-18",
            "phase_1_targets": phase1_items,
            "phase_1_all_met": all(i["status"] == "MET" for i in phase1_items),
            "phase_2_deadline": "2036-08-18",
            "phase_2_targets": phase2_items,
            "phase_2_all_met": all(i["status"] == "MET" for i in phase2_items),
        }

    def _section_phase_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build regulatory phase compliance section."""
        tracking = self._section_target_tracking(data)
        reporting_year = data.get("reporting_year", utcnow().year)

        phases: List[Dict[str, Any]] = [
            {
                "phase": "Documentation & Declaration",
                "article": "Art 8(1)",
                "deadline": "2025-08-18",
                "description": "Recycled content documentation in technical file",
                "status": self._phase_status(reporting_year, 2025, True),
            },
            {
                "phase": "Phase 1 - Minimum Recycled Content",
                "article": "Art 8(4)(a)",
                "deadline": "2031-08-18",
                "description": "Co 16%, Pb 85%, Li 6%, Ni 6%",
                "status": self._phase_status(
                    reporting_year, 2031, tracking["phase_1_all_met"]
                ),
            },
            {
                "phase": "Phase 2 - Increased Recycled Content",
                "article": "Art 8(4)(b)",
                "deadline": "2036-08-18",
                "description": "Co 26%, Pb 85%, Li 12%, Ni 15%",
                "status": self._phase_status(
                    reporting_year, 2036, tracking["phase_2_all_met"]
                ),
            },
        ]

        return {
            "title": "Regulatory Phase Compliance",
            "reporting_year": reporting_year,
            "phases": phases,
            "overall_compliance": all(
                p["status"] in ("COMPLIANT", "NOT_YET_APPLICABLE") for p in phases
            ),
        }

    def _section_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build actionable recommendations section."""
        content_sec = self._section_per_material_content(data)
        recommendations: List[Dict[str, Any]] = []
        priority_rank = 0

        for mat in content_sec["materials"]:
            if not mat["meets_2031_target"] and mat["target_2031_pct"] > 0:
                priority_rank += 1
                recommendations.append({
                    "rank": priority_rank,
                    "priority": "HIGH",
                    "material": mat["name"],
                    "action": (
                        f"Increase {mat['name']} recycled content from "
                        f"{mat['recycled_content_pct']:.1f}% to "
                        f"{mat['target_2031_pct']:.1f}% (2031 target)"
                    ),
                    "gap_pct": mat["gap_to_2031_pct"],
                    "deadline": "2031-08-18",
                    "suggested_measures": self._suggest_measures(
                        mat["name"], mat["gap_to_2031_pct"]
                    ),
                })
            elif not mat["meets_2036_target"] and mat["target_2036_pct"] > 0:
                priority_rank += 1
                recommendations.append({
                    "rank": priority_rank,
                    "priority": "MEDIUM",
                    "material": mat["name"],
                    "action": (
                        f"Plan to increase {mat['name']} recycled content from "
                        f"{mat['recycled_content_pct']:.1f}% to "
                        f"{mat['target_2036_pct']:.1f}% (2036 target)"
                    ),
                    "gap_pct": mat["gap_to_2036_pct"],
                    "deadline": "2036-08-18",
                    "suggested_measures": self._suggest_measures(
                        mat["name"], mat["gap_to_2036_pct"]
                    ),
                })

        if not any(m.get("documentation_available") for m in content_sec["materials"]):
            priority_rank += 1
            recommendations.append({
                "rank": priority_rank,
                "priority": "HIGH",
                "material": "All",
                "action": "Establish recycled content documentation and audit trail",
                "gap_pct": 0.0,
                "deadline": "2025-08-18",
                "suggested_measures": [
                    "Implement material traceability system",
                    "Obtain third-party verification of recycled content claims",
                    "Establish supplier recycled content certificates",
                ],
            })

        return {
            "title": "Recommendations",
            "total_recommendations": len(recommendations),
            "high_priority_count": sum(
                1 for r in recommendations if r["priority"] == "HIGH"
            ),
            "recommendations": recommendations,
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Recycled Content Report\n"
            f"## EU Battery Regulation (EU) 2023/1542 - Article 8\n\n"
            f"**Manufacturer:** {data.get('entity_name', '')}  \n"
            f"**Battery Model:** {data.get('battery_model', '')}  \n"
            f"**Battery Type:** {data.get('battery_type', 'ev_battery')}  \n"
            f"**Generated:** {ts}"
        )

    def _md_material_inventory(self, data: Dict[str, Any]) -> str:
        """Render material inventory as markdown."""
        sec = self._section_material_inventory(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Battery Mass:** {sec['total_battery_mass_kg']:.3f} kg  \n"
            f"**Regulated Materials:** {sec['regulated_material_count']} of "
            f"{sec['material_count']}\n",
            "| Material | Mass (kg) | % of Battery | Regulated |",
            "|----------|----------:|------------:|:---------:|",
        ]
        for mat in sec["materials"]:
            reg = "Yes" if mat["is_regulated_material"] else "No"
            lines.append(
                f"| {mat['name']} | {mat['total_mass_kg']:.3f} | "
                f"{mat['percentage_of_battery_mass']:.2f}% | {reg} |"
            )
        return "\n".join(lines)

    def _md_per_material_content(self, data: Dict[str, Any]) -> str:
        """Render per-material recycled content as markdown."""
        sec = self._section_per_material_content(data)
        lines = [
            f"## {sec['title']}\n",
            "| Material | Recycled % | 2031 Target | 2036 Target | "
            "2031 Status | 2036 Status |",
            "|----------|----------:|----------:|----------:|:----------:|:----------:|",
        ]
        for mat in sec["materials"]:
            s31 = "PASS" if mat["meets_2031_target"] else f"GAP -{mat['gap_to_2031_pct']:.1f}%"
            s36 = "PASS" if mat["meets_2036_target"] else f"GAP -{mat['gap_to_2036_pct']:.1f}%"
            lines.append(
                f"| {mat['name']} | {mat['recycled_content_pct']:.2f}% | "
                f"{mat['target_2031_pct']:.1f}% | {mat['target_2036_pct']:.1f}% | "
                f"{s31} | {s36} |"
            )
        return "\n".join(lines)

    def _md_target_tracking(self, data: Dict[str, Any]) -> str:
        """Render target tracking as markdown."""
        sec = self._section_target_tracking(data)
        lines = [
            f"## {sec['title']}\n",
            f"### Phase 1 - Deadline: {sec['phase_1_deadline']}\n",
            "| Material | Current | Target | Gap | Status |",
            "|----------|--------:|-------:|----:|:------:|",
        ]
        for item in sec["phase_1_targets"]:
            lines.append(
                f"| {item['material']} | {item['current_pct']:.1f}% | "
                f"{item['target_pct']:.1f}% | {item['gap_pct']:.1f}% | {item['status']} |"
            )
        status1 = "ALL MET" if sec["phase_1_all_met"] else "GAPS REMAIN"
        lines.append(f"\n**Phase 1 Overall:** {status1}\n")

        lines.append(f"### Phase 2 - Deadline: {sec['phase_2_deadline']}\n")
        lines.append("| Material | Current | Target | Gap | Status |")
        lines.append("|----------|--------:|-------:|----:|:------:|")
        for item in sec["phase_2_targets"]:
            lines.append(
                f"| {item['material']} | {item['current_pct']:.1f}% | "
                f"{item['target_pct']:.1f}% | {item['gap_pct']:.1f}% | {item['status']} |"
            )
        status2 = "ALL MET" if sec["phase_2_all_met"] else "GAPS REMAIN"
        lines.append(f"\n**Phase 2 Overall:** {status2}")
        return "\n".join(lines)

    def _md_phase_compliance(self, data: Dict[str, Any]) -> str:
        """Render phase compliance as markdown."""
        sec = self._section_phase_compliance(data)
        lines = [
            f"## {sec['title']}\n",
            "| Phase | Article | Deadline | Description | Status |",
            "|-------|---------|----------|-------------|:------:|",
        ]
        for phase in sec["phases"]:
            lines.append(
                f"| {phase['phase']} | {phase['article']} | {phase['deadline']} | "
                f"{phase['description']} | {phase['status']} |"
            )
        overall = "COMPLIANT" if sec["overall_compliance"] else "ACTION REQUIRED"
        lines.append(f"\n**Overall:** {overall}")
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations as markdown."""
        sec = self._section_recommendations(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Actions:** {sec['total_recommendations']}  \n"
            f"**High Priority:** {sec['high_priority_count']}\n",
        ]
        for rec in sec["recommendations"]:
            lines.append(f"### {rec['rank']}. [{rec['priority']}] {rec['material']}\n")
            lines.append(f"{rec['action']}\n")
            if rec["suggested_measures"]:
                lines.append("**Suggested Measures:**")
                for measure in rec["suggested_measures"]:
                    lines.append(f"  - {measure}")
            lines.append("")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n*Report generated by PACK-020 Battery Passport Prep Pack on {ts}*\n"
            f"*Regulation: EU Battery Regulation (EU) 2023/1542, Article 8*"
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
            ".warn{color:#e65100;font-weight:bold}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Recycled Content Report</h1>\n"
            f"<p>EU Battery Regulation (EU) 2023/1542 - Article 8</p>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> | "
            f"{data.get('battery_model', '')}</p>"
        )

    def _html_material_inventory(self, data: Dict[str, Any]) -> str:
        """Render material inventory HTML."""
        sec = self._section_material_inventory(data)
        rows = "".join(
            f"<tr><td>{m['name']}</td><td>{m['total_mass_kg']:.3f}</td>"
            f"<td>{m['percentage_of_battery_mass']:.2f}%</td>"
            f"<td>{'Yes' if m['is_regulated_material'] else 'No'}</td></tr>"
            for m in sec["materials"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total battery mass: {sec['total_battery_mass_kg']:.3f} kg</p>\n"
            f"<table><tr><th>Material</th><th>Mass (kg)</th><th>%</th>"
            f"<th>Regulated</th></tr>{rows}</table>"
        )

    def _html_per_material_content(self, data: Dict[str, Any]) -> str:
        """Render per-material content HTML."""
        sec = self._section_per_material_content(data)
        rows = ""
        for m in sec["materials"]:
            cls31 = "pass" if m["meets_2031_target"] else "fail"
            cls36 = "pass" if m["meets_2036_target"] else "fail"
            rows += (
                f"<tr><td>{m['name']}</td><td>{m['recycled_content_pct']:.2f}%</td>"
                f"<td>{m['target_2031_pct']:.1f}%</td>"
                f"<td>{m['target_2036_pct']:.1f}%</td>"
                f"<td class='{cls31}'>{'PASS' if m['meets_2031_target'] else 'GAP'}</td>"
                f"<td class='{cls36}'>{'PASS' if m['meets_2036_target'] else 'GAP'}</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Material</th><th>Recycled %</th><th>2031 Target</th>"
            f"<th>2036 Target</th><th>2031</th><th>2036</th></tr>{rows}</table>"
        )

    def _html_target_tracking(self, data: Dict[str, Any]) -> str:
        """Render target tracking HTML."""
        sec = self._section_target_tracking(data)
        rows = "".join(
            f"<tr><td>{i['material']}</td><td>{i['current_pct']:.1f}%</td>"
            f"<td>{i['target_pct']:.1f}%</td><td>{i['gap_pct']:.1f}%</td>"
            f"<td class='{'pass' if i['status'] == 'MET' else 'fail'}'>{i['status']}</td></tr>"
            for i in sec["phase_1_targets"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<h3>Phase 1 - {sec['phase_1_deadline']}</h3>\n"
            f"<table><tr><th>Material</th><th>Current</th><th>Target</th>"
            f"<th>Gap</th><th>Status</th></tr>{rows}</table>"
        )

    def _html_phase_compliance(self, data: Dict[str, Any]) -> str:
        """Render phase compliance HTML."""
        sec = self._section_phase_compliance(data)
        rows = "".join(
            f"<tr><td>{p['phase']}</td><td>{p['article']}</td><td>{p['deadline']}</td>"
            f"<td>{p['description']}</td><td>{p['status']}</td></tr>"
            for p in sec["phases"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Phase</th><th>Article</th><th>Deadline</th>"
            f"<th>Description</th><th>Status</th></tr>{rows}</table>"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _phase_status(self, reporting_year: int, deadline_year: int, met: bool) -> str:
        """Determine phase compliance status."""
        if reporting_year < deadline_year:
            return "NOT_YET_APPLICABLE"
        return "COMPLIANT" if met else "NON_COMPLIANT"

    def _suggest_measures(self, material: str, gap_pct: float) -> List[str]:
        """Generate deterministic sourcing recommendations based on material and gap."""
        measures: List[str] = []
        mat_lower = material.lower()
        if gap_pct <= 0:
            return ["Maintain current recycled content sourcing"]
        measures.append(f"Secure additional recycled {mat_lower} supply contracts")
        measures.append(f"Qualify secondary {mat_lower} suppliers from recycling facilities")
        if gap_pct > 5.0:
            measures.append("Invest in closed-loop battery recycling partnerships")
        if mat_lower == "lithium":
            measures.append("Explore hydrometallurgical lithium recovery processes")
        elif mat_lower == "cobalt":
            measures.append("Partner with urban mining and e-waste recycling operators")
        elif mat_lower == "nickel":
            measures.append("Source recycled nickel from stainless steel recyclers")
        elif mat_lower == "lead":
            measures.append("Verify existing lead recycling loop meets documentation standards")
        measures.append("Implement chain-of-custody tracking for recycled material claims")
        return measures
