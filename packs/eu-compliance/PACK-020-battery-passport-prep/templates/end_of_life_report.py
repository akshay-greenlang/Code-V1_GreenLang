# -*- coding: utf-8 -*-
"""
EndOfLifeReportTemplate - EU Battery Regulation Art 56-71 End-of-Life Management Report

Renders the end-of-life management compliance report covering collection,
recycling, and material recovery requirements per Articles 56-71 and Annex XII
of Regulation (EU) 2023/1542. Tracks collection rates against mandatory targets
(portable: 63% by 2027, 73% by 2030; LMT: 51% by 2028, 61% by 2031),
recycling efficiency against minimums (Li-ion 65% by 2025, 70% by 2030;
lead-acid 75%), material recovery (Co 95%, Cu 95%, Ni 95%, Li 80%),
and second-life battery assessment readiness.

Sections:
    1. Collection Rates - Current vs. target collection rates by category
    2. Recycling Efficiency - Process efficiency against regulatory minimums
    3. Material Recovery - Per-material recovery rates vs. targets
    4. Second-Life Assessment - Repurposing feasibility and requirements

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
    "collection_rates",
    "recycling_efficiency",
    "material_recovery",
    "second_life_assessment",
]

# Collection rate targets per Art 59
_COLLECTION_TARGETS: Dict[str, List[Dict[str, Any]]] = {
    "portable": [
        {"year": 2024, "target_pct": 45.0, "article": "Art 59(3)(a)"},
        {"year": 2027, "target_pct": 63.0, "article": "Art 59(3)(b)"},
        {"year": 2030, "target_pct": 73.0, "article": "Art 59(3)(c)"},
    ],
    "lmt": [
        {"year": 2028, "target_pct": 51.0, "article": "Art 59(3a)(a)"},
        {"year": 2031, "target_pct": 61.0, "article": "Art 59(3a)(b)"},
    ],
    "automotive": [
        {"year": 2024, "target_pct": 100.0, "article": "Art 59(4)"},
    ],
    "industrial_ev": [
        {"year": 2024, "target_pct": 100.0, "article": "Art 59(5)"},
    ],
}

# Recycling efficiency targets per Art 67 + Annex XII Part B
_RECYCLING_EFFICIENCY_TARGETS: List[Dict[str, Any]] = [
    {"chemistry": "lithium_ion", "target_2025_pct": 65.0,
     "target_2030_pct": 70.0, "article": "Annex XII, Part B"},
    {"chemistry": "lead_acid", "target_2025_pct": 75.0,
     "target_2030_pct": 80.0, "article": "Annex XII, Part B"},
    {"chemistry": "nickel_cadmium", "target_2025_pct": 80.0,
     "target_2030_pct": 80.0, "article": "Annex XII, Part B"},
    {"chemistry": "other", "target_2025_pct": 50.0,
     "target_2030_pct": 50.0, "article": "Annex XII, Part B"},
]

# Material recovery targets per Art 67 + Annex XII Part C
_MATERIAL_RECOVERY_TARGETS: Dict[str, List[Dict[str, Any]]] = {
    "cobalt": [
        {"year": 2027, "target_pct": 90.0, "article": "Annex XII, Part C"},
        {"year": 2031, "target_pct": 95.0, "article": "Annex XII, Part C"},
    ],
    "copper": [
        {"year": 2027, "target_pct": 90.0, "article": "Annex XII, Part C"},
        {"year": 2031, "target_pct": 95.0, "article": "Annex XII, Part C"},
    ],
    "nickel": [
        {"year": 2027, "target_pct": 90.0, "article": "Annex XII, Part C"},
        {"year": 2031, "target_pct": 95.0, "article": "Annex XII, Part C"},
    ],
    "lithium": [
        {"year": 2027, "target_pct": 50.0, "article": "Annex XII, Part C"},
        {"year": 2031, "target_pct": 80.0, "article": "Annex XII, Part C"},
    ],
    "lead": [
        {"year": 2027, "target_pct": 90.0, "article": "Annex XII, Part C"},
        {"year": 2031, "target_pct": 95.0, "article": "Annex XII, Part C"},
    ],
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

class EndOfLifeReportTemplate:
    """
    End-of-Life Management report template per EU Battery Regulation Art 56-71.

    Generates an end-of-life compliance report covering battery collection,
    recycling efficiency, material recovery, and second-life assessment.
    Tracks performance against mandatory regulatory targets and identifies
    gaps requiring corrective action.

    Regulatory References:
        - Regulation (EU) 2023/1542, Articles 56-71
        - Regulation (EU) 2023/1542, Annex XII (Recycling efficiencies and
          material recovery targets)
        - Directive 2006/66/EC (repealed predecessor)

    Example:
        >>> tpl = EndOfLifeReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize EndOfLifeReportTemplate."""
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
        if not data.get("reporting_year"):
            errors.append("reporting_year is required")
        if "collection_data" not in data:
            warnings.append("collection_data missing; collection section limited")
        if "recycling_data" not in data:
            warnings.append("recycling_data missing; recycling section limited")
        if "material_recovery_data" not in data:
            warnings.append("material_recovery_data missing; recovery section limited")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render end-of-life report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_collection_rates(data),
            self._md_recycling_efficiency(data),
            self._md_material_recovery(data),
            self._md_second_life(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render end-of-life report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_collection_rates(data),
            self._html_recycling_efficiency(data),
            self._html_material_recovery(data),
            self._html_second_life(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>End-of-Life Management Report - Art 56-71</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render end-of-life report as JSON."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "end_of_life_report",
            "regulation_reference": "EU Battery Regulation 2023/1542, Art 56-71",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "collection_rates": self._section_collection_rates(data),
            "recycling_efficiency": self._section_recycling_efficiency(data),
            "material_recovery": self._section_material_recovery(data),
            "second_life_assessment": self._section_second_life_assessment(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_collection_rates(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build collection rates section."""
        collection = data.get("collection_data", {})
        reporting_year = data.get("reporting_year", utcnow().year)
        category_results: List[Dict[str, Any]] = []

        for category, targets in _COLLECTION_TARGETS.items():
            cat_data = collection.get(category, {})
            placed_on_market_kg = cat_data.get("placed_on_market_kg", 0.0)
            collected_kg = cat_data.get("collected_kg", 0.0)
            actual_rate = (
                round(collected_kg / placed_on_market_kg * 100, 2)
                if placed_on_market_kg > 0 else 0.0
            )

            applicable_target = self._find_applicable_target(targets, reporting_year)
            target_pct = applicable_target.get("target_pct", 0.0) if applicable_target else 0.0
            meets_target = actual_rate >= target_pct if target_pct > 0 else True
            gap_pct = max(0.0, round(target_pct - actual_rate, 2))

            category_results.append({
                "category": category,
                "category_label": category.replace("_", " ").title(),
                "placed_on_market_kg": round(placed_on_market_kg, 2),
                "collected_kg": round(collected_kg, 2),
                "actual_rate_pct": actual_rate,
                "target_pct": target_pct,
                "target_article": applicable_target.get("article", "") if applicable_target else "",
                "meets_target": meets_target,
                "gap_pct": gap_pct,
                "collection_points": cat_data.get("collection_points", 0),
                "future_targets": [
                    {"year": t["year"], "target_pct": t["target_pct"]}
                    for t in targets if t["year"] > reporting_year
                ],
            })

        all_met = all(r["meets_target"] for r in category_results)
        return {
            "title": "Collection Rates",
            "reporting_year": reporting_year,
            "categories": category_results,
            "all_targets_met": all_met,
            "total_collected_kg": round(
                sum(r["collected_kg"] for r in category_results), 2
            ),
            "total_placed_kg": round(
                sum(r["placed_on_market_kg"] for r in category_results), 2
            ),
        }

    def _section_recycling_efficiency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build recycling efficiency section."""
        recycling = data.get("recycling_data", {})
        reporting_year = data.get("reporting_year", utcnow().year)
        efficiency_results: List[Dict[str, Any]] = []

        for target_def in _RECYCLING_EFFICIENCY_TARGETS:
            chemistry = target_def["chemistry"]
            chem_data = recycling.get(chemistry, {})
            input_mass_kg = chem_data.get("input_mass_kg", 0.0)
            output_mass_kg = chem_data.get("output_mass_kg", 0.0)
            actual_eff = (
                round(output_mass_kg / input_mass_kg * 100, 2)
                if input_mass_kg > 0 else 0.0
            )

            target_pct = (
                target_def["target_2030_pct"]
                if reporting_year >= 2030
                else target_def["target_2025_pct"]
            )
            meets_target = actual_eff >= target_pct if input_mass_kg > 0 else True

            efficiency_results.append({
                "chemistry": chemistry,
                "chemistry_label": chemistry.replace("_", " ").title(),
                "input_mass_kg": round(input_mass_kg, 2),
                "output_mass_kg": round(output_mass_kg, 2),
                "actual_efficiency_pct": actual_eff,
                "target_pct": target_pct,
                "target_2025_pct": target_def["target_2025_pct"],
                "target_2030_pct": target_def["target_2030_pct"],
                "article": target_def["article"],
                "meets_target": meets_target,
                "gap_pct": max(0.0, round(target_pct - actual_eff, 2)),
                "recycling_process": chem_data.get("process", ""),
                "recycler_name": chem_data.get("recycler", ""),
            })

        return {
            "title": "Recycling Efficiency",
            "reporting_year": reporting_year,
            "chemistries": efficiency_results,
            "all_targets_met": all(r["meets_target"] for r in efficiency_results),
            "total_input_kg": round(
                sum(r["input_mass_kg"] for r in efficiency_results), 2
            ),
            "total_output_kg": round(
                sum(r["output_mass_kg"] for r in efficiency_results), 2
            ),
        }

    def _section_material_recovery(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build material recovery rates section."""
        recovery = data.get("material_recovery_data", {})
        reporting_year = data.get("reporting_year", utcnow().year)
        material_results: List[Dict[str, Any]] = []

        for material, targets in _MATERIAL_RECOVERY_TARGETS.items():
            mat_data = recovery.get(material, {})
            input_kg = mat_data.get("input_kg", 0.0)
            recovered_kg = mat_data.get("recovered_kg", 0.0)
            actual_recovery = (
                round(recovered_kg / input_kg * 100, 2)
                if input_kg > 0 else 0.0
            )

            applicable_target = self._find_applicable_target(targets, reporting_year)
            target_pct = (
                applicable_target.get("target_pct", 0.0) if applicable_target else 0.0
            )
            meets_target = actual_recovery >= target_pct if input_kg > 0 else True

            material_results.append({
                "material": material,
                "material_label": material.title(),
                "input_kg": round(input_kg, 3),
                "recovered_kg": round(recovered_kg, 3),
                "actual_recovery_pct": actual_recovery,
                "target_pct": target_pct,
                "meets_target": meets_target,
                "gap_pct": max(0.0, round(target_pct - actual_recovery, 2)),
                "recovery_method": mat_data.get("method", ""),
                "all_targets": [
                    {"year": t["year"], "target_pct": t["target_pct"]}
                    for t in targets
                ],
            })

        return {
            "title": "Material Recovery Rates",
            "reporting_year": reporting_year,
            "materials": material_results,
            "all_targets_met": all(r["meets_target"] for r in material_results),
            "total_input_kg": round(
                sum(r["input_kg"] for r in material_results), 3
            ),
            "total_recovered_kg": round(
                sum(r["recovered_kg"] for r in material_results), 3
            ),
        }

    def _section_second_life_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build second-life battery assessment section."""
        second_life = data.get("second_life", {})
        batteries_assessed = second_life.get("batteries_assessed", 0)
        eligible_count = second_life.get("eligible_for_second_life", 0)
        repurposed_count = second_life.get("repurposed_count", 0)

        eligibility_criteria: List[Dict[str, Any]] = [
            {
                "criterion": "State of Health above threshold",
                "threshold": f">= {second_life.get('soh_threshold_pct', 70.0)}%",
                "met": second_life.get("soh_criterion_met", False),
            },
            {
                "criterion": "Safety assessment passed",
                "threshold": "No critical safety issues",
                "met": second_life.get("safety_assessment_passed", False),
            },
            {
                "criterion": "Performance data available",
                "threshold": "Complete BMS data history",
                "met": second_life.get("performance_data_available", False),
            },
            {
                "criterion": "Remaining useful life assessment",
                "threshold": ">= 5 years for stationary applications",
                "met": second_life.get("rul_criterion_met", False),
            },
            {
                "criterion": "Liability and warranty framework",
                "threshold": "New CE marking per Art 69(3)",
                "met": second_life.get("warranty_framework_established", False),
            },
        ]

        criteria_met = sum(1 for c in eligibility_criteria if c["met"])
        return {
            "title": "Second-Life Battery Assessment",
            "batteries_assessed": batteries_assessed,
            "eligible_for_second_life": eligible_count,
            "eligibility_rate_pct": (
                round(eligible_count / batteries_assessed * 100, 1)
                if batteries_assessed > 0 else 0.0
            ),
            "repurposed_count": repurposed_count,
            "repurposing_rate_pct": (
                round(repurposed_count / eligible_count * 100, 1)
                if eligible_count > 0 else 0.0
            ),
            "eligibility_criteria": eligibility_criteria,
            "criteria_met": criteria_met,
            "criteria_total": len(eligibility_criteria),
            "criteria_compliance_pct": round(
                criteria_met / len(eligibility_criteria) * 100, 1
            ),
            "target_applications": second_life.get("target_applications", []),
            "economic_viability": second_life.get("economic_viability", ""),
            "environmental_benefit_kgco2e_saved": second_life.get(
                "co2e_saved_kg", 0.0
            ),
            "art_reference": "Art 69 (Repurposing and remanufacturing)",
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# End-of-Life Management Report\n"
            f"## EU Battery Regulation (EU) 2023/1542 - Articles 56-71\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Generated:** {ts}"
        )

    def _md_collection_rates(self, data: Dict[str, Any]) -> str:
        """Render collection rates as markdown."""
        sec = self._section_collection_rates(data)
        lines = [
            f"## {sec['title']}\n",
            f"**All Targets Met:** {'Yes' if sec['all_targets_met'] else 'No'}  \n"
            f"**Total Collected:** {sec['total_collected_kg']:,.2f} kg\n",
            "| Category | Placed (kg) | Collected (kg) | Rate | Target | Status |",
            "|----------|----------:|-------------:|-----:|-------:|:------:|",
        ]
        for cat in sec["categories"]:
            status = "PASS" if cat["meets_target"] else "FAIL"
            lines.append(
                f"| {cat['category_label']} | {cat['placed_on_market_kg']:,.2f} | "
                f"{cat['collected_kg']:,.2f} | {cat['actual_rate_pct']:.1f}% | "
                f"{cat['target_pct']:.1f}% | {status} |"
            )
        return "\n".join(lines)

    def _md_recycling_efficiency(self, data: Dict[str, Any]) -> str:
        """Render recycling efficiency as markdown."""
        sec = self._section_recycling_efficiency(data)
        lines = [
            f"## {sec['title']}\n",
            f"**All Targets Met:** {'Yes' if sec['all_targets_met'] else 'No'}\n",
            "| Chemistry | Input (kg) | Output (kg) | Efficiency | Target | Status |",
            "|-----------|----------:|----------:|----------:|-------:|:------:|",
        ]
        for chem in sec["chemistries"]:
            status = "PASS" if chem["meets_target"] else "FAIL"
            lines.append(
                f"| {chem['chemistry_label']} | {chem['input_mass_kg']:,.2f} | "
                f"{chem['output_mass_kg']:,.2f} | {chem['actual_efficiency_pct']:.1f}% | "
                f"{chem['target_pct']:.1f}% | {status} |"
            )
        return "\n".join(lines)

    def _md_material_recovery(self, data: Dict[str, Any]) -> str:
        """Render material recovery as markdown."""
        sec = self._section_material_recovery(data)
        lines = [
            f"## {sec['title']}\n",
            f"**All Targets Met:** {'Yes' if sec['all_targets_met'] else 'No'}\n",
            "| Material | Input (kg) | Recovered (kg) | Recovery | Target | Status |",
            "|----------|----------:|--------------:|--------:|-------:|:------:|",
        ]
        for mat in sec["materials"]:
            status = "PASS" if mat["meets_target"] else "FAIL"
            lines.append(
                f"| {mat['material_label']} | {mat['input_kg']:,.3f} | "
                f"{mat['recovered_kg']:,.3f} | {mat['actual_recovery_pct']:.1f}% | "
                f"{mat['target_pct']:.1f}% | {status} |"
            )
        return "\n".join(lines)

    def _md_second_life(self, data: Dict[str, Any]) -> str:
        """Render second-life assessment as markdown."""
        sec = self._section_second_life_assessment(data)
        lines = [
            f"## {sec['title']}\n*{sec['art_reference']}*\n",
            f"**Batteries Assessed:** {sec['batteries_assessed']}  \n"
            f"**Eligible for 2nd Life:** {sec['eligible_for_second_life']} "
            f"({sec['eligibility_rate_pct']:.1f}%)  \n"
            f"**Repurposed:** {sec['repurposed_count']} "
            f"({sec['repurposing_rate_pct']:.1f}%)  \n"
            f"**CO2e Saved:** {sec['environmental_benefit_kgco2e_saved']:,.1f} kg\n",
            "### Eligibility Criteria\n",
            "| Criterion | Threshold | Met |",
            "|-----------|-----------|:---:|",
        ]
        for crit in sec["eligibility_criteria"]:
            met = "Yes" if crit["met"] else "No"
            lines.append(
                f"| {crit['criterion']} | {crit['threshold']} | {met} |"
            )
        lines.append(
            f"\n**Criteria Met:** {sec['criteria_met']}/{sec['criteria_total']} "
            f"({sec['criteria_compliance_pct']:.1f}%)"
        )
        if sec["target_applications"]:
            lines.append("\n**Target Applications:**")
            for app in sec["target_applications"]:
                lines.append(f"  - {app}")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n*Report generated by PACK-020 Battery Passport Prep Pack on {ts}*\n"
            f"*Regulation: EU Battery Regulation (EU) 2023/1542, Articles 56-71*"
        )

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:1100px;margin:auto}"
            "h1{color:#0d47a1;border-bottom:2px solid #0d47a1;padding-bottom:.3em}"
            "h2{color:#1565c0;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e3f2fd}"
            ".pass{color:#2e7d32;font-weight:bold}"
            ".fail{color:#c62828;font-weight:bold}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>End-of-Life Management Report</h1>\n"
            f"<p>EU Battery Regulation (EU) 2023/1542 - Articles 56-71</p>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> | "
            f"Year: {data.get('reporting_year', '')}</p>"
        )

    def _html_collection_rates(self, data: Dict[str, Any]) -> str:
        """Render collection rates HTML."""
        sec = self._section_collection_rates(data)
        rows = "".join(
            f"<tr><td>{c['category_label']}</td>"
            f"<td>{c['placed_on_market_kg']:,.2f}</td>"
            f"<td>{c['collected_kg']:,.2f}</td>"
            f"<td>{c['actual_rate_pct']:.1f}%</td>"
            f"<td>{c['target_pct']:.1f}%</td>"
            f"<td class='{'pass' if c['meets_target'] else 'fail'}'>"
            f"{'PASS' if c['meets_target'] else 'FAIL'}</td></tr>"
            for c in sec["categories"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Category</th><th>Placed (kg)</th><th>Collected (kg)</th>"
            f"<th>Rate</th><th>Target</th><th>Status</th></tr>{rows}</table>"
        )

    def _html_recycling_efficiency(self, data: Dict[str, Any]) -> str:
        """Render recycling efficiency HTML."""
        sec = self._section_recycling_efficiency(data)
        rows = "".join(
            f"<tr><td>{c['chemistry_label']}</td>"
            f"<td>{c['input_mass_kg']:,.2f}</td>"
            f"<td>{c['actual_efficiency_pct']:.1f}%</td>"
            f"<td>{c['target_pct']:.1f}%</td>"
            f"<td class='{'pass' if c['meets_target'] else 'fail'}'>"
            f"{'PASS' if c['meets_target'] else 'FAIL'}</td></tr>"
            for c in sec["chemistries"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Chemistry</th><th>Input (kg)</th>"
            f"<th>Efficiency</th><th>Target</th><th>Status</th></tr>{rows}</table>"
        )

    def _html_material_recovery(self, data: Dict[str, Any]) -> str:
        """Render material recovery HTML."""
        sec = self._section_material_recovery(data)
        rows = "".join(
            f"<tr><td>{m['material_label']}</td>"
            f"<td>{m['input_kg']:,.3f}</td>"
            f"<td>{m['recovered_kg']:,.3f}</td>"
            f"<td>{m['actual_recovery_pct']:.1f}%</td>"
            f"<td>{m['target_pct']:.1f}%</td>"
            f"<td class='{'pass' if m['meets_target'] else 'fail'}'>"
            f"{'PASS' if m['meets_target'] else 'FAIL'}</td></tr>"
            for m in sec["materials"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Material</th><th>Input (kg)</th><th>Recovered (kg)</th>"
            f"<th>Recovery</th><th>Target</th><th>Status</th></tr>{rows}</table>"
        )

    def _html_second_life(self, data: Dict[str, Any]) -> str:
        """Render second-life assessment HTML."""
        sec = self._section_second_life_assessment(data)
        criteria_rows = "".join(
            f"<tr><td>{c['criterion']}</td><td>{c['threshold']}</td>"
            f"<td class='{'pass' if c['met'] else 'fail'}'>"
            f"{'Yes' if c['met'] else 'No'}</td></tr>"
            for c in sec["eligibility_criteria"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Assessed: {sec['batteries_assessed']} | "
            f"Eligible: {sec['eligible_for_second_life']} | "
            f"Repurposed: {sec['repurposed_count']}</p>\n"
            f"<table><tr><th>Criterion</th><th>Threshold</th><th>Met</th></tr>"
            f"{criteria_rows}</table>"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_applicable_target(
        self,
        targets: List[Dict[str, Any]],
        reporting_year: int,
    ) -> Optional[Dict[str, Any]]:
        """Find the most recent applicable target for the reporting year."""
        applicable = [t for t in targets if t["year"] <= reporting_year]
        if not applicable:
            return None
        return max(applicable, key=lambda t: t["year"])
