# -*- coding: utf-8 -*-
"""
TechnologyRoadmapReportTemplate - Technology transition roadmap for PACK-028.

Renders a sector-specific technology transition roadmap report with IEA
milestone tracking, technology adoption schedules, CapEx phasing timelines,
technology dependency graphs, and risk assessment. Multi-format output
(Markdown, HTML, JSON, PDF-ready).

Sections:
    1.  Executive Summary
    2.  Technology Inventory (Current State)
    3.  IEA Milestone Mapping
    4.  Technology Adoption Schedule
    5.  S-Curve Adoption Modeling
    6.  CapEx Phasing Timeline
    7.  OpEx Impact Analysis
    8.  Technology Dependency Graph
    9.  Technology Readiness Assessment
    10. Risk Assessment
    11. Implementation Priorities
    12. XBRL Tagging Summary
    13. Audit Trail & Provenance

Author: GreenLang Team
Version: 28.0.0
Pack: PACK-028 Sector Pathway Pack
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "28.0.0"
_PACK_ID = "PACK-028"
_TEMPLATE_ID = "technology_roadmap_report"

_PRIMARY = "#0a4a3a"
_SECONDARY = "#167a5b"
_ACCENT = "#1db954"
_LIGHT = "#e0f5ed"
_LIGHTER = "#f0faf5"
_WARN = "#ef6c00"
_DANGER = "#c62828"
_SUCCESS = "#2e7d32"

TRL_LEVELS: Dict[int, str] = {
    1: "Basic principles observed",
    2: "Technology concept formulated",
    3: "Experimental proof of concept",
    4: "Technology validated in lab",
    5: "Technology validated in relevant environment",
    6: "Technology demonstrated in relevant environment",
    7: "System prototype demonstrated in operational environment",
    8: "System complete and qualified",
    9: "Actual system proven in operational environment",
}

SECTOR_TECHNOLOGIES: Dict[str, List[Dict[str, Any]]] = {
    "power": [
        {"id": "solar_pv", "name": "Solar PV", "trl": 9, "category": "Renewable"},
        {"id": "onshore_wind", "name": "Onshore Wind", "trl": 9, "category": "Renewable"},
        {"id": "offshore_wind", "name": "Offshore Wind", "trl": 9, "category": "Renewable"},
        {"id": "battery_storage", "name": "Grid Battery Storage", "trl": 8, "category": "Storage"},
        {"id": "green_hydrogen", "name": "Green Hydrogen (Electrolysis)", "trl": 7, "category": "Hydrogen"},
        {"id": "smr_nuclear", "name": "Small Modular Reactors", "trl": 6, "category": "Nuclear"},
        {"id": "ccs_power", "name": "CCS for Gas Power", "trl": 7, "category": "CCS"},
    ],
    "steel": [
        {"id": "eaf_scrap", "name": "Electric Arc Furnace (Scrap)", "trl": 9, "category": "Process"},
        {"id": "dri_hydrogen", "name": "DRI with Green Hydrogen", "trl": 7, "category": "Hydrogen"},
        {"id": "ccs_bf", "name": "CCS for Blast Furnace", "trl": 6, "category": "CCS"},
        {"id": "waste_heat", "name": "Waste Heat Recovery", "trl": 9, "category": "Efficiency"},
        {"id": "electrolysis_iron", "name": "Iron Electrolysis", "trl": 4, "category": "Emerging"},
    ],
    "cement": [
        {"id": "alt_fuels", "name": "Alternative Fuels (Biomass/Waste)", "trl": 9, "category": "Fuel"},
        {"id": "clinker_sub", "name": "Clinker Substitution", "trl": 9, "category": "Process"},
        {"id": "ccus_cement", "name": "CCUS for Cement", "trl": 6, "category": "CCS"},
        {"id": "he_kiln", "name": "High-Efficiency Kiln", "trl": 8, "category": "Efficiency"},
        {"id": "geopolymer", "name": "Geopolymer Cement", "trl": 5, "category": "Emerging"},
    ],
    "aviation": [
        {"id": "saf", "name": "Sustainable Aviation Fuel (SAF)", "trl": 8, "category": "Fuel"},
        {"id": "fuel_efficient", "name": "Next-Gen Fuel-Efficient Aircraft", "trl": 7, "category": "Fleet"},
        {"id": "electric_short", "name": "Electric Aircraft (Short-Haul)", "trl": 5, "category": "Electrification"},
        {"id": "h2_aircraft", "name": "Hydrogen Aircraft", "trl": 4, "category": "Hydrogen"},
        {"id": "ops_efficiency", "name": "Operational Efficiency (Routing)", "trl": 9, "category": "Efficiency"},
    ],
    "shipping": [
        {"id": "lng_fuel", "name": "LNG Propulsion", "trl": 9, "category": "Fuel"},
        {"id": "methanol", "name": "Green Methanol", "trl": 7, "category": "Fuel"},
        {"id": "ammonia_fuel", "name": "Ammonia Propulsion", "trl": 5, "category": "Fuel"},
        {"id": "wind_assist", "name": "Wind-Assisted Propulsion", "trl": 7, "category": "Efficiency"},
        {"id": "shore_power", "name": "Shore Power (Port Electrification)", "trl": 8, "category": "Electrification"},
    ],
}

XBRL_TECH_TAGS: Dict[str, str] = {
    "tech_count": "gl:TechnologyCountInRoadmap",
    "total_capex": "gl:TotalCapExInvestment",
    "milestones_on_track": "gl:IEAMilestonesOnTrack",
    "milestones_total": "gl:IEAMilestonesTotal",
    "avg_trl": "gl:AverageTechnologyReadinessLevel",
    "adoption_pct": "gl:TechnologyAdoptionPercentage",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dec(val: Any, places: int = 2) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)


def _dec_comma(val: Any, places: int = 0) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        rounded = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(rounded).split(".")
        int_part = parts[0]
        negative = int_part.startswith("-")
        if negative:
            int_part = int_part[1:]
        formatted = ""
        for i, ch in enumerate(reversed(int_part)):
            if i > 0 and i % 3 == 0:
                formatted = "," + formatted
            formatted = ch + formatted
        if negative:
            formatted = "-" + formatted
        if len(parts) > 1:
            formatted += "." + parts[1]
        return formatted
    except Exception:
        return str(val)


def _s_curve(year: int, start_year: int, inflection_year: int, end_year: int, max_pct: float = 100.0) -> float:
    """S-curve adoption model returning percentage adopted."""
    if year <= start_year:
        return 0.0
    if year >= end_year:
        return max_pct
    k = 0.5
    sigmoid = 1.0 / (1.0 + math.exp(-k * (year - inflection_year)))
    sigmoid_start = 1.0 / (1.0 + math.exp(-k * (start_year - inflection_year)))
    sigmoid_end = 1.0 / (1.0 + math.exp(-k * (end_year - inflection_year)))
    norm = (sigmoid - sigmoid_start) / (sigmoid_end - sigmoid_start)
    return max_pct * norm


def _trl_bar(trl: int) -> str:
    filled = trl
    empty = 9 - trl
    return "[" + "#" * filled + "." * empty + f"] TRL {trl}"


class TechnologyRoadmapReportTemplate:
    """
    Technology transition roadmap report template for sector pathways.

    Renders comprehensive technology roadmap with IEA milestone tracking,
    adoption schedules (S-curve), CapEx phasing, dependency mapping, and
    TRL assessment. Supports MD, HTML, JSON, and PDF.

    Example:
        >>> template = TechnologyRoadmapReportTemplate()
        >>> data = {
        ...     "org_name": "PowerGen Corp",
        ...     "sector_id": "power",
        ...     "technologies": [...],
        ...     "capex_plan": [...],
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_tech_inventory(data),
            self._md_iea_milestones(data),
            self._md_adoption_schedule(data),
            self._md_s_curve(data),
            self._md_capex_phasing(data),
            self._md_opex_impact(data),
            self._md_dependency_graph(data),
            self._md_trl_assessment(data),
            self._md_risk_assessment(data),
            self._md_priorities(data),
            self._md_xbrl_tags(data),
            self._md_audit_trail(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        body_parts = [
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_tech_inventory(data),
            self._html_iea_milestones(data),
            self._html_adoption_schedule(data),
            self._html_s_curve(data),
            self._html_capex_phasing(data),
            self._html_opex_impact(data),
            self._html_dependency_graph(data),
            self._html_trl_assessment(data),
            self._html_risk_assessment(data),
            self._html_priorities(data),
            self._html_xbrl_tags(data),
            self._html_audit_trail(data),
            self._html_footer(data),
        ]
        body = "\n".join(body_parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Technology Roadmap - {data.get("org_name", "")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        sector_id = data.get("sector_id", "")
        techs = data.get("technologies", SECTOR_TECHNOLOGIES.get(sector_id, []))
        milestones = data.get("iea_milestones", [])
        capex = data.get("capex_plan", [])
        total_capex = sum(float(c.get("amount", 0)) for c in capex)
        on_track = sum(1 for m in milestones if m.get("status") == "on_track")
        avg_trl = sum(t.get("trl", 0) for t in techs) / max(1, len(techs))

        result = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "sector_id": sector_id,
            "summary": {
                "technology_count": len(techs),
                "total_capex_eur": str(total_capex),
                "iea_milestones_total": len(milestones),
                "iea_milestones_on_track": on_track,
                "average_trl": str(round(avg_trl, 1)),
            },
            "technologies": techs,
            "iea_milestones": milestones,
            "capex_plan": capex,
            "opex_impact": data.get("opex_impact", []),
            "dependencies": data.get("dependencies", []),
            "risks": data.get("risks", []),
            "priorities": data.get("priorities", []),
            "xbrl_tags": {k: XBRL_TECH_TAGS[k] for k in XBRL_TECH_TAGS},
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "format": "pdf",
            "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {
                "title": f"Technology Roadmap - {data.get('org_name', '')}",
                "author": "GreenLang PACK-028",
                "subject": "Technology Transition Roadmap",
                "creator": f"GreenLang v{_MODULE_VERSION}",
            },
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_techs(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        sector_id = data.get("sector_id", "")
        return data.get("technologies", SECTOR_TECHNOLOGIES.get(sector_id, []))

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Technology Transition Roadmap\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Sector:** {data.get('sector_id', '').replace('_', ' ').title()}  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-028 Sector Pathway Pack v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        techs = self._get_techs(data)
        milestones = data.get("iea_milestones", [])
        capex = data.get("capex_plan", [])
        total_capex = sum(float(c.get("amount", 0)) for c in capex)
        on_track = sum(1 for m in milestones if m.get("status") == "on_track")
        avg_trl = sum(t.get("trl", 0) for t in techs) / max(1, len(techs))
        lines = [
            "## 1. Executive Summary\n",
            f"| KPI | Value |",
            f"|-----|-------|",
            f"| Technologies in Roadmap | {len(techs)} |",
            f"| Average TRL | {_dec(avg_trl, 1)} / 9 |",
            f"| IEA Milestones Mapped | {len(milestones)} |",
            f"| Milestones On Track | {on_track} ({_dec(on_track / max(1, len(milestones)) * 100)}%) |",
            f"| Total CapEx Investment | EUR {_dec_comma(total_capex)} |",
            f"| Technology Categories | {len(set(t.get('category', '') for t in techs))} |",
        ]
        return "\n".join(lines)

    def _md_tech_inventory(self, data: Dict[str, Any]) -> str:
        techs = self._get_techs(data)
        lines = [
            "## 2. Technology Inventory\n",
            "| # | Technology | Category | TRL | Status | Current Adoption |",
            "|---|-----------|----------|----:|--------|-----------------|",
        ]
        for i, t in enumerate(techs, 1):
            trl = t.get("trl", 0)
            status = "Mature" if trl >= 8 else ("Demonstrating" if trl >= 6 else ("Developing" if trl >= 4 else "Early R&D"))
            adoption = t.get("current_adoption_pct", 0)
            lines.append(
                f"| {i} | {t.get('name', '')} | {t.get('category', '')} "
                f"| {trl} | {status} | {_dec(adoption)}% |"
            )
        return "\n".join(lines)

    def _md_iea_milestones(self, data: Dict[str, Any]) -> str:
        milestones = data.get("iea_milestones", [])
        lines = [
            "## 3. IEA Milestone Mapping\n",
            "| Year | Milestone | Technology | Status | Gap | IEA Chapter |",
            "|------|-----------|-----------|--------|-----|------------|",
        ]
        for m in milestones:
            status = m.get("status", "pending")
            icon = "ON TRACK" if status == "on_track" else ("OFF TRACK" if status == "off_track" else "PENDING")
            lines.append(
                f"| {m.get('year', '')} | {m.get('description', '')} "
                f"| {m.get('technology', '')} | {icon} "
                f"| {m.get('gap', '-')} | {m.get('iea_chapter', '-')} |"
            )
        if not milestones:
            lines.append("| - | _No milestones mapped yet_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_adoption_schedule(self, data: Dict[str, Any]) -> str:
        schedule = data.get("adoption_schedule", [])
        lines = [
            "## 4. Technology Adoption Schedule\n",
            "| Technology | Start | Inflection | Full Adoption | Max Share (%) |",
            "|-----------|------:|----------:|--------------:|--------------:|",
        ]
        for s in schedule:
            lines.append(
                f"| {s.get('technology', '')} | {s.get('start_year', '')} "
                f"| {s.get('inflection_year', '')} | {s.get('end_year', '')} "
                f"| {_dec(s.get('max_share_pct', 100))}% |"
            )
        if not schedule:
            lines.append("| _No adoption schedules defined_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_s_curve(self, data: Dict[str, Any]) -> str:
        schedule = data.get("adoption_schedule", [])
        if not schedule:
            return "## 5. S-Curve Adoption Modeling\n\n_Requires adoption schedule data._"
        lines = [
            "## 5. S-Curve Adoption Modeling\n",
            "| Technology |" + "".join(f" {yr} |" for yr in range(2025, 2055, 5)),
            "|-----------|" + "".join("------:|" for _ in range(2025, 2055, 5)),
        ]
        for s in schedule:
            name = s.get("technology", "")
            start = int(s.get("start_year", 2025))
            inflection = int(s.get("inflection_year", 2035))
            end = int(s.get("end_year", 2050))
            max_pct = float(s.get("max_share_pct", 100))
            vals = []
            for yr in range(2025, 2055, 5):
                pct = _s_curve(yr, start, inflection, end, max_pct)
                vals.append(f" {_dec(pct, 1)}%")
            lines.append(f"| {name} |" + " |".join(vals) + " |")
        return "\n".join(lines)

    def _md_capex_phasing(self, data: Dict[str, Any]) -> str:
        capex = data.get("capex_plan", [])
        lines = [
            "## 6. CapEx Phasing Timeline\n",
            "| Year | Technology | Amount (EUR) | Category | Cumulative (EUR) |",
            "|------|-----------|-------------:|----------|----------------:|",
        ]
        cumulative = 0
        for c in sorted(capex, key=lambda x: x.get("year", 0)):
            amt = float(c.get("amount", 0))
            cumulative += amt
            lines.append(
                f"| {c.get('year', '')} | {c.get('technology', '')} "
                f"| {_dec_comma(amt)} | {c.get('category', '')} "
                f"| {_dec_comma(cumulative)} |"
            )
        if capex:
            total = sum(float(c.get("amount", 0)) for c in capex)
            lines.append(f"| | **Total** | **{_dec_comma(total)}** | | |")
        else:
            lines.append("| - | _No CapEx plan defined_ | - | - | - |")

        # Phase breakdown
        phases = data.get("capex_phases", {})
        if phases:
            lines.append("\n### Phase Summary\n")
            lines.append("| Phase | Period | Amount (EUR) | Share (%) |")
            lines.append("|-------|--------|-------------:|----------:|")
            total_all = sum(float(v) for v in phases.values())
            for phase_name, amount in phases.items():
                amt = float(amount)
                share = (amt / total_all * 100) if total_all > 0 else 0
                lines.append(f"| {phase_name} | - | {_dec_comma(amt)} | {_dec(share)}% |")
        return "\n".join(lines)

    def _md_opex_impact(self, data: Dict[str, Any]) -> str:
        opex = data.get("opex_impact", [])
        lines = [
            "## 7. OpEx Impact Analysis\n",
            "| Technology | Current OpEx (EUR/yr) | Post-Adoption OpEx | Delta | Payback (yrs) |",
            "|-----------|---------------------:|--------------------|------:|--------------:|",
        ]
        for o in opex:
            current = float(o.get("current_opex", 0))
            post = float(o.get("post_adoption_opex", 0))
            delta = post - current
            lines.append(
                f"| {o.get('technology', '')} | {_dec_comma(current)} "
                f"| {_dec_comma(post)} | {'+' if delta > 0 else ''}{_dec_comma(delta)} "
                f"| {_dec(o.get('payback_years', 0), 1)} |"
            )
        if not opex:
            lines.append("| _No OpEx impact data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_dependency_graph(self, data: Dict[str, Any]) -> str:
        deps = data.get("dependencies", [])
        lines = [
            "## 8. Technology Dependency Graph\n",
            "Dependencies between technologies in the roadmap:\n",
        ]
        if deps:
            lines.append("| Technology | Depends On | Dependency Type | Critical Path |")
            lines.append("|-----------|-----------|----------------|:-------------:|")
            for d in deps:
                lines.append(
                    f"| {d.get('technology', '')} | {d.get('depends_on', '')} "
                    f"| {d.get('type', 'prerequisite')} "
                    f"| {'Yes' if d.get('critical_path', False) else 'No'} |"
                )
        else:
            lines.append("_No technology dependencies defined._")
        return "\n".join(lines)

    def _md_trl_assessment(self, data: Dict[str, Any]) -> str:
        techs = self._get_techs(data)
        lines = [
            "## 9. Technology Readiness Assessment\n",
            "| Technology | TRL | Level | Description |",
            "|-----------|----:|-------|-------------|",
        ]
        for t in techs:
            trl = t.get("trl", 0)
            desc = TRL_LEVELS.get(trl, "Unknown")
            lines.append(f"| {t.get('name', '')} | {trl} | {_trl_bar(trl)} | {desc} |")
        avg_trl = sum(t.get("trl", 0) for t in techs) / max(1, len(techs))
        lines.append(f"\n**Average TRL:** {_dec(avg_trl, 1)} / 9")
        lines.append(f"**TRL >= 7 (deployment-ready):** {sum(1 for t in techs if t.get('trl', 0) >= 7)} / {len(techs)}")
        return "\n".join(lines)

    def _md_risk_assessment(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        lines = [
            "## 10. Risk Assessment\n",
            "| Risk | Likelihood | Impact | Technology | Mitigation |",
            "|------|-----------|--------|-----------|-----------|",
        ]
        for r in risks:
            lines.append(
                f"| {r.get('risk', '')} | {r.get('likelihood', 'Medium')} "
                f"| {r.get('impact', 'Medium')} | {r.get('technology', 'Multiple')} "
                f"| {r.get('mitigation', '')} |"
            )
        if not risks:
            lines.append("| Technology maturity risk | Medium | High | All emerging techs | Phased deployment with pilot validation |")
            lines.append("| Supply chain constraints | Medium | Medium | Green hydrogen | Multi-supplier strategy |")
            lines.append("| Cost overrun | Medium | High | CCS | Contingency budget (15-20%) |")
            lines.append("| Policy/regulatory risk | Low | High | All | Active policy engagement |")
        return "\n".join(lines)

    def _md_priorities(self, data: Dict[str, Any]) -> str:
        priorities = data.get("priorities", [])
        lines = [
            "## 11. Implementation Priorities\n",
        ]
        if priorities:
            lines.append("| Priority | Technology | Timeframe | Impact (tCO2e) | Investment (EUR) | Quick Win |")
            lines.append("|----------|-----------|-----------|---------------:|----------------:|:---------:|")
            for i, p in enumerate(priorities, 1):
                lines.append(
                    f"| {i} | {p.get('technology', '')} | {p.get('timeframe', '')} "
                    f"| {_dec_comma(p.get('impact_tco2e', 0))} "
                    f"| {_dec_comma(p.get('investment', 0))} "
                    f"| {'Yes' if p.get('quick_win', False) else 'No'} |"
                )
        else:
            lines.append("_Priorities will be determined based on abatement cost curves and TRL assessment._")
        return "\n".join(lines)

    def _md_xbrl_tags(self, data: Dict[str, Any]) -> str:
        techs = self._get_techs(data)
        milestones = data.get("iea_milestones", [])
        capex = data.get("capex_plan", [])
        total_capex = sum(float(c.get("amount", 0)) for c in capex)
        on_track = sum(1 for m in milestones if m.get("status") == "on_track")
        avg_trl = sum(t.get("trl", 0) for t in techs) / max(1, len(techs))
        lines = [
            "## 12. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |",
            "|------------|----------|-------|",
            f"| Technology Count | {XBRL_TECH_TAGS['tech_count']} | {len(techs)} |",
            f"| Total CapEx | {XBRL_TECH_TAGS['total_capex']} | EUR {_dec_comma(total_capex)} |",
            f"| Milestones On Track | {XBRL_TECH_TAGS['milestones_on_track']} | {on_track} |",
            f"| Milestones Total | {XBRL_TECH_TAGS['milestones_total']} | {len(milestones)} |",
            f"| Average TRL | {XBRL_TECH_TAGS['avg_trl']} | {_dec(avg_trl, 1)} |",
        ]
        return "\n".join(lines)

    def _md_audit_trail(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            "## 13. Audit Trail & Provenance\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n| Input Hash | `{dh[:16]}...` |\n"
            f"| Engine | Deterministic (zero-hallucination) |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-028 Sector Pathway Pack on {ts}*  \n"
            f"*Technology roadmap aligned with IEA NZE 2050 milestones.*"
        )

    # ------------------------------------------------------------------
    # HTML Sections
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            f"background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            f"border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"h3{{color:{_ACCENT};}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f9fbe7;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));"
            f"gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;"
            f"padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.5em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.75em;color:{_ACCENT};}}"
            f".trl-bar{{height:16px;border-radius:3px;display:inline-block;background:#e0e0e0;width:100%;}}"
            f".trl-fill{{height:16px;border-radius:3px;background:{_ACCENT};}}"
            f".status-on_track{{color:{_SUCCESS};font-weight:600;}}"
            f".status-off_track{{color:{_DANGER};font-weight:600;}}"
            f".status-pending{{color:{_WARN};font-style:italic;}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};"
            f"color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Technology Transition Roadmap</h1>\n'
            f'<p><strong>Organization:</strong> {data.get("org_name", "")} | '
            f'<strong>Sector:</strong> {data.get("sector_id", "").replace("_", " ").title()} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        techs = self._get_techs(data)
        milestones = data.get("iea_milestones", [])
        capex = data.get("capex_plan", [])
        total_capex = sum(float(c.get("amount", 0)) for c in capex)
        on_track = sum(1 for m in milestones if m.get("status") == "on_track")
        avg_trl = sum(t.get("trl", 0) for t in techs) / max(1, len(techs))
        return (
            f'<h2>1. Executive Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Technologies</div><div class="card-value">{len(techs)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Avg TRL</div><div class="card-value">{_dec(avg_trl, 1)}</div><div class="card-unit">/ 9</div></div>\n'
            f'  <div class="card"><div class="card-label">IEA Milestones</div><div class="card-value">{on_track}/{len(milestones)}</div><div class="card-unit">on track</div></div>\n'
            f'  <div class="card"><div class="card-label">Total CapEx</div><div class="card-value">EUR {_dec_comma(total_capex)}</div></div>\n'
            f'</div>'
        )

    def _html_tech_inventory(self, data: Dict[str, Any]) -> str:
        techs = self._get_techs(data)
        rows = ""
        for i, t in enumerate(techs, 1):
            trl = t.get("trl", 0)
            pct = trl / 9 * 100
            rows += (
                f'<tr><td>{i}</td><td>{t.get("name", "")}</td><td>{t.get("category", "")}</td>'
                f'<td><div class="trl-bar"><div class="trl-fill" style="width:{pct}%"></div></div> TRL {trl}</td>'
                f'<td>{_dec(t.get("current_adoption_pct", 0))}%</td></tr>\n'
            )
        return (
            f'<h2>2. Technology Inventory</h2>\n'
            f'<table>\n<tr><th>#</th><th>Technology</th><th>Category</th><th>TRL</th><th>Adoption</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_iea_milestones(self, data: Dict[str, Any]) -> str:
        milestones = data.get("iea_milestones", [])
        rows = ""
        for m in milestones:
            s = m.get("status", "pending")
            cls = f"status-{s}"
            rows += (
                f'<tr><td>{m.get("year", "")}</td><td>{m.get("description", "")}</td>'
                f'<td>{m.get("technology", "")}</td>'
                f'<td class="{cls}">{s.replace("_", " ").upper()}</td>'
                f'<td>{m.get("gap", "-")}</td></tr>\n'
            )
        return (
            f'<h2>3. IEA Milestones</h2>\n'
            f'<table>\n<tr><th>Year</th><th>Milestone</th><th>Technology</th><th>Status</th><th>Gap</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_adoption_schedule(self, data: Dict[str, Any]) -> str:
        schedule = data.get("adoption_schedule", [])
        rows = ""
        for s in schedule:
            rows += (
                f'<tr><td>{s.get("technology", "")}</td><td>{s.get("start_year", "")}</td>'
                f'<td>{s.get("inflection_year", "")}</td><td>{s.get("end_year", "")}</td>'
                f'<td>{_dec(s.get("max_share_pct", 100))}%</td></tr>\n'
            )
        return (
            f'<h2>4. Adoption Schedule</h2>\n'
            f'<table>\n<tr><th>Technology</th><th>Start</th><th>Inflection</th><th>Full Adoption</th><th>Max Share</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_s_curve(self, data: Dict[str, Any]) -> str:
        schedule = data.get("adoption_schedule", [])
        if not schedule:
            return '<h2>5. S-Curve Modeling</h2>\n<p><em>Requires adoption schedule data.</em></p>'
        header = "<th>Technology</th>" + "".join(f"<th>{yr}</th>" for yr in range(2025, 2055, 5))
        rows = ""
        for s in schedule:
            name = s.get("technology", "")
            start = int(s.get("start_year", 2025))
            inflection = int(s.get("inflection_year", 2035))
            end = int(s.get("end_year", 2050))
            max_pct = float(s.get("max_share_pct", 100))
            cells = ""
            for yr in range(2025, 2055, 5):
                pct = _s_curve(yr, start, inflection, end, max_pct)
                cells += f'<td>{_dec(pct, 1)}%</td>'
            rows += f'<tr><td>{name}</td>{cells}</tr>\n'
        return (
            f'<h2>5. S-Curve Modeling</h2>\n'
            f'<table>\n<tr>{header}</tr>\n{rows}</table>'
        )

    def _html_capex_phasing(self, data: Dict[str, Any]) -> str:
        capex = data.get("capex_plan", [])
        rows = ""
        cum = 0
        for c in sorted(capex, key=lambda x: x.get("year", 0)):
            amt = float(c.get("amount", 0))
            cum += amt
            rows += (
                f'<tr><td>{c.get("year", "")}</td><td>{c.get("technology", "")}</td>'
                f'<td>EUR {_dec_comma(amt)}</td><td>{c.get("category", "")}</td>'
                f'<td>EUR {_dec_comma(cum)}</td></tr>\n'
            )
        return (
            f'<h2>6. CapEx Phasing</h2>\n'
            f'<table>\n<tr><th>Year</th><th>Technology</th><th>Amount</th><th>Category</th><th>Cumulative</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_opex_impact(self, data: Dict[str, Any]) -> str:
        opex = data.get("opex_impact", [])
        rows = ""
        for o in opex:
            c = float(o.get("current_opex", 0))
            p = float(o.get("post_adoption_opex", 0))
            d = p - c
            rows += (
                f'<tr><td>{o.get("technology", "")}</td><td>EUR {_dec_comma(c)}</td>'
                f'<td>EUR {_dec_comma(p)}</td><td>{"+" if d > 0 else ""}{_dec_comma(d)}</td>'
                f'<td>{_dec(o.get("payback_years", 0), 1)}</td></tr>\n'
            )
        return (
            f'<h2>7. OpEx Impact</h2>\n'
            f'<table>\n<tr><th>Technology</th><th>Current</th><th>Post-Adoption</th><th>Delta</th><th>Payback</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_dependency_graph(self, data: Dict[str, Any]) -> str:
        deps = data.get("dependencies", [])
        rows = ""
        for d in deps:
            cp = "Yes" if d.get("critical_path", False) else "No"
            rows += (
                f'<tr><td>{d.get("technology", "")}</td><td>{d.get("depends_on", "")}</td>'
                f'<td>{d.get("type", "prerequisite")}</td><td>{cp}</td></tr>\n'
            )
        return (
            f'<h2>8. Dependencies</h2>\n'
            f'<table>\n<tr><th>Technology</th><th>Depends On</th><th>Type</th><th>Critical Path</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_trl_assessment(self, data: Dict[str, Any]) -> str:
        techs = self._get_techs(data)
        rows = ""
        for t in techs:
            trl = t.get("trl", 0)
            pct = trl / 9 * 100
            desc = TRL_LEVELS.get(trl, "Unknown")
            rows += (
                f'<tr><td>{t.get("name", "")}</td><td>{trl}</td>'
                f'<td><div class="trl-bar"><div class="trl-fill" style="width:{pct}%"></div></div></td>'
                f'<td>{desc}</td></tr>\n'
            )
        return (
            f'<h2>9. TRL Assessment</h2>\n'
            f'<table>\n<tr><th>Technology</th><th>TRL</th><th>Level</th><th>Description</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_risk_assessment(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        rows = ""
        for r in risks:
            rows += (
                f'<tr><td>{r.get("risk", "")}</td><td>{r.get("likelihood", "Medium")}</td>'
                f'<td>{r.get("impact", "Medium")}</td><td>{r.get("technology", "")}</td>'
                f'<td>{r.get("mitigation", "")}</td></tr>\n'
            )
        return (
            f'<h2>10. Risk Assessment</h2>\n'
            f'<table>\n<tr><th>Risk</th><th>Likelihood</th><th>Impact</th><th>Technology</th><th>Mitigation</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_priorities(self, data: Dict[str, Any]) -> str:
        priorities = data.get("priorities", [])
        rows = ""
        for i, p in enumerate(priorities, 1):
            rows += (
                f'<tr><td>{i}</td><td>{p.get("technology", "")}</td><td>{p.get("timeframe", "")}</td>'
                f'<td>{_dec_comma(p.get("impact_tco2e", 0))}</td>'
                f'<td>EUR {_dec_comma(p.get("investment", 0))}</td>'
                f'<td>{"Yes" if p.get("quick_win", False) else "No"}</td></tr>\n'
            )
        return (
            f'<h2>11. Priorities</h2>\n'
            f'<table>\n<tr><th>#</th><th>Technology</th><th>Timeframe</th>'
            f'<th>Impact (tCO2e)</th><th>Investment</th><th>Quick Win</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_xbrl_tags(self, data: Dict[str, Any]) -> str:
        techs = self._get_techs(data)
        milestones = data.get("iea_milestones", [])
        capex = data.get("capex_plan", [])
        total_capex = sum(float(c.get("amount", 0)) for c in capex)
        on_track = sum(1 for m in milestones if m.get("status") == "on_track")
        avg_trl = sum(t.get("trl", 0) for t in techs) / max(1, len(techs))
        return (
            f'<h2>12. XBRL Tags</h2>\n'
            f'<table>\n<tr><th>Point</th><th>Tag</th><th>Value</th></tr>\n'
            f'<tr><td>Technology Count</td><td><code>{XBRL_TECH_TAGS["tech_count"]}</code></td><td>{len(techs)}</td></tr>\n'
            f'<tr><td>Total CapEx</td><td><code>{XBRL_TECH_TAGS["total_capex"]}</code></td><td>EUR {_dec_comma(total_capex)}</td></tr>\n'
            f'<tr><td>Milestones On Track</td><td><code>{XBRL_TECH_TAGS["milestones_on_track"]}</code></td><td>{on_track}</td></tr>\n'
            f'<tr><td>Average TRL</td><td><code>{XBRL_TECH_TAGS["avg_trl"]}</code></td><td>{_dec(avg_trl, 1)}</td></tr>\n'
            f'</table>'
        )

    def _html_audit_trail(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            f'<h2>13. Audit Trail</h2>\n'
            f'<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n'
            f'<tr><td>Generated</td><td>{ts}</td></tr>\n'
            f'<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n'
            f'<tr><td>Version</td><td>{_MODULE_VERSION}</td></tr>\n'
            f'<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n'
            f'</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-028 Sector Pathway Pack on {ts}<br>'
            f'Technology roadmap aligned with IEA NZE 2050 milestones'
            f'</div>'
        )
