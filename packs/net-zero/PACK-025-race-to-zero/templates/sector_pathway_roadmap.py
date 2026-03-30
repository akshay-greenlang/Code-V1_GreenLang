# -*- coding: utf-8 -*-
"""
SectorPathwayRoadmapTemplate - Sector decarbonization pathway for PACK-025.

Renders a sector-specific emissions trajectory (2025-2050) with technology
adoption curves, abatement priority ranking, peer benchmark comparison,
and investment requirements by phase.

Sections:
    1. Sector Overview
    2. Emissions Trajectory (2025-2050)
    3. Key Milestones & Technology Adoption
    4. Abatement Priority Ranking (MAC Curve)
    5. Peer Benchmark Comparison
    6. Investment Requirements by Phase
    7. Sector-Specific Risks
    8. Policy & Regulatory Landscape

Author: GreenLang Team
Version: 25.0.0
Pack: PACK-025 Race to Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "25.0.0"
_PACK_ID = "PACK-025"
_TEMPLATE_ID = "sector_pathway_roadmap"

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
        r = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(r).split(".")
        ip = parts[0]
        neg = ip.startswith("-")
        if neg:
            ip = ip[1:]
        f = ""
        for i, ch in enumerate(reversed(ip)):
            if i > 0 and i % 3 == 0:
                f = "," + f
            f = ch + f
        if neg:
            f = "-" + f
        if len(parts) > 1:
            f += "." + parts[1]
        return f
    except Exception:
        return str(val)

def _pct(val: Any) -> str:
    try:
        return _dec(val, 1) + "%"
    except Exception:
        return str(val)

def _safe_div(n: Any, d: Any) -> float:
    try:
        dv = float(d)
        return float(n) / dv if dv != 0 else 0.0
    except Exception:
        return 0.0

class SectorPathwayRoadmapTemplate:
    """Sector-specific pathway roadmap template for PACK-025.

    Generates sector-specific decarbonization pathways with emissions
    trajectories, technology adoption curves, marginal abatement cost
    analysis, peer benchmarking, and phased investment requirements.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID

    # Default sector pathways reference
    SECTOR_PATHWAYS = {
        "Power & Utilities": {"pathway": "IEA NZE 2050", "interim_reduction": 75, "key_tech": "Renewables, CCS, Storage"},
        "Oil & Gas": {"pathway": "IEA NZE 2050", "interim_reduction": 40, "key_tech": "CCUS, Electrification, H2"},
        "Steel & Iron": {"pathway": "SDA / MPP", "interim_reduction": 30, "key_tech": "DRI-H2, EAF, CCUS"},
        "Cement": {"pathway": "SDA / GCCA", "interim_reduction": 25, "key_tech": "CCUS, Alt. fuels, Novel cements"},
        "Chemicals": {"pathway": "SDA / IEA", "interim_reduction": 30, "key_tech": "Electrification, Bio-feedstocks"},
        "Transport (Aviation)": {"pathway": "SDA / ICAO", "interim_reduction": 25, "key_tech": "SAF, Efficiency, Electric"},
        "Transport (Shipping)": {"pathway": "SDA / IMO", "interim_reduction": 30, "key_tech": "Green ammonia, LNG, Wind"},
        "Transport (Road)": {"pathway": "SDA / IEA", "interim_reduction": 50, "key_tech": "EV, H2 trucks, Efficiency"},
        "Buildings": {"pathway": "SDA / IEA", "interim_reduction": 45, "key_tech": "Heat pumps, Insulation, Smart"},
        "Agriculture": {"pathway": "IPCC AR6", "interim_reduction": 25, "key_tech": "Precision ag, Feed additives"},
        "Financial Services": {"pathway": "PCAF / NZBA", "interim_reduction": 50, "key_tech": "Portfolio alignment, Green finance"},
        "Technology": {"pathway": "SDA / SBTi", "interim_reduction": 55, "key_tech": "Renewables, Efficiency, Circular"},
        "General": {"pathway": "Cross-sector 1.5C", "interim_reduction": 43, "key_tech": "Sector-dependent"},
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    #  Public render methods                                               #
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the sector pathway roadmap as Markdown."""
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_sector_overview(data),
            self._md_emissions_trajectory(data),
            self._md_milestones_tech(data),
            self._md_abatement_ranking(data),
            self._md_peer_benchmark(data),
            self._md_investment_requirements(data),
            self._md_sector_risks(data),
            self._md_policy_landscape(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the sector pathway roadmap as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_sector_overview(data),
            self._html_trajectory(data),
            self._html_milestones(data),
            self._html_abatement(data),
            self._html_benchmark(data),
            self._html_investment(data),
            self._html_risks(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Race to Zero - Sector Pathway Roadmap</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the sector pathway roadmap as structured JSON."""
        self.generated_at = utcnow()
        sector = data.get("sector", "General")
        pathway_ref = self.SECTOR_PATHWAYS.get(sector, self.SECTOR_PATHWAYS["General"])

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "sector": sector,
            "pathway_reference": pathway_ref,
            "baseline_year": data.get("baseline_year", ""),
            "baseline_tco2e": data.get("baseline_tco2e", 0),
            "trajectory": data.get("trajectory", []),
            "abatement_levers": data.get("abatement_levers", []),
            "peer_benchmarks": data.get("peer_benchmarks", []),
            "investment_phases": data.get("investment_phases", []),
            "technology_adoption": data.get("technology_adoption", []),
            "sector_risks": data.get("sector_risks", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel_data(self, data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Return structured data for Excel/openpyxl export."""
        self.generated_at = utcnow()
        sheets: Dict[str, List[Dict[str, Any]]] = {}

        # Sheet 1: Trajectory
        trajectory = data.get("trajectory", [])
        traj_rows: List[Dict[str, Any]] = []
        for t in trajectory:
            traj_rows.append({
                "Year": t.get("year", ""),
                "Sector Pathway (tCO2e)": t.get("sector_target_tco2e", 0),
                "Organization Target (tCO2e)": t.get("org_target_tco2e", 0),
                "Actual (tCO2e)": t.get("actual_tco2e", ""),
                "Reduction from Baseline (%)": t.get("reduction_pct", 0),
            })
        sheets["Emissions Trajectory"] = traj_rows

        # Sheet 2: Abatement Levers
        levers = data.get("abatement_levers", [])
        lever_rows: List[Dict[str, Any]] = []
        for lever in levers:
            lever_rows.append({
                "Rank": lever.get("rank", ""),
                "Lever": lever.get("lever", ""),
                "Reduction (tCO2e)": lever.get("reduction_tco2e", 0),
                "Marginal Cost (USD/tCO2e)": lever.get("cost_per_tco2e", 0),
                "CAPEX (USD)": lever.get("capex_usd", 0),
                "Maturity": lever.get("maturity", ""),
                "Timeline": lever.get("timeline", ""),
            })
        sheets["Abatement Levers"] = lever_rows

        # Sheet 3: Peer Benchmarks
        peers = data.get("peer_benchmarks", [])
        peer_rows: List[Dict[str, Any]] = []
        for peer in peers:
            peer_rows.append({
                "Peer": peer.get("name", ""),
                "Sector": peer.get("sector", ""),
                "Emissions (tCO2e)": peer.get("emissions_tco2e", 0),
                "Intensity": peer.get("intensity", ""),
                "Target Year": peer.get("target_year", ""),
                "Reduction (%)": peer.get("reduction_pct", 0),
                "SBTi Validated": "Yes" if peer.get("sbti_validated") else "No",
            })
        sheets["Peer Benchmarks"] = peer_rows

        # Sheet 4: Investment Phases
        phases = data.get("investment_phases", [])
        phase_rows: List[Dict[str, Any]] = []
        for phase in phases:
            phase_rows.append({
                "Phase": phase.get("phase", ""),
                "Period": phase.get("period", ""),
                "Investment (USD)": phase.get("investment_usd", 0),
                "Focus Areas": phase.get("focus", ""),
                "Expected Reduction (%)": phase.get("expected_reduction_pct", 0),
            })
        sheets["Investment Phases"] = phase_rows

        # Sheet 5: Technology Adoption
        tech = data.get("technology_adoption", [])
        tech_rows: List[Dict[str, Any]] = []
        for t in tech:
            tech_rows.append({
                "Technology": t.get("technology", ""),
                "TRL": t.get("trl", ""),
                "Adoption Stage": t.get("adoption_stage", ""),
                "Expected Commercial Date": t.get("commercial_date", ""),
                "Impact (tCO2e)": t.get("impact_tco2e", 0),
                "Cost Trajectory": t.get("cost_trajectory", ""),
            })
        sheets["Technology Adoption"] = tech_rows

        return sheets

    # ------------------------------------------------------------------ #
    #  Markdown sections                                                   #
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        sector = data.get("sector", "General")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Race to Zero -- Sector Pathway Roadmap\n\n"
            f"**Organization:** {org}  \n"
            f"**Sector:** {sector}  \n"
            f"**Pathway Period:** 2025-2050  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_sector_overview(self, data: Dict[str, Any]) -> str:
        sector = data.get("sector", "General")
        pathway_ref = self.SECTOR_PATHWAYS.get(sector, self.SECTOR_PATHWAYS["General"])
        baseline = data.get("baseline_tco2e", 0)
        sector_info = data.get("sector_info", {})

        return (
            f"## 1. Sector Overview\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Sector | {sector} |\n"
            f"| Reference Pathway | {pathway_ref['pathway']} |\n"
            f"| Sector Interim Reduction (2030) | {_pct(pathway_ref['interim_reduction'])} |\n"
            f"| Key Technologies | {pathway_ref['key_tech']} |\n"
            f"| Organization Baseline | {_dec_comma(baseline)} tCO2e |\n"
            f"| Global Sector Emissions | {_dec_comma(sector_info.get('global_emissions_mtco2e', 0))} MtCO2e |\n"
            f"| Sector Share of Global | {_pct(sector_info.get('global_share_pct', 0))} |\n"
            f"| Sector Decarbonization Difficulty | {sector_info.get('difficulty', 'Medium')} |"
        )

    def _md_emissions_trajectory(self, data: Dict[str, Any]) -> str:
        trajectory = data.get("trajectory", [])
        lines = [
            "## 2. Emissions Trajectory (2025-2050)\n",
        ]

        if trajectory:
            lines.extend([
                "| Year | Sector Pathway | Org Target | Actual | Reduction (%) | On Track |",
                "|:----:|---------------:|-----------:|-------:|:-------------:|:--------:|",
            ])
            for t in trajectory:
                actual = t.get("actual_tco2e", "")
                on_track = ""
                if actual != "":
                    on_track = "YES" if float(actual) <= float(t.get("org_target_tco2e", 0)) else "NO"
                    actual_str = _dec_comma(actual)
                else:
                    actual_str = "--"
                    on_track = "--"
                lines.append(
                    f"| {t.get('year', '-')} "
                    f"| {_dec_comma(t.get('sector_target_tco2e', 0))} "
                    f"| {_dec_comma(t.get('org_target_tco2e', 0))} "
                    f"| {actual_str} "
                    f"| {_pct(t.get('reduction_pct', 0))} "
                    f"| {on_track} |"
                )
        else:
            lines.append("_Trajectory data to be populated from sector pathway model._")

        return "\n".join(lines)

    def _md_milestones_tech(self, data: Dict[str, Any]) -> str:
        tech = data.get("technology_adoption", [])
        milestones = data.get("sector_milestones", [])

        lines = [
            "## 3. Key Milestones & Technology Adoption\n",
        ]

        if milestones:
            lines.extend([
                "### Sector Milestones\n",
                "| Year | Milestone | Impact | Confidence |",
                "|:----:|-----------|--------|:----------:|",
            ])
            for ms in milestones:
                lines.append(
                    f"| {ms.get('year', '-')} | {ms.get('milestone', '-')} "
                    f"| {ms.get('impact', '-')} | {ms.get('confidence', 'Medium')} |"
                )

        if tech:
            lines.extend([
                "\n### Technology Adoption Curve\n",
                "| Technology | TRL | Stage | Commercial | Impact (tCO2e) | Cost Trend |",
                "|------------|:---:|-------|:----------:|---------------:|------------|",
            ])
            for t in tech:
                lines.append(
                    f"| {t.get('technology', '-')} | {t.get('trl', '-')} "
                    f"| {t.get('adoption_stage', '-')} | {t.get('commercial_date', '-')} "
                    f"| {_dec_comma(t.get('impact_tco2e', 0))} "
                    f"| {t.get('cost_trajectory', '-')} |"
                )
        else:
            sector = data.get("sector", "General")
            pathway_ref = self.SECTOR_PATHWAYS.get(sector, self.SECTOR_PATHWAYS["General"])
            lines.append(f"\nKey technologies for {sector}: {pathway_ref['key_tech']}")

        return "\n".join(lines)

    def _md_abatement_ranking(self, data: Dict[str, Any]) -> str:
        levers = data.get("abatement_levers", [])
        lines = [
            "## 4. Abatement Priority Ranking (MAC Analysis)\n",
        ]

        if levers:
            # Sort by cost per tCO2e for MAC curve ordering
            sorted_levers = sorted(levers, key=lambda x: x.get("cost_per_tco2e", 0))
            total_reduction = sum(l.get("reduction_tco2e", 0) for l in sorted_levers)
            total_capex = sum(l.get("capex_usd", 0) for l in sorted_levers)

            lines.extend([
                f"**Total Abatement Potential:** {_dec_comma(total_reduction)} tCO2e  \n"
                f"**Total Investment Required:** ${_dec_comma(total_capex)}\n",
                "| Rank | Lever | Reduction (tCO2e) | Cost ($/tCO2e) | CAPEX | Maturity | Timeline |",
                "|:----:|-------|------------------:|---------------:|------:|----------|:--------:|",
            ])
            cumulative = 0
            for i, lever in enumerate(sorted_levers, 1):
                reduction = lever.get("reduction_tco2e", 0)
                cumulative += reduction
                lines.append(
                    f"| {i} | {lever.get('lever', '-')} "
                    f"| {_dec_comma(reduction)} "
                    f"| ${_dec(lever.get('cost_per_tco2e', 0))} "
                    f"| ${_dec_comma(lever.get('capex_usd', 0))} "
                    f"| {lever.get('maturity', '-')} "
                    f"| {lever.get('timeline', '-')} |"
                )

            # Cumulative summary
            lines.append(
                f"\n**Cumulative Abatement:** {_dec_comma(cumulative)} tCO2e "
                f"(**{_pct(_safe_div(cumulative, max(data.get('baseline_tco2e', 1), 1)) * 100)}** of baseline)"
            )
        else:
            lines.append("_Abatement lever analysis to be conducted._")

        return "\n".join(lines)

    def _md_peer_benchmark(self, data: Dict[str, Any]) -> str:
        peers = data.get("peer_benchmarks", [])
        org_name = data.get("org_name", "Organization")

        lines = [
            "## 5. Peer Benchmark Comparison\n",
        ]

        if peers:
            lines.extend([
                "| Company | Emissions (tCO2e) | Intensity | Target Year | Reduction | SBTi | R2Z |",
                "|---------|------------------:|-----------|:-----------:|:---------:|:----:|:---:|",
            ])
            for peer in peers:
                is_self = peer.get("name", "") == org_name
                prefix = "**" if is_self else ""
                lines.append(
                    f"| {prefix}{peer.get('name', '-')}{prefix} "
                    f"| {prefix}{_dec_comma(peer.get('emissions_tco2e', 0))}{prefix} "
                    f"| {peer.get('intensity', '-')} "
                    f"| {peer.get('target_year', '-')} "
                    f"| {_pct(peer.get('reduction_pct', 0))} "
                    f"| {'Yes' if peer.get('sbti_validated') else 'No'} "
                    f"| {'Yes' if peer.get('race_to_zero') else 'No'} |"
                )
        else:
            lines.append("_Peer benchmark data to be populated._")

        return "\n".join(lines)

    def _md_investment_requirements(self, data: Dict[str, Any]) -> str:
        phases = data.get("investment_phases", [])
        lines = [
            "## 6. Investment Requirements by Phase\n",
        ]

        if phases:
            total_investment = sum(p.get("investment_usd", 0) for p in phases)
            lines.extend([
                f"**Total Investment (2025-2050):** ${_dec_comma(total_investment)}\n",
                "| Phase | Period | Investment (USD) | % of Total | Focus Areas | Expected Reduction |",
                "|-------|:------:|----------------:|:----------:|-------------|:------------------:|",
            ])
            for phase in phases:
                inv = phase.get("investment_usd", 0)
                lines.append(
                    f"| {phase.get('phase', '-')} "
                    f"| {phase.get('period', '-')} "
                    f"| ${_dec_comma(inv)} "
                    f"| {_pct(_safe_div(inv, max(total_investment, 1)) * 100)} "
                    f"| {phase.get('focus', '-')} "
                    f"| {_pct(phase.get('expected_reduction_pct', 0))} |"
                )
        else:
            lines.extend([
                "### Indicative Phase Structure\n",
                "| Phase | Period | Focus |",
                "|-------|:------:|-------|",
                "| Foundation | 2025-2027 | Quick wins, data quality, governance |",
                "| Acceleration | 2028-2030 | Major reduction projects, technology deployment |",
                "| Deep Decarb | 2031-2040 | Hard-to-abate sectors, innovation |",
                "| Net-Zero | 2041-2050 | Residual emissions, carbon removals |",
            ])

        return "\n".join(lines)

    def _md_sector_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("sector_risks", [])
        lines = ["## 7. Sector-Specific Risks\n"]

        if risks:
            lines.extend([
                "| # | Risk | Category | Likelihood | Impact | Mitigation |",
                "|---|------|----------|:----------:|:------:|------------|",
            ])
            for i, risk in enumerate(risks, 1):
                lines.append(
                    f"| {i} | {risk.get('risk', '-')} "
                    f"| {risk.get('category', '-')} "
                    f"| {risk.get('likelihood', '-')} "
                    f"| {risk.get('impact', '-')} "
                    f"| {risk.get('mitigation', '-')} |"
                )
        else:
            lines.append("_Sector-specific risk assessment to be conducted._")

        return "\n".join(lines)

    def _md_policy_landscape(self, data: Dict[str, Any]) -> str:
        policies = data.get("policy_landscape", [])
        lines = ["## 8. Policy & Regulatory Landscape\n"]

        if policies:
            lines.extend([
                "| Regulation | Jurisdiction | Status | Impact | Timeline |",
                "|------------|-------------|:------:|--------|:--------:|",
            ])
            for p in policies:
                lines.append(
                    f"| {p.get('regulation', '-')} "
                    f"| {p.get('jurisdiction', '-')} "
                    f"| {p.get('status', '-')} "
                    f"| {p.get('impact', '-')} "
                    f"| {p.get('timeline', '-')} |"
                )
        else:
            lines.append("_Policy landscape analysis to be completed._")

        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-025 Race to Zero Pack on {ts}.*  \n"
            f"*Sector pathways referenced from IEA, SBTi SDA, and IPCC AR6.*"
        )

    # ------------------------------------------------------------------ #
    #  HTML sections                                                       #
    # ------------------------------------------------------------------ #

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            "background:#f0f4f0;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;padding-left:12px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".mac-bar{height:24px;border-radius:4px;display:inline-block;margin:2px 0;"
            "min-width:20px;}"
            ".mac-negative{background:linear-gradient(90deg,#43a047,#66bb6a);}"
            ".mac-positive{background:linear-gradient(90deg,#ff9800,#ffb74d);}"
            ".mac-high{background:linear-gradient(90deg,#ef5350,#ef9a9a);}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        sector = data.get("sector", "General")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Race to Zero -- Sector Pathway Roadmap</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Sector:</strong> {sector} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_sector_overview(self, data: Dict[str, Any]) -> str:
        sector = data.get("sector", "General")
        pathway_ref = self.SECTOR_PATHWAYS.get(sector, self.SECTOR_PATHWAYS["General"])
        return (
            f'<h2>1. Sector Overview</h2>\n'
            f'<div class="cards">\n'
            f'  <div class="card"><div class="card-label">Sector</div>'
            f'<div class="card-value">{sector}</div></div>\n'
            f'  <div class="card"><div class="card-label">Pathway</div>'
            f'<div class="card-value">{pathway_ref["pathway"]}</div></div>\n'
            f'  <div class="card"><div class="card-label">2030 Reduction</div>'
            f'<div class="card-value">{_pct(pathway_ref["interim_reduction"])}</div></div>\n'
            f'  <div class="card"><div class="card-label">Baseline</div>'
            f'<div class="card-value">{_dec_comma(data.get("baseline_tco2e", 0))}</div>tCO2e</div>\n'
            f'</div>'
        )

    def _html_trajectory(self, data: Dict[str, Any]) -> str:
        trajectory = data.get("trajectory", [])
        rows = ""
        for t in trajectory:
            actual = t.get("actual_tco2e", "")
            actual_str = _dec_comma(actual) if actual != "" else "--"
            rows += (f'<tr><td>{t.get("year", "-")}</td>'
                     f'<td>{_dec_comma(t.get("org_target_tco2e", 0))}</td>'
                     f'<td>{actual_str}</td>'
                     f'<td>{_pct(t.get("reduction_pct", 0))}</td></tr>\n')
        if not rows:
            rows = '<tr><td colspan="4"><em>Trajectory data pending</em></td></tr>'
        return (
            f'<h2>2. Emissions Trajectory</h2>\n'
            f'<table><tr><th>Year</th><th>Target (tCO2e)</th><th>Actual</th><th>Reduction</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_milestones(self, data: Dict[str, Any]) -> str:
        tech = data.get("technology_adoption", [])
        rows = ""
        for t in tech:
            rows += (f'<tr><td>{t.get("technology", "-")}</td>'
                     f'<td>TRL {t.get("trl", "-")}</td>'
                     f'<td>{t.get("adoption_stage", "-")}</td>'
                     f'<td>{_dec_comma(t.get("impact_tco2e", 0))}</td></tr>\n')
        if not rows:
            rows = '<tr><td colspan="4"><em>Technology data pending</em></td></tr>'
        return (
            f'<h2>3. Technology Adoption</h2>\n'
            f'<table><tr><th>Technology</th><th>TRL</th><th>Stage</th><th>Impact (tCO2e)</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_abatement(self, data: Dict[str, Any]) -> str:
        levers = data.get("abatement_levers", [])
        sorted_levers = sorted(levers, key=lambda x: x.get("cost_per_tco2e", 0))
        rows = ""
        for lever in sorted_levers:
            cost = lever.get("cost_per_tco2e", 0)
            css_class = "mac-negative" if cost < 0 else ("mac-positive" if cost < 100 else "mac-high")
            width = min(abs(cost) / 2, 100)
            rows += (
                f'<tr><td>{lever.get("lever", "-")}</td>'
                f'<td>{_dec_comma(lever.get("reduction_tco2e", 0))}</td>'
                f'<td>${_dec(cost)}/tCO2e '
                f'<span class="mac-bar {css_class}" style="width:{width}px"></span></td></tr>\n'
            )
        if not rows:
            rows = '<tr><td colspan="3"><em>MAC analysis pending</em></td></tr>'
        return (
            f'<h2>4. Abatement Ranking</h2>\n'
            f'<table><tr><th>Lever</th><th>Reduction (tCO2e)</th><th>Marginal Cost</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_benchmark(self, data: Dict[str, Any]) -> str:
        peers = data.get("peer_benchmarks", [])
        rows = ""
        for peer in peers:
            rows += (f'<tr><td>{peer.get("name", "-")}</td>'
                     f'<td>{_dec_comma(peer.get("emissions_tco2e", 0))}</td>'
                     f'<td>{_pct(peer.get("reduction_pct", 0))}</td>'
                     f'<td>{"Yes" if peer.get("sbti_validated") else "No"}</td></tr>\n')
        if not rows:
            rows = '<tr><td colspan="4"><em>Benchmark data pending</em></td></tr>'
        return (
            f'<h2>5. Peer Benchmarks</h2>\n'
            f'<table><tr><th>Company</th><th>Emissions</th><th>Reduction</th><th>SBTi</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_investment(self, data: Dict[str, Any]) -> str:
        phases = data.get("investment_phases", [])
        rows = ""
        for phase in phases:
            rows += (f'<tr><td>{phase.get("phase", "-")}</td><td>{phase.get("period", "-")}</td>'
                     f'<td>${_dec_comma(phase.get("investment_usd", 0))}</td>'
                     f'<td>{_pct(phase.get("expected_reduction_pct", 0))}</td></tr>\n')
        if not rows:
            rows = '<tr><td colspan="4"><em>Investment data pending</em></td></tr>'
        return (
            f'<h2>6. Investment Requirements</h2>\n'
            f'<table><tr><th>Phase</th><th>Period</th><th>Investment</th><th>Reduction</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("sector_risks", [])
        rows = ""
        for risk in risks:
            rows += (f'<tr><td>{risk.get("risk", "-")}</td>'
                     f'<td>{risk.get("likelihood", "-")}</td>'
                     f'<td>{risk.get("impact", "-")}</td>'
                     f'<td>{risk.get("mitigation", "-")}</td></tr>\n')
        if not rows:
            rows = '<tr><td colspan="4"><em>Risk data pending</em></td></tr>'
        return (
            f'<h2>7. Sector Risks</h2>\n'
            f'<table><tr><th>Risk</th><th>Likelihood</th><th>Impact</th><th>Mitigation</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-025 Race to Zero Pack on {ts}'
            f'</div>'
        )
