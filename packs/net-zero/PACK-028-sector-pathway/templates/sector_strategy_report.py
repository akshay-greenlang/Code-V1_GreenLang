# -*- coding: utf-8 -*-
"""
SectorStrategyReportTemplate - Executive sector transition strategy for PACK-028.

Renders a comprehensive executive-level sector transition strategy document
consolidating pathway analysis, technology roadmap, investment priorities,
benchmarking, risk assessment, and implementation roadmap into a board-ready
report. Multi-format (MD, HTML, JSON, PDF).

Sections:
    1.  Executive Summary (CEO-level)
    2.  Strategic Context
    3.  Sector Pathway Summary
    4.  Key Findings & Insights
    5.  Technology Strategy
    6.  Investment Priorities
    7.  Abatement Strategy
    8.  Competitive Position
    9.  Risk Assessment & Mitigation
    10. Implementation Roadmap
    11. Governance & Accountability
    12. KPIs & Milestones
    13. Board Recommendations
    14. XBRL Tagging Summary
    15. Audit Trail & Provenance

Author: GreenLang Team
Version: 28.0.0
Pack: PACK-028 Sector Pathway Pack
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

_MODULE_VERSION = "28.0.0"
_PACK_ID = "PACK-028"
_TEMPLATE_ID = "sector_strategy_report"

_PRIMARY = "#0a4a3a"
_SECONDARY = "#167a5b"
_ACCENT = "#1db954"
_LIGHT = "#e0f5ed"
_LIGHTER = "#f0faf5"
_SUCCESS = "#2e7d32"
_WARN = "#ef6c00"
_DANGER = "#c62828"

STRATEGY_PILLARS: List[Dict[str, str]] = [
    {"id": "pathway", "name": "Sector Pathway Alignment", "icon": "Target"},
    {"id": "technology", "name": "Technology Transition", "icon": "Innovation"},
    {"id": "investment", "name": "Climate Investment", "icon": "Finance"},
    {"id": "operations", "name": "Operational Excellence", "icon": "Operations"},
    {"id": "engagement", "name": "Stakeholder Engagement", "icon": "People"},
    {"id": "governance", "name": "Climate Governance", "icon": "Governance"},
]

XBRL_STRATEGY_TAGS: Dict[str, str] = {
    "strategy_score": "gl:SectorStrategyOverallScore",
    "investment_total": "gl:TotalClimateInvestment",
    "target_year": "gl:NetZeroTargetYear",
    "pathway_alignment": "gl:PathwayAlignmentPercentage",
    "risk_level": "gl:OverallClimateRiskLevel",
    "implementation_phase": "gl:CurrentImplementationPhase",
}

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
        ip = parts[0]
        neg = ip.startswith("-")
        if neg: ip = ip[1:]
        fmt = ""
        for i, ch in enumerate(reversed(ip)):
            if i > 0 and i % 3 == 0: fmt = "," + fmt
            fmt = ch + fmt
        if neg: fmt = "-" + fmt
        if len(parts) > 1: fmt += "." + parts[1]
        return fmt
    except Exception:
        return str(val)

def _rag(status: str) -> str:
    return {"green": "GREEN", "amber": "AMBER", "red": "RED"}.get(status.lower(), status.upper())

class SectorStrategyReportTemplate:
    """
    Executive sector transition strategy report template.

    Consolidates all PACK-028 analyses into a board-ready strategy document
    with strategic recommendations, investment priorities, implementation
    roadmap, and governance framework. Supports MD, HTML, JSON, and PDF.

    Example:
        >>> template = SectorStrategyReportTemplate()
        >>> data = {"org_name": "SteelCo", "sector_id": "steel", ...}
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        sections = [
            self._md_header(data), self._md_exec_summary(data),
            self._md_strategic_context(data), self._md_pathway_summary(data),
            self._md_key_findings(data), self._md_tech_strategy(data),
            self._md_investment_priorities(data), self._md_abatement_strategy(data),
            self._md_competitive_position(data), self._md_risk_assessment(data),
            self._md_implementation_roadmap(data), self._md_governance(data),
            self._md_kpis(data), self._md_board_recommendations(data),
            self._md_xbrl(data), self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_exec_summary(data),
            self._html_strategic_context(data), self._html_pathway_summary(data),
            self._html_key_findings(data), self._html_tech_strategy(data),
            self._html_investment_priorities(data), self._html_abatement_strategy(data),
            self._html_competitive_position(data), self._html_risk_assessment(data),
            self._html_implementation_roadmap(data), self._html_governance(data),
            self._html_kpis(data), self._html_board_recommendations(data),
            self._html_xbrl(data), self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Sector Strategy - {data.get("org_name","")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        result = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION,
            "pack_id": _PACK_ID, "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(), "org_name": data.get("org_name", ""),
            "sector_id": data.get("sector_id", ""),
            "strategy_pillars": STRATEGY_PILLARS,
            "executive_summary": data.get("executive_summary", {}),
            "pathway_summary": data.get("pathway_summary", {}),
            "key_findings": data.get("key_findings", []),
            "technology_strategy": data.get("technology_strategy", {}),
            "investment_priorities": data.get("investment_priorities", []),
            "abatement_strategy": data.get("abatement_strategy", {}),
            "competitive_position": data.get("competitive_position", {}),
            "risks": data.get("risks", []),
            "implementation_roadmap": data.get("implementation_roadmap", []),
            "governance": data.get("governance", {}),
            "kpis": data.get("kpis", []),
            "board_recommendations": data.get("board_recommendations", []),
            "xbrl_tags": {k: XBRL_STRATEGY_TAGS[k] for k in XBRL_STRATEGY_TAGS},
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"format": "pdf", "html_content": self.render_html(data),
                "structured_data": self.render_json(data),
                "metadata": {"title": f"Sector Strategy - {data.get('org_name','')}", "author": "GreenLang PACK-028"}}

    # -- Markdown --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Sector Transition Strategy\n\n"
            f"**Organization:** {data.get('org_name','')}  \n"
            f"**Sector:** {data.get('sector_id','').replace('_',' ').title()}  \n"
            f"**Classification:** CONFIDENTIAL - Board Document  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-028 Sector Pathway Pack v{_MODULE_VERSION}\n\n---"
        )

    def _md_exec_summary(self, data: Dict[str, Any]) -> str:
        es = data.get("executive_summary", {})
        lines = [
            "## 1. Executive Summary\n",
            f"{es.get('narrative', 'This document presents the sector-specific decarbonization strategy, outlining the pathway to net-zero aligned with science-based targets.')}\n",
            f"| Strategic KPI | Value | Status |",
            f"|--------------|-------|--------|",
            f"| Sector Pathway | {es.get('pathway_scenario', '1.5C NZE')} | {_rag(es.get('pathway_status', 'amber'))} |",
            f"| Current Intensity Gap | {_dec(es.get('gap_pct', 0))}% | {_rag(es.get('gap_status', 'amber'))} |",
            f"| SBTi Readiness | {_dec(es.get('sbti_readiness_pct', 0))}% | {_rag(es.get('sbti_status', 'amber'))} |",
            f"| Total Investment Required | EUR {_dec_comma(es.get('total_investment', 0))} | - |",
            f"| Net-Zero Target Year | {es.get('target_year', 2050)} | - |",
            f"| Sector Percentile | {es.get('percentile', 'N/A')} | - |",
            f"| Abatement Potential | {_dec_comma(es.get('abatement_potential', 0))} tCO2e | - |",
        ]
        return "\n".join(lines)

    def _md_strategic_context(self, data: Dict[str, Any]) -> str:
        ctx = data.get("strategic_context", {})
        lines = [
            "## 2. Strategic Context\n",
            f"### Regulatory Landscape\n{ctx.get('regulatory', 'Key regulations include SBTi Corporate Standard, EU CSRD/ESRS E1, SEC Climate Rule, and sector-specific requirements.')}\n",
            f"### Market Dynamics\n{ctx.get('market', 'Carbon pricing, green premiums, and investor expectations are reshaping the competitive landscape.')}\n",
            f"### Sector Challenges\n{ctx.get('challenges', 'Sector-specific challenges include technology maturity, supply chain dependencies, and capital intensity.')}\n",
            "### Strategy Pillars\n",
            "| # | Pillar | Focus Area |",
            "|---|--------|-----------|",
        ]
        for i, p in enumerate(STRATEGY_PILLARS, 1):
            lines.append(f"| {i} | {p['name']} | {p['icon']} |")
        return "\n".join(lines)

    def _md_pathway_summary(self, data: Dict[str, Any]) -> str:
        ps = data.get("pathway_summary", {})
        lines = [
            "## 3. Sector Pathway Summary\n",
            f"| Parameter | Value |", f"|-----------|-------|",
            f"| Sector | {data.get('sector_id', '').replace('_', ' ').title()} |",
            f"| SDA Methodology | {ps.get('sda_method', 'N/A')} |",
            f"| Scenario | {ps.get('scenario', 'NZE 1.5C')} |",
            f"| Base Year Intensity | {_dec(ps.get('base_intensity', 0), 4)} |",
            f"| Current Intensity | {_dec(ps.get('current_intensity', 0), 4)} |",
            f"| 2030 Target | {_dec(ps.get('target_2030', 0), 4)} |",
            f"| 2050 Target | {_dec(ps.get('target_2050', 0), 4)} |",
            f"| Gap to Pathway | {_dec(ps.get('gap_pct', 0))}% |",
            f"| Required Annual Reduction | {_dec(ps.get('required_rate', 0))}% |",
            f"| Convergence Method | {ps.get('convergence_method', 'Linear')} |",
        ]
        return "\n".join(lines)

    def _md_key_findings(self, data: Dict[str, Any]) -> str:
        findings = data.get("key_findings", [])
        lines = ["## 4. Key Findings & Insights\n"]
        if findings:
            for i, f in enumerate(findings, 1):
                severity = f.get("severity", "medium")
                lines.append(
                    f"{i}. **{f.get('finding', '')}** [{severity.upper()}]  \n"
                    f"   {f.get('implication', '')}  \n"
                    f"   _Action: {f.get('action', '')}_\n"
                )
        else:
            lines.extend([
                "1. **Intensity gap requires acceleration** - Current trajectory insufficient for 1.5C alignment",
                "2. **Technology readiness varies** - Key technologies at TRL 6-8; phased deployment recommended",
                "3. **Investment front-loading needed** - 60% of CapEx required by 2030 for pathway compliance",
                "4. **Competitive advantage opportunity** - Early movers capturing green premiums",
                "5. **Regulatory tightening accelerating** - EU ETS reform and CBAM increasing urgency",
            ])
        return "\n".join(lines)

    def _md_tech_strategy(self, data: Dict[str, Any]) -> str:
        ts = data.get("technology_strategy", {})
        priorities = ts.get("priorities", [])
        lines = [
            "## 5. Technology Strategy\n",
            f"**Technology Readiness:** {ts.get('overall_readiness', 'Moderate')}  \n"
            f"**Key Technologies:** {ts.get('key_tech_count', 0)}  \n"
            f"**Investment in Technology:** EUR {_dec_comma(ts.get('tech_investment', 0))}\n",
        ]
        if priorities:
            lines.append("| Priority | Technology | Phase | TRL | Investment (EUR) | Impact |")
            lines.append("|----------|-----------|-------|----:|----------------:|--------|")
            for i, p in enumerate(priorities, 1):
                lines.append(
                    f"| {i} | {p.get('technology', '')} | {p.get('phase', '')} "
                    f"| {p.get('trl', 0)} | {_dec_comma(p.get('investment', 0))} | {p.get('impact', '')} |"
                )
        return "\n".join(lines)

    def _md_investment_priorities(self, data: Dict[str, Any]) -> str:
        inv = data.get("investment_priorities", [])
        lines = [
            "## 6. Investment Priorities\n",
            "| Priority | Category | Amount (EUR) | Phase | ROI | Payback |",
            "|----------|----------|-------------:|-------|----:|--------:|",
        ]
        total = 0
        for i, item in enumerate(inv, 1):
            amt = float(item.get("amount", 0))
            total += amt
            lines.append(
                f"| {i} | {item.get('category', '')} | {_dec_comma(amt)} "
                f"| {item.get('phase', '')} | {_dec(item.get('roi_pct', 0))}% "
                f"| {_dec(item.get('payback_years', 0), 1)} yr |"
            )
        if inv:
            lines.append(f"| | **Total** | **EUR {_dec_comma(total)}** | | | |")
        return "\n".join(lines)

    def _md_abatement_strategy(self, data: Dict[str, Any]) -> str:
        ab = data.get("abatement_strategy", {})
        levers = ab.get("top_levers", [])
        lines = [
            "## 7. Abatement Strategy\n",
            f"**Total Abatement Potential:** {_dec_comma(ab.get('total_abatement', 0))} tCO2e  \n"
            f"**Quick Wins:** {ab.get('quick_win_count', 0)} levers  \n"
            f"**Average MAC:** EUR {_dec(ab.get('avg_mac', 0))}/tCO2e\n",
        ]
        if levers:
            lines.append("| Rank | Lever | Abatement (tCO2e) | MAC (EUR/tCO2e) | Phase |")
            lines.append("|------|-------|------------------:|----------------:|-------|")
            for i, l in enumerate(levers, 1):
                lines.append(
                    f"| {i} | {l.get('name', '')} | {_dec_comma(l.get('abatement', 0))} "
                    f"| {_dec(l.get('mac', 0))} | {l.get('phase', '')} |"
                )
        return "\n".join(lines)

    def _md_competitive_position(self, data: Dict[str, Any]) -> str:
        cp = data.get("competitive_position", {})
        lines = [
            "## 8. Competitive Position\n",
            f"| Dimension | Your Position | Sector Average | Leader | Quartile |",
            f"|-----------|--------------|:--------------:|:------:|----------|",
            f"| Intensity | {_dec(cp.get('your_intensity', 0), 4)} | {_dec(cp.get('avg_intensity', 0), 4)} | {_dec(cp.get('leader_intensity', 0), 4)} | {cp.get('quartile', 'Q2')} |",
            f"| Reduction Rate | {_dec(cp.get('your_rate', 0))}%/yr | {_dec(cp.get('avg_rate', 0))}%/yr | {_dec(cp.get('leader_rate', 0))}%/yr | {cp.get('rate_quartile', 'Q2')} |",
            f"| Technology | {cp.get('your_tech', 'Moderate')} | {cp.get('avg_tech', 'Moderate')} | {cp.get('leader_tech', 'Advanced')} | {cp.get('tech_quartile', 'Q2')} |",
            f"| Disclosure | {cp.get('your_disclosure', 'Medium')} | {cp.get('avg_disclosure', 'Medium')} | {cp.get('leader_disclosure', 'High')} | {cp.get('disc_quartile', 'Q2')} |",
        ]
        return "\n".join(lines)

    def _md_risk_assessment(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        lines = [
            "## 9. Risk Assessment & Mitigation\n",
            "| Risk | Category | Likelihood | Impact | Mitigation | Owner |",
            "|------|----------|-----------|--------|-----------|-------|",
        ]
        for r in risks:
            lines.append(
                f"| {r.get('risk', '')} | {r.get('category', '')} "
                f"| {r.get('likelihood', 'Medium')} | {r.get('impact', 'Medium')} "
                f"| {r.get('mitigation', '')} | {r.get('owner', '')} |"
            )
        if not risks:
            lines.extend([
                "| Transition risk (stranded assets) | Strategic | Medium | High | Accelerated asset depreciation | CFO |",
                "| Technology risk (TRL delays) | Technology | Medium | Medium | Multi-technology hedging | CTO |",
                "| Regulatory risk (tightening) | Regulatory | High | High | Active policy engagement | Legal |",
                "| Market risk (green premium erosion) | Market | Low | Medium | Cost reduction roadmap | COO |",
                "| Physical risk (climate impacts) | Physical | Medium | High | Adaptation planning | Risk |",
            ])
        return "\n".join(lines)

    def _md_implementation_roadmap(self, data: Dict[str, Any]) -> str:
        roadmap = data.get("implementation_roadmap", [])
        lines = [
            "## 10. Implementation Roadmap\n",
        ]
        if roadmap:
            lines.append("| Phase | Period | Key Actions | Investment | Milestone | Owner |")
            lines.append("|-------|--------|-------------|------------|-----------|-------|")
            for r in roadmap:
                lines.append(
                    f"| {r.get('phase', '')} | {r.get('period', '')} | {r.get('actions', '')} "
                    f"| EUR {_dec_comma(r.get('investment', 0))} | {r.get('milestone', '')} | {r.get('owner', '')} |"
                )
        else:
            lines.extend([
                "| Phase 1: Foundation | 2025-2026 | Baseline validation, target setting, governance | EUR TBD | SBTi submission | CSO |",
                "| Phase 2: Quick Wins | 2026-2027 | Efficiency upgrades, renewable procurement | EUR TBD | 10% reduction | COO |",
                "| Phase 3: Transition | 2027-2030 | Technology deployment, process changes | EUR TBD | Near-term target | CTO |",
                "| Phase 4: Acceleration | 2030-2040 | Scale-up, supply chain transformation | EUR TBD | 50% pathway | CEO |",
                "| Phase 5: Net Zero | 2040-2050 | Final technology, residual neutralization | EUR TBD | Net-zero | Board |",
            ])
        return "\n".join(lines)

    def _md_governance(self, data: Dict[str, Any]) -> str:
        gov = data.get("governance", {})
        lines = [
            "## 11. Governance & Accountability\n",
            f"| Role | Responsibility | Reporting Frequency |",
            f"|------|---------------|-------------------|",
            f"| Board of Directors | Strategic oversight, target approval | Quarterly |",
            f"| CEO | Overall accountability, strategy direction | Monthly |",
            f"| CSO / Chief Sustainability Officer | Strategy execution, reporting | Monthly |",
            f"| CFO | Investment allocation, financial risk | Quarterly |",
            f"| CTO | Technology strategy, R&D | Monthly |",
            f"| COO | Operational execution | Weekly |",
        ]
        committees = gov.get("committees", [])
        if committees:
            lines.append("\n### Committees\n")
            for c in committees:
                lines.append(f"- **{c.get('name', '')}**: {c.get('mandate', '')} ({c.get('frequency', 'Quarterly')})")
        return "\n".join(lines)

    def _md_kpis(self, data: Dict[str, Any]) -> str:
        kpis = data.get("kpis", [])
        lines = [
            "## 12. KPIs & Milestones\n",
            "| KPI | Current | 2025 Target | 2030 Target | 2050 Target | RAG |",
            "|-----|---------|------------|------------|------------|-----|",
        ]
        for k in kpis:
            lines.append(
                f"| {k.get('name', '')} | {k.get('current', '')} "
                f"| {k.get('target_2025', '-')} | {k.get('target_2030', '-')} "
                f"| {k.get('target_2050', '-')} | {_rag(k.get('rag', 'amber'))} |"
            )
        if not kpis:
            lines.extend([
                "| Emission intensity | - | - | - | - | AMBER |",
                "| Absolute emissions | - | - | - | - | AMBER |",
                "| Renewable share | - | - | - | - | GREEN |",
                "| SBTi alignment | - | - | - | - | AMBER |",
                "| Technology deployment | - | - | - | - | AMBER |",
                "| Climate investment | - | - | - | - | AMBER |",
            ])
        return "\n".join(lines)

    def _md_board_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("board_recommendations", [])
        lines = ["## 13. Board Recommendations\n"]
        if recs:
            lines.append("### Decisions Required\n")
            for i, r in enumerate(recs, 1):
                lines.append(
                    f"{i}. **{r.get('title', '')}**  \n"
                    f"   {r.get('description', '')}  \n"
                    f"   _Investment: EUR {_dec_comma(r.get('investment', 0))} | "
                    f"Timeline: {r.get('timeline', 'TBD')} | "
                    f"Priority: {r.get('priority', 'High')}_\n"
                )
        else:
            lines.extend([
                "### Decisions Required\n",
                "1. **Approve SBTi target submission** - Submit SDA-aligned sector targets for validation",
                "2. **Approve Phase 1 investment** - EUR [TBD] for foundation and quick-win initiatives",
                "3. **Establish Climate Committee** - Board-level governance for transition oversight",
                "4. **Mandate technology assessment** - Detailed feasibility study for key technologies",
                "5. **Set internal carbon price** - EUR [TBD]/tCO2e for investment decision-making",
            ])
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        es = data.get("executive_summary", {})
        lines = [
            "## 14. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |", "|------------|----------|-------|",
            f"| Total Investment | {XBRL_STRATEGY_TAGS['investment_total']} | EUR {_dec_comma(es.get('total_investment', 0))} |",
            f"| Target Year | {XBRL_STRATEGY_TAGS['target_year']} | {es.get('target_year', 2050)} |",
            f"| Pathway Alignment | {XBRL_STRATEGY_TAGS['pathway_alignment']} | {_dec(es.get('pathway_alignment_pct', 0))}% |",
        ]
        return "\n".join(lines)

    def _md_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f"## 15. Audit Trail\n\n| Parameter | Value |\n|-----------|-------|\n| Report ID | `{rid}` |\n| Generated | {ts} |\n| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n| Hash | `{dh[:16]}...` |"

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-028 Sector Pathway Pack on {ts}*  \n"
            f"*Executive sector transition strategy - CONFIDENTIAL*  \n"
            f"*All calculations deterministic - zero LLM in computation path.*"
        )

    # -- HTML --

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;font-size:1.8em;}}"
            f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"h3{{color:{_ACCENT};}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f9fbe7;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.5em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.75em;color:{_ACCENT};}}"
            f".rag-green{{background:#c8e6c9;color:#1b5e20;font-weight:700;padding:4px 12px;border-radius:4px;}}"
            f".rag-amber{{background:#fff3e0;color:#e65100;font-weight:700;padding:4px 12px;border-radius:4px;}}"
            f".rag-red{{background:#ffcdd2;color:#c62828;font-weight:700;padding:4px 12px;border-radius:4px;}}"
            f".confidential{{background:#fff3e0;border:2px solid #ef6c00;border-radius:8px;padding:12px;text-align:center;color:#e65100;font-weight:700;margin:20px 0;}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="confidential">CONFIDENTIAL - Board Document</div>\n'
            f'<h1>Sector Transition Strategy</h1>\n'
            f'<p><strong>Organization:</strong> {data.get("org_name","")} | '
            f'<strong>Sector:</strong> {data.get("sector_id","").replace("_"," ").title()} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_exec_summary(self, data: Dict[str, Any]) -> str:
        es = data.get("executive_summary", {})
        gap = float(es.get("gap_pct", 0))
        gap_rag = "rag-green" if gap <= 5 else ("rag-amber" if gap <= 15 else "rag-red")
        return (
            f'<h2>1. Executive Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Pathway</div><div class="card-value">{es.get("pathway_scenario","1.5C")}</div></div>\n'
            f'<div class="card"><div class="card-label">Gap</div><div class="card-value">{_dec(gap)}%</div><div class="card-unit"><span class="{gap_rag}">{_rag(es.get("gap_status","amber"))}</span></div></div>\n'
            f'<div class="card"><div class="card-label">SBTi Ready</div><div class="card-value">{_dec(es.get("sbti_readiness_pct",0))}%</div></div>\n'
            f'<div class="card"><div class="card-label">Investment</div><div class="card-value">EUR {_dec_comma(es.get("total_investment",0))}</div></div>\n'
            f'<div class="card"><div class="card-label">Target Year</div><div class="card-value">{es.get("target_year",2050)}</div></div>\n'
            f'</div>'
        )

    def _html_strategic_context(self, data: Dict[str, Any]) -> str:
        ctx = data.get("strategic_context", {})
        rows = "".join(f'<tr><td>{i}</td><td>{p["name"]}</td><td>{p["icon"]}</td></tr>\n' for i, p in enumerate(STRATEGY_PILLARS, 1))
        return (
            f'<h2>2. Strategic Context</h2>\n'
            f'<p>{ctx.get("regulatory", "Regulatory landscape analysis pending.")}</p>\n'
            f'<table>\n<tr><th>#</th><th>Pillar</th><th>Focus</th></tr>\n{rows}</table>'
        )

    def _html_pathway_summary(self, data: Dict[str, Any]) -> str:
        ps = data.get("pathway_summary", {})
        return (
            f'<h2>3. Pathway Summary</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Base Intensity</td><td>{_dec(ps.get("base_intensity",0),4)}</td></tr>\n'
            f'<tr><td>Current Intensity</td><td>{_dec(ps.get("current_intensity",0),4)}</td></tr>\n'
            f'<tr><td>2030 Target</td><td>{_dec(ps.get("target_2030",0),4)}</td></tr>\n'
            f'<tr><td>2050 Target</td><td>{_dec(ps.get("target_2050",0),4)}</td></tr>\n'
            f'<tr><td>Gap</td><td>{_dec(ps.get("gap_pct",0))}%</td></tr>\n'
            f'<tr><td>Required Rate</td><td>{_dec(ps.get("required_rate",0))}%/yr</td></tr>\n</table>'
        )

    def _html_key_findings(self, data: Dict[str, Any]) -> str:
        findings = data.get("key_findings", [])
        items = "".join(f'<tr><td>{i}</td><td>{f.get("finding","")}</td><td>{f.get("severity","medium").upper()}</td><td>{f.get("action","")}</td></tr>\n' for i, f in enumerate(findings, 1))
        return f'<h2>4. Key Findings</h2>\n<table>\n<tr><th>#</th><th>Finding</th><th>Severity</th><th>Action</th></tr>\n{items}</table>'

    def _html_tech_strategy(self, data: Dict[str, Any]) -> str:
        ts = data.get("technology_strategy", {})
        priorities = ts.get("priorities", [])
        rows = "".join(f'<tr><td>{i}</td><td>{p.get("technology","")}</td><td>{p.get("trl",0)}</td><td>EUR {_dec_comma(p.get("investment",0))}</td><td>{p.get("impact","")}</td></tr>\n' for i, p in enumerate(priorities, 1))
        return f'<h2>5. Technology Strategy</h2>\n<table>\n<tr><th>#</th><th>Technology</th><th>TRL</th><th>Investment</th><th>Impact</th></tr>\n{rows}</table>'

    def _html_investment_priorities(self, data: Dict[str, Any]) -> str:
        inv = data.get("investment_priorities", [])
        rows = "".join(f'<tr><td>{i}</td><td>{item.get("category","")}</td><td>EUR {_dec_comma(item.get("amount",0))}</td><td>{item.get("phase","")}</td><td>{_dec(item.get("roi_pct",0))}%</td></tr>\n' for i, item in enumerate(inv, 1))
        return f'<h2>6. Investment Priorities</h2>\n<table>\n<tr><th>#</th><th>Category</th><th>Amount</th><th>Phase</th><th>ROI</th></tr>\n{rows}</table>'

    def _html_abatement_strategy(self, data: Dict[str, Any]) -> str:
        ab = data.get("abatement_strategy", {})
        levers = ab.get("top_levers", [])
        rows = "".join(f'<tr><td>{i}</td><td>{l.get("name","")}</td><td>{_dec_comma(l.get("abatement",0))}</td><td>EUR {_dec(l.get("mac",0))}</td></tr>\n' for i, l in enumerate(levers, 1))
        return (
            f'<h2>7. Abatement Strategy</h2>\n'
            f'<p><strong>Total:</strong> {_dec_comma(ab.get("total_abatement",0))} tCO2e | <strong>Avg MAC:</strong> EUR {_dec(ab.get("avg_mac",0))}/tCO2e</p>\n'
            f'<table>\n<tr><th>#</th><th>Lever</th><th>Abatement</th><th>MAC</th></tr>\n{rows}</table>'
        )

    def _html_competitive_position(self, data: Dict[str, Any]) -> str:
        cp = data.get("competitive_position", {})
        return (
            f'<h2>8. Competitive Position</h2>\n<table>\n<tr><th>Dimension</th><th>You</th><th>Average</th><th>Leader</th></tr>\n'
            f'<tr><td>Intensity</td><td>{_dec(cp.get("your_intensity",0),4)}</td><td>{_dec(cp.get("avg_intensity",0),4)}</td><td>{_dec(cp.get("leader_intensity",0),4)}</td></tr>\n'
            f'<tr><td>Reduction Rate</td><td>{_dec(cp.get("your_rate",0))}%</td><td>{_dec(cp.get("avg_rate",0))}%</td><td>{_dec(cp.get("leader_rate",0))}%</td></tr>\n'
            f'</table>'
        )

    def _html_risk_assessment(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        rows = "".join(f'<tr><td>{r.get("risk","")}</td><td>{r.get("likelihood","Medium")}</td><td>{r.get("impact","Medium")}</td><td>{r.get("mitigation","")}</td></tr>\n' for r in risks)
        return f'<h2>9. Risk Assessment</h2>\n<table>\n<tr><th>Risk</th><th>Likelihood</th><th>Impact</th><th>Mitigation</th></tr>\n{rows}</table>'

    def _html_implementation_roadmap(self, data: Dict[str, Any]) -> str:
        roadmap = data.get("implementation_roadmap", [])
        rows = "".join(f'<tr><td>{r.get("phase","")}</td><td>{r.get("period","")}</td><td>{r.get("actions","")}</td><td>EUR {_dec_comma(r.get("investment",0))}</td></tr>\n' for r in roadmap)
        return f'<h2>10. Implementation Roadmap</h2>\n<table>\n<tr><th>Phase</th><th>Period</th><th>Actions</th><th>Investment</th></tr>\n{rows}</table>'

    def _html_governance(self, data: Dict[str, Any]) -> str:
        return (
            f'<h2>11. Governance</h2>\n<table>\n<tr><th>Role</th><th>Responsibility</th><th>Frequency</th></tr>\n'
            f'<tr><td>Board</td><td>Strategic oversight</td><td>Quarterly</td></tr>\n'
            f'<tr><td>CEO</td><td>Overall accountability</td><td>Monthly</td></tr>\n'
            f'<tr><td>CSO</td><td>Strategy execution</td><td>Monthly</td></tr>\n'
            f'<tr><td>CFO</td><td>Investment allocation</td><td>Quarterly</td></tr>\n'
            f'</table>'
        )

    def _html_kpis(self, data: Dict[str, Any]) -> str:
        kpis = data.get("kpis", [])
        rows = ""
        for k in kpis:
            rag_val = k.get("rag", "amber")
            rag_cls = f"rag-{rag_val}"
            rows += f'<tr><td>{k.get("name","")}</td><td>{k.get("current","")}</td><td>{k.get("target_2030","")}</td><td>{k.get("target_2050","")}</td><td><span class="{rag_cls}">{_rag(rag_val)}</span></td></tr>\n'
        return f'<h2>12. KPIs</h2>\n<table>\n<tr><th>KPI</th><th>Current</th><th>2030</th><th>2050</th><th>RAG</th></tr>\n{rows}</table>'

    def _html_board_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("board_recommendations", [])
        rows = "".join(f'<tr><td>{i}</td><td><strong>{r.get("title","")}</strong></td><td>{r.get("description","")}</td><td>EUR {_dec_comma(r.get("investment",0))}</td><td>{r.get("priority","High")}</td></tr>\n' for i, r in enumerate(recs, 1))
        return f'<h2>13. Board Recommendations</h2>\n<table>\n<tr><th>#</th><th>Decision</th><th>Description</th><th>Investment</th><th>Priority</th></tr>\n{rows}</table>'

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        es = data.get("executive_summary", {})
        return f'<h2>14. XBRL Tags</h2>\n<table>\n<tr><th>Point</th><th>Tag</th><th>Value</th></tr>\n<tr><td>Investment</td><td><code>{XBRL_STRATEGY_TAGS["investment_total"]}</code></td><td>EUR {_dec_comma(es.get("total_investment",0))}</td></tr>\n<tr><td>Target Year</td><td><code>{XBRL_STRATEGY_TAGS["target_year"]}</code></td><td>{es.get("target_year",2050)}</td></tr>\n</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>15. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n<tr><td>Version</td><td>{_MODULE_VERSION}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-028 Sector Pathway Pack on {ts}<br>Executive sector transition strategy - CONFIDENTIAL</div>'
