# -*- coding: utf-8 -*-
"""
TCFDStrategyTemplate - TCFD Strategy Pillar Template for PACK-030.

Renders TCFD Strategy pillar disclosure covering climate-related risks
and opportunities, scenario analysis (1.5C, 2C, 4C), financial impact
assessment, strategic resilience, and transition planning. Multi-format
output (MD, HTML, JSON, PDF) with SHA-256 provenance hashing.

Sections:
    1.  Executive Summary
    2.  Climate-Related Risks Identified
    3.  Climate-Related Opportunities Identified
    4.  Impact on Business, Strategy & Financial Planning
    5.  Scenario Analysis - Methodology
    6.  Scenario Analysis - Results (1.5C / 2C / 4C)
    7.  Strategic Resilience Assessment
    8.  Transition Plan Summary
    9.  Financial Impact Quantification
    10. XBRL Tagging Summary
    11. Audit Trail & Provenance

Author: GreenLang Team
Version: 30.0.0
Pack: PACK-030 Net Zero Reporting Pack
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

_MODULE_VERSION = "30.0.0"
_PACK_ID = "PACK-030"
_TEMPLATE_ID = "tcfd_strategy"

_PRIMARY = "#1a237e"
_SECONDARY = "#283593"
_ACCENT = "#42a5f5"
_LIGHT = "#e8eaf6"
_LIGHTER = "#f5f5ff"

SCENARIO_DEFINITIONS = {
    "1.5C": {"name": "Net Zero 2050 (1.5C)", "warming": "1.5C", "source": "IEA NZE 2050", "carbon_price_2030": "$130/tCO2", "carbon_price_2050": "$250/tCO2"},
    "2C": {"name": "Announced Pledges (2C)", "warming": "~2C", "source": "IEA APS", "carbon_price_2030": "$90/tCO2", "carbon_price_2050": "$200/tCO2"},
    "4C": {"name": "Stated Policies (4C)", "warming": ">3C", "source": "IEA STEPS", "carbon_price_2030": "$30/tCO2", "carbon_price_2050": "$50/tCO2"},
}

XBRL_TAGS: Dict[str, str] = {
    "risks_count": "gl:TCFDStrategyRisksCount",
    "opportunities_count": "gl:TCFDStrategyOpportunitiesCount",
    "scenario_analysis": "gl:TCFDStrategyScenarioAnalysis",
    "financial_impact": "gl:TCFDStrategyFinancialImpact",
    "resilience_score": "gl:TCFDStrategyResilienceScore",
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

def _dec_comma(val: Any, places: int = 2) -> str:
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

class TCFDStrategyTemplate:
    """
    TCFD Strategy pillar template for PACK-030 Net Zero Reporting Pack.

    Generates TCFD-aligned strategy disclosure covering risks, opportunities,
    scenario analysis, and strategic resilience. Supports MD, HTML, JSON, PDF.

    Example:
        >>> template = TCFDStrategyTemplate()
        >>> data = {
        ...     "org_name": "GreenCorp", "reporting_year": 2025,
        ...     "risks": [{"category": "Emerging regulation", "description": "Carbon tax",
        ...                "time_horizon": "Medium", "financial_impact": "$5M-10M"}],
        ...     "opportunities": [{"category": "Energy source", "description": "Renewables"}],
        ...     "scenarios": {"1.5C": {"revenue_impact": -5}, "2C": {"revenue_impact": -3}},
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render TCFD strategy report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_risks(data), self._md_opportunities(data),
            self._md_business_impact(data), self._md_scenario_methodology(data),
            self._md_scenario_results(data), self._md_resilience(data),
            self._md_transition_plan(data), self._md_financial_impact(data),
            self._md_xbrl(data), self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render TCFD strategy report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_executive_summary(data),
            self._html_risks(data), self._html_opportunities(data),
            self._html_scenario_results(data), self._html_resilience(data),
            self._html_financial_impact(data), self._html_xbrl(data),
            self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>TCFD Strategy - {data.get("org_name", "")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render as structured JSON."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION,
            "pack_id": _PACK_ID, "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "framework": "TCFD", "pillar": "Strategy",
            "risks": data.get("risks", []),
            "opportunities": data.get("opportunities", []),
            "scenarios": data.get("scenarios", {}),
            "scenario_definitions": SCENARIO_DEFINITIONS,
            "transition_plan": data.get("transition_plan", {}),
            "financial_impacts": data.get("financial_impacts", []),
            "resilience_assessment": data.get("resilience_assessment", {}),
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return PDF-ready data."""
        return {
            "format": "pdf", "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {"title": f"TCFD Strategy - {data.get('org_name', '')}", "author": "GreenLang PACK-030", "framework": "TCFD"},
        }

    # -- Markdown sections --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# TCFD Strategy Pillar Disclosure\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Framework:** TCFD | **Pillar:** Strategy  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-030 v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        opps = data.get("opportunities", [])
        scenarios = data.get("scenarios", {})
        lines = [
            "## 1. Executive Summary\n",
            "| Metric | Value |", "|--------|-------|",
            f"| Risks Identified | {len(risks)} |",
            f"| Opportunities Identified | {len(opps)} |",
            f"| Scenarios Analyzed | {len(scenarios)} |",
            f"| Transition Plan | {data.get('transition_plan', {}).get('status', 'In development')} |",
            f"| Financial Impact Quantified | {'Yes' if data.get('financial_impacts') else 'No'} |",
        ]
        return "\n".join(lines)

    def _md_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        lines = [
            "## 2. Climate-Related Risks Identified\n",
            "| # | Category | Type | Description | Time Horizon | Likelihood | Magnitude | Financial Impact |",
            "|---|----------|------|-------------|:------------:|------------|-----------|------------------|",
        ]
        for i, r in enumerate(risks, 1):
            lines.append(
                f"| {i} | {r.get('category', '')} | {r.get('type', 'Transition')} "
                f"| {r.get('description', '')} | {r.get('time_horizon', '')} "
                f"| {r.get('likelihood', '')} | {r.get('magnitude', '')} "
                f"| {r.get('financial_impact', '')} |"
            )
        if not risks:
            lines.append("| - | _No risks disclosed_ | - | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_opportunities(self, data: Dict[str, Any]) -> str:
        opps = data.get("opportunities", [])
        lines = [
            "## 3. Climate-Related Opportunities Identified\n",
            "| # | Category | Description | Time Horizon | Financial Impact | Strategy |",
            "|---|----------|-------------|:------------:|------------------|----------|",
        ]
        for i, o in enumerate(opps, 1):
            lines.append(
                f"| {i} | {o.get('category', '')} | {o.get('description', '')} "
                f"| {o.get('time_horizon', '')} | {o.get('financial_impact', '')} "
                f"| {o.get('strategy', '')} |"
            )
        if not opps:
            lines.append("| - | _No opportunities disclosed_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_business_impact(self, data: Dict[str, Any]) -> str:
        impact = data.get("business_impact", {})
        lines = [
            "## 4. Impact on Business, Strategy & Financial Planning\n",
            "| Dimension | Impact |", "|-----------|--------|",
            f"| Products & Services | {impact.get('products', 'Shift toward low-carbon offerings')} |",
            f"| Supply Chain | {impact.get('supply_chain', 'Decarbonization requirements for suppliers')} |",
            f"| Capital Allocation | {impact.get('capital', 'Increased investment in clean technology')} |",
            f"| R&D | {impact.get('rd', 'Increased focus on emission reduction technologies')} |",
            f"| Operations | {impact.get('operations', 'Energy efficiency and renewable procurement')} |",
            f"| M&A Strategy | {impact.get('ma', 'Climate due diligence in acquisitions')} |",
        ]
        return "\n".join(lines)

    def _md_scenario_methodology(self, data: Dict[str, Any]) -> str:
        methodology = data.get("scenario_methodology", {})
        lines = [
            "## 5. Scenario Analysis - Methodology\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Scenarios Used | {methodology.get('scenarios', '1.5C (NZE), 2C (APS), 4C (STEPS)')} |",
            f"| Source | {methodology.get('source', 'IEA World Energy Outlook 2024')} |",
            f"| Time Horizons | {methodology.get('time_horizons', '2030, 2040, 2050')} |",
            f"| Scope | {methodology.get('scope', 'All business segments, all geographies')} |",
            f"| Methodology | {methodology.get('methodology', 'Quantitative financial impact modeling')} |",
        ]
        lines.extend([
            "\n### Scenario Definitions\n",
            "| Scenario | Warming | Source | Carbon Price 2030 | Carbon Price 2050 |",
            "|----------|:-------:|--------|:-----------------:|:-----------------:|",
        ])
        for key, sdef in SCENARIO_DEFINITIONS.items():
            lines.append(
                f"| {sdef['name']} | {sdef['warming']} | {sdef['source']} "
                f"| {sdef['carbon_price_2030']} | {sdef['carbon_price_2050']} |"
            )
        return "\n".join(lines)

    def _md_scenario_results(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", {})
        lines = [
            "## 6. Scenario Analysis - Results\n",
            "| Scenario | Revenue Impact (%) | Cost Impact (%) | Asset Impairment | Capex Required | Overall Rating |",
            "|----------|-------------------:|----------------:|:----------------:|:--------------:|:--------------:|",
        ]
        for key, sdef in SCENARIO_DEFINITIONS.items():
            s = scenarios.get(key, {})
            lines.append(
                f"| {sdef['name']} | {'+' if s.get('revenue_impact', 0) > 0 else ''}{_dec(s.get('revenue_impact', 0))}% "
                f"| {'+' if s.get('cost_impact', 0) > 0 else ''}{_dec(s.get('cost_impact', 0))}% "
                f"| {s.get('asset_impairment', 'N/A')} | {s.get('capex_required', 'N/A')} "
                f"| {s.get('overall_rating', 'N/A')} |"
            )
        return "\n".join(lines)

    def _md_resilience(self, data: Dict[str, Any]) -> str:
        resilience = data.get("resilience_assessment", {})
        lines = [
            "## 7. Strategic Resilience Assessment\n",
            "| Dimension | Assessment | Score |",
            "|-----------|-----------|:-----:|",
            f"| Revenue Resilience | {resilience.get('revenue', 'Moderate')} | {resilience.get('revenue_score', 'N/A')} |",
            f"| Supply Chain Resilience | {resilience.get('supply_chain', 'Moderate')} | {resilience.get('supply_chain_score', 'N/A')} |",
            f"| Asset Resilience | {resilience.get('assets', 'High')} | {resilience.get('assets_score', 'N/A')} |",
            f"| Regulatory Preparedness | {resilience.get('regulatory', 'High')} | {resilience.get('regulatory_score', 'N/A')} |",
            f"| Technology Adaptability | {resilience.get('technology', 'Moderate')} | {resilience.get('technology_score', 'N/A')} |",
            f"| **Overall Resilience** | {resilience.get('overall', 'Moderate')} | {resilience.get('overall_score', 'N/A')} |",
        ]
        return "\n".join(lines)

    def _md_transition_plan(self, data: Dict[str, Any]) -> str:
        plan = data.get("transition_plan", {})
        phases = plan.get("phases", [])
        lines = [
            "## 8. Transition Plan Summary\n",
            f"**Status:** {plan.get('status', 'In development')}  \n"
            f"**Net-Zero Target Year:** {plan.get('net_zero_year', '')}  \n"
            f"**Total Investment Required:** {plan.get('total_investment', '')}  \n",
        ]
        if phases:
            lines.extend([
                "### Transition Phases\n",
                "| # | Phase | Period | Key Actions | Investment |",
                "|---|-------|--------|-------------|-----------|",
            ])
            for i, p in enumerate(phases, 1):
                lines.append(
                    f"| {i} | {p.get('name', '')} | {p.get('period', '')} "
                    f"| {p.get('actions', '')} | {p.get('investment', '')} |"
                )
        return "\n".join(lines)

    def _md_financial_impact(self, data: Dict[str, Any]) -> str:
        impacts = data.get("financial_impacts", [])
        lines = [
            "## 9. Financial Impact Quantification\n",
            "| # | Risk/Opportunity | Category | Impact Type | Estimated Impact | Timeframe |",
            "|---|-----------------|----------|-------------|-----------------|-----------|",
        ]
        for i, f in enumerate(impacts, 1):
            lines.append(
                f"| {i} | {f.get('name', '')} | {f.get('category', '')} "
                f"| {f.get('impact_type', '')} | {f.get('estimated_impact', '')} "
                f"| {f.get('timeframe', '')} |"
            )
        if not impacts:
            lines.append("| - | _No financial impacts quantified_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 10. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |", "|------------|----------|-------|",
        ]
        for key, tag in XBRL_TAGS.items():
            lines.append(f"| {key.replace('_', ' ').title()} | {tag} | - |")
        return "\n".join(lines)

    def _md_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            "## 11. Audit Trail & Provenance\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n| Data Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-030 on {ts}*  \n*TCFD Strategy pillar disclosure.*"

    # -- HTML --

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"h3{{color:{_ACCENT};}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #c5cae9;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f3f4fb;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.5em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>TCFD Strategy Pillar</h1>\n<p><strong>{data.get("org_name", "")}</strong> | {data.get("reporting_year", "")} | {ts}</p>'

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        return (
            f'<h2>1. Executive Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Risks</div><div class="card-value">{len(data.get("risks", []))}</div></div>\n'
            f'<div class="card"><div class="card-label">Opportunities</div><div class="card-value">{len(data.get("opportunities", []))}</div></div>\n'
            f'<div class="card"><div class="card-label">Scenarios</div><div class="card-value">{len(data.get("scenarios", {}))}</div></div>\n'
            f'</div>'
        )

    def _html_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        rows = ""
        for i, r in enumerate(risks, 1):
            rows += f'<tr><td>{i}</td><td>{r.get("category", "")}</td><td>{r.get("description", "")}</td><td>{r.get("time_horizon", "")}</td><td>{r.get("financial_impact", "")}</td></tr>\n'
        return f'<h2>2. Risks</h2>\n<table>\n<tr><th>#</th><th>Category</th><th>Description</th><th>Horizon</th><th>Impact</th></tr>\n{rows}</table>'

    def _html_opportunities(self, data: Dict[str, Any]) -> str:
        opps = data.get("opportunities", [])
        rows = ""
        for i, o in enumerate(opps, 1):
            rows += f'<tr><td>{i}</td><td>{o.get("category", "")}</td><td>{o.get("description", "")}</td><td>{o.get("financial_impact", "")}</td></tr>\n'
        return f'<h2>3. Opportunities</h2>\n<table>\n<tr><th>#</th><th>Category</th><th>Description</th><th>Impact</th></tr>\n{rows}</table>'

    def _html_scenario_results(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("scenarios", {})
        rows = ""
        for key, sdef in SCENARIO_DEFINITIONS.items():
            s = scenarios.get(key, {})
            rows += f'<tr><td>{sdef["name"]}</td><td>{_dec(s.get("revenue_impact", 0))}%</td><td>{_dec(s.get("cost_impact", 0))}%</td><td>{s.get("overall_rating", "N/A")}</td></tr>\n'
        return f'<h2>4. Scenario Results</h2>\n<table>\n<tr><th>Scenario</th><th>Revenue</th><th>Cost</th><th>Rating</th></tr>\n{rows}</table>'

    def _html_resilience(self, data: Dict[str, Any]) -> str:
        r = data.get("resilience_assessment", {})
        return (
            f'<h2>5. Resilience</h2>\n<table>\n<tr><th>Dimension</th><th>Assessment</th></tr>\n'
            f'<tr><td>Revenue</td><td>{r.get("revenue", "Moderate")}</td></tr>\n'
            f'<tr><td>Supply Chain</td><td>{r.get("supply_chain", "Moderate")}</td></tr>\n'
            f'<tr><td>Assets</td><td>{r.get("assets", "High")}</td></tr>\n'
            f'<tr><td>Overall</td><td>{r.get("overall", "Moderate")}</td></tr>\n'
            f'</table>'
        )

    def _html_financial_impact(self, data: Dict[str, Any]) -> str:
        impacts = data.get("financial_impacts", [])
        rows = ""
        for i, f in enumerate(impacts, 1):
            rows += f'<tr><td>{i}</td><td>{f.get("name", "")}</td><td>{f.get("estimated_impact", "")}</td><td>{f.get("timeframe", "")}</td></tr>\n'
        return f'<h2>6. Financial Impact</h2>\n<table>\n<tr><th>#</th><th>Item</th><th>Impact</th><th>Timeframe</th></tr>\n{rows}</table>'

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        rows = ""
        for key, tag in XBRL_TAGS.items():
            rows += f'<tr><td>{key.replace("_", " ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>7. XBRL Tags</h2>\n<table>\n<tr><th>Data Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>8. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-030 on {ts} - TCFD Strategy</div>'
