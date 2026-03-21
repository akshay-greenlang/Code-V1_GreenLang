# -*- coding: utf-8 -*-
"""
CDPClimateResponseTemplate - CDP Climate Change questionnaire response (PACK-027).

Renders a full CDP Climate Change questionnaire response covering all modules
(C0-C15). Auto-populates from pack data for C4 (targets), C5 (emissions
methodology), C6 (emissions data), C7 (energy), C8 (energy expenditures),
C12 (engagement), C15 (biodiversity). Maximizes scoring toward A-list.

Sections / Modules:
    C0  - Introduction
    C1  - Governance
    C2  - Risks and Opportunities
    C3  - Business Strategy
    C4  - Targets and Performance
    C5  - Emissions Methodology
    C6  - Emissions Data
    C7  - Energy
    C8  - Energy-Related Expenditures
    C9  - Additional Metrics
    C10 - Verification
    C11 - Carbon Pricing
    C12 - Engagement
    C14 - Portfolio Impact (Financial sector only)
    C15 - Biodiversity

Output: Markdown, HTML, JSON
Provenance: SHA-256 hash on all outputs

Author: GreenLang Team
Version: 27.0.0
Pack: PACK-027 Enterprise Net Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "27.0.0"
_PACK_ID = "PACK-027"
_TEMPLATE_ID = "cdp_climate_response"

_PRIMARY = "#0d3b2e"
_SECONDARY = "#1a6b4f"
_ACCENT = "#2e8b6e"
_LIGHT = "#e0f2ee"
_LIGHTER = "#f0f9f6"
_CARD_BG = "#b2dfdb"

CDP_MODULES = [
    {"id": "C0", "name": "Introduction", "auto_populate": True},
    {"id": "C1", "name": "Governance", "auto_populate": False},
    {"id": "C2", "name": "Risks and Opportunities", "auto_populate": False},
    {"id": "C3", "name": "Business Strategy", "auto_populate": False},
    {"id": "C4", "name": "Targets and Performance", "auto_populate": True},
    {"id": "C5", "name": "Emissions Methodology", "auto_populate": True},
    {"id": "C6", "name": "Emissions Data", "auto_populate": True},
    {"id": "C7", "name": "Energy", "auto_populate": True},
    {"id": "C8", "name": "Energy-Related Expenditures", "auto_populate": True},
    {"id": "C9", "name": "Additional Metrics", "auto_populate": False},
    {"id": "C10", "name": "Verification", "auto_populate": False},
    {"id": "C11", "name": "Carbon Pricing", "auto_populate": True},
    {"id": "C12", "name": "Engagement", "auto_populate": True},
    {"id": "C14", "name": "Portfolio Impact", "auto_populate": False},
    {"id": "C15", "name": "Biodiversity", "auto_populate": True},
]

CDP_SCORING_GUIDANCE = {
    "A": "Leadership: comprehensive disclosure, awareness, management, leadership actions",
    "A-": "Strong leadership with minor gaps",
    "B": "Management: evidence of climate management actions",
    "B-": "Good management with minor gaps",
    "C": "Awareness: demonstrates climate awareness",
    "D": "Disclosure: minimum disclosure level",
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
        return str(round(float(val), 1)) + "%"
    except Exception:
        return str(val)


def _safe_div(num: Any, den: Any, default: float = 0.0) -> float:
    try:
        d = float(den)
        return float(num) / d if d != 0 else default
    except Exception:
        return default


class CDPClimateResponseTemplate:
    """
    CDP Climate Change questionnaire response template.

    Auto-populates CDP modules C0-C15 from enterprise pack data,
    optimized for A-list scoring. Supports Markdown, HTML, and JSON output.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID
    FORMATS = ["markdown", "html", "json"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_module_status(data),
            self._md_c0_introduction(data),
            self._md_c1_governance(data),
            self._md_c2_risks(data),
            self._md_c3_strategy(data),
            self._md_c4_targets(data),
            self._md_c5_methodology(data),
            self._md_c6_emissions(data),
            self._md_c7_energy(data),
            self._md_c8_expenditures(data),
            self._md_c9_metrics(data),
            self._md_c10_verification(data),
            self._md_c11_carbon_pricing(data),
            self._md_c12_engagement(data),
            self._md_c15_biodiversity(data),
            self._md_scoring_tips(data),
            self._md_citations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(s for s in sections if s)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- SHA-256 Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:24px;"
            f"background:#f5f7f5;color:#1a1a2e;line-height:1.6;}}"
            f".report{{max-width:1100px;margin:0 auto;background:#fff;padding:40px;"
            f"border-radius:12px;box-shadow:0 2px 16px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:28px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"h3{{color:{_ACCENT};margin-top:16px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid #ddd;padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};color:{_PRIMARY};font-weight:600;}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".auto{{background:#e8f5e9;color:#2e7d32;padding:2px 8px;border-radius:4px;font-size:0.8em;}}"
            f".manual{{background:#fff3e0;color:#e65100;padding:2px 8px;border-radius:4px;font-size:0.8em;}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#607d8b;font-size:0.8em;text-align:center;}}"
        )

        body_parts = [
            f'<h1>CDP Climate Change Response</h1>',
            f'<p><strong>{data.get("org_name", "")}</strong> | '
            f'Reporting Year: {data.get("reporting_year", "")} | '
            f'Generated: {self.generated_at.strftime("%Y-%m-%d %H:%M UTC")}</p>',
            self._html_module_table(data),
            self._html_c6_summary(data),
            f'<div class="footer">Generated by GreenLang PACK-027 | CDP Climate Change | SHA-256</div>',
        ]
        body = "\n".join(body_parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n<title>CDP Climate Response</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- SHA-256 Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        s1 = float(data.get("scope1_tco2e", 0))
        s2_loc = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s2_mkt = float(data.get("scope2_market_tco2e", s2_loc))
        s3 = float(data.get("scope3_tco2e", 0))

        modules = {}
        for mod in CDP_MODULES:
            mod_data = data.get(f"cdp_{mod['id'].lower()}", {})
            modules[mod["id"]] = {
                "name": mod["name"],
                "auto_populated": mod["auto_populate"],
                "status": mod_data.get("status", "auto" if mod["auto_populate"] else "manual_required"),
                "responses": mod_data.get("responses", {}),
            }

        modules["C6"]["responses"] = {
            "scope1_tco2e": round(s1, 2),
            "scope2_location_tco2e": round(s2_loc, 2),
            "scope2_market_tco2e": round(s2_mkt, 2),
            "scope3_tco2e": round(s3, 2),
            "total_tco2e": round(s1 + s2_loc + s3, 2),
            "scope3_categories": data.get("scope3_categories", []),
        }

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "organization": {
                "name": data.get("org_name", ""),
                "sector": data.get("sector", ""),
                "country": data.get("country", ""),
                "employees": data.get("employees", 0),
                "revenue": data.get("revenue", 0),
                "currency": data.get("currency", "USD"),
            },
            "reporting_year": data.get("reporting_year", ""),
            "cdp_modules": modules,
            "scoring_estimate": data.get("cdp_scoring_estimate", "B"),
            "prior_year_score": data.get("cdp_prior_year_score", ""),
            "target_score": data.get("cdp_target_score", "A"),
            "citations": data.get("citations", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Markdown sections
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# CDP Climate Change Questionnaire Response\n\n"
            f"## {data.get('org_name', 'Enterprise')}\n\n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**CDP Account:** {data.get('cdp_account_number', 'N/A')}  \n"
            f"**Prior Year Score:** {data.get('cdp_prior_year_score', 'N/A')}  \n"
            f"**Target Score:** {data.get('cdp_target_score', 'A')}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_module_status(self, data: Dict[str, Any]) -> str:
        lines = [
            "## Module Completion Status\n",
            "| Module | Name | Auto-Populated | Status |",
            "|:------:|------|:--------------:|:------:|",
        ]
        for mod in CDP_MODULES:
            mod_data = data.get(f"cdp_{mod['id'].lower()}", {})
            status = mod_data.get("status", "Auto" if mod["auto_populate"] else "Manual Required")
            auto = "Yes" if mod["auto_populate"] else "No"
            lines.append(f"| {mod['id']} | {mod['name']} | {auto} | {status} |")
        return "\n".join(lines)

    def _md_c0_introduction(self, data: Dict[str, Any]) -> str:
        return (
            f"## C0 - Introduction\n\n"
            f"**C0.1 Organization:** {data.get('org_name', '')}  \n"
            f"**C0.2 Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**C0.3 Country:** {data.get('country', '')}  \n"
            f"**C0.4 Currency:** {data.get('currency', 'USD')}  \n"
            f"**C0.5 Base Year:** {data.get('base_year', '')}  \n"
            f"**C0.8 Sector:** {data.get('sector', '')}  "
        )

    def _md_c1_governance(self, data: Dict[str, Any]) -> str:
        gov = data.get("cdp_c1", {})
        return (
            f"## C1 - Governance\n\n"
            f"**C1.1a Board-level oversight:** {gov.get('board_oversight', 'Board committee has climate responsibility')}  \n"
            f"**C1.1b Frequency of climate reviews:** {gov.get('review_frequency', 'Quarterly')}  \n"
            f"**C1.2 Highest management-level position:** {gov.get('management_position', 'Chief Sustainability Officer')}  \n"
            f"**C1.3 Incentivized climate management:** {gov.get('incentives', 'Yes - linked to executive compensation')}  "
        )

    def _md_c2_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("climate_risks", [])
        opportunities = data.get("climate_opportunities", [])
        lines = [
            "## C2 - Risks and Opportunities\n",
            "### C2.3a Physical Risks\n",
            "| Risk Type | Description | Timeframe | Likelihood | Financial Impact |",
            "|-----------|-------------|:---------:|:----------:|:---------------:|",
        ]
        for r in risks[:5]:
            lines.append(
                f"| {r.get('type', 'Acute')} | {r.get('description', '')} "
                f"| {r.get('timeframe', 'Medium-term')} "
                f"| {r.get('likelihood', 'Likely')} "
                f"| {r.get('financial_impact', 'Medium')} |"
            )
        lines.append("\n### C2.4a Opportunities\n")
        lines.append("| Opportunity | Description | Timeframe | Financial Impact |")
        lines.append("|-------------|-------------|:---------:|:---------------:|")
        for o in opportunities[:5]:
            lines.append(
                f"| {o.get('type', '')} | {o.get('description', '')} "
                f"| {o.get('timeframe', '')} | {o.get('financial_impact', '')} |"
            )
        return "\n".join(lines)

    def _md_c3_strategy(self, data: Dict[str, Any]) -> str:
        strategy = data.get("cdp_c3", {})
        return (
            f"## C3 - Business Strategy\n\n"
            f"**C3.1 Climate transition plan:** {strategy.get('has_transition_plan', 'Yes')}  \n"
            f"**C3.2 Scenario analysis conducted:** {strategy.get('scenario_analysis', 'Yes - 1.5C, 2C, and 4C scenarios')}  \n"
            f"**C3.3 Financial planning impact:** {strategy.get('financial_planning', 'Yes - internal carbon price applied to CapEx decisions')}  \n"
            f"**C3.4 Low-carbon products revenue:** {strategy.get('low_carbon_revenue_pct', 'N/A')}  "
        )

    def _md_c4_targets(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        lines = [
            "## C4 - Targets and Performance\n",
            "### C4.1 Absolute Emissions Targets\n",
            "| Target | Scope | Base Year | Target Year | Base Emissions | Target Emissions | Reduction |",
            "|--------|-------|:---------:|:-----------:|---------------:|----------------:|:---------:|",
        ]
        for t in targets:
            lines.append(
                f"| {t.get('name', '')} | {t.get('scope', '')} "
                f"| {t.get('base_year', '')} | {t.get('target_year', '')} "
                f"| {_dec_comma(t.get('base_tco2e', 0))} "
                f"| {_dec_comma(t.get('target_tco2e', 0))} "
                f"| {_pct(t.get('reduction_pct', 0))} |"
            )
        if not targets:
            lines.append("| *See SBTi target submission* | | | | | | |")
        return "\n".join(lines)

    def _md_c5_methodology(self, data: Dict[str, Any]) -> str:
        return (
            f"## C5 - Emissions Methodology\n\n"
            f"**C5.1 Reporting standard:** GHG Protocol Corporate Standard  \n"
            f"**C5.2 Base year:** {data.get('base_year', '')}  \n"
            f"**C5.2a Consolidation approach:** {data.get('consolidation_approach', 'Operational control')}  \n"
            f"**C5.3 Emission factors:** {data.get('emission_factor_sources', 'DEFRA, EPA, IEA, ecoinvent')}  \n"
            f"**C5.3a GWP values:** {data.get('gwp_source', 'IPCC AR6 GWP-100')}  "
        )

    def _md_c6_emissions(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2_loc = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s2_mkt = float(data.get("scope2_market_tco2e", s2_loc))
        s3 = float(data.get("scope3_tco2e", 0))
        total = s1 + s2_loc + s3

        cats = data.get("scope3_categories", [])
        lines = [
            "## C6 - Emissions Data\n",
            "### C6.1 Gross Global Scope 1\n",
            f"**{_dec_comma(s1)} tCO2e**\n",
            "### C6.3 Scope 2 (Location-based)\n",
            f"**{_dec_comma(s2_loc)} tCO2e**\n",
            "### C6.3 Scope 2 (Market-based)\n",
            f"**{_dec_comma(s2_mkt)} tCO2e**\n",
            "### C6.5 Scope 3 by Category\n",
            "| Cat | Category | tCO2e | % of S3 | Evaluation Status |",
            "|:---:|----------|------:|:-------:|:----------------:|",
        ]
        for cat in cats:
            lines.append(
                f"| {cat.get('category', '')} | {cat.get('name', '')} "
                f"| {_dec_comma(cat.get('tco2e', 0))} "
                f"| {_pct(_safe_div(cat.get('tco2e', 0), s3) * 100)} "
                f"| {cat.get('evaluation_status', 'Relevant, calculated')} |"
            )
        lines.append(f"\n**Total Scope 3:** {_dec_comma(s3)} tCO2e  ")
        lines.append(f"**Total Scope 1+2+3:** {_dec_comma(total)} tCO2e")
        return "\n".join(lines)

    def _md_c7_energy(self, data: Dict[str, Any]) -> str:
        energy = data.get("cdp_c7", {})
        return (
            f"## C7 - Energy\n\n"
            f"**C7.1 Total energy consumption:** {_dec_comma(energy.get('total_mwh', 0))} MWh  \n"
            f"**C7.2 Electricity from renewables:** {_pct(energy.get('renewable_pct', 0))}  \n"
            f"**C7.3 Total fuel consumption:** {_dec_comma(energy.get('fuel_mwh', 0))} MWh  \n"
            f"**C7.5 Energy intensity:** {energy.get('intensity_mwh_per_revenue', 'N/A')} MWh/{data.get('currency', 'USD')}M revenue  "
        )

    def _md_c8_expenditures(self, data: Dict[str, Any]) -> str:
        c8 = data.get("cdp_c8", {})
        return (
            f"## C8 - Energy-Related Expenditures\n\n"
            f"**C8.1 Total energy spend:** {data.get('currency', 'USD')} "
            f"{_dec_comma(c8.get('total_energy_spend', 0))}  \n"
            f"**C8.2a Low-carbon investment:** {data.get('currency', 'USD')} "
            f"{_dec_comma(c8.get('low_carbon_investment', 0))}  \n"
            f"**C8.2b Internal carbon price applied:** {c8.get('carbon_price_applied', 'Yes')}  "
        )

    def _md_c9_metrics(self, data: Dict[str, Any]) -> str:
        return "## C9 - Additional Metrics\n\nSee sector-specific metrics in detailed response."

    def _md_c10_verification(self, data: Dict[str, Any]) -> str:
        c10 = data.get("cdp_c10", {})
        return (
            f"## C10 - Verification\n\n"
            f"**C10.1 Scope 1 verified:** {c10.get('scope1_verified', 'Yes')}  \n"
            f"**C10.1a Verification standard:** {c10.get('verification_standard', 'ISO 14064-3:2019')}  \n"
            f"**C10.1b Assurance level:** {c10.get('assurance_level', 'Limited assurance')}  \n"
            f"**C10.2 Scope 2 verified:** {c10.get('scope2_verified', 'Yes')}  \n"
            f"**C10.2a Scope 3 verified:** {c10.get('scope3_verified', 'Partial -- material categories')}  "
        )

    def _md_c11_carbon_pricing(self, data: Dict[str, Any]) -> str:
        c11 = data.get("cdp_c11", {})
        return (
            f"## C11 - Carbon Pricing\n\n"
            f"**C11.1 Uses internal carbon price:** {c11.get('uses_icp', 'Yes')}  \n"
            f"**C11.1a Price type:** {c11.get('price_type', 'Shadow price')}  \n"
            f"**C11.1b Price level:** {c11.get('price_level', '$75-100/tCO2e')}  \n"
            f"**C11.1c Application:** {c11.get('application', 'Capital allocation decisions')}  \n"
            f"**C11.2 ETS exposure:** {c11.get('ets_exposure', 'Yes -- EU ETS')}  \n"
            f"**C11.3 Carbon credits purchased:** {c11.get('credits_purchased', 'Yes')}  "
        )

    def _md_c12_engagement(self, data: Dict[str, Any]) -> str:
        c12 = data.get("cdp_c12", {})
        return (
            f"## C12 - Engagement\n\n"
            f"### C12.1 Value Chain Engagement\n\n"
            f"**C12.1a Supplier engagement:** {c12.get('supplier_engagement', 'Yes -- CDP Supply Chain')}  \n"
            f"**Suppliers requested to disclose:** {_dec_comma(c12.get('suppliers_requested', 0))}  \n"
            f"**Response rate:** {_pct(c12.get('response_rate', 0))}  \n"
            f"**Suppliers with SBTi targets:** {_dec_comma(c12.get('suppliers_with_sbti', 0))}  \n"
            f"**Scope 3 covered by engagement:** {_pct(c12.get('scope3_covered_pct', 0))}  \n\n"
            f"### C12.3 Policy Engagement\n\n"
            f"**Trade associations:** {c12.get('trade_associations', 'See detailed response')}  "
        )

    def _md_c15_biodiversity(self, data: Dict[str, Any]) -> str:
        c15 = data.get("cdp_c15", {})
        return (
            f"## C15 - Biodiversity\n\n"
            f"**C15.1 Biodiversity assessment:** {c15.get('assessment', 'Yes')}  \n"
            f"**C15.2 Deforestation policy:** {c15.get('deforestation_policy', 'Zero deforestation by 2025')}  \n"
            f"**C15.3 TNFD reporting:** {c15.get('tnfd', 'Aligned with TNFD LEAP approach')}  "
        )

    def _md_scoring_tips(self, data: Dict[str, Any]) -> str:
        current_est = data.get("cdp_scoring_estimate", "B")
        lines = [
            "## Scoring Optimization Notes\n",
            f"**Current Estimated Score:** {current_est}  ",
            f"**Target Score:** {data.get('cdp_target_score', 'A')}\n",
            "### A-List Requirements\n",
            "- All management-level questions answered with evidence",
            "- Targets in line with 1.5C pathway (SBTi validated preferred)",
            "- Scope 3 reported for all relevant categories",
            "- Engagement activities documented with measurable outcomes",
            "- Verification/assurance obtained for Scope 1+2",
            "- Transition plan aligned with TCFD/ISSB S2",
            "- Board-level governance with climate-linked incentives",
        ]
        return "\n".join(lines)

    def _md_citations(self, data: Dict[str, Any]) -> str:
        citations = data.get("citations", [
            {"ref": "CDP-001", "source": "CDP Climate Change Questionnaire 2025", "year": "2025"},
            {"ref": "CDP-002", "source": "CDP Scoring Methodology", "year": "2024"},
            {"ref": "CDP-003", "source": "CDP Supply Chain Program", "year": "2024"},
        ])
        lines = ["## Citations\n"]
        for c in citations:
            lines.append(f"- [{c.get('ref', '')}] {c.get('source', '')} ({c.get('year', '')})")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-027 Enterprise Net Zero Pack on {ts}*  \n"
            f"*CDP Climate Change Questionnaire auto-population. SHA-256 provenance.*"
        )

    # ------------------------------------------------------------------ #
    # HTML helpers
    # ------------------------------------------------------------------ #

    def _html_module_table(self, data: Dict[str, Any]) -> str:
        rows = ""
        for mod in CDP_MODULES:
            cls = "auto" if mod["auto_populate"] else "manual"
            label = "Auto" if mod["auto_populate"] else "Manual"
            rows += (
                f'<tr><td>{mod["id"]}</td><td>{mod["name"]}</td>'
                f'<td><span class="{cls}">{label}</span></td></tr>\n'
            )
        return (
            f'<h2>Module Status</h2>\n'
            f'<table><tr><th>Module</th><th>Name</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_c6_summary(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1_tco2e", 0))
        s2 = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s3 = float(data.get("scope3_tco2e", 0))
        return (
            f'<h2>C6 - Emissions Summary</h2>\n'
            f'<table><tr><th>Scope</th><th>tCO2e</th></tr>\n'
            f'<tr><td>Scope 1</td><td>{_dec_comma(s1)}</td></tr>\n'
            f'<tr><td>Scope 2 (Location)</td><td>{_dec_comma(s2)}</td></tr>\n'
            f'<tr><td>Scope 3</td><td>{_dec_comma(s3)}</td></tr>\n'
            f'<tr><td><strong>Total</strong></td><td><strong>{_dec_comma(s1+s2+s3)}</strong></td></tr>\n'
            f'</table>'
        )
