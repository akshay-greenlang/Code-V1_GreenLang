# -*- coding: utf-8 -*-
"""
PledgeCommitmentLetterTemplate - Race to Zero pledge commitment for PACK-025.

Renders a formal pledge commitment letter with executive summary, organization
profile, baseline emissions statement, interim and long-term target commitments,
starting line compliance checklist, action plan timeline, verification schedule,
and signatory authorization.

Sections:
    1. Executive Summary & Organization Profile
    2. Baseline Emissions Statement (S1/S2/S3)
    3. Interim Target Commitment (2030)
    4. Long-Term Target Commitment (2045-2050)
    5. Starting Line Compliance Checklist
    6. Action Plan Commitment Timeline
    7. Verification Schedule
    8. Signatory Authorization

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

logger = logging.getLogger(__name__)

_MODULE_VERSION = "25.0.0"
_PACK_ID = "PACK-025"
_TEMPLATE_ID = "pledge_commitment_letter"


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


def _safe_div(numerator: Any, denominator: Any) -> float:
    try:
        d = float(denominator)
        return float(numerator) / d if d != 0 else 0.0
    except Exception:
        return 0.0


class PledgeCommitmentLetterTemplate:
    """Race to Zero pledge commitment letter template for PACK-025.

    Generates formal pledge commitment letters in multiple output formats
    (Markdown, HTML, JSON) suitable for Race to Zero campaign submission.
    Covers organizational profile, baseline emissions, target commitments,
    starting line compliance, action plan timeline, verification schedule,
    and signatory authorization.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID

    # Starting line criteria per Race to Zero
    STARTING_LINE_CRITERIA = [
        {"id": "SL-01", "category": "Pledge", "criterion": "Public pledge to reach net-zero by 2050 at the latest"},
        {"id": "SL-02", "category": "Pledge", "criterion": "Pledge covers all GHG emissions scopes (S1, S2, S3)"},
        {"id": "SL-03", "category": "Pledge", "criterion": "Interim target set for 2030 or sooner"},
        {"id": "SL-04", "category": "Pledge", "criterion": "Targets aligned with science-based 1.5C pathways"},
        {"id": "SL-05", "category": "Pledge", "criterion": "No planned use of offsets for interim targets"},
        {"id": "SL-06", "category": "Plan", "criterion": "Transition plan developed within 12 months of joining"},
        {"id": "SL-07", "category": "Plan", "criterion": "Plan includes specific actions and milestones"},
        {"id": "SL-08", "category": "Plan", "criterion": "Governance structure for climate action in place"},
        {"id": "SL-09", "category": "Plan", "criterion": "Budget allocated for decarbonization activities"},
        {"id": "SL-10", "category": "Plan", "criterion": "Scope 3 engagement strategy documented"},
        {"id": "SL-11", "category": "Proceed", "criterion": "Immediate actions initiated within 12 months"},
        {"id": "SL-12", "category": "Proceed", "criterion": "Annual emissions inventory conducted"},
        {"id": "SL-13", "category": "Proceed", "criterion": "Reduction actions aligned with transition plan"},
        {"id": "SL-14", "category": "Proceed", "criterion": "Investment in low-carbon technologies commenced"},
        {"id": "SL-15", "category": "Proceed", "criterion": "Supply chain engagement initiated"},
        {"id": "SL-16", "category": "Publish", "criterion": "Annual progress reporting committed"},
        {"id": "SL-17", "category": "Publish", "criterion": "Emissions data publicly disclosed"},
        {"id": "SL-18", "category": "Publish", "criterion": "Progress against targets reported transparently"},
        {"id": "SL-19", "category": "Publish", "criterion": "Independent verification planned or completed"},
        {"id": "SL-20", "category": "Publish", "criterion": "Use of offsets transparently disclosed"},
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    #  Public render methods                                               #
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the pledge commitment letter as Markdown."""
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_baseline_emissions(data),
            self._md_interim_target(data),
            self._md_longterm_target(data),
            self._md_starting_line(data),
            self._md_action_plan_timeline(data),
            self._md_verification_schedule(data),
            self._md_signatory(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the pledge commitment letter as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_baseline_emissions(data),
            self._html_interim_target(data),
            self._html_longterm_target(data),
            self._html_starting_line(data),
            self._html_action_plan_timeline(data),
            self._html_verification_schedule(data),
            self._html_signatory(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Race to Zero - Pledge Commitment Letter</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the pledge commitment letter as structured JSON."""
        self.generated_at = _utcnow()
        org = data.get("org_name", "")
        sector = data.get("sector", "")
        baseline = data.get("baseline", {})
        interim = data.get("interim_target", {})
        longterm = data.get("longterm_target", {})
        starting_line = data.get("starting_line", {})
        sl_items = starting_line.get("items", [])

        met_count = sum(1 for it in sl_items if it.get("met", False))
        total_criteria = len(sl_items) if sl_items else len(self.STARTING_LINE_CRITERIA)

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": org,
            "sector": sector,
            "campaign": data.get("campaign", "Race to Zero"),
            "partner_initiative": data.get("partner_initiative", ""),
            "baseline": {
                "year": baseline.get("year", ""),
                "scope1_tco2e": baseline.get("scope1_tco2e", 0),
                "scope2_tco2e": baseline.get("scope2_tco2e", 0),
                "scope3_tco2e": baseline.get("scope3_tco2e", 0),
                "total_tco2e": baseline.get("total_tco2e", 0),
                "methodology": baseline.get("methodology", "GHG Protocol"),
                "boundary": baseline.get("boundary", "Operational control"),
            },
            "interim_target": {
                "year": interim.get("year", 2030),
                "reduction_pct": interim.get("reduction_pct", 0),
                "base_year": interim.get("base_year", ""),
                "scope_coverage": interim.get("scope_coverage", "S1+S2"),
                "pathway": interim.get("pathway", "1.5C aligned"),
            },
            "longterm_target": {
                "year": longterm.get("year", 2050),
                "target": longterm.get("target", "Net-zero emissions"),
                "residual_emissions_pct": longterm.get("residual_emissions_pct", 10),
                "neutralization_strategy": longterm.get("neutralization_strategy", ""),
            },
            "starting_line_compliance": {
                "criteria_met": met_count,
                "total_criteria": total_criteria,
                "compliance_pct": _safe_div(met_count, total_criteria) * 100,
                "is_compliant": met_count >= total_criteria,
            },
            "action_plan_milestones": data.get("milestones", []),
            "verification_schedule": data.get("verification_schedule", []),
            "signatories": data.get("signatories", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel_data(self, data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Return structured data suitable for Excel/openpyxl export.

        Returns a dict of sheet_name -> list of row dicts.
        """
        self.generated_at = _utcnow()
        sheets: Dict[str, List[Dict[str, Any]]] = {}

        # Sheet 1: Organization Profile
        org_info = data.get("org_profile", {})
        sheets["Organization Profile"] = [
            {"Field": "Organization Name", "Value": data.get("org_name", "")},
            {"Field": "Sector", "Value": data.get("sector", "")},
            {"Field": "Country", "Value": org_info.get("country", "")},
            {"Field": "Employees", "Value": org_info.get("employees", "")},
            {"Field": "Revenue (USD)", "Value": org_info.get("revenue_usd", "")},
            {"Field": "Campaign", "Value": data.get("campaign", "Race to Zero")},
            {"Field": "Partner Initiative", "Value": data.get("partner_initiative", "")},
            {"Field": "Pledge Date", "Value": data.get("pledge_date", "")},
        ]

        # Sheet 2: Baseline Emissions
        baseline = data.get("baseline", {})
        sheets["Baseline Emissions"] = [
            {"Scope": "Scope 1", "Emissions (tCO2e)": baseline.get("scope1_tco2e", 0),
             "Percentage": _safe_div(baseline.get("scope1_tco2e", 0), max(baseline.get("total_tco2e", 1), 1)) * 100},
            {"Scope": "Scope 2 (Location)", "Emissions (tCO2e)": baseline.get("scope2_location_tco2e", 0),
             "Percentage": _safe_div(baseline.get("scope2_location_tco2e", 0), max(baseline.get("total_tco2e", 1), 1)) * 100},
            {"Scope": "Scope 2 (Market)", "Emissions (tCO2e)": baseline.get("scope2_market_tco2e", baseline.get("scope2_tco2e", 0)),
             "Percentage": _safe_div(baseline.get("scope2_market_tco2e", baseline.get("scope2_tco2e", 0)), max(baseline.get("total_tco2e", 1), 1)) * 100},
            {"Scope": "Scope 3", "Emissions (tCO2e)": baseline.get("scope3_tco2e", 0),
             "Percentage": _safe_div(baseline.get("scope3_tco2e", 0), max(baseline.get("total_tco2e", 1), 1)) * 100},
            {"Scope": "TOTAL", "Emissions (tCO2e)": baseline.get("total_tco2e", 0), "Percentage": 100.0},
        ]

        # Sheet 3: Targets
        interim = data.get("interim_target", {})
        longterm = data.get("longterm_target", {})
        sheets["Targets"] = [
            {"Target Type": "Interim", "Year": interim.get("year", 2030),
             "Reduction (%)": interim.get("reduction_pct", 0),
             "Scope Coverage": interim.get("scope_coverage", "S1+S2"),
             "Pathway": interim.get("pathway", "1.5C aligned"),
             "Base Year": interim.get("base_year", "")},
            {"Target Type": "Long-Term", "Year": longterm.get("year", 2050),
             "Reduction (%)": longterm.get("reduction_pct", 90),
             "Scope Coverage": longterm.get("scope_coverage", "S1+S2+S3"),
             "Pathway": longterm.get("pathway", "Net-zero"),
             "Base Year": longterm.get("base_year", "")},
        ]

        # Sheet 4: Starting Line Compliance
        sl_items = data.get("starting_line", {}).get("items", [])
        sl_rows: List[Dict[str, Any]] = []
        criteria_list = sl_items if sl_items else [
            {"id": c["id"], "category": c["category"], "criterion": c["criterion"], "met": False}
            for c in self.STARTING_LINE_CRITERIA
        ]
        for item in criteria_list:
            sl_rows.append({
                "ID": item.get("id", ""),
                "Category": item.get("category", ""),
                "Criterion": item.get("criterion", ""),
                "Met": "Yes" if item.get("met", False) else "No",
                "Evidence": item.get("evidence", ""),
            })
        sheets["Starting Line Compliance"] = sl_rows

        # Sheet 5: Action Plan Timeline
        milestones = data.get("milestones", [])
        ms_rows: List[Dict[str, Any]] = []
        for ms in milestones:
            ms_rows.append({
                "Phase": ms.get("phase", ""),
                "Milestone": ms.get("milestone", ""),
                "Target Date": ms.get("target_date", ""),
                "Responsible": ms.get("responsible", ""),
                "Status": ms.get("status", "Planned"),
            })
        sheets["Action Plan Timeline"] = ms_rows

        # Sheet 6: Verification Schedule
        verifications = data.get("verification_schedule", [])
        v_rows: List[Dict[str, Any]] = []
        for v in verifications:
            v_rows.append({
                "Year": v.get("year", ""),
                "Type": v.get("type", ""),
                "Provider": v.get("provider", ""),
                "Scope": v.get("scope", ""),
                "Status": v.get("status", "Planned"),
            })
        sheets["Verification Schedule"] = v_rows

        return sheets

    # ------------------------------------------------------------------ #
    #  Markdown sections                                                   #
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        campaign = data.get("campaign", "Race to Zero")
        partner = data.get("partner_initiative", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        partner_line = f"**Partner Initiative:** {partner}  \n" if partner else ""
        return (
            f"# Race to Zero -- Pledge Commitment Letter\n\n"
            f"**Organization:** {org}  \n"
            f"**Campaign:** {campaign}  \n"
            f"{partner_line}"
            f"**Date:** {ts}  \n"
            f"**Document ID:** {_new_uuid()}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        sector = data.get("sector", "")
        profile = data.get("org_profile", {})
        baseline = data.get("baseline", {})
        interim = data.get("interim_target", {})
        longterm = data.get("longterm_target", {})

        return (
            f"## 1. Executive Summary & Organization Profile\n\n"
            f"### Organization Profile\n\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| Organization | {org} |\n"
            f"| Sector | {sector} |\n"
            f"| Country | {profile.get('country', '')} |\n"
            f"| Employees | {_dec_comma(profile.get('employees', 0))} |\n"
            f"| Revenue | ${_dec_comma(profile.get('revenue_usd', 0))} |\n"
            f"| Reporting Standard | {profile.get('reporting_standard', 'GHG Protocol')} |\n"
            f"| Boundary | {profile.get('boundary', 'Operational control')} |\n\n"
            f"### Pledge Summary\n\n"
            f"{org} hereby pledges to join the Race to Zero campaign and commits to:\n\n"
            f"- Achieving **{_pct(interim.get('reduction_pct', 50))} reduction** "
            f"by **{interim.get('year', 2030)}** (interim target)\n"
            f"- Reaching **net-zero emissions** by **{longterm.get('year', 2050)}** at the latest\n"
            f"- Total baseline emissions: **{_dec_comma(baseline.get('total_tco2e', 0))} tCO2e** "
            f"(base year {baseline.get('year', 'N/A')})\n"
            f"- Publicly reporting progress annually and submitting to independent verification"
        )

    def _md_baseline_emissions(self, data: Dict[str, Any]) -> str:
        b = data.get("baseline", {})
        total = b.get("total_tco2e", 0)
        s1 = b.get("scope1_tco2e", 0)
        s2 = b.get("scope2_tco2e", 0)
        s3 = b.get("scope3_tco2e", 0)
        s3_cats = b.get("scope3_categories", [])

        lines = [
            f"## 2. Baseline Emissions Statement\n",
            f"**Base Year:** {b.get('year', 'N/A')}  \n"
            f"**Methodology:** {b.get('methodology', 'GHG Protocol Corporate Standard')}  \n"
            f"**Boundary:** {b.get('boundary', 'Operational control')}  \n"
            f"**GWP Values:** {b.get('gwp', 'IPCC AR6 100-year')}\n",
            "### Scope Summary\n",
            "| Scope | Emissions (tCO2e) | % of Total | Data Quality |",
            "|-------|------------------:|:----------:|:------------:|",
            f"| Scope 1 (Direct) | {_dec_comma(s1)} | {_pct(_safe_div(s1, max(total, 1)) * 100)} | {b.get('scope1_dq', 'Tier 1')} |",
            f"| Scope 2 (Indirect Energy) | {_dec_comma(s2)} | {_pct(_safe_div(s2, max(total, 1)) * 100)} | {b.get('scope2_dq', 'Tier 1')} |",
            f"| Scope 3 (Value Chain) | {_dec_comma(s3)} | {_pct(_safe_div(s3, max(total, 1)) * 100)} | {b.get('scope3_dq', 'Tier 2')} |",
            f"| **Total** | **{_dec_comma(total)}** | **100%** | |",
        ]

        if s3_cats:
            lines.append("\n### Scope 3 Category Breakdown\n")
            lines.append("| Cat | Category | Emissions (tCO2e) | % of S3 | Included |")
            lines.append("|:---:|----------|------------------:|:-------:|:--------:|")
            for cat in s3_cats:
                lines.append(
                    f"| {cat.get('id', '-')} | {cat.get('name', '-')} "
                    f"| {_dec_comma(cat.get('tco2e', 0))} "
                    f"| {_pct(cat.get('pct_of_s3', 0))} "
                    f"| {'Yes' if cat.get('included', True) else 'No'} |"
                )

        return "\n".join(lines)

    def _md_interim_target(self, data: Dict[str, Any]) -> str:
        t = data.get("interim_target", {})
        b = data.get("baseline", {})
        base_total = b.get("total_tco2e", 0)
        reduction_pct = t.get("reduction_pct", 50)
        target_emissions = base_total * (1 - reduction_pct / 100.0) if base_total else 0

        milestones_2030 = t.get("milestones", [])
        ml_text = ""
        if milestones_2030:
            ml_lines = ["\n### Key Milestones to 2030\n"]
            for ms in milestones_2030:
                ml_lines.append(f"- **{ms.get('year', '')}**: {ms.get('milestone', '')}")
            ml_text = "\n".join(ml_lines)

        return (
            f"## 3. Interim Target Commitment (2030)\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Target Year | {t.get('year', 2030)} |\n"
            f"| Reduction Target | {_pct(reduction_pct)} from base year |\n"
            f"| Base Year | {t.get('base_year', b.get('year', 'N/A'))} |\n"
            f"| Base Year Emissions | {_dec_comma(base_total)} tCO2e |\n"
            f"| Target Emissions | {_dec_comma(target_emissions)} tCO2e |\n"
            f"| Scope Coverage | {t.get('scope_coverage', 'S1+S2')} |\n"
            f"| Pathway Alignment | {t.get('pathway', '1.5C aligned (no/limited overshoot)')} |\n"
            f"| Offset Usage | {t.get('offset_policy', 'No offsets for interim target')} |\n"
            f"| Validation | {t.get('validation', 'To be validated by SBTi or equivalent')} |"
            f"{ml_text}"
        )

    def _md_longterm_target(self, data: Dict[str, Any]) -> str:
        lt = data.get("longterm_target", {})
        b = data.get("baseline", {})

        return (
            f"## 4. Long-Term Target Commitment ({lt.get('year', '2045-2050')})\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Target Year | {lt.get('year', 2050)} |\n"
            f"| Target | {lt.get('target', 'Net-zero GHG emissions')} |\n"
            f"| Base Year | {lt.get('base_year', b.get('year', 'N/A'))} |\n"
            f"| Minimum Reduction | {_pct(lt.get('min_reduction_pct', 90))} from base year |\n"
            f"| Maximum Residual | {_pct(lt.get('residual_emissions_pct', 10))} of base year emissions |\n"
            f"| Scope Coverage | {lt.get('scope_coverage', 'S1+S2+S3')} |\n"
            f"| Neutralization | {lt.get('neutralization_strategy', 'Carbon removal for residual emissions')} |\n"
            f"| Pathway | {lt.get('pathway', 'IPCC SR1.5 P1/P2 pathway')} |\n"
            f"| Just Transition | {lt.get('just_transition', 'Integrated into transition planning')} |"
        )

    def _md_starting_line(self, data: Dict[str, Any]) -> str:
        sl = data.get("starting_line", {})
        items = sl.get("items", [])
        criteria_list = items if items else [
            {"id": c["id"], "category": c["category"], "criterion": c["criterion"], "met": False, "evidence": ""}
            for c in self.STARTING_LINE_CRITERIA
        ]

        met_count = sum(1 for it in criteria_list if it.get("met", False))
        total = len(criteria_list)
        compliance_pct = _safe_div(met_count, total) * 100

        lines = [
            f"## 5. Starting Line Compliance Checklist\n",
            f"**Compliance Status:** {met_count}/{total} criteria met ({_pct(compliance_pct)})\n",
            "| ID | Category | Criterion | Met | Evidence |",
            "|:--:|:--------:|-----------|:---:|----------|",
        ]

        for item in criteria_list:
            met_symbol = "YES" if item.get("met", False) else "NO"
            evidence = item.get("evidence", "")
            lines.append(
                f"| {item.get('id', '-')} | {item.get('category', '-')} "
                f"| {item.get('criterion', '-')} | {met_symbol} "
                f"| {evidence} |"
            )

        # Summary by category
        categories = {}
        for item in criteria_list:
            cat = item.get("category", "Other")
            if cat not in categories:
                categories[cat] = {"total": 0, "met": 0}
            categories[cat]["total"] += 1
            if item.get("met", False):
                categories[cat]["met"] += 1

        lines.append("\n### Compliance by Category (4P Framework)\n")
        lines.append("| Category | Met | Total | Compliance |")
        lines.append("|----------|:---:|:-----:|:----------:|")
        for cat_name in ["Pledge", "Plan", "Proceed", "Publish"]:
            cat_data = categories.get(cat_name, {"met": 0, "total": 0})
            cat_pct = _safe_div(cat_data["met"], cat_data["total"]) * 100
            lines.append(
                f"| {cat_name} | {cat_data['met']} | {cat_data['total']} | {_pct(cat_pct)} |"
            )

        return "\n".join(lines)

    def _md_action_plan_timeline(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        lines = [
            "## 6. Action Plan Commitment Timeline\n",
            "| # | Phase | Milestone | Target Date | Responsible | Status |",
            "|---|-------|-----------|:-----------:|-------------|:------:|",
        ]

        if milestones:
            for i, ms in enumerate(milestones, 1):
                lines.append(
                    f"| {i} | {ms.get('phase', '-')} "
                    f"| {ms.get('milestone', '-')} "
                    f"| {ms.get('target_date', '-')} "
                    f"| {ms.get('responsible', '-')} "
                    f"| {ms.get('status', 'Planned')} |"
                )
        else:
            # Default timeline
            default_milestones = [
                ("Year 1 Q1", "Complete GHG baseline inventory", "Sustainability Team", "Planned"),
                ("Year 1 Q2", "Submit Race to Zero pledge", "CEO / Board", "Planned"),
                ("Year 1 Q2", "Develop transition plan", "Strategy Team", "Planned"),
                ("Year 1 Q3", "Set science-based interim targets", "Sustainability Team", "Planned"),
                ("Year 1 Q4", "Initiate Scope 1&2 reduction projects", "Operations", "Planned"),
                ("Year 2 Q1", "Begin Scope 3 supplier engagement", "Procurement", "Planned"),
                ("Year 2 Q2", "First annual progress report", "Sustainability Team", "Planned"),
                ("Year 2 Q4", "Independent verification", "External Verifier", "Planned"),
                ("Year 3+", "Annual reporting and continuous improvement", "All Teams", "Planned"),
            ]
            for i, (phase, milestone, resp, status) in enumerate(default_milestones, 1):
                lines.append(f"| {i} | {phase} | {milestone} | TBD | {resp} | {status} |")

        return "\n".join(lines)

    def _md_verification_schedule(self, data: Dict[str, Any]) -> str:
        schedule = data.get("verification_schedule", [])
        lines = [
            "## 7. Verification Schedule\n",
            "| Year | Verification Type | Provider | Scope | Standard | Status |",
            "|:----:|-------------------|----------|-------|----------|:------:|",
        ]

        if schedule:
            for v in schedule:
                lines.append(
                    f"| {v.get('year', '-')} | {v.get('type', '-')} "
                    f"| {v.get('provider', '-')} | {v.get('scope', '-')} "
                    f"| {v.get('standard', 'ISO 14064-3')} | {v.get('status', 'Planned')} |"
                )
        else:
            defaults = [
                ("Year 1", "GHG Inventory", "TBD", "S1+S2+S3", "ISO 14064-3:2019"),
                ("Year 2", "Progress Report", "TBD", "Targets + Actions", "Race to Zero criteria"),
                ("Year 3", "Full Verification", "TBD", "S1+S2+S3 + Targets", "ISO 14064-3:2019"),
                ("Year 5", "Comprehensive Review", "TBD", "All scopes + pathway", "SBTi + R2Z"),
            ]
            for year, vtype, provider, scope, standard in defaults:
                lines.append(f"| {year} | {vtype} | {provider} | {scope} | {standard} | Planned |")

        lines.append(
            "\n> *All verifications will be conducted by accredited third-party verification "
            "bodies in accordance with ISO 14064-3:2019 or equivalent standards.*"
        )

        return "\n".join(lines)

    def _md_signatory(self, data: Dict[str, Any]) -> str:
        signatories = data.get("signatories", [])
        org = data.get("org_name", "Organization")
        pledge_date = data.get("pledge_date", self.generated_at.strftime("%Y-%m-%d") if self.generated_at else "")

        lines = [
            "## 8. Signatory Authorization\n",
            f"We, the undersigned, on behalf of **{org}**, hereby commit to the Race to Zero "
            f"campaign and the pledges, plans, and commitments described in this letter.\n",
        ]

        if signatories:
            for sig in signatories:
                lines.append(
                    f"**{sig.get('name', '')}**  \n"
                    f"*{sig.get('title', '')}*  \n"
                    f"Date: {sig.get('date', pledge_date)}\n"
                )
        else:
            lines.append(
                "**[Signatory Name]**  \n"
                "*[Title / Position]*  \n"
                f"Date: {pledge_date}\n\n"
                "**[Board Chair / Director Name]**  \n"
                "*Chair of the Board*  \n"
                f"Date: {pledge_date}"
            )

        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*This pledge commitment letter has been generated by GreenLang PACK-025 "
            f"Race to Zero Pack on {ts}.*  \n"
            f"*Race to Zero is a global campaign led by the UNFCCC High-Level Champions.*  \n"
            f"*For more information: https://unfccc.int/climate-action/race-to-zero-campaign*"
        )

    # ------------------------------------------------------------------ #
    #  HTML sections                                                       #
    # ------------------------------------------------------------------ #

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;"
            "padding:20px;background:#f0f4f0;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;"
            "font-size:1.8em;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;"
            "padding-left:12px;}"
            "h3{color:#388e3c;margin-top:20px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;letter-spacing:0.5px;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".badge-met{background:#43a047;color:#fff;padding:2px 10px;border-radius:4px;font-size:0.85em;}"
            ".badge-not-met{background:#ef5350;color:#fff;padding:2px 10px;border-radius:4px;font-size:0.85em;}"
            ".pledge-box{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);padding:30px;"
            "border-radius:12px;border-left:5px solid #2e7d32;margin:20px 0;}"
            ".pledge-title{font-size:1.4em;font-weight:700;color:#1b5e20;margin-bottom:8px;}"
            ".signatory-block{border:2px solid #c8e6c9;border-radius:8px;padding:20px;margin:10px 0;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        campaign = data.get("campaign", "Race to Zero")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Race to Zero -- Pledge Commitment Letter</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Campaign:</strong> {campaign} | '
            f'<strong>Date:</strong> {ts}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        baseline = data.get("baseline", {})
        interim = data.get("interim_target", {})
        longterm = data.get("longterm_target", {})
        total = baseline.get("total_tco2e", 0)
        return (
            f'<h2>1. Executive Summary</h2>\n'
            f'<div class="pledge-box">\n'
            f'  <div class="pledge-title">Net-Zero Pledge</div>\n'
            f'  <p>Baseline: <strong>{_dec_comma(total)} tCO2e</strong> | '
            f'Interim: <strong>{_pct(interim.get("reduction_pct", 50))} by {interim.get("year", 2030)}</strong> | '
            f'Net-Zero: <strong>{longterm.get("year", 2050)}</strong></p>\n'
            f'</div>\n'
            f'<div class="cards">\n'
            f'  <div class="card"><div class="card-label">Baseline Emissions</div>'
            f'<div class="card-value">{_dec_comma(total)}</div>tCO2e</div>\n'
            f'  <div class="card"><div class="card-label">Scope 1</div>'
            f'<div class="card-value">{_dec_comma(baseline.get("scope1_tco2e", 0))}</div>tCO2e</div>\n'
            f'  <div class="card"><div class="card-label">Scope 2</div>'
            f'<div class="card-value">{_dec_comma(baseline.get("scope2_tco2e", 0))}</div>tCO2e</div>\n'
            f'  <div class="card"><div class="card-label">Scope 3</div>'
            f'<div class="card-value">{_dec_comma(baseline.get("scope3_tco2e", 0))}</div>tCO2e</div>\n'
            f'</div>'
        )

    def _html_baseline_emissions(self, data: Dict[str, Any]) -> str:
        b = data.get("baseline", {})
        total = b.get("total_tco2e", 0)
        s1 = b.get("scope1_tco2e", 0)
        s2 = b.get("scope2_tco2e", 0)
        s3 = b.get("scope3_tco2e", 0)
        return (
            f'<h2>2. Baseline Emissions</h2>\n'
            f'<p><strong>Base Year:</strong> {b.get("year", "N/A")} | '
            f'<strong>Methodology:</strong> {b.get("methodology", "GHG Protocol")}</p>\n'
            f'<table><tr><th>Scope</th><th>Emissions (tCO2e)</th><th>% of Total</th></tr>\n'
            f'<tr><td>Scope 1</td><td>{_dec_comma(s1)}</td><td>{_pct(_safe_div(s1, max(total, 1)) * 100)}</td></tr>\n'
            f'<tr><td>Scope 2</td><td>{_dec_comma(s2)}</td><td>{_pct(_safe_div(s2, max(total, 1)) * 100)}</td></tr>\n'
            f'<tr><td>Scope 3</td><td>{_dec_comma(s3)}</td><td>{_pct(_safe_div(s3, max(total, 1)) * 100)}</td></tr>\n'
            f'<tr style="font-weight:bold"><td>Total</td><td>{_dec_comma(total)}</td><td>100%</td></tr>\n'
            f'</table>'
        )

    def _html_interim_target(self, data: Dict[str, Any]) -> str:
        t = data.get("interim_target", {})
        return (
            f'<h2>3. Interim Target (2030)</h2>\n'
            f'<div class="cards">\n'
            f'  <div class="card"><div class="card-label">Reduction Target</div>'
            f'<div class="card-value">{_pct(t.get("reduction_pct", 50))}</div></div>\n'
            f'  <div class="card"><div class="card-label">Target Year</div>'
            f'<div class="card-value">{t.get("year", 2030)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Scope Coverage</div>'
            f'<div class="card-value">{t.get("scope_coverage", "S1+S2")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Pathway</div>'
            f'<div class="card-value">{t.get("pathway", "1.5C")}</div></div>\n'
            f'</div>'
        )

    def _html_longterm_target(self, data: Dict[str, Any]) -> str:
        lt = data.get("longterm_target", {})
        return (
            f'<h2>4. Long-Term Target ({lt.get("year", 2050)})</h2>\n'
            f'<div class="cards">\n'
            f'  <div class="card"><div class="card-label">Target</div>'
            f'<div class="card-value">{lt.get("target", "Net-Zero")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Min Reduction</div>'
            f'<div class="card-value">{_pct(lt.get("min_reduction_pct", 90))}</div></div>\n'
            f'  <div class="card"><div class="card-label">Max Residual</div>'
            f'<div class="card-value">{_pct(lt.get("residual_emissions_pct", 10))}</div></div>\n'
            f'</div>'
        )

    def _html_starting_line(self, data: Dict[str, Any]) -> str:
        sl = data.get("starting_line", {})
        items = sl.get("items", [])
        criteria_list = items if items else [
            {"id": c["id"], "category": c["category"], "criterion": c["criterion"], "met": False}
            for c in self.STARTING_LINE_CRITERIA
        ]

        met_count = sum(1 for it in criteria_list if it.get("met", False))
        total = len(criteria_list)

        rows = ""
        for item in criteria_list:
            badge = "badge-met" if item.get("met", False) else "badge-not-met"
            label = "MET" if item.get("met", False) else "NOT MET"
            rows += (
                f'<tr><td>{item.get("id", "-")}</td><td>{item.get("category", "-")}</td>'
                f'<td>{item.get("criterion", "-")}</td>'
                f'<td><span class="{badge}">{label}</span></td></tr>\n'
            )

        return (
            f'<h2>5. Starting Line Compliance</h2>\n'
            f'<p><strong>{met_count}/{total}</strong> criteria met '
            f'({_pct(_safe_div(met_count, total) * 100)})</p>\n'
            f'<table><tr><th>ID</th><th>Category</th><th>Criterion</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_action_plan_timeline(self, data: Dict[str, Any]) -> str:
        milestones = data.get("milestones", [])
        rows = ""
        if milestones:
            for ms in milestones:
                rows += (
                    f'<tr><td>{ms.get("phase", "-")}</td>'
                    f'<td>{ms.get("milestone", "-")}</td>'
                    f'<td>{ms.get("target_date", "-")}</td>'
                    f'<td>{ms.get("status", "Planned")}</td></tr>\n'
                )
        else:
            rows = '<tr><td colspan="4"><em>Action plan timeline to be developed</em></td></tr>'

        return (
            f'<h2>6. Action Plan Timeline</h2>\n'
            f'<table><tr><th>Phase</th><th>Milestone</th><th>Target Date</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_verification_schedule(self, data: Dict[str, Any]) -> str:
        schedule = data.get("verification_schedule", [])
        rows = ""
        if schedule:
            for v in schedule:
                rows += (
                    f'<tr><td>{v.get("year", "-")}</td>'
                    f'<td>{v.get("type", "-")}</td>'
                    f'<td>{v.get("provider", "-")}</td>'
                    f'<td>{v.get("status", "Planned")}</td></tr>\n'
                )
        else:
            rows = '<tr><td colspan="4"><em>Verification schedule pending</em></td></tr>'

        return (
            f'<h2>7. Verification Schedule</h2>\n'
            f'<table><tr><th>Year</th><th>Type</th><th>Provider</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_signatory(self, data: Dict[str, Any]) -> str:
        signatories = data.get("signatories", [])
        org = data.get("org_name", "Organization")
        blocks = ""
        if signatories:
            for sig in signatories:
                blocks += (
                    f'<div class="signatory-block">'
                    f'<strong>{sig.get("name", "")}</strong><br>'
                    f'<em>{sig.get("title", "")}</em><br>'
                    f'Date: {sig.get("date", "")}</div>\n'
                )
        else:
            blocks = (
                '<div class="signatory-block">'
                '<strong>[Signatory Name]</strong><br>'
                '<em>[Title / Position]</em><br>'
                'Date: _______________</div>'
            )

        return (
            f'<h2>8. Signatory Authorization</h2>\n'
            f'<p>On behalf of <strong>{org}</strong>:</p>\n{blocks}'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-025 Race to Zero Pack on {ts}<br>'
            f'Race to Zero Campaign -- UNFCCC High-Level Champions'
            f'</div>'
        )
