# -*- coding: utf-8 -*-
"""
SMEGrantReportTemplate - Matched grants and funding opportunities for PACK-026.

Renders a report of matched grant opportunities for SMEs pursuing
decarbonization, with eligibility scores, application timelines,
requirements checklists, pre-filled data fields, and a grant
application calendar.

Sections:
    1. Grants Summary (top 3-5 matched grants)
    2. Detailed Grant Cards (per grant)
    3. Grant Application Calendar (timeline view)
    4. Pre-Filled Data Fields
    5. Application Checklist

Author: GreenLang Team
Version: 26.0.0
Pack: PACK-026 SME Net Zero Pack
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

_MODULE_VERSION = "26.0.0"
_PACK_ID = "PACK-026"
_TEMPLATE_ID = "sme_grant_report"

_PRIMARY = "#1b5e20"
_SECONDARY = "#2e7d32"
_ACCENT = "#43a047"
_LIGHT = "#e8f5e9"
_LIGHTER = "#f1f8e9"
_CARD_BG = "#c8e6c9"

# ---------------------------------------------------------------------------
# Eligibility scoring helpers
# ---------------------------------------------------------------------------
_ELIG_BANDS = {
    "excellent": {"min": 80, "label": "Excellent Match", "color": "#4caf50"},
    "good": {"min": 60, "label": "Good Match", "color": "#8bc34a"},
    "moderate": {"min": 40, "label": "Moderate Match", "color": "#ffc107"},
    "low": {"min": 0, "label": "Low Match", "color": "#ff9800"},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def _safe_div(num: Any, den: Any, default: float = 0.0) -> float:
    try:
        d = float(den)
        return float(num) / d if d != 0 else default
    except Exception:
        return default

def _elig_band(score: float) -> Dict[str, Any]:
    """Return eligibility band for a numeric score 0-100."""
    if score >= 80:
        return _ELIG_BANDS["excellent"]
    elif score >= 60:
        return _ELIG_BANDS["good"]
    elif score >= 40:
        return _ELIG_BANDS["moderate"]
    return _ELIG_BANDS["low"]

def _progress_bar_ascii(score: float, width: int = 20) -> str:
    """Render an ASCII progress bar for eligibility score."""
    filled = int(round(score / 100 * width))
    filled = max(0, min(width, filled))
    return "[" + "#" * filled + "." * (width - filled) + f"] {_dec(score, 0)}/100"

# ===========================================================================
# Template Class
# ===========================================================================

class SMEGrantReportTemplate:
    """
    SME grant opportunities report template.

    Renders matched grants with eligibility scores, requirements,
    timelines, pre-filled application data, and a calendar view
    across Markdown, HTML, JSON, and Excel formats.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID
    FORMATS = ["markdown", "html", "json", "excel"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    #  Public render methods                                               #
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the grant report as Markdown."""
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_grants_summary(data),
            self._md_grant_cards(data),
            self._md_application_calendar(data),
            self._md_prefilled_data(data),
            self._md_checklist(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the grant report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_grants_summary(data),
            self._html_grant_cards(data),
            self._html_application_calendar(data),
            self._html_prefilled_data(data),
            self._html_checklist(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'<title>SME Grant Opportunities</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the grant report as structured JSON."""
        self.generated_at = utcnow()
        grants = data.get("grants", [])
        currency = data.get("currency", "GBP")

        processed_grants = []
        total_min_funding = 0.0
        total_max_funding = 0.0

        for g in grants:
            band = _elig_band(float(g.get("eligibility_score", 0)))
            min_f = float(g.get("funding_min", 0))
            max_f = float(g.get("funding_max", 0))
            total_min_funding += min_f
            total_max_funding += max_f

            processed_grants.append({
                "name": g.get("name", ""),
                "funding_body": g.get("funding_body", ""),
                "program_type": g.get("program_type", ""),
                "funding_min": round(min_f, 2),
                "funding_max": round(max_f, 2),
                "currency": currency,
                "eligibility_score": round(float(g.get("eligibility_score", 0)), 1),
                "eligibility_band": band["label"],
                "application_deadline": g.get("application_deadline", ""),
                "requirements": g.get("requirements", []),
                "documentation_needed": g.get("documentation_needed", []),
                "estimated_application_hours": int(g.get("estimated_hours", 0)),
                "match_reasons": g.get("match_reasons", []),
                "website": g.get("website", ""),
            })

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "organization": {
                "name": data.get("org_name", ""),
                "sector": data.get("sector", ""),
                "employees": data.get("employees", 0),
                "region": data.get("region", ""),
                "baseline_tco2e": data.get("baseline_tco2e", 0),
            },
            "grants_matched": len(processed_grants),
            "total_potential_funding": {
                "min": round(total_min_funding, 2),
                "max": round(total_max_funding, 2),
                "currency": currency,
            },
            "grants": processed_grants,
            "prefilled_data": data.get("prefilled_data", {}),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render Excel-ready data structure."""
        self.generated_at = utcnow()
        grants = data.get("grants", [])
        currency = data.get("currency", "GBP")

        grants_sheet = {
            "name": "Matched Grants",
            "headers": [
                "Grant Name", "Funding Body", "Program Type",
                f"Min Funding ({currency})", f"Max Funding ({currency})",
                "Eligibility Score", "Match Level",
                "Deadline", "Est. Hours to Apply",
            ],
            "rows": [],
        }
        for g in grants:
            band = _elig_band(float(g.get("eligibility_score", 0)))
            grants_sheet["rows"].append([
                g.get("name", ""),
                g.get("funding_body", ""),
                g.get("program_type", ""),
                round(float(g.get("funding_min", 0)), 2),
                round(float(g.get("funding_max", 0)), 2),
                round(float(g.get("eligibility_score", 0)), 1),
                band["label"],
                g.get("application_deadline", ""),
                int(g.get("estimated_hours", 0)),
            ])

        requirements_sheet = {
            "name": "Requirements",
            "headers": ["Grant Name", "Requirement", "Status"],
            "rows": [],
        }
        for g in grants:
            for req in g.get("requirements", []):
                requirements_sheet["rows"].append([
                    g.get("name", ""),
                    req.get("requirement", req) if isinstance(req, dict) else str(req),
                    req.get("status", "Pending") if isinstance(req, dict) else "Pending",
                ])

        docs_sheet = {
            "name": "Documentation Needed",
            "headers": ["Grant Name", "Document", "Available"],
            "rows": [],
        }
        for g in grants:
            for doc in g.get("documentation_needed", []):
                docs_sheet["rows"].append([
                    g.get("name", ""),
                    doc.get("document", doc) if isinstance(doc, dict) else str(doc),
                    doc.get("available", "No") if isinstance(doc, dict) else "No",
                ])

        calendar_sheet = {
            "name": "Application Calendar",
            "headers": ["Grant Name", "Deadline", "Prep Start", "Est. Hours"],
            "rows": [],
        }
        for g in grants:
            calendar_sheet["rows"].append([
                g.get("name", ""),
                g.get("application_deadline", ""),
                g.get("prep_start_date", ""),
                int(g.get("estimated_hours", 0)),
            ])

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "filename": f"sme_grants_{data.get('org_name', 'org').replace(' ', '_')}.xlsx",
            "worksheets": [grants_sheet, requirements_sheet, docs_sheet, calendar_sheet],
            "chart_definitions": [
                {
                    "type": "bar",
                    "title": "Funding Range by Grant",
                    "worksheet": "Matched Grants",
                    "data_range": "D2:E" + str(len(grants) + 1),
                    "labels_range": "A2:A" + str(len(grants) + 1),
                    "colors": [_SECONDARY, _ACCENT],
                },
            ],
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    #  Markdown sections                                                   #
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Grant & Funding Opportunities\n\n"
            f"**Organization:** {data.get('org_name', 'Your Company')}  \n"
            f"**Sector:** {data.get('sector', '')}  \n"
            f"**Region:** {data.get('region', '')}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_grants_summary(self, data: Dict[str, Any]) -> str:
        grants = data.get("grants", [])
        currency = data.get("currency", "GBP")

        total_min = sum(float(g.get("funding_min", 0)) for g in grants)
        total_max = sum(float(g.get("funding_max", 0)) for g in grants)

        lines = [
            f"## Matched Grants Summary\n",
            f"**{len(grants)} grant(s) matched** to your profile "
            f"| Potential funding: {currency} {_dec_comma(total_min)} - "
            f"{currency} {_dec_comma(total_max)}\n",
            f"| # | Grant | Funding Body | Funding Range | Eligibility | Deadline |",
            f"|--:|-------|-------------|:-------------:|:-----------:|:--------:|",
        ]
        for idx, g in enumerate(grants, 1):
            band = _elig_band(float(g.get("eligibility_score", 0)))
            lines.append(
                f"| {idx} "
                f"| {g.get('name', '')} "
                f"| {g.get('funding_body', '')} "
                f"| {currency} {_dec_comma(g.get('funding_min', 0))} - "
                f"{_dec_comma(g.get('funding_max', 0))} "
                f"| {_dec(g.get('eligibility_score', 0), 0)}/100 ({band['label']}) "
                f"| {g.get('application_deadline', '')} |"
            )

        return "\n".join(lines)

    def _md_grant_cards(self, data: Dict[str, Any]) -> str:
        grants = data.get("grants", [])
        currency = data.get("currency", "GBP")

        lines = ["## Detailed Grant Information\n"]
        for idx, g in enumerate(grants, 1):
            band = _elig_band(float(g.get("eligibility_score", 0)))
            score = float(g.get("eligibility_score", 0))

            lines.append(f"### {idx}. {g.get('name', '')}")
            lines.append(
                f"**Funding Body:** {g.get('funding_body', '')} | "
                f"**Program:** {g.get('program_type', '')}  \n"
                f"**Funding Range:** {currency} {_dec_comma(g.get('funding_min', 0))} - "
                f"{currency} {_dec_comma(g.get('funding_max', 0))}  \n"
                f"**Eligibility Score:** {_progress_bar_ascii(score)} ({band['label']})  \n"
                f"**Deadline:** {g.get('application_deadline', '')}  \n"
                f"**Estimated Time to Apply:** {g.get('estimated_hours', 0)} hours"
            )

            # Requirements checklist
            reqs = g.get("requirements", [])
            if reqs:
                lines.append("\n**Key Requirements:**")
                for req in reqs:
                    if isinstance(req, dict):
                        status = "x" if req.get("met", False) else " "
                        lines.append(f"- [{status}] {req.get('requirement', '')}")
                    else:
                        lines.append(f"- [ ] {req}")

            # Documentation needed
            docs = g.get("documentation_needed", [])
            if docs:
                lines.append("\n**Documentation Needed:**")
                for doc in docs:
                    if isinstance(doc, dict):
                        avail = "x" if doc.get("available", False) else " "
                        lines.append(f"- [{avail}] {doc.get('document', '')}")
                    else:
                        lines.append(f"- [ ] {doc}")

            # Match reasons
            reasons = g.get("match_reasons", [])
            if reasons:
                lines.append(f"\n**Why this matches you:** {'; '.join(reasons)}")

            lines.append("")

        return "\n".join(lines)

    def _md_application_calendar(self, data: Dict[str, Any]) -> str:
        grants = data.get("grants", [])

        lines = [
            "## Application Calendar\n",
            "| Grant | Deadline | Prep Start | Hours Needed | Priority |",
            "|-------|:--------:|:----------:|:------------:|:--------:|",
        ]
        for g in sorted(grants, key=lambda x: x.get("application_deadline", "9999")):
            score = float(g.get("eligibility_score", 0))
            priority = "HIGH" if score >= 80 else ("MEDIUM" if score >= 60 else "LOW")
            lines.append(
                f"| {g.get('name', '')} "
                f"| {g.get('application_deadline', '')} "
                f"| {g.get('prep_start_date', 'TBD')} "
                f"| {g.get('estimated_hours', 0)} hrs "
                f"| {priority} |"
            )

        lines.append("")
        lines.append("> **Tip:** Start with the highest-scoring grants first. "
                     "Many grants share similar documentation - prepare once, apply multiple times.")

        return "\n".join(lines)

    def _md_prefilled_data(self, data: Dict[str, Any]) -> str:
        prefilled = data.get("prefilled_data", {})
        if not prefilled:
            return ""

        lines = [
            "## Pre-Filled Application Data\n",
            "The following data from your baseline can be reused across grant applications:\n",
            "| Field | Value | Source |",
            "|-------|-------|--------|",
        ]
        for field in prefilled.get("fields", []):
            lines.append(
                f"| {field.get('name', '')} "
                f"| {field.get('value', '')} "
                f"| {field.get('source', 'Baseline report')} |"
            )

        return "\n".join(lines)

    def _md_checklist(self, data: Dict[str, Any]) -> str:
        lines = [
            "## Application Readiness Checklist\n",
            "- [ ] Company registration documents (Companies House, etc.)",
            "- [ ] Latest annual accounts / financial statements",
            "- [ ] GHG emissions baseline report",
            "- [ ] Energy bills (last 12 months)",
            "- [ ] Decarbonization action plan / quick wins report",
            "- [ ] Board resolution / letter of support",
            "- [ ] Quotes from suppliers (for capital equipment)",
            "- [ ] Bank details for grant payments",
            "- [ ] Evidence of match funding (if required)",
            "- [ ] Site plan / floor plan (for building upgrades)",
        ]
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-026 SME Net Zero Pack on {ts}*  \n"
            f"*Grant eligibility scores are indicative. Always verify requirements "
            f"with the funding body before applying.*"
        )

    # ------------------------------------------------------------------ #
    #  HTML sections                                                       #
    # ------------------------------------------------------------------ #

    def _css(self) -> str:
        return (
            "*, *::before, *::after{box-sizing:border-box;}"
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            f"background:#f5f7f5;color:#1a1a2e;line-height:1.6;}}"
            f".report{{max-width:900px;margin:0 auto;background:#fff;padding:32px;"
            f"border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:28px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"h3{{color:#388e3c;margin-top:20px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid {_CARD_BG};padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".grant-card{{background:{_LIGHTER};border:1px solid {_CARD_BG};"
            f"border-left:5px solid {_ACCENT};border-radius:0 10px 10px 0;"
            f"padding:16px;margin:16px 0;}}"
            f".grant-header{{display:flex;justify-content:space-between;align-items:center;"
            f"flex-wrap:wrap;gap:8px;margin-bottom:8px;}}"
            f".grant-name{{font-size:1.15em;font-weight:700;color:{_PRIMARY};}}"
            f".elig-badge{{display:inline-block;padding:4px 12px;border-radius:12px;"
            f"font-size:0.8em;font-weight:600;color:#fff;}}"
            f".grant-meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));"
            f"gap:8px;margin:10px 0;}}"
            f".grant-meta-item{{padding:8px;background:#fff;border-radius:6px;text-align:center;}}"
            f".meta-label{{font-size:0.7em;color:#689f38;text-transform:uppercase;}}"
            f".meta-value{{font-size:1em;font-weight:600;color:{_PRIMARY};}}"
            f".progress-track{{height:12px;background:#e0e0e0;border-radius:6px;overflow:hidden;"
            f"margin:4px 0;}}"
            f".progress-fill{{height:100%;border-radius:6px;transition:width 0.3s ease;}}"
            f".req-list{{list-style:none;padding:0;margin:8px 0;}}"
            f".req-list li{{padding:4px 0;font-size:0.9em;}}"
            f".req-list li::before{{content:'\\2610  ';color:{_ACCENT};font-weight:700;}}"
            f".req-list li.met::before{{content:'\\2611  ';color:{_PRIMARY};}}"
            f".calendar-card{{background:{_LIGHT};padding:12px;border-radius:8px;margin:6px 0;"
            f"display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;}}"
            f".priority-high{{color:#1b5e20;font-weight:700;}}"
            f".priority-medium{{color:#f57f17;font-weight:600;}}"
            f".priority-low{{color:#9e9e9e;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));"
            f"gap:12px;margin:16px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_CARD_BG});border-radius:10px;"
            f"padding:16px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.75em;color:#558b2f;text-transform:uppercase;}}"
            f".card-value{{font-size:1.4em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.7em;color:#689f38;}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#689f38;font-size:0.8em;text-align:center;}}"
            f"@media(max-width:600px){{.grant-meta{{grid-template-columns:1fr;}}"
            f".report{{padding:16px;}}}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Grant & Funding Opportunities</h1>\n'
            f'<p><strong>{data.get("org_name", "Your Company")}</strong> | '
            f'{data.get("sector", "")} | {data.get("region", "")} | '
            f'Generated: {ts}</p>'
        )

    def _html_grants_summary(self, data: Dict[str, Any]) -> str:
        grants = data.get("grants", [])
        currency = data.get("currency", "GBP")
        total_min = sum(float(g.get("funding_min", 0)) for g in grants)
        total_max = sum(float(g.get("funding_max", 0)) for g in grants)

        return (
            f'<h2>Matched Grants Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Grants Matched</div>'
            f'<div class="card-value">{len(grants)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Min Funding</div>'
            f'<div class="card-value">{currency} {_dec_comma(total_min)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Max Funding</div>'
            f'<div class="card-value">{currency} {_dec_comma(total_max)}</div></div>\n'
            f'</div>'
        )

    def _html_grant_cards(self, data: Dict[str, Any]) -> str:
        grants = data.get("grants", [])
        currency = data.get("currency", "GBP")
        cards = ""

        for idx, g in enumerate(grants, 1):
            score = float(g.get("eligibility_score", 0))
            band = _elig_band(score)

            # Requirements
            reqs_html = ""
            reqs = g.get("requirements", [])
            if reqs:
                items = ""
                for req in reqs:
                    if isinstance(req, dict):
                        cls = " met" if req.get("met", False) else ""
                        items += f'<li class="{cls}">{req.get("requirement", "")}</li>\n'
                    else:
                        items += f'<li>{req}</li>\n'
                reqs_html = f'<p style="font-weight:600;margin-top:8px;">Requirements:</p><ul class="req-list">{items}</ul>'

            # Documentation
            docs_html = ""
            docs = g.get("documentation_needed", [])
            if docs:
                items = ""
                for doc in docs:
                    if isinstance(doc, dict):
                        cls = " met" if doc.get("available", False) else ""
                        items += f'<li class="{cls}">{doc.get("document", "")}</li>\n'
                    else:
                        items += f'<li>{doc}</li>\n'
                docs_html = f'<p style="font-weight:600;margin-top:8px;">Documentation Needed:</p><ul class="req-list">{items}</ul>'

            cards += (
                f'<div class="grant-card">\n'
                f'  <div class="grant-header">\n'
                f'    <span class="grant-name">{idx}. {g.get("name", "")}</span>\n'
                f'    <span class="elig-badge" style="background:{band["color"]}">'
                f'{band["label"]} ({_dec(score, 0)}/100)</span>\n'
                f'  </div>\n'
                f'  <div class="grant-meta">\n'
                f'    <div class="grant-meta-item"><div class="meta-label">Funding Body</div>'
                f'<div class="meta-value">{g.get("funding_body", "")}</div></div>\n'
                f'    <div class="grant-meta-item"><div class="meta-label">Funding Range</div>'
                f'<div class="meta-value">{currency} {_dec_comma(g.get("funding_min", 0))} - '
                f'{_dec_comma(g.get("funding_max", 0))}</div></div>\n'
                f'    <div class="grant-meta-item"><div class="meta-label">Deadline</div>'
                f'<div class="meta-value">{g.get("application_deadline", "")}</div></div>\n'
                f'    <div class="grant-meta-item"><div class="meta-label">Time to Apply</div>'
                f'<div class="meta-value">{g.get("estimated_hours", 0)} hours</div></div>\n'
                f'  </div>\n'
                f'  <div class="progress-track">'
                f'<div class="progress-fill" style="width:{score}%;background:{band["color"]}">'
                f'</div></div>\n'
                f'  {reqs_html}\n'
                f'  {docs_html}\n'
                f'</div>\n'
            )

        return f'<h2>Detailed Grant Information</h2>\n{cards}'

    def _html_application_calendar(self, data: Dict[str, Any]) -> str:
        grants = data.get("grants", [])
        cards = ""

        for g in sorted(grants, key=lambda x: x.get("application_deadline", "9999")):
            score = float(g.get("eligibility_score", 0))
            if score >= 80:
                priority_cls = "priority-high"
                priority_label = "HIGH"
            elif score >= 60:
                priority_cls = "priority-medium"
                priority_label = "MEDIUM"
            else:
                priority_cls = "priority-low"
                priority_label = "LOW"

            cards += (
                f'<div class="calendar-card">\n'
                f'  <div><strong>{g.get("name", "")}</strong></div>\n'
                f'  <div>Deadline: {g.get("application_deadline", "TBD")}</div>\n'
                f'  <div>{g.get("estimated_hours", 0)} hours</div>\n'
                f'  <div class="{priority_cls}">{priority_label}</div>\n'
                f'</div>\n'
            )

        return (
            f'<h2>Application Calendar</h2>\n{cards}\n'
            f'<p style="background:{_LIGHTER};padding:12px;border-radius:6px;">'
            f'<strong>Tip:</strong> Start with the highest-scoring grants. '
            f'Many share similar documentation - prepare once, apply multiple times.</p>'
        )

    def _html_prefilled_data(self, data: Dict[str, Any]) -> str:
        prefilled = data.get("prefilled_data", {})
        if not prefilled:
            return ""

        fields = prefilled.get("fields", [])
        rows = ""
        for f in fields:
            rows += (
                f'<tr><td>{f.get("name", "")}</td>'
                f'<td><strong>{f.get("value", "")}</strong></td>'
                f'<td>{f.get("source", "Baseline")}</td></tr>\n'
            )

        return (
            f'<h2>Pre-Filled Application Data</h2>\n'
            f'<p>Reuse this data across grant applications:</p>\n'
            f'<table>\n'
            f'<tr><th>Field</th><th>Value</th><th>Source</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_checklist(self, data: Dict[str, Any]) -> str:
        items = [
            "Company registration documents",
            "Latest annual accounts / financial statements",
            "GHG emissions baseline report",
            "Energy bills (last 12 months)",
            "Decarbonization action plan",
            "Board resolution / letter of support",
            "Quotes from suppliers (for capital equipment)",
            "Bank details for grant payments",
            "Evidence of match funding (if required)",
            "Site plan / floor plan (for building upgrades)",
        ]
        li = "".join(f'<li>{item}</li>\n' for item in items)
        return (
            f'<h2>Application Readiness Checklist</h2>\n'
            f'<ul class="req-list">\n{li}</ul>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-026 SME Net Zero Pack on {ts}<br>'
            f'Eligibility scores are indicative - verify with funding body before applying.'
            f'</div>'
        )
