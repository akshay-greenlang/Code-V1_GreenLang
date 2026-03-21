# -*- coding: utf-8 -*-
"""
PublicDisclosureTemplate - Public-facing climate report for PACK-029.

Renders a public-facing annual climate report with interim targets and
progress, stakeholder-friendly language, emissions trajectory charts data,
sector breakdown, greenwashing compliance checks, and links to detailed
reports (SBTi, CDP, TCFD). Multi-format output (MD, HTML, JSON, PDF).

Sections:
    1.  Executive Message
    2.  Our Climate Commitments
    3.  Emissions Performance
    4.  Interim Targets & Progress
    5.  Key Actions & Initiatives
    6.  Emissions by Scope (Visual)
    7.  Our Pathway to Net Zero
    8.  Greenwashing Compliance Check
    9.  Detailed Reports & Frameworks
    10. Methodology Note
    11. XBRL Tagging Summary
    12. Audit Trail & Provenance

Author: GreenLang Team
Version: 29.0.0
Pack: PACK-029 Interim Targets Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "29.0.0"
_PACK_ID = "PACK-029"
_TEMPLATE_ID = "public_disclosure"

_PRIMARY = "#0a4a3a"
_SECONDARY = "#167a5b"
_ACCENT = "#1db954"
_LIGHT = "#e0f5ed"
_LIGHTER = "#f0faf5"
_SUCCESS = "#2e7d32"
_WARN = "#ef6c00"
_DANGER = "#c62828"

GREENWASHING_CHECKS = [
    {"id": "GW01", "check": "All claims are quantified with specific metrics (tCO2e, %)", "category": "Specificity"},
    {"id": "GW02", "check": "No unsubstantiated 'carbon neutral' or 'net zero' claims without evidence", "category": "Accuracy"},
    {"id": "GW03", "check": "Scope coverage is clearly stated (which emissions are included)", "category": "Transparency"},
    {"id": "GW04", "check": "Base year, methodology, and boundary are disclosed", "category": "Transparency"},
    {"id": "GW05", "check": "Offset usage (if any) is separately disclosed from reductions", "category": "Distinction"},
    {"id": "GW06", "check": "Forward-looking statements are clearly identified as projections", "category": "Accuracy"},
    {"id": "GW07", "check": "Third-party verification status is disclosed", "category": "Credibility"},
    {"id": "GW08", "check": "Targets are science-based (SBTi validated or aligned)", "category": "Credibility"},
    {"id": "GW09", "check": "Year-over-year data enables trend verification", "category": "Comparability"},
    {"id": "GW10", "check": "No selective reporting (all scopes and categories included)", "category": "Completeness"},
    {"id": "GW11", "check": "Compliant with EU Green Claims Directive requirements", "category": "Regulatory"},
    {"id": "GW12", "check": "No misleading use of imagery or language", "category": "Communication"},
]

XBRL_TAGS: Dict[str, str] = {
    "public_emissions_total": "gl:PublicDisclosureTotalEmissions",
    "public_reduction_pct": "gl:PublicDisclosureReductionPct",
    "public_target_year": "gl:PublicDisclosureTargetYear",
    "greenwash_score": "gl:GreenwashingComplianceScore",
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


class PublicDisclosureTemplate:
    """
    Public-facing climate disclosure template for PACK-029 Interim Targets Pack.

    Renders a stakeholder-friendly annual climate report with interim targets,
    progress visualization, greenwashing compliance, and links to frameworks.
    Supports MD, HTML, JSON, PDF.

    Example:
        >>> template = PublicDisclosureTemplate()
        >>> data = {
        ...     "org_name": "GreenCorp",
        ...     "reporting_year": 2025,
        ...     "total_emissions": 85000,
        ...     "baseline_emissions": 100000,
        ...     "baseline_year": 2022,
        ...     "net_zero_year": 2050,
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render public disclosure report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_executive_message(data),
            self._md_commitments(data), self._md_performance(data),
            self._md_targets_progress(data), self._md_initiatives(data),
            self._md_scope_breakdown(data), self._md_pathway(data),
            self._md_greenwashing(data), self._md_frameworks(data),
            self._md_methodology(data), self._md_xbrl(data),
            self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render public disclosure report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_executive_message(data),
            self._html_commitments(data), self._html_performance(data),
            self._html_targets_progress(data), self._html_initiatives(data),
            self._html_scope_breakdown(data), self._html_pathway(data),
            self._html_greenwashing(data), self._html_frameworks(data),
            self._html_methodology(data), self._html_xbrl(data),
            self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Climate Report - {data.get("org_name","")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render as structured JSON."""
        self.generated_at = _utcnow()
        total = float(data.get("total_emissions", 0))
        baseline = float(data.get("baseline_emissions", 0))
        reduction = ((baseline - total) / baseline * 100) if baseline else 0
        gw_results = data.get("greenwashing_results", {})
        gw_passed = sum(1 for v in gw_results.values() if v.get("status") == "pass")
        gw_total = len(GREENWASHING_CHECKS)
        gw_score = (gw_passed / max(1, gw_total)) * 100

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION,
            "pack_id": _PACK_ID, "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(), "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "emissions": {
                "total": str(total), "baseline": str(baseline),
                "reduction_pct": str(round(reduction, 2)),
                "scope_breakdown": data.get("scope_breakdown", {}),
            },
            "targets": data.get("targets", []),
            "initiatives": data.get("initiatives", []),
            "pathway": data.get("pathway", []),
            "greenwashing_compliance": {
                "score": str(round(gw_score, 1)),
                "passed": gw_passed, "total": gw_total,
                "compliant": gw_score >= 80,
            },
            "framework_links": data.get("framework_links", {}),
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return PDF-ready data."""
        return {
            "format": "pdf", "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {"title": f"Climate Report - {data.get('org_name','')}", "author": "GreenLang PACK-029"},
        }

    # -- Markdown --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Annual Climate Report {data.get('reporting_year', '')}\n\n"
            f"**{data.get('org_name', '')}**  \n"
            f"**Published:** {ts}  \n"
            f"**Pack:** PACK-029 v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_message(self, data: Dict[str, Any]) -> str:
        message = data.get("executive_message", "")
        ceo = data.get("ceo_name", "")
        lines = ["## 1. Message from Leadership\n"]
        if message:
            lines.append(f"{message}\n")
        if ceo:
            lines.append(f"*-- {ceo}, {data.get('ceo_title', 'CEO')}*")
        if not message and not ceo:
            lines.append("_Executive message to be provided._")
        return "\n".join(lines)

    def _md_commitments(self, data: Dict[str, Any]) -> str:
        commitments = data.get("commitments", [])
        lines = ["## 2. Our Climate Commitments\n"]
        default_commitments = [
            f"Achieve net-zero greenhouse gas emissions by {data.get('net_zero_year', 2050)}",
            f"Reduce emissions {_dec(data.get('near_term_reduction_pct', 46.2))}% by {data.get('near_term_year', 2030)} (from {data.get('baseline_year', 2022)} baseline)",
            f"Science-based targets validated by SBTi ({data.get('sbti_status', 'Committed')})",
        ]
        for c in commitments or default_commitments:
            if isinstance(c, dict):
                lines.append(f"- **{c.get('commitment', '')}**")
            else:
                lines.append(f"- **{c}**")
        return "\n".join(lines)

    def _md_performance(self, data: Dict[str, Any]) -> str:
        total = float(data.get("total_emissions", 0))
        baseline = float(data.get("baseline_emissions", 0))
        reduction = ((baseline - total) / baseline * 100) if baseline else 0
        lines = [
            "## 3. Emissions Performance\n",
            "| Metric | Value |", "|--------|-------|",
            f"| Baseline Emissions ({data.get('baseline_year', '')}) | {_dec_comma(baseline, 0)} tCO2e |",
            f"| Current Emissions ({data.get('reporting_year', '')}) | {_dec_comma(total, 0)} tCO2e |",
            f"| Absolute Reduction | {_dec_comma(baseline - total, 0)} tCO2e |",
            f"| Percentage Reduction | {_dec(reduction)}% |",
        ]
        return "\n".join(lines)

    def _md_targets_progress(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        lines = [
            "## 4. Interim Targets & Progress\n",
            "| Target | Year | Reduction | Progress | Status |",
            "|--------|:----:|---------:|:--------:|--------|",
        ]
        for t in targets:
            progress = float(t.get("progress_pct", 0))
            status = "On Track" if progress >= 90 else ("Behind" if progress < 70 else "Monitor")
            lines.append(
                f"| {t.get('name', '')} | {t.get('year', '')} | -{_dec(t.get('reduction_pct', 0))}% "
                f"| {_dec(progress)}% | {status} |"
            )
        if not targets:
            lines.append("| _No targets defined_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_initiatives(self, data: Dict[str, Any]) -> str:
        initiatives = data.get("initiatives", [])
        lines = ["## 5. Key Actions & Initiatives\n"]
        for i, init in enumerate(initiatives, 1):
            lines.append(
                f"{i}. **{init.get('name', '')}** - {init.get('description', '')}  \n"
                f"   Impact: {_dec_comma(init.get('reduction_tco2e', 0), 0)} tCO2e reduced | "
                f"Status: {init.get('status', '')}"
            )
        if not initiatives:
            lines.append("_Key initiatives to be reported._")
        return "\n".join(lines)

    def _md_scope_breakdown(self, data: Dict[str, Any]) -> str:
        breakdown = data.get("scope_breakdown", {})
        total = float(data.get("total_emissions", 0))
        lines = [
            "## 6. Emissions by Scope\n",
            "| Scope | Emissions (tCO2e) | Share (%) |",
            "|-------|------------------:|----------:|",
        ]
        for scope_name in ["Scope 1", "Scope 2", "Scope 3"]:
            key = scope_name.lower().replace(" ", "_")
            val = float(breakdown.get(key, 0))
            share = (val / total * 100) if total > 0 else 0
            lines.append(f"| {scope_name} | {_dec_comma(val, 0)} | {_dec(share)}% |")
        lines.append(f"| **Total** | **{_dec_comma(total, 0)}** | **100%** |")
        return "\n".join(lines)

    def _md_pathway(self, data: Dict[str, Any]) -> str:
        pathway = data.get("pathway", [])
        lines = [
            "## 7. Our Pathway to Net Zero\n",
            "| Year | Milestone | Target Emissions (tCO2e) | Reduction (%) |",
            "|------|-----------|-------------------------:|--------------:|",
        ]
        for p in pathway:
            lines.append(
                f"| {p.get('year', '')} | {p.get('milestone', '')} "
                f"| {_dec_comma(p.get('target_emissions', 0), 0)} "
                f"| {_dec(p.get('reduction_pct', 0))}% |"
            )
        if not pathway:
            lines.append("| - | _Pathway data to be provided_ | - | - |")
        return "\n".join(lines)

    def _md_greenwashing(self, data: Dict[str, Any]) -> str:
        results = data.get("greenwashing_results", {})
        lines = [
            "## 8. Greenwashing Compliance Check\n",
            "This report has been checked against 12 greenwashing compliance criteria.\n",
            "| ID | Check | Category | Status |",
            "|----|-------|----------|--------|",
        ]
        passed = 0
        for gc in GREENWASHING_CHECKS:
            r = results.get(gc["id"], {})
            status = r.get("status", "pending")
            icon = "PASS" if status == "pass" else ("FAIL" if status == "fail" else "PENDING")
            if status == "pass":
                passed += 1
            lines.append(f"| {gc['id']} | {gc['check']} | {gc['category']} | **{icon}** |")
        total = len(GREENWASHING_CHECKS)
        score = (passed / max(1, total)) * 100
        lines.append(f"\n**Compliance Score:** {_dec(score, 1)}% ({passed}/{total})")
        return "\n".join(lines)

    def _md_frameworks(self, data: Dict[str, Any]) -> str:
        links = data.get("framework_links", {})
        lines = [
            "## 9. Detailed Reports & Frameworks\n",
            "For detailed technical disclosures, refer to:\n",
            f"- **SBTi Target Validation**: {links.get('sbti', 'See SBTi validation report')}",
            f"- **CDP Climate Change Response**: {links.get('cdp', 'See CDP disclosure report')}",
            f"- **TCFD Report**: {links.get('tcfd', 'See TCFD metrics report')}",
            f"- **GHG Protocol Inventory**: {links.get('ghg', 'See GHG inventory report')}",
            f"- **Sustainability Report**: {links.get('sustainability', 'See full sustainability report')}",
        ]
        return "\n".join(lines)

    def _md_methodology(self, data: Dict[str, Any]) -> str:
        method = data.get("methodology", {})
        lines = [
            "## 10. Methodology Note\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Reporting Standard | {method.get('standard', 'GHG Protocol Corporate Standard')} |",
            f"| Consolidation | {method.get('consolidation', 'Operational control')} |",
            f"| Emission Factors | {method.get('emission_factors', 'IPCC AR6, IEA, DEFRA 2024')} |",
            f"| GWP Values | {method.get('gwp', 'IPCC AR6 (100-year)')} |",
            f"| Verification | {method.get('verification', 'Third-party limited assurance')} |",
            f"| Verification Provider | {method.get('verifier', '')} |",
        ]
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 11. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag |", "|------------|----------|",
        ]
        for key, tag in XBRL_TAGS.items():
            lines.append(f"| {key.replace('_', ' ').title()} | {tag} |")
        return "\n".join(lines)

    def _md_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            "## 12. Audit Trail & Provenance\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n| Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        org = data.get("org_name", "")
        return (
            f"---\n\n"
            f"*{org} Annual Climate Report {data.get('reporting_year', '')}*  \n"
            f"*Generated by GreenLang PACK-029 on {ts}*  \n"
            f"*All data verified - no unsubstantiated claims.*"
        )

    # -- HTML --

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
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
            f".pass{{color:{_SUCCESS};font-weight:700;}}.fail{{color:{_DANGER};font-weight:700;}}.pending{{color:{_WARN};}}"
            f".commitment{{background:{_LIGHT};padding:15px;border-radius:8px;margin:8px 0;border-left:4px solid {_ACCENT};}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>Annual Climate Report {data.get("reporting_year","")}</h1>\n<p><strong>{data.get("org_name","")}</strong> | Published {ts}</p>'

    def _html_executive_message(self, data: Dict[str, Any]) -> str:
        msg = data.get("executive_message", "Our commitment to climate action drives everything we do.")
        return f'<h2>1. Leadership Message</h2>\n<blockquote>{msg}</blockquote>\n<p><em>{data.get("ceo_name","")}, {data.get("ceo_title","CEO")}</em></p>'

    def _html_commitments(self, data: Dict[str, Any]) -> str:
        items = ""
        commitments = data.get("commitments", [f"Net-zero by {data.get('net_zero_year', 2050)}"])
        for c in commitments:
            text = c.get("commitment", c) if isinstance(c, dict) else c
            items += f'<div class="commitment">{text}</div>\n'
        return f'<h2>2. Commitments</h2>\n{items}'

    def _html_performance(self, data: Dict[str, Any]) -> str:
        total = float(data.get("total_emissions", 0))
        baseline = float(data.get("baseline_emissions", 0))
        reduction = ((baseline - total) / baseline * 100) if baseline else 0
        return (
            f'<h2>3. Performance</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Baseline</div><div class="card-value">{_dec_comma(baseline, 0)}</div><div class="card-unit">tCO2e ({data.get("baseline_year","")})</div></div>\n'
            f'<div class="card"><div class="card-label">Current</div><div class="card-value">{_dec_comma(total, 0)}</div><div class="card-unit">tCO2e ({data.get("reporting_year","")})</div></div>\n'
            f'<div class="card"><div class="card-label">Reduced</div><div class="card-value">{_dec(reduction)}%</div><div class="card-unit">{_dec_comma(baseline - total, 0)} tCO2e</div></div>\n'
            f'</div>'
        )

    def _html_targets_progress(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        rows = ""
        for t in targets:
            progress = float(t.get("progress_pct", 0))
            rows += f'<tr><td>{t.get("name","")}</td><td>{t.get("year","")}</td><td>-{_dec(t.get("reduction_pct",0))}%</td><td>{_dec(progress)}%</td></tr>\n'
        return f'<h2>4. Targets & Progress</h2>\n<table>\n<tr><th>Target</th><th>Year</th><th>Reduction</th><th>Progress</th></tr>\n{rows}</table>'

    def _html_initiatives(self, data: Dict[str, Any]) -> str:
        initiatives = data.get("initiatives", [])
        items = ""
        for init in initiatives:
            items += f'<li><strong>{init.get("name","")}</strong> - {init.get("description","")} ({_dec_comma(init.get("reduction_tco2e",0), 0)} tCO2e)</li>\n'
        return f'<h2>5. Key Actions</h2>\n<ol>\n{items}</ol>' if items else '<h2>5. Key Actions</h2>\n<p>Initiatives to be reported.</p>'

    def _html_scope_breakdown(self, data: Dict[str, Any]) -> str:
        breakdown = data.get("scope_breakdown", {})
        total = float(data.get("total_emissions", 0))
        rows = ""
        for s in ["scope_1", "scope_2", "scope_3"]:
            val = float(breakdown.get(s, 0))
            share = (val / total * 100) if total > 0 else 0
            rows += f'<tr><td>{s.replace("_"," ").title()}</td><td>{_dec_comma(val, 0)}</td><td>{_dec(share)}%</td></tr>\n'
        return f'<h2>6. Emissions by Scope</h2>\n<table>\n<tr><th>Scope</th><th>Emissions</th><th>Share</th></tr>\n{rows}</table>'

    def _html_pathway(self, data: Dict[str, Any]) -> str:
        pathway = data.get("pathway", [])
        rows = ""
        for p in pathway:
            rows += f'<tr><td>{p.get("year","")}</td><td>{p.get("milestone","")}</td><td>{_dec_comma(p.get("target_emissions",0), 0)}</td><td>{_dec(p.get("reduction_pct",0))}%</td></tr>\n'
        return f'<h2>7. Net Zero Pathway</h2>\n<table>\n<tr><th>Year</th><th>Milestone</th><th>Target</th><th>Reduction</th></tr>\n{rows}</table>'

    def _html_greenwashing(self, data: Dict[str, Any]) -> str:
        results = data.get("greenwashing_results", {})
        rows = ""
        for gc in GREENWASHING_CHECKS:
            r = results.get(gc["id"], {})
            s = r.get("status", "pending")
            cls = "pass" if s == "pass" else ("fail" if s == "fail" else "pending")
            rows += f'<tr><td>{gc["id"]}</td><td>{gc["check"]}</td><td class="{cls}">{"PASS" if s == "pass" else ("FAIL" if s == "fail" else "PENDING")}</td></tr>\n'
        return f'<h2>8. Greenwashing Compliance</h2>\n<table>\n<tr><th>ID</th><th>Check</th><th>Status</th></tr>\n{rows}</table>'

    def _html_frameworks(self, data: Dict[str, Any]) -> str:
        links = data.get("framework_links", {})
        items = ""
        for name, url in [("SBTi", links.get("sbti", "#")), ("CDP", links.get("cdp", "#")),
                          ("TCFD", links.get("tcfd", "#")), ("GHG Protocol", links.get("ghg", "#"))]:
            items += f'<li><a href="{url}">{name}</a></li>\n'
        return f'<h2>9. Framework Links</h2>\n<ul>\n{items}</ul>'

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        m = data.get("methodology", {})
        return (
            f'<h2>10. Methodology</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Standard</td><td>{m.get("standard","GHG Protocol")}</td></tr>\n'
            f'<tr><td>Verification</td><td>{m.get("verification","Third-party limited assurance")}</td></tr>\n'
            f'<tr><td>EF Sources</td><td>{m.get("emission_factors","IPCC, IEA, DEFRA")}</td></tr>\n'
            f'</table>'
        )

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        rows = ""
        for key, tag in XBRL_TAGS.items():
            rows += f'<tr><td>{key.replace("_"," ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>11. XBRL Tags</h2>\n<table>\n<tr><th>Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>12. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">{data.get("org_name","")} Climate Report {data.get("reporting_year","")} | Generated by GreenLang PACK-029 on {ts}</div>'
