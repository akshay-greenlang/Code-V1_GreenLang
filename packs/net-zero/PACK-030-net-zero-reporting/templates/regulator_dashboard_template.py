# -*- coding: utf-8 -*-
"""
RegulatorDashboardTemplate - Regulator Dashboard Template for PACK-030.

Renders a regulator-focused compliance dashboard covering CSRD and SEC
climate disclosure requirements. Provides compliance status tracking,
mandatory disclosure coverage, audit trail and evidence links, data quality
indicators, filing timeline adherence, and cross-framework mapping.
Multi-format output (MD, HTML, JSON, PDF) with SHA-256 provenance.

Sections:
    1.  Executive Summary
    2.  Compliance Status Overview
    3.  CSRD / ESRS E1 Coverage
    4.  SEC Reg S-K Climate Coverage
    5.  Mandatory Disclosure Checklist
    6.  Data Quality Indicators
    7.  Filing Timeline & Deadlines
    8.  Audit Trail & Evidence
    9.  Enforcement Risk Assessment
    10. Cross-Framework Mapping
    11. Provenance & Audit

Author: GreenLang Team
Version: 30.0.0
Pack: PACK-030 Net Zero Reporting Pack
"""

import hashlib, json, logging, uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_MODULE_VERSION = "30.0.0"
_PACK_ID = "PACK-030"
_TEMPLATE_ID = "regulator_dashboard"
_PRIMARY = "#1a237e"
_SECONDARY = "#283593"
_ACCENT = "#c62828"
_LIGHT = "#e8eaf6"
_LIGHTER = "#f3f4fb"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow(): return datetime.now(timezone.utc).replace(microsecond=0)
def _new_uuid(): return str(uuid.uuid4())
def _compute_hash(data):
    raw = json.dumps(data, sort_keys=True, default=str) if isinstance(data, dict) else str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
def _dec(val, places=2):
    try: return str(Decimal(str(val)).quantize(Decimal("0." + "0" * places), rounding=ROUND_HALF_UP))
    except: return str(val)
def _dec_comma(val, places=2):
    try:
        rounded = Decimal(str(val)).quantize(Decimal("0." + "0" * places if places > 0 else "0"), rounding=ROUND_HALF_UP)
        parts = str(rounded).split("."); ip = parts[0]; neg = ip.startswith("-")
        if neg: ip = ip[1:]
        fmt = ""
        for i, c in enumerate(reversed(ip)):
            if i > 0 and i % 3 == 0: fmt = "," + fmt
            fmt = c + fmt
        if neg: fmt = "-" + fmt
        return fmt + ("." + parts[1] if len(parts) > 1 else "")
    except: return str(val)

# ---------------------------------------------------------------------------
# Regulatory Framework Requirements
# ---------------------------------------------------------------------------

CSRD_ESRS_E1_REQUIREMENTS: List[Dict[str, str]] = [
    {"code": "E1-1", "title": "Transition plan for climate change mitigation", "mandatory": "Yes"},
    {"code": "E1-2", "title": "Policies related to climate change", "mandatory": "Yes"},
    {"code": "E1-3", "title": "Actions and resources", "mandatory": "Yes"},
    {"code": "E1-4", "title": "GHG emission reduction targets", "mandatory": "Yes"},
    {"code": "E1-5", "title": "Energy consumption and mix", "mandatory": "Yes"},
    {"code": "E1-6", "title": "Gross Scopes 1, 2, 3 GHG emissions", "mandatory": "Yes"},
    {"code": "E1-7", "title": "GHG removals and carbon credits", "mandatory": "If material"},
    {"code": "E1-8", "title": "Internal carbon pricing", "mandatory": "If applied"},
    {"code": "E1-9", "title": "Anticipated financial effects", "mandatory": "Yes"},
]

SEC_REG_SK_REQUIREMENTS: List[Dict[str, str]] = [
    {"code": "1502", "title": "Governance of climate-related risks", "mandatory": "Yes"},
    {"code": "1503", "title": "Strategy, business model, outlook", "mandatory": "Yes"},
    {"code": "1504", "title": "Risk management processes", "mandatory": "Yes"},
    {"code": "1505", "title": "GHG emissions metrics", "mandatory": "LAF/AF only"},
    {"code": "1506", "title": "Targets and goals", "mandatory": "If set"},
]

RISK_LEVELS = {"High": "RED", "Medium": "AMBER", "Low": "GREEN"}


class RegulatorDashboardTemplate:
    """Regulator compliance dashboard template for PACK-030. Supports MD, HTML, JSON, PDF."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # -----------------------------------------------------------------------
    # Public Render Methods
    # -----------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_compliance_status(data), self._md_csrd_coverage(data),
            self._md_sec_coverage(data), self._md_checklist(data),
            self._md_data_quality(data), self._md_timeline(data),
            self._md_audit_trail(data), self._md_enforcement(data),
            self._md_cross_framework(data), self._md_provenance(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_executive_summary(data),
            self._html_compliance_status(data), self._html_csrd_coverage(data),
            self._html_sec_coverage(data), self._html_data_quality(data),
            self._html_provenance(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Regulator Dashboard - {data.get("org_name","")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n'
            f'<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {_compute_hash(body)} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        csrd_comp = self._calculate_csrd_compliance(data)
        sec_comp = self._calculate_sec_compliance(data)
        result = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION, "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(), "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""), "reporting_year": data.get("reporting_year", ""),
            "stakeholder": "regulator",
            "csrd_compliance": csrd_comp, "sec_compliance": sec_comp,
            "overall_compliance_pct": _dec(
                (csrd_comp["score"] + sec_comp["score"]) / 2 if (csrd_comp["total"] + sec_comp["total"]) else 0
            ),
            "data_quality": data.get("data_quality", {}),
            "filing_deadlines": data.get("filing_deadlines", []),
            "enforcement_risks": data.get("enforcement_risks", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "format": "pdf",
            "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {
                "title": f"Regulator Dashboard - {data.get('org_name', '')}",
                "author": "GreenLang PACK-030",
            },
        }

    # -----------------------------------------------------------------------
    # Compliance Calculators
    # -----------------------------------------------------------------------

    def _calculate_csrd_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        csrd = data.get("csrd_compliance", {})
        checks = {}
        for req in CSRD_ESRS_E1_REQUIREMENTS:
            code = req["code"]
            checks[code] = bool(csrd.get(code.lower().replace("-", "_"))) or bool(csrd.get(code, {}).get("disclosed"))
        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        return {"checks": checks, "passed": passed, "total": total,
                "score": round(passed / total * 100, 1) if total else 0}

    def _calculate_sec_compliance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        sec = data.get("sec_compliance", {})
        checks = {}
        for req in SEC_REG_SK_REQUIREMENTS:
            code = req["code"]
            checks[code] = bool(sec.get(f"item_{code}")) or bool(sec.get(code, {}).get("disclosed"))
        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        return {"checks": checks, "passed": passed, "total": total,
                "score": round(passed / total * 100, 1) if total else 0}

    # -----------------------------------------------------------------------
    # Markdown Section Renderers
    # -----------------------------------------------------------------------

    def _md_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Regulator Compliance Dashboard\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Audience:** Regulators & Compliance Officers  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-030 v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data):
        csrd = self._calculate_csrd_compliance(data)
        sec = self._calculate_sec_compliance(data)
        overall = (csrd["score"] + sec["score"]) / 2 if (csrd["total"] + sec["total"]) else 0
        lines = [
            "## 1. Executive Summary\n",
            "| Metric | Value |", "|--------|-------|",
            f"| CSRD ESRS E1 Coverage | {csrd['score']}% ({csrd['passed']}/{csrd['total']}) |",
            f"| SEC Reg S-K Coverage | {sec['score']}% ({sec['passed']}/{sec['total']}) |",
            f"| Overall Compliance | {_dec(overall)}% |",
            f"| Filing Status | {data.get('filing_status', 'On Track')} |",
            f"| Data Quality Level | {data.get('data_quality', {}).get('level', 'Reasonable')} |",
            f"| Assurance Level | {data.get('assurance_level', 'Limited')} |",
        ]
        return "\n".join(lines)

    def _md_compliance_status(self, data):
        frameworks = data.get("frameworks", [])
        lines = [
            "## 2. Compliance Status Overview\n",
            "| Framework | Jurisdiction | Status | Deadline | Risk |",
            "|-----------|-------------|--------|----------|:----:|",
        ]
        for f in frameworks:
            risk = f.get("risk", "Low")
            lines.append(
                f"| {f.get('name', '')} | {f.get('jurisdiction', '')} | "
                f"{f.get('status', '')} | {f.get('deadline', '')} | "
                f"{RISK_LEVELS.get(risk, risk)} |"
            )
        if not frameworks:
            lines.extend([
                "| CSRD / ESRS E1 | EU | See below | 2026 | - |",
                "| SEC Climate | US | See below | 2026 | - |",
            ])
        return "\n".join(lines)

    def _md_csrd_coverage(self, data):
        csrd = self._calculate_csrd_compliance(data)
        lines = [
            "## 3. CSRD / ESRS E1 Coverage\n",
            f"**Score:** {csrd['score']}% ({csrd['passed']}/{csrd['total']})\n",
            "| Code | Requirement | Mandatory | Status |",
            "|------|------------|:---------:|--------|",
        ]
        for req in CSRD_ESRS_E1_REQUIREMENTS:
            status = "Disclosed" if csrd["checks"].get(req["code"]) else "Missing"
            lines.append(
                f"| {req['code']} | {req['title']} | {req['mandatory']} | {status} |"
            )
        return "\n".join(lines)

    def _md_sec_coverage(self, data):
        sec = self._calculate_sec_compliance(data)
        lines = [
            "## 4. SEC Reg S-K Climate Coverage\n",
            f"**Score:** {sec['score']}% ({sec['passed']}/{sec['total']})\n",
            "| Item | Requirement | Mandatory | Status |",
            "|------|------------|:---------:|--------|",
        ]
        for req in SEC_REG_SK_REQUIREMENTS:
            status = "Disclosed" if sec["checks"].get(req["code"]) else "Missing"
            lines.append(
                f"| {req['code']} | {req['title']} | {req['mandatory']} | {status} |"
            )
        return "\n".join(lines)

    def _md_checklist(self, data):
        items = data.get("mandatory_checklist", [])
        lines = [
            "## 5. Mandatory Disclosure Checklist\n",
            "| # | Item | Framework | Required | Provided | Gap |",
            "|---|------|-----------|:--------:|:--------:|:---:|",
        ]
        for i, item in enumerate(items, 1):
            provided = item.get("provided", False)
            gap = "No" if provided else "Yes"
            lines.append(
                f"| {i} | {item.get('item', '')} | {item.get('framework', '')} | "
                f"{'Yes' if item.get('required', True) else 'No'} | "
                f"{'Yes' if provided else 'No'} | {gap} |"
            )
        if not items:
            lines.append("| - | _Populate mandatory_checklist in data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_data_quality(self, data):
        dq = data.get("data_quality", {})
        indicators = dq.get("indicators", [])
        lines = [
            "## 6. Data Quality Indicators\n",
            f"**Overall Level:** {dq.get('level', 'Reasonable')}  \n"
            f"**Score:** {_dec(dq.get('score', 0))}%\n",
            "| # | Indicator | Measure | Score | Rating |",
            "|---|-----------|---------|:-----:|:------:|",
        ]
        for i, ind in enumerate(indicators, 1):
            lines.append(
                f"| {i} | {ind.get('name', '')} | {ind.get('measure', '')} | "
                f"{_dec(ind.get('score', 0))}% | {ind.get('rating', '')} |"
            )
        if not indicators:
            lines.extend([
                "| 1 | Completeness | Data coverage across scopes | N/A | - |",
                "| 2 | Accuracy | Variance from verified data | N/A | - |",
                "| 3 | Timeliness | Data freshness vs filing date | N/A | - |",
                "| 4 | Consistency | Cross-source reconciliation | N/A | - |",
            ])
        return "\n".join(lines)

    def _md_timeline(self, data):
        deadlines = data.get("filing_deadlines", [])
        lines = [
            "## 7. Filing Timeline & Deadlines\n",
            "| # | Filing | Framework | Deadline | Status | Days Remaining |",
            "|---|--------|-----------|----------|--------|:--------------:|",
        ]
        for i, dl in enumerate(deadlines, 1):
            lines.append(
                f"| {i} | {dl.get('filing', '')} | {dl.get('framework', '')} | "
                f"{dl.get('deadline', '')} | {dl.get('status', 'Pending')} | "
                f"{dl.get('days_remaining', '')} |"
            )
        if not deadlines:
            lines.extend([
                "| 1 | CSRD Annual Report | CSRD | Annual | Pending | - |",
                "| 2 | SEC 10-K Climate | SEC | Annual | Pending | - |",
            ])
        return "\n".join(lines)

    def _md_audit_trail(self, data):
        evidence = data.get("audit_evidence", [])
        lines = [
            "## 8. Audit Trail & Evidence\n",
            "| # | Evidence Item | Source | Status | Hash |",
            "|---|--------------|--------|--------|------|",
        ]
        for i, ev in enumerate(evidence, 1):
            h = _compute_hash(ev)
            lines.append(
                f"| {i} | {ev.get('item', '')} | {ev.get('source', '')} | "
                f"{ev.get('status', 'Verified')} | `{h[:12]}...` |"
            )
        if not evidence:
            lines.append("| - | _No evidence items provided_ | - | - | - |")
        return "\n".join(lines)

    def _md_enforcement(self, data):
        risks = data.get("enforcement_risks", [])
        lines = [
            "## 9. Enforcement Risk Assessment\n",
            "| # | Risk | Area | Severity | Mitigation | Status |",
            "|---|------|------|:--------:|-----------|--------|",
        ]
        for i, r in enumerate(risks, 1):
            lines.append(
                f"| {i} | {r.get('risk', '')} | {r.get('area', '')} | "
                f"{r.get('severity', 'Medium')} | {r.get('mitigation', '')} | "
                f"{r.get('status', 'Open')} |"
            )
        if not risks:
            lines.append("| - | _No enforcement risks identified_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_cross_framework(self, data):
        mappings = data.get("cross_framework_mapping", [])
        lines = [
            "## 10. Cross-Framework Mapping\n",
            "| Disclosure | CSRD | SEC | CDP | TCFD | GRI | ISSB |",
            "|------------|:----:|:---:|:---:|:----:|:---:|:----:|",
        ]
        for m in mappings:
            lines.append(
                f"| {m.get('disclosure', '')} | {m.get('csrd', '-')} | "
                f"{m.get('sec', '-')} | {m.get('cdp', '-')} | "
                f"{m.get('tcfd', '-')} | {m.get('gri', '-')} | "
                f"{m.get('issb', '-')} |"
            )
        if not mappings:
            lines.extend([
                "| GHG Scope 1 | E1-6 | 1505 | C6.1 | M-a | 305-1 | S2.29 |",
                "| GHG Scope 2 | E1-6 | 1505 | C6.3 | M-a | 305-2 | S2.29 |",
                "| GHG Scope 3 | E1-6 | - | C6.5 | M-a | 305-3 | S2.29 |",
                "| Targets | E1-4 | 1506 | C4.1 | M-b | - | S2.33 |",
                "| Transition Plan | E1-1 | 1503 | C3.3 | S-b | - | S2.14 |",
                "| Governance | GOV-1 | 1502 | C1.1 | G-a | 2-9 | S2.6 |",
            ])
        return "\n".join(lines)

    def _md_provenance(self, data):
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            f"## 11. Provenance & Audit\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n"
            f"| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n"
            f"| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n"
            f"| Data Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-030 on {ts}*  \n"
            f"*Regulator compliance dashboard - CSRD + SEC focus.*"
        )

    # -----------------------------------------------------------------------
    # HTML Section Renderers
    # -----------------------------------------------------------------------

    def _css(self):
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            f"background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            f"border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_ACCENT};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};"
            f"padding-left:12px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #c5cae9;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#e8eaf6;}}"
            f".status-ok{{color:#2e7d32;font-weight:600;}}"
            f".status-warn{{color:#f57f17;font-weight:600;}}"
            f".status-err{{color:{_ACCENT};font-weight:600;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));"
            f"gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});"
            f"border-radius:10px;padding:18px;text-align:center;"
            f"border-left:4px solid {_ACCENT};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.4em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};"
            f"color:{_SECONDARY};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Regulator Compliance Dashboard</h1>\n'
            f'<p><strong>{data.get("org_name", "")}</strong> | '
            f'{data.get("reporting_year", "")} | {ts}</p>'
        )

    def _html_executive_summary(self, data):
        csrd = self._calculate_csrd_compliance(data)
        sec = self._calculate_sec_compliance(data)
        overall = (csrd["score"] + sec["score"]) / 2 if (csrd["total"] + sec["total"]) else 0
        return (
            f'<h2>1. Executive Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">CSRD E1</div>'
            f'<div class="card-value">{csrd["score"]}%</div></div>\n'
            f'<div class="card"><div class="card-label">SEC</div>'
            f'<div class="card-value">{sec["score"]}%</div></div>\n'
            f'<div class="card"><div class="card-label">Overall</div>'
            f'<div class="card-value">{_dec(overall)}%</div></div>\n'
            f'<div class="card"><div class="card-label">Filing</div>'
            f'<div class="card-value">{data.get("filing_status", "On Track")}</div></div>\n'
            f'</div>'
        )

    def _html_compliance_status(self, data):
        frameworks = data.get("frameworks", [])
        rows = ""
        for f in frameworks:
            risk = f.get("risk", "Low")
            cls = "status-ok" if risk == "Low" else ("status-warn" if risk == "Medium" else "status-err")
            rows += (
                f'<tr><td>{f.get("name", "")}</td><td>{f.get("jurisdiction", "")}</td>'
                f'<td>{f.get("status", "")}</td><td>{f.get("deadline", "")}</td>'
                f'<td class="{cls}">{risk}</td></tr>\n'
            )
        return (
            f'<h2>2. Compliance Status</h2>\n<table>\n'
            f'<tr><th>Framework</th><th>Jurisdiction</th><th>Status</th>'
            f'<th>Deadline</th><th>Risk</th></tr>\n{rows}</table>'
        )

    def _html_csrd_coverage(self, data):
        csrd = self._calculate_csrd_compliance(data)
        rows = ""
        for req in CSRD_ESRS_E1_REQUIREMENTS:
            disclosed = csrd["checks"].get(req["code"])
            cls = "status-ok" if disclosed else "status-err"
            label = "Disclosed" if disclosed else "Missing"
            rows += (
                f'<tr><td>{req["code"]}</td><td>{req["title"]}</td>'
                f'<td>{req["mandatory"]}</td><td class="{cls}">{label}</td></tr>\n'
            )
        return (
            f'<h2>3. CSRD / ESRS E1</h2>\n<p>Score: {csrd["score"]}%</p>\n<table>\n'
            f'<tr><th>Code</th><th>Requirement</th><th>Mandatory</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_sec_coverage(self, data):
        sec = self._calculate_sec_compliance(data)
        rows = ""
        for req in SEC_REG_SK_REQUIREMENTS:
            disclosed = sec["checks"].get(req["code"])
            cls = "status-ok" if disclosed else "status-err"
            label = "Disclosed" if disclosed else "Missing"
            rows += (
                f'<tr><td>{req["code"]}</td><td>{req["title"]}</td>'
                f'<td>{req["mandatory"]}</td><td class="{cls}">{label}</td></tr>\n'
            )
        return (
            f'<h2>4. SEC Reg S-K</h2>\n<p>Score: {sec["score"]}%</p>\n<table>\n'
            f'<tr><th>Item</th><th>Requirement</th><th>Mandatory</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_data_quality(self, data):
        dq = data.get("data_quality", {})
        indicators = dq.get("indicators", [])
        rows = ""
        for ind in indicators:
            rows += (
                f'<tr><td>{ind.get("name", "")}</td><td>{_dec(ind.get("score", 0))}%</td>'
                f'<td>{ind.get("rating", "")}</td></tr>\n'
            )
        return (
            f'<h2>5. Data Quality</h2>\n<p>Level: {dq.get("level", "Reasonable")}</p>\n'
            f'<table>\n<tr><th>Indicator</th><th>Score</th><th>Rating</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_provenance(self, data):
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            f'<h2>6. Provenance</h2>\n<table>\n'
            f'<tr><th>Param</th><th>Value</th></tr>\n'
            f'<tr><td>ID</td><td><code>{rid}</code></td></tr>\n'
            f'<tr><td>Generated</td><td>{ts}</td></tr>\n'
            f'<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'
        )

    def _html_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-030 on {ts} '
            f'- Regulator Dashboard</div>'
        )
