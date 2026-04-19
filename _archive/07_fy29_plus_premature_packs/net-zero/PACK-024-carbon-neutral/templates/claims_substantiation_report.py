# -*- coding: utf-8 -*-
"""
ClaimsSubstantiationReportTemplate - Claims substantiation for PACK-024.

Renders the claims substantiation documentation required for PAS 2060,
VCMI Claims Code, ISO 14021, and EU Green Claims Directive compliance.

Sections:
    1. Claim Statement
    2. Compliance Assessment (PAS 2060, VCMI, ISO 14021)
    3. Evidence Summary
    4. Risk Assessment
    5. Approval Status

Author: GreenLang Team
Version: 24.0.0
"""

import hashlib, json, logging, uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)
_MODULE_VERSION = "24.0.0"

def _new_uuid(): return str(uuid.uuid4())
def _compute_hash(d):
    r = json.dumps(d, sort_keys=True, default=str) if isinstance(d, dict) else str(d)
    return hashlib.sha256(r.encode("utf-8")).hexdigest()
def _dec(v, p=2):
    try: return str(Decimal(str(v)).quantize(Decimal("0."+"0"*p), rounding=ROUND_HALF_UP))
    except: return str(v)
def _dec_comma(v, p=0):
    try:
        d = Decimal(str(v)); q = "0."+"0"*p if p > 0 else "0"
        r = d.quantize(Decimal(q), rounding=ROUND_HALF_UP); parts = str(r).split(".")
        ip = parts[0]; neg = ip.startswith("-")
        if neg: ip = ip[1:]
        f = ""
        for i, ch in enumerate(reversed(ip)):
            if i > 0 and i % 3 == 0: f = "," + f
            f = ch + f
        if neg: f = "-" + f
        if len(parts) > 1: f += "." + parts[1]
        return f
    except: return str(v)
def _pct(v):
    try: return _dec(v, 1) + "%"
    except: return str(v)

class ClaimsSubstantiationReportTemplate:
    """Claims substantiation report template for PACK-024."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        sections = [
            self._md_header(data), self._md_claim(data), self._md_compliance(data),
            self._md_evidence(data), self._md_risk(data), self._md_approval(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        css = ("body{font-family:'Segoe UI',sans-serif;padding:20px;background:#f0f4f0;}"
               ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;}"
               "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;}"
               "h2{color:#2e7d32;border-left:4px solid #43a047;padding-left:12px;}"
               "table{width:100%;border-collapse:collapse;margin:15px 0;}"
               "th,td{border:1px solid #c8e6c9;padding:10px;}"
               "th{background:#e8f5e9;color:#1b5e20;}"
               ".pass{color:#2e7d32;font-weight:bold;}.fail{color:#c62828;font-weight:bold;}"
               ".footer{margin-top:40px;border-top:2px solid #c8e6c9;padding-top:20px;color:#689f38;text-align:center;}")
        body = f'<h1>Claims Substantiation Report</h1>\n{self._html_claim(data)}\n{self._html_compliance(data)}'
        return f'<!DOCTYPE html>\n<html><head><style>{css}</style></head><body><div class="report">{body}</div></body></html>'

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        result = {"template": "claims_substantiation_report", "version": _MODULE_VERSION,
                  "generated_at": self.generated_at.isoformat(), "report_id": _new_uuid(),
                  "claim": data.get("claim", {}), "compliance": data.get("compliance", []),
                  "is_valid": data.get("is_valid", False)}
        result["provenance_hash"] = _compute_hash(result)
        return result

    def _md_header(self, data):
        org = data.get("org_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"# Claims Substantiation Report\n\n**Organization:** {org}  \n**Generated:** {ts}\n\n---"

    def _md_claim(self, data):
        claim = data.get("claim", {})
        return (f"## 1. Claim Statement\n\n"
                f"| Field | Value |\n|-------|-------|\n"
                f"| Claim Type | {claim.get('type', 'Carbon Neutral Organization')} |\n"
                f"| Subject | {claim.get('subject', 'N/A')} |\n"
                f"| Period | {claim.get('period', 'N/A')} |\n"
                f"| Total Emissions | {_dec_comma(claim.get('total_tco2e', 0))} tCO2e |\n"
                f"| Credits Retired | {_dec_comma(claim.get('credits_tco2e', 0))} tCO2e |\n"
                f"| Substantiated | {'Yes' if claim.get('substantiated', False) else 'No'} |\n"
                f"| Status | {claim.get('status', 'Under Review')} |\n\n"
                f"> **Claim Text:** {claim.get('text', '_No claim text provided._')}")

    def _md_compliance(self, data):
        checks = data.get("compliance", [])
        lines = ["## 2. Compliance Assessment\n",
                  "| Framework | Requirement | Critical | Met | Notes |",
                  "|-----------|------------|:--------:|:---:|-------|"]
        for c in checks:
            met = "PASS" if c.get("met", False) else "FAIL"
            lines.append(
                f"| {c.get('framework', '-')} | {c.get('requirement', '-')} "
                f"| {'Yes' if c.get('critical', True) else 'No'} | {met} "
                f"| {c.get('notes', '-')} |")
        total = len(checks)
        passed = sum(1 for c in checks if c.get("met", False))
        lines.append(f"\n**Compliance Rate:** {_pct(passed/max(total,1)*100)} ({passed}/{total} requirements met)")
        return "\n".join(lines)

    def _md_evidence(self, data):
        evidence = data.get("evidence", [])
        lines = ["## 3. Evidence Summary\n",
                  "| # | Item | Verified | Source |",
                  "|---|------|:--------:|--------|"]
        for i, e in enumerate(evidence, 1):
            lines.append(
                f"| {i} | {e.get('description', '-')} "
                f"| {'Yes' if e.get('verified', False) else 'No'} "
                f"| {e.get('source', '-')} |")
        return "\n".join(lines)

    def _md_risk(self, data):
        risk = data.get("risk_assessment", {})
        return (f"## 4. Risk Assessment\n\n| Risk | Level | Mitigation |\n|------|:-----:|------------|\n"
                f"| Greenwashing | {risk.get('greenwashing', 'low')} | PAS 2060 compliance, third-party verification |\n"
                f"| Regulatory | {risk.get('regulatory', 'moderate')} | EU Green Claims Directive alignment |\n"
                f"| Reputational | {risk.get('reputational', 'low')} | Transparent disclosure, VCMI compliance |\n"
                f"| Litigation | {risk.get('litigation', 'low')} | Complete evidence documentation |")

    def _md_approval(self, data):
        approval = data.get("approval", {})
        return (f"## 5. Approval Status\n\n| Field | Value |\n|-------|-------|\n"
                f"| Decision | {approval.get('decision', 'Pending')} |\n"
                f"| Risk Level | {approval.get('risk_level', 'N/A')} |\n"
                f"| Approved By | {approval.get('approved_by', 'N/A')} |\n"
                f"| Approval Date | {approval.get('date', 'N/A')} |\n"
                f"| Valid Until | {approval.get('valid_until', 'N/A')} |")

    def _md_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-024 Carbon Neutral Pack on {ts}*"

    def _html_claim(self, data):
        claim = data.get("claim", {})
        return f'<h2>Claim</h2><p><strong>{claim.get("type", "Carbon Neutral")}</strong>: {claim.get("text", "")}</p>'

    def _html_compliance(self, data):
        checks = data.get("compliance", [])
        rows = ""
        for c in checks:
            cls = "pass" if c.get("met") else "fail"
            rows += f'<tr><td>{c.get("framework","")}</td><td>{c.get("requirement","")}</td><td class="{cls}">{"PASS" if c.get("met") else "FAIL"}</td></tr>\n'
        return f'<h2>Compliance</h2><table><tr><th>Framework</th><th>Requirement</th><th>Status</th></tr>{rows}</table>'
