# -*- coding: utf-8 -*-
"""
VerificationPackageReportTemplate - Verification package for PACK-024.

Renders verification documentation with package completeness, verification
body details, findings summary, resolution tracking, and verification
opinion/certificate.

Sections:
    1. Verification Overview
    2. Package Completeness
    3. Verification Body
    4. Findings Summary
    5. Verification Opinion
    6. Certificate Details

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
def _pct(v):
    try: return _dec(v, 1) + "%"
    except: return str(v)

class VerificationPackageReportTemplate:
    """Verification package report template for PACK-024."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = utcnow()
        sections = [
            self._md_header(data), self._md_overview(data), self._md_completeness(data),
            self._md_body(data), self._md_findings(data), self._md_opinion(data),
            self._md_certificate(data), self._md_footer(data),
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
               ".badge-pass{background:#43a047;color:#fff;padding:2px 8px;border-radius:4px;}"
               ".badge-fail{background:#ef5350;color:#fff;padding:2px 8px;border-radius:4px;}"
               ".footer{margin-top:40px;border-top:2px solid #c8e6c9;padding-top:20px;color:#689f38;text-align:center;}")
        body = f'<h1>Verification Package</h1>\n{self._html_overview(data)}\n{self._html_opinion(data)}'
        return f'<!DOCTYPE html>\n<html><head><style>{css}</style></head><body><div class="report">{body}</div></body></html>'

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = utcnow()
        result = {"template": "verification_package_report", "version": _MODULE_VERSION,
                  "generated_at": self.generated_at.isoformat(), "report_id": _new_uuid(),
                  "opinion": data.get("opinion", {}), "findings": data.get("findings", []),
                  "is_verified": data.get("is_verified", False)}
        result["provenance_hash"] = _compute_hash(result)
        return result

    def _md_header(self, data):
        org = data.get("org_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"# Verification Package Report\n\n**Organization:** {org}  \n**Generated:** {ts}\n\n---"

    def _md_overview(self, data):
        v = data.get("verification", {})
        return (f"## 1. Verification Overview\n\n| Metric | Value |\n|--------|-------|\n"
                f"| Assurance Level | {v.get('assurance_level', 'Limited')} |\n"
                f"| Standard | {v.get('standard', 'ISO 14064-3:2019')} |\n"
                f"| Reporting Year | {v.get('reporting_year', 'N/A')} |\n"
                f"| Total Emissions Verified | {v.get('total_tco2e', 0):,.0f} tCO2e |\n"
                f"| Materiality Threshold | {_pct(v.get('materiality_pct', 5))} |\n"
                f"| Status | {'Verified' if v.get('is_verified', False) else 'Pending'} |")

    def _md_completeness(self, data):
        items = data.get("package_items", [])
        lines = ["## 2. Package Completeness\n",
                  "| # | Document | Category | Required | Available |",
                  "|---|----------|----------|:--------:|:---------:|"]
        for i, item in enumerate(items, 1):
            lines.append(
                f"| {i} | {item.get('name', '-')} | {item.get('category', '-')} "
                f"| {'Yes' if item.get('required', True) else 'No'} "
                f"| {'Yes' if item.get('available', False) else 'No'} |")
        avail = sum(1 for it in items if it.get("available"))
        lines.append(f"\n**Completeness:** {_pct(avail/max(len(items),1)*100)} ({avail}/{len(items)})")
        return "\n".join(lines)

    def _md_body(self, data):
        vb = data.get("verification_body", {})
        return (f"## 3. Verification Body\n\n| Field | Value |\n|-------|-------|\n"
                f"| Name | {vb.get('name', 'N/A')} |\n"
                f"| Tier | {vb.get('tier', 'N/A')} |\n"
                f"| Accreditations | {', '.join(vb.get('accreditations', []))} |\n"
                f"| Lead Verifier | {vb.get('lead_verifier', 'N/A')} |\n"
                f"| Engagement Date | {vb.get('engagement_date', 'N/A')} |")

    def _md_findings(self, data):
        findings = data.get("findings", [])
        lines = ["## 4. Findings Summary\n",
                  "| # | Title | Category | Severity | Status |",
                  "|---|-------|----------|:--------:|:------:|"]
        for i, f in enumerate(findings, 1):
            lines.append(
                f"| {i} | {f.get('title', '-')} | {f.get('category', '-')} "
                f"| {f.get('severity', '-')} | {f.get('status', '-')} |")
        if not findings:
            lines.append("| - | _No findings_ | - | - | - |")
        critical = sum(1 for f in findings if f.get("severity") == "critical")
        lines.append(f"\n**Total:** {len(findings)} findings, **Critical:** {critical}")
        return "\n".join(lines)

    def _md_opinion(self, data):
        op = data.get("opinion", {})
        return (f"## 5. Verification Opinion\n\n| Field | Value |\n|-------|-------|\n"
                f"| Opinion Type | {op.get('type', 'N/A')} |\n"
                f"| Assurance Level | {op.get('assurance_level', 'Limited')} |\n"
                f"| Date | {op.get('date', 'N/A')} |\n"
                f"| Verifier | {op.get('verifier', 'N/A')} |\n\n"
                f"> {op.get('text', '_Opinion text pending._')}")

    def _md_certificate(self, data):
        cert = data.get("certificate", {})
        return (f"## 6. Certificate Details\n\n| Field | Value |\n|-------|-------|\n"
                f"| Certificate Number | {cert.get('number', 'N/A')} |\n"
                f"| Issue Date | {cert.get('issue_date', 'N/A')} |\n"
                f"| Valid Until | {cert.get('valid_until', 'N/A')} |\n"
                f"| Scope | {cert.get('scope', 'Carbon neutrality claim')} |")

    def _md_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-024 Carbon Neutral Pack on {ts}*"

    def _html_overview(self, data):
        v = data.get("verification", {})
        return (f'<h2>Overview</h2><table><tr><th>Metric</th><th>Value</th></tr>'
                f'<tr><td>Assurance</td><td>{v.get("assurance_level", "Limited")}</td></tr>'
                f'<tr><td>Status</td><td>{"Verified" if v.get("is_verified") else "Pending"}</td></tr></table>')

    def _html_opinion(self, data):
        op = data.get("opinion", {})
        badge = "badge-pass" if op.get("type") == "unmodified" else "badge-fail"
        return f'<h2>Opinion</h2><p><span class="{badge}">{op.get("type", "N/A").upper()}</span></p><p>{op.get("text", "")}</p>'
