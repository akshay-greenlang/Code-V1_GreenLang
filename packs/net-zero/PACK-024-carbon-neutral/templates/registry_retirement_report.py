# -*- coding: utf-8 -*-
"""
RegistryRetirementReportTemplate - Credit retirement report for PACK-024.

Renders registry retirement documentation with serial number tracking,
retirement certificates, registry confirmations, beneficiary details,
and PAS 2060 compliance verification.

Sections:
    1. Retirement Summary
    2. Retirement Records (serial numbers, volumes)
    3. Registry Confirmations
    4. Beneficiary Details
    5. PAS 2060 Compliance
    6. Certificate References

Author: GreenLang Team
Version: 24.0.0
"""

import hashlib, json, logging, uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
_MODULE_VERSION = "24.0.0"

def _utcnow(): return datetime.now(timezone.utc).replace(microsecond=0)
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


class RegistryRetirementReportTemplate:
    """Registry retirement report template for PACK-024."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_summary(data), self._md_records(data),
            self._md_confirmations(data), self._md_beneficiary(data),
            self._md_pas2060(data), self._md_certificates(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = ("body{font-family:'Segoe UI',sans-serif;padding:20px;background:#f0f4f0;}"
               ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;}"
               "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;}"
               "h2{color:#2e7d32;border-left:4px solid #43a047;padding-left:12px;}"
               "table{width:100%;border-collapse:collapse;margin:15px 0;}"
               "th,td{border:1px solid #c8e6c9;padding:10px;}"
               "th{background:#e8f5e9;color:#1b5e20;}"
               ".footer{margin-top:40px;border-top:2px solid #c8e6c9;padding-top:20px;color:#689f38;text-align:center;}")
        body = f'<h1>Registry Retirement Report</h1>\n{self._html_summary(data)}\n{self._html_records(data)}'
        return f'<!DOCTYPE html>\n<html><head><style>{css}</style></head><body><div class="report">{body}</div></body></html>'

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        result = {"template": "registry_retirement_report", "version": _MODULE_VERSION,
                  "generated_at": self.generated_at.isoformat(), "report_id": _new_uuid(),
                  "records": data.get("records", []), "confirmations": data.get("confirmations", []),
                  "total_retired_tco2e": data.get("total_retired_tco2e", 0)}
        result["provenance_hash"] = _compute_hash(result)
        return result

    def _md_header(self, data):
        org = data.get("org_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"# Registry Retirement Report\n\n**Organization:** {org}  \n**Generated:** {ts}\n\n---"

    def _md_summary(self, data):
        return (f"## 1. Retirement Summary\n\n| Metric | Value |\n|--------|-------|\n"
                f"| Total Retired | {_dec_comma(data.get('total_retired_tco2e', 0))} tCO2e |\n"
                f"| Records | {len(data.get('records', []))} |\n"
                f"| Registries | {len(set(r.get('registry', '') for r in data.get('records', [])))} |\n"
                f"| Certificates | {len(data.get('confirmations', []))} |\n"
                f"| Status | {'All Confirmed' if data.get('all_confirmed', False) else 'Pending'} |")

    def _md_records(self, data):
        records = data.get("records", [])
        lines = ["## 2. Retirement Records\n",
                  "| # | Registry | Project | Volume (tCO2e) | Vintage | Serial Range | Status |",
                  "|---|----------|---------|---------------:|:-------:|:------------:|:------:|"]
        for i, r in enumerate(records, 1):
            lines.append(
                f"| {i} | {r.get('registry', '-')} | {r.get('project_name', '-')} "
                f"| {_dec_comma(r.get('volume_tco2e', 0))} | {r.get('vintage_year', '-')} "
                f"| {r.get('serial_range', '-')} | {r.get('status', '-')} |")
        return "\n".join(lines)

    def _md_confirmations(self, data):
        confs = data.get("confirmations", [])
        lines = ["## 3. Registry Confirmations\n",
                  "| # | Registry | Transaction ID | Date | Volume | Certificate |",
                  "|---|----------|:--------------:|:----:|-------:|:-----------:|"]
        for i, c in enumerate(confs, 1):
            lines.append(
                f"| {i} | {c.get('registry', '-')} | {c.get('transaction_id', '-')} "
                f"| {c.get('date', '-')} | {_dec_comma(c.get('volume_tco2e', 0))} "
                f"| [Link]({c.get('certificate_url', '#')}) |")
        return "\n".join(lines)

    def _md_beneficiary(self, data):
        b = data.get("beneficiary", {})
        return (f"## 4. Beneficiary Details\n\n| Field | Value |\n|-------|-------|\n"
                f"| Name | {b.get('name', 'N/A')} |\n"
                f"| Country | {b.get('country', 'N/A')} |\n"
                f"| Purpose | {b.get('purpose', 'Carbon Neutrality')} |\n"
                f"| Retirement Reason | {b.get('reason', 'PAS 2060 carbon neutrality')} |")

    def _md_pas2060(self, data):
        checks = data.get("pas2060_checks", {})
        return (f"## 5. PAS 2060 Compliance\n\n| Requirement | Status |\n|-------------|:------:|\n"
                f"| Eligible registry | {checks.get('eligible_registry', 'PASS')} |\n"
                f"| Beneficiary named | {checks.get('beneficiary_named', 'PASS')} |\n"
                f"| Purpose stated | {checks.get('purpose_stated', 'PASS')} |\n"
                f"| Vintage acceptable | {checks.get('vintage_ok', 'PASS')} |\n"
                f"| Retirement timing | {checks.get('timing_ok', 'PASS')} |")

    def _md_certificates(self, data):
        certs = data.get("certificates", [])
        lines = ["## 6. Certificate References\n"]
        if certs:
            lines.append("| # | Certificate ID | Registry | URL |")
            lines.append("|---|:--------------:|----------|-----|")
            for i, c in enumerate(certs, 1):
                lines.append(f"| {i} | {c.get('id', '-')} | {c.get('registry', '-')} | [View]({c.get('url', '#')}) |")
        else:
            lines.append("_Certificate references linked from retirement confirmations above._")
        return "\n".join(lines)

    def _md_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-024 Carbon Neutral Pack on {ts}*"

    def _html_summary(self, data):
        return (f'<h2>Retirement Summary</h2><table><tr><th>Metric</th><th>Value</th></tr>'
                f'<tr><td>Total Retired</td><td>{_dec_comma(data.get("total_retired_tco2e", 0))} tCO2e</td></tr>'
                f'<tr><td>Records</td><td>{len(data.get("records", []))}</td></tr></table>')

    def _html_records(self, data):
        records = data.get("records", [])
        rows = ""
        for r in records:
            rows += f'<tr><td>{r.get("registry", "-")}</td><td>{_dec_comma(r.get("volume_tco2e", 0))}</td><td>{r.get("status", "-")}</td></tr>\n'
        return f'<h2>Records</h2><table><tr><th>Registry</th><th>Volume</th><th>Status</th></tr>{rows}</table>'
