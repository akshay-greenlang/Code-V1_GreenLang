# -*- coding: utf-8 -*-
"""
PermanenceAssessmentReportTemplate - Credit permanence assessment for PACK-024.

Renders permanence risk assessment for carbon credit portfolio including
reversal risk by project type, buffer pool adequacy, insurance coverage,
monitoring frequency, and long-term storage guarantees.

Sections:
    1. Portfolio Permanence Overview
    2. Risk by Project Type
    3. Buffer Pool Assessment
    4. Reversal Risk Factors
    5. Monitoring Schedule
    6. Recommendations

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

# Permanence periods by project type (years)
PERMANENCE_PERIODS: Dict[str, int] = {
    "direct_air_capture": 1000, "enhanced_weathering": 1000, "biochar": 100,
    "arr": 40, "redd_plus": 30, "blue_carbon": 50, "soil_carbon": 20,
    "renewable_energy": 0, "energy_efficiency": 0, "methane_capture": 0,
}

# Reversal risk by type (%)
REVERSAL_RISK: Dict[str, float] = {
    "direct_air_capture": 0.5, "enhanced_weathering": 1.0, "biochar": 5.0,
    "arr": 15.0, "redd_plus": 20.0, "blue_carbon": 10.0, "soil_carbon": 25.0,
    "renewable_energy": 0.0, "energy_efficiency": 0.0, "methane_capture": 0.0,
}


class PermanenceAssessmentReportTemplate:
    """Permanence assessment report template for PACK-024."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_overview(data), self._md_by_type(data),
            self._md_buffer(data), self._md_risk_factors(data),
            self._md_monitoring(data), self._md_recommendations(data), self._md_footer(data),
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
               ".risk-low{color:#2e7d32;}.risk-mod{color:#ff9800;}.risk-high{color:#c62828;}"
               ".footer{margin-top:40px;border-top:2px solid #c8e6c9;padding-top:20px;color:#689f38;text-align:center;}")
        body = f'<h1>Permanence Assessment</h1>\n{self._html_overview(data)}\n{self._html_by_type(data)}'
        return f'<!DOCTYPE html>\n<html><head><style>{css}</style></head><body><div class="report">{body}</div></body></html>'

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        result = {"template": "permanence_assessment_report", "version": _MODULE_VERSION,
                  "generated_at": self.generated_at.isoformat(), "report_id": _new_uuid(),
                  "projects": data.get("projects", []),
                  "overall_risk": data.get("overall_risk", "moderate")}
        result["provenance_hash"] = _compute_hash(result)
        return result

    def _md_header(self, data):
        org = data.get("org_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"# Permanence Assessment Report\n\n**Organization:** {org}  \n**Generated:** {ts}\n\n---"

    def _md_overview(self, data):
        o = data.get("overview", {})
        return (f"## 1. Portfolio Permanence Overview\n\n| Metric | Value |\n|--------|-------|\n"
                f"| Total Credits | {_dec_comma(o.get('total_tco2e', 0))} tCO2e |\n"
                f"| Weighted Avg Permanence | {o.get('avg_permanence_years', 'N/A')} years |\n"
                f"| Weighted Reversal Risk | {_pct(o.get('avg_reversal_risk_pct', 0))} |\n"
                f"| Buffer Pool Allocation | {_pct(o.get('buffer_pct', 0))} |\n"
                f"| Overall Risk Rating | {o.get('overall_risk', 'moderate')} |")

    def _md_by_type(self, data):
        projects = data.get("projects", [])
        lines = ["## 2. Risk by Project Type\n",
                  "| Project Type | Volume (tCO2e) | Permanence (yr) | Reversal Risk | Buffer | Rating |",
                  "|-------------|---------------:|:---------------:|:-------------:|:------:|:------:|"]
        for p in projects:
            pt = p.get("type", "unknown")
            perm = PERMANENCE_PERIODS.get(pt, 0)
            rev = REVERSAL_RISK.get(pt, 10.0)
            rating = "Low" if rev < 5 else ("Moderate" if rev < 15 else "High")
            lines.append(
                f"| {pt.replace('_', ' ').title()} | {_dec_comma(p.get('volume', 0))} "
                f"| {perm if perm > 0 else 'N/A'} | {_pct(rev)} "
                f"| {_pct(p.get('buffer_pct', 5))} | {rating} |")
        return "\n".join(lines)

    def _md_buffer(self, data):
        buffer = data.get("buffer_pool", {})
        return (f"## 3. Buffer Pool Assessment\n\n| Metric | Value |\n|--------|-------|\n"
                f"| Total Buffer Pool | {_dec_comma(buffer.get('total_tco2e', 0))} tCO2e |\n"
                f"| Buffer as % of Portfolio | {_pct(buffer.get('pct_of_portfolio', 0))} |\n"
                f"| Minimum Required | {_pct(buffer.get('min_required_pct', 5))} |\n"
                f"| Adequacy | {'Adequate' if buffer.get('is_adequate', True) else 'Insufficient'} |")

    def _md_risk_factors(self, data):
        factors = data.get("risk_factors", [])
        lines = ["## 4. Reversal Risk Factors\n",
                  "| Factor | Impact | Likelihood | Mitigation |",
                  "|--------|:------:|:----------:|------------|"]
        defaults = [
            ("Wildfire", "High", "Moderate", "Buffer pool, geographic diversification"),
            ("Disease/Pest", "Moderate", "Moderate", "Species diversification, monitoring"),
            ("Political/Governance", "High", "Low", "Multi-jurisdiction portfolio"),
            ("Climate Change", "Moderate", "High", "Climate-resilient project selection"),
            ("Market Risk", "Low", "Moderate", "Long-term contracts"),
        ]
        items = factors if factors else [{"factor": f, "impact": im, "likelihood": l, "mitigation": m} for f, im, l, m in defaults]
        for f in items:
            lines.append(f"| {f.get('factor', '-')} | {f.get('impact', '-')} | {f.get('likelihood', '-')} | {f.get('mitigation', '-')} |")
        return "\n".join(lines)

    def _md_monitoring(self, data):
        return (f"## 5. Monitoring Schedule\n\n"
                f"| Activity | Frequency | Responsible |\n|----------|:---------:|-------------|\n"
                f"| Buffer pool adequacy review | Quarterly | Risk Manager |\n"
                f"| Project site monitoring | Annual | Project Developer |\n"
                f"| Reversal risk reassessment | Annual | Sustainability Director |\n"
                f"| Registry status verification | Monthly | Operations |\n"
                f"| Insurance coverage review | Annual | Risk Manager |")

    def _md_recommendations(self, data):
        recs = data.get("recommendations", [
            "Increase removal credit share for higher permanence",
            "Diversify across project types and geographies",
            "Maintain buffer pool above 10% for high-risk projects",
            "Consider engineered removal credits for long-term permanence",
        ])
        lines = ["## 6. Recommendations\n"]
        for i, r in enumerate(recs, 1):
            lines.append(f"{i}. {r}")
        return "\n".join(lines)

    def _md_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-024 Carbon Neutral Pack on {ts}*"

    def _html_overview(self, data):
        o = data.get("overview", {})
        return (f'<h2>Overview</h2><table><tr><th>Metric</th><th>Value</th></tr>'
                f'<tr><td>Reversal Risk</td><td>{_pct(o.get("avg_reversal_risk_pct", 0))}</td></tr>'
                f'<tr><td>Rating</td><td>{o.get("overall_risk", "moderate")}</td></tr></table>')

    def _html_by_type(self, data):
        projects = data.get("projects", [])
        rows = ""
        for p in projects:
            pt = p.get("type", ""); rev = REVERSAL_RISK.get(pt, 10)
            cls = "risk-low" if rev < 5 else ("risk-mod" if rev < 15 else "risk-high")
            rows += f'<tr><td>{pt.replace("_", " ").title()}</td><td>{_dec_comma(p.get("volume", 0))}</td><td class="{cls}">{_pct(rev)}</td></tr>\n'
        return f'<h2>By Project Type</h2><table><tr><th>Type</th><th>Volume</th><th>Risk</th></tr>{rows}</table>'
