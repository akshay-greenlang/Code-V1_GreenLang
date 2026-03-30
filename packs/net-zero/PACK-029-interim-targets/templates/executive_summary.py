# -*- coding: utf-8 -*-
"""
ExecutiveSummaryTemplate - 1-page executive summary for PACK-029.

Renders a concise 1-page executive summary for board/C-suite consumption
with headline metrics, performance summary, key achievements, key risks,
and recommendations. Designed for maximum impact with minimal reading time.
Multi-format output (MD, HTML, JSON, PDF).

Sections:
    1.  Headline Metrics
    2.  Performance Summary (On-Track / Ahead / Behind)
    3.  Key Achievements
    4.  Key Risks
    5.  Recommendations
    6.  Next Steps
    7.  XBRL Tagging Summary
    8.  Audit Trail & Provenance

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

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "29.0.0"
_PACK_ID = "PACK-029"
_TEMPLATE_ID = "executive_summary"

_PRIMARY = "#0a4a3a"
_SECONDARY = "#167a5b"
_ACCENT = "#1db954"
_LIGHT = "#e0f5ed"
_LIGHTER = "#f0faf5"
_SUCCESS = "#2e7d32"
_WARN = "#ef6c00"
_DANGER = "#c62828"

XBRL_TAGS: Dict[str, str] = {
    "total_emissions": "gl:ExecutiveTotalEmissions",
    "target_emissions": "gl:ExecutiveTargetEmissions",
    "variance_pct": "gl:ExecutiveVariancePct",
    "performance_status": "gl:ExecutivePerformanceStatus",
    "reduction_from_baseline": "gl:ExecutiveReductionFromBaseline",
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

def _performance_status(actual: float, target: float) -> str:
    if target <= 0:
        return "N/A"
    pct = (actual - target) / target * 100
    if pct <= -5:
        return "Ahead"
    elif pct <= 5:
        return "On Track"
    return "Behind"

class ExecutiveSummaryTemplate:
    """
    1-page executive summary template for PACK-029 Interim Targets Pack.

    Renders headline metrics, performance summary, key achievements,
    key risks, and corrective action recommendations for board/C-suite.
    Supports MD, HTML, JSON, PDF.

    Example:
        >>> template = ExecutiveSummaryTemplate()
        >>> data = {
        ...     "org_name": "GreenCorp",
        ...     "reporting_period": "FY2025",
        ...     "total_emissions": 85000,
        ...     "target_emissions": 82000,
        ...     "baseline_emissions": 100000,
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render 1-page executive summary as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data), self._md_headline_metrics(data),
            self._md_performance(data), self._md_achievements(data),
            self._md_risks(data), self._md_recommendations(data),
            self._md_next_steps(data), self._md_xbrl(data),
            self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render executive summary as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_headline_metrics(data),
            self._html_performance(data), self._html_achievements(data),
            self._html_risks(data), self._html_recommendations(data),
            self._html_next_steps(data), self._html_xbrl(data),
            self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Executive Summary - {data.get("org_name","")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render as structured JSON."""
        self.generated_at = utcnow()
        total = float(data.get("total_emissions", 0))
        target = float(data.get("target_emissions", 0))
        baseline = float(data.get("baseline_emissions", 0))
        variance = total - target
        var_pct = (variance / target * 100) if target else 0
        reduction = ((baseline - total) / baseline * 100) if baseline else 0
        status = _performance_status(total, target)

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION,
            "pack_id": _PACK_ID, "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(), "org_name": data.get("org_name", ""),
            "reporting_period": data.get("reporting_period", ""),
            "metrics": {
                "total_emissions": str(total), "target_emissions": str(target),
                "baseline_emissions": str(baseline), "variance": str(variance),
                "variance_pct": str(round(var_pct, 2)),
                "reduction_from_baseline_pct": str(round(reduction, 2)),
                "performance_status": status,
            },
            "achievements": data.get("achievements", []),
            "risks": data.get("risks", []),
            "recommendations": data.get("recommendations", []),
            "next_steps": data.get("next_steps", []),
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return PDF-ready data."""
        return {
            "format": "pdf", "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {"title": f"Executive Summary - {data.get('org_name','')}", "author": "GreenLang PACK-029"},
        }

    # -- Markdown --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Executive Summary: Interim Targets Progress\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Period:** {data.get('reporting_period', '')}  \n"
            f"**Date:** {ts}  \n"
            f"**Pack:** PACK-029 v{_MODULE_VERSION}\n\n---"
        )

    def _md_headline_metrics(self, data: Dict[str, Any]) -> str:
        total = float(data.get("total_emissions", 0))
        target = float(data.get("target_emissions", 0))
        baseline = float(data.get("baseline_emissions", 0))
        variance = total - target
        var_pct = (variance / target * 100) if target else 0
        reduction = ((baseline - total) / baseline * 100) if baseline else 0
        status = _performance_status(total, target)
        lines = [
            "## 1. Headline Metrics\n",
            "| Metric | Value |", "|--------|-------|",
            f"| Total Emissions | {_dec_comma(total, 0)} tCO2e |",
            f"| Annual Target | {_dec_comma(target, 0)} tCO2e |",
            f"| Variance | {'+' if variance > 0 else ''}{_dec_comma(variance, 0)} tCO2e ({'+' if var_pct > 0 else ''}{_dec(var_pct)}%) |",
            f"| Reduction from Baseline | {_dec(reduction)}% |",
            f"| Performance | **{status}** |",
        ]
        return "\n".join(lines)

    def _md_performance(self, data: Dict[str, Any]) -> str:
        total = float(data.get("total_emissions", 0))
        target = float(data.get("target_emissions", 0))
        status = _performance_status(total, target)
        scope_perf = data.get("scope_performance", {})
        lines = [
            "## 2. Performance Summary\n",
            f"**Overall Status: {status}**\n",
            "| Scope | Actual (tCO2e) | Target (tCO2e) | Status |",
            "|-------|---------------:|---------------:|--------|",
        ]
        for scope in ["scope1", "scope2", "scope3"]:
            sp = scope_perf.get(scope, {})
            actual = float(sp.get("actual", 0))
            tgt = float(sp.get("target", 0))
            s = _performance_status(actual, tgt)
            lines.append(
                f"| {scope.replace('scope', 'Scope ').title()} | {_dec_comma(actual, 0)} "
                f"| {_dec_comma(tgt, 0)} | **{s}** |"
            )
        return "\n".join(lines)

    def _md_achievements(self, data: Dict[str, Any]) -> str:
        achievements = data.get("achievements", [])
        lines = ["## 3. Key Achievements\n"]
        for i, a in enumerate(achievements, 1):
            lines.append(
                f"{i}. **{a.get('title', '')}** - {a.get('description', '')} "
                f"({_dec_comma(a.get('impact_tco2e', 0), 0)} tCO2e reduced)"
            )
        if not achievements:
            lines.append("_No achievements reported for this period._")
        return "\n".join(lines)

    def _md_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        lines = [
            "## 4. Key Risks\n",
            "| # | Risk | Impact | Likelihood | Mitigation |",
            "|---|------|--------|:----------:|-----------|",
        ]
        for i, r in enumerate(risks, 1):
            lines.append(
                f"| {i} | {r.get('risk', '')} | {r.get('impact', 'Medium')} "
                f"| {r.get('likelihood', 'Medium')} | {r.get('mitigation', '')} |"
            )
        if not risks:
            lines.append("| - | _No critical risks identified_ | - | - | - |")
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        lines = ["## 5. Recommendations\n"]
        for i, r in enumerate(recs, 1):
            lines.append(
                f"{i}. **{r.get('recommendation', '')}** ({r.get('priority', 'Medium')} priority)  \n"
                f"   _{r.get('rationale', '')}_"
            )
        if not recs:
            lines.append("_No recommendations at this time._")
        return "\n".join(lines)

    def _md_next_steps(self, data: Dict[str, Any]) -> str:
        steps = data.get("next_steps", [])
        lines = ["## 6. Next Steps\n"]
        for i, s in enumerate(steps, 1):
            lines.append(
                f"{i}. {s.get('step', '')} - **{s.get('owner', 'TBD')}** by {s.get('deadline', 'TBD')}"
            )
        if not steps:
            lines.append("_Next steps to be defined._")
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 7. XBRL Tagging Summary\n",
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
            "## 8. Audit Trail\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n"
            f"| Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-029 on {ts}*  \n*Executive summary for board/C-suite.*"

    # -- HTML --

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1000px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;font-size:1.6em;}}"
            f"h2{{color:{_SECONDARY};margin-top:28px;border-left:4px solid {_ACCENT};padding-left:12px;font-size:1.2em;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid #c8e6c9;padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f9fbe7;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:16px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:14px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.75em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.3em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.7em;color:{_ACCENT};}}"
            f".status-ahead{{background:#c8e6c9;color:#1b5e20;font-weight:700;padding:4px 12px;border-radius:4px;display:inline-block;}}"
            f".status-on-track{{background:#e8f5e9;color:#2e7d32;font-weight:700;padding:4px 12px;border-radius:4px;display:inline-block;}}"
            f".status-behind{{background:#ffcdd2;color:#c62828;font-weight:700;padding:4px 12px;border-radius:4px;display:inline-block;}}"
            f".footer{{margin-top:30px;padding-top:15px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>Executive Summary: Interim Targets</h1>\n<p><strong>{data.get("org_name","")}</strong> | {data.get("reporting_period","")} | {ts}</p>'

    def _html_headline_metrics(self, data: Dict[str, Any]) -> str:
        total = float(data.get("total_emissions", 0))
        target = float(data.get("target_emissions", 0))
        baseline = float(data.get("baseline_emissions", 0))
        variance = total - target
        var_pct = (variance / target * 100) if target else 0
        reduction = ((baseline - total) / baseline * 100) if baseline else 0
        status = _performance_status(total, target)
        status_cls = f"status-{status.lower().replace(' ', '-')}"
        return (
            f'<h2>1. Headline Metrics</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Emissions</div><div class="card-value">{_dec_comma(total, 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Target</div><div class="card-value">{_dec_comma(target, 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Variance</div><div class="card-value">{_dec(var_pct)}%</div><div class="card-unit">{_dec_comma(variance, 0)} tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Reduction</div><div class="card-value">{_dec(reduction)}%</div><div class="card-unit">from baseline</div></div>\n'
            f'<div class="card"><div class="card-label">Status</div><div class="card-value"><span class="{status_cls}">{status}</span></div></div>\n'
            f'</div>'
        )

    def _html_performance(self, data: Dict[str, Any]) -> str:
        scope_perf = data.get("scope_performance", {})
        rows = ""
        for scope in ["scope1", "scope2", "scope3"]:
            sp = scope_perf.get(scope, {})
            actual = float(sp.get("actual", 0))
            tgt = float(sp.get("target", 0))
            s = _performance_status(actual, tgt)
            cls = f"status-{s.lower().replace(' ', '-')}"
            rows += f'<tr><td>{scope.replace("scope","Scope ").title()}</td><td>{_dec_comma(actual, 0)}</td><td>{_dec_comma(tgt, 0)}</td><td><span class="{cls}">{s}</span></td></tr>\n'
        return f'<h2>2. Performance</h2>\n<table>\n<tr><th>Scope</th><th>Actual</th><th>Target</th><th>Status</th></tr>\n{rows}</table>'

    def _html_achievements(self, data: Dict[str, Any]) -> str:
        achs = data.get("achievements", [])
        items = ""
        for a in achs:
            items += f'<li><strong>{a.get("title","")}</strong> - {a.get("description","")} ({_dec_comma(a.get("impact_tco2e",0), 0)} tCO2e)</li>\n'
        return f'<h2>3. Achievements</h2>\n<ol>\n{items}</ol>' if items else '<h2>3. Achievements</h2>\n<p>No achievements reported.</p>'

    def _html_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        rows = ""
        for i, r in enumerate(risks, 1):
            rows += f'<tr><td>{i}</td><td>{r.get("risk","")}</td><td>{r.get("impact","Medium")}</td><td>{r.get("mitigation","")}</td></tr>\n'
        return f'<h2>4. Risks</h2>\n<table>\n<tr><th>#</th><th>Risk</th><th>Impact</th><th>Mitigation</th></tr>\n{rows}</table>'

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        items = ""
        for r in recs:
            items += f'<li><strong>{r.get("recommendation","")}</strong> ({r.get("priority","Medium")}) - {r.get("rationale","")}</li>\n'
        return f'<h2>5. Recommendations</h2>\n<ol>\n{items}</ol>' if items else '<h2>5. Recommendations</h2>\n<p>None at this time.</p>'

    def _html_next_steps(self, data: Dict[str, Any]) -> str:
        steps = data.get("next_steps", [])
        items = ""
        for s in steps:
            items += f'<li>{s.get("step","")} - <strong>{s.get("owner","TBD")}</strong> by {s.get("deadline","TBD")}</li>\n'
        return f'<h2>6. Next Steps</h2>\n<ol>\n{items}</ol>' if items else '<h2>6. Next Steps</h2>\n<p>To be defined.</p>'

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        rows = ""
        for key, tag in XBRL_TAGS.items():
            rows += f'<tr><td>{key.replace("_"," ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>7. XBRL Tags</h2>\n<table>\n<tr><th>Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>8. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-029 on {ts} - Executive summary</div>'
