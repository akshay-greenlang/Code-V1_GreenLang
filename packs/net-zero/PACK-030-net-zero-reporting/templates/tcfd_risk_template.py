# -*- coding: utf-8 -*-
"""
TCFDRiskTemplate - TCFD Risk Management Pillar Template for PACK-030.

Renders TCFD Risk Management pillar disclosure covering processes for
identifying, assessing, and managing climate-related risks, integration
with enterprise risk management (ERM), risk appetite, and mitigation
strategies. Multi-format output (MD, HTML, JSON, PDF) with SHA-256
provenance hashing.

Sections:
    1.  Executive Summary
    2.  Risk Identification Process
    3.  Risk Assessment Framework
    4.  Risk Prioritization & Materiality
    5.  Climate Risk Register
    6.  Risk Mitigation Strategies
    7.  ERM Integration
    8.  Risk Appetite & Tolerance
    9.  Emerging Risk Monitoring
    10. XBRL Tagging Summary
    11. Audit Trail & Provenance

Author: GreenLang Team
Version: 30.0.0
Pack: PACK-030 Net Zero Reporting Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "30.0.0"
_PACK_ID = "PACK-030"
_TEMPLATE_ID = "tcfd_risk"

_PRIMARY = "#1a237e"
_SECONDARY = "#283593"
_ACCENT = "#42a5f5"
_LIGHT = "#e8eaf6"
_LIGHTER = "#f5f5ff"

RISK_CATEGORIES = {
    "transition": ["Policy & legal", "Technology", "Market", "Reputation"],
    "physical": ["Acute (extreme weather)", "Chronic (long-term shifts)"],
}

LIKELIHOOD_SCALE = ["Very unlikely", "Unlikely", "Possible", "Likely", "Very likely"]
IMPACT_SCALE = ["Insignificant", "Minor", "Moderate", "Major", "Catastrophic"]

XBRL_TAGS: Dict[str, str] = {
    "risk_process_exists": "gl:TCFDRiskProcessExists",
    "erm_integration": "gl:TCFDRiskERMIntegration",
    "risks_identified": "gl:TCFDRisksIdentified",
    "transition_risks": "gl:TCFDTransitionRisks",
    "physical_risks": "gl:TCFDPhysicalRisks",
    "risk_score": "gl:TCFDRiskManagementScore",
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


class TCFDRiskTemplate:
    """
    TCFD Risk Management pillar template for PACK-030 Net Zero Reporting Pack.

    Generates TCFD-aligned risk management disclosure covering risk
    identification, assessment, management processes, and ERM integration.
    Supports MD, HTML, JSON, PDF.

    Example:
        >>> template = TCFDRiskTemplate()
        >>> data = {
        ...     "org_name": "GreenCorp", "reporting_year": 2025,
        ...     "risk_process": {"exists": True, "frequency": "Quarterly"},
        ...     "risks": [{"name": "Carbon tax", "category": "Policy & legal",
        ...                "type": "Transition", "likelihood": "Likely", "impact": "Major"}],
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_identification(data), self._md_assessment(data),
            self._md_prioritization(data), self._md_risk_register(data),
            self._md_mitigation(data), self._md_erm_integration(data),
            self._md_risk_appetite(data), self._md_emerging_risks(data),
            self._md_xbrl(data), self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_executive_summary(data),
            self._html_risk_register(data), self._html_mitigation(data),
            self._html_erm_integration(data), self._html_xbrl(data),
            self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>TCFD Risk Management - {data.get("org_name", "")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        risks = data.get("risks", [])
        transition = [r for r in risks if r.get("type") == "Transition"]
        physical = [r for r in risks if r.get("type") == "Physical"]
        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION,
            "pack_id": _PACK_ID, "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""), "reporting_year": data.get("reporting_year", ""),
            "framework": "TCFD", "pillar": "Risk Management",
            "risk_process": data.get("risk_process", {}),
            "risks": risks, "transition_risks": len(transition), "physical_risks": len(physical),
            "mitigations": data.get("mitigations", []),
            "erm_integration": data.get("erm_integration", {}),
            "risk_appetite": data.get("risk_appetite", {}),
            "emerging_risks": data.get("emerging_risks", []),
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "format": "pdf", "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {"title": f"TCFD Risk - {data.get('org_name', '')}", "author": "GreenLang PACK-030", "framework": "TCFD"},
        }

    # -- Markdown --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# TCFD Risk Management Pillar Disclosure\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Framework:** TCFD | **Pillar:** Risk Management  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-030 v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        transition = [r for r in risks if r.get("type") == "Transition"]
        physical = [r for r in risks if r.get("type") == "Physical"]
        process = data.get("risk_process", {})
        lines = [
            "## 1. Executive Summary\n",
            "| Metric | Value |", "|--------|-------|",
            f"| Risk Process Exists | {process.get('exists', 'Yes')} |",
            f"| Process Frequency | {process.get('frequency', 'Quarterly')} |",
            f"| Total Risks Identified | {len(risks)} |",
            f"| Transition Risks | {len(transition)} |",
            f"| Physical Risks | {len(physical)} |",
            f"| ERM Integration | {data.get('erm_integration', {}).get('level', 'Fully integrated')} |",
            f"| Mitigations Defined | {len(data.get('mitigations', []))} |",
        ]
        return "\n".join(lines)

    def _md_identification(self, data: Dict[str, Any]) -> str:
        process = data.get("risk_process", {})
        lines = [
            "## 2. Risk Identification Process\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Process | {process.get('description', 'Annual climate risk assessment')} |",
            f"| Frequency | {process.get('frequency', 'Quarterly')} |",
            f"| Scope | {process.get('scope', 'All operations and value chain')} |",
            f"| Time Horizons | {process.get('time_horizons', 'Short (<3yr), Medium (3-10yr), Long (>10yr)')} |",
            f"| Data Sources | {process.get('data_sources', 'IPCC, IEA, internal models')} |",
            f"| Stakeholder Input | {process.get('stakeholder_input', 'Yes - investors, suppliers, regulators')} |",
        ]
        return "\n".join(lines)

    def _md_assessment(self, data: Dict[str, Any]) -> str:
        framework = data.get("assessment_framework", {})
        lines = [
            "## 3. Risk Assessment Framework\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Framework | {framework.get('name', 'TCFD-aligned risk matrix')} |",
            f"| Likelihood Scale | {', '.join(LIKELIHOOD_SCALE)} |",
            f"| Impact Scale | {', '.join(IMPACT_SCALE)} |",
            f"| Quantitative Assessment | {framework.get('quantitative', 'Yes - financial impact modeling')} |",
            f"| Scenario Analysis | {framework.get('scenario_analysis', 'Yes - 1.5C, 2C, 4C')} |",
        ]
        return "\n".join(lines)

    def _md_prioritization(self, data: Dict[str, Any]) -> str:
        materiality = data.get("materiality", {})
        lines = [
            "## 4. Risk Prioritization & Materiality\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Materiality Threshold | {materiality.get('threshold', '>1% of revenue or >$10M impact')} |",
            f"| Priority Classification | {materiality.get('classification', 'Critical / High / Medium / Low')} |",
            f"| Review Frequency | {materiality.get('review_frequency', 'Quarterly')} |",
            f"| Board Escalation Criteria | {materiality.get('escalation', 'Critical and High risks')} |",
        ]
        return "\n".join(lines)

    def _md_risk_register(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        lines = [
            "## 5. Climate Risk Register\n",
            "| # | Risk | Type | Category | Likelihood | Impact | Rating | Time Horizon | Mitigation |",
            "|---|------|------|----------|------------|--------|--------|:------------:|-----------|",
        ]
        for i, r in enumerate(risks, 1):
            lines.append(
                f"| {i} | {r.get('name', '')} | {r.get('type', '')} | {r.get('category', '')} "
                f"| {r.get('likelihood', '')} | {r.get('impact', '')} | {r.get('rating', '')} "
                f"| {r.get('time_horizon', '')} | {r.get('mitigation', '')} |"
            )
        if not risks:
            lines.append("| - | _No risks registered_ | - | - | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_mitigation(self, data: Dict[str, Any]) -> str:
        mitigations = data.get("mitigations", [])
        lines = [
            "## 6. Risk Mitigation Strategies\n",
            "| # | Risk | Strategy | Owner | Status | Effectiveness |",
            "|---|------|----------|-------|--------|:-------------:|",
        ]
        for i, m in enumerate(mitigations, 1):
            lines.append(
                f"| {i} | {m.get('risk', '')} | {m.get('strategy', '')} "
                f"| {m.get('owner', '')} | {m.get('status', '')} | {m.get('effectiveness', '')} |"
            )
        if not mitigations:
            lines.append("| - | _No mitigations defined_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_erm_integration(self, data: Dict[str, Any]) -> str:
        erm = data.get("erm_integration", {})
        lines = [
            "## 7. ERM Integration\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Integration Level | {erm.get('level', 'Fully integrated')} |",
            f"| ERM Framework | {erm.get('framework', 'COSO ERM 2017')} |",
            f"| Climate in Risk Taxonomy | {erm.get('in_taxonomy', 'Yes')} |",
            f"| Reporting to Board | {erm.get('board_reporting', 'Quarterly')} |",
            f"| Key Risk Indicators (KRIs) | {erm.get('kris', 'Carbon price, regulatory changes, physical exposure')} |",
        ]
        return "\n".join(lines)

    def _md_risk_appetite(self, data: Dict[str, Any]) -> str:
        appetite = data.get("risk_appetite", {})
        lines = [
            "## 8. Risk Appetite & Tolerance\n",
            "| Dimension | Appetite | Tolerance |",
            "|-----------|----------|-----------|",
            f"| Transition Risk | {appetite.get('transition_appetite', 'Moderate')} | {appetite.get('transition_tolerance', 'Accept with mitigation')} |",
            f"| Physical Risk | {appetite.get('physical_appetite', 'Low')} | {appetite.get('physical_tolerance', 'Minimize exposure')} |",
            f"| Regulatory Risk | {appetite.get('regulatory_appetite', 'Very Low')} | {appetite.get('regulatory_tolerance', 'Full compliance')} |",
            f"| Reputational Risk | {appetite.get('reputational_appetite', 'Very Low')} | {appetite.get('reputational_tolerance', 'Zero tolerance')} |",
        ]
        return "\n".join(lines)

    def _md_emerging_risks(self, data: Dict[str, Any]) -> str:
        emerging = data.get("emerging_risks", [])
        lines = [
            "## 9. Emerging Risk Monitoring\n",
            "| # | Emerging Risk | Category | Potential Impact | Monitoring |",
            "|---|-------------- |----------|------------------|-----------|",
        ]
        for i, r in enumerate(emerging, 1):
            lines.append(
                f"| {i} | {r.get('name', '')} | {r.get('category', '')} "
                f"| {r.get('potential_impact', '')} | {r.get('monitoring', '')} |"
            )
        if not emerging:
            lines.append("| - | _No emerging risks identified_ | - | - | - |")
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        lines = [
            "## 10. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |", "|------------|----------|-------|",
        ]
        for key, tag in XBRL_TAGS.items():
            lines.append(f"| {key.replace('_', ' ').title()} | {tag} | - |")
        return "\n".join(lines)

    def _md_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            "## 11. Audit Trail & Provenance\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n| Data Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-030 on {ts}*  \n*TCFD Risk Management pillar disclosure.*"

    # -- HTML --

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #c5cae9;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f3f4fb;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.5em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>TCFD Risk Management</h1>\n<p><strong>{data.get("org_name", "")}</strong> | {data.get("reporting_year", "")} | {ts}</p>'

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        return (
            f'<h2>1. Executive Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Total Risks</div><div class="card-value">{len(risks)}</div></div>\n'
            f'<div class="card"><div class="card-label">Transition</div><div class="card-value">{len([r for r in risks if r.get("type") == "Transition"])}</div></div>\n'
            f'<div class="card"><div class="card-label">Physical</div><div class="card-value">{len([r for r in risks if r.get("type") == "Physical"])}</div></div>\n'
            f'<div class="card"><div class="card-label">Mitigations</div><div class="card-value">{len(data.get("mitigations", []))}</div></div>\n'
            f'</div>'
        )

    def _html_risk_register(self, data: Dict[str, Any]) -> str:
        risks = data.get("risks", [])
        rows = ""
        for i, r in enumerate(risks, 1):
            rows += f'<tr><td>{i}</td><td>{r.get("name", "")}</td><td>{r.get("type", "")}</td><td>{r.get("likelihood", "")}</td><td>{r.get("impact", "")}</td><td>{r.get("rating", "")}</td></tr>\n'
        return f'<h2>2. Risk Register</h2>\n<table>\n<tr><th>#</th><th>Risk</th><th>Type</th><th>Likelihood</th><th>Impact</th><th>Rating</th></tr>\n{rows}</table>'

    def _html_mitigation(self, data: Dict[str, Any]) -> str:
        mitigations = data.get("mitigations", [])
        rows = ""
        for i, m in enumerate(mitigations, 1):
            rows += f'<tr><td>{i}</td><td>{m.get("risk", "")}</td><td>{m.get("strategy", "")}</td><td>{m.get("status", "")}</td></tr>\n'
        return f'<h2>3. Mitigations</h2>\n<table>\n<tr><th>#</th><th>Risk</th><th>Strategy</th><th>Status</th></tr>\n{rows}</table>'

    def _html_erm_integration(self, data: Dict[str, Any]) -> str:
        erm = data.get("erm_integration", {})
        return (
            f'<h2>4. ERM Integration</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n'
            f'<tr><td>Level</td><td>{erm.get("level", "Fully integrated")}</td></tr>\n'
            f'<tr><td>Framework</td><td>{erm.get("framework", "COSO ERM 2017")}</td></tr>\n'
            f'</table>'
        )

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        rows = ""
        for key, tag in XBRL_TAGS.items():
            rows += f'<tr><td>{key.replace("_", " ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>5. XBRL Tags</h2>\n<table>\n<tr><th>Data Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>6. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-030 on {ts} - TCFD Risk Management</div>'
