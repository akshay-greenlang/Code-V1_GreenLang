# -*- coding: utf-8 -*-
"""
AssuranceEvidenceTemplate - Assurance Evidence Package Template for PACK-030.

Renders an assurance-ready evidence package for third-party verifiers,
including provenance hash chains, data lineage diagrams, methodology
documentation, control matrix, calculation trails, evidence hierarchy,
materiality assessment, data quality scoring, and ISO 14064-3 aligned
workpapers. Multi-format output (MD, HTML, JSON, PDF) with SHA-256
provenance hashing.

Sections:
    1.  Executive Summary
    2.  Provenance Hash Chain
    3.  Data Lineage Diagram
    4.  Methodology Documentation
    5.  Control Matrix
    6.  Calculation Trail
    7.  Evidence Hierarchy
    8.  Materiality Assessment
    9.  Data Quality Score
    10. Assurance Readiness Checklist
    11. ISO 14064-3 Workpaper Summary
    12. Audit Trail & Provenance

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
_TEMPLATE_ID = "assurance_evidence"
_PRIMARY = "#4a148c"
_SECONDARY = "#6a1b9a"
_ACCENT = "#ce93d8"
_LIGHT = "#f3e5f5"
_LIGHTER = "#faf0fc"

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
# Constants
# ---------------------------------------------------------------------------

EVIDENCE_TIERS: List[Dict[str, str]] = [
    {"tier": "1", "name": "Measured", "description": "Directly measured via calibrated instruments", "quality": "Highest"},
    {"tier": "2", "name": "Calculated", "description": "Calculated from primary activity data + emission factors", "quality": "High"},
    {"tier": "3", "name": "Estimated-Specific", "description": "Industry-specific estimation methods", "quality": "Medium"},
    {"tier": "4", "name": "Estimated-Generic", "description": "Generic estimation methods or proxy data", "quality": "Low"},
    {"tier": "5", "name": "Extrapolated", "description": "Extrapolated from partial data or benchmarks", "quality": "Lowest"},
]

CONTROL_CATEGORIES: List[str] = [
    "Data Collection", "Data Entry & Transfer", "Calculation & Processing",
    "Review & Approval", "Storage & Retention", "Reporting & Disclosure",
]

ASSURANCE_CHECKLIST: List[Dict[str, str]] = [
    {"id": "AC-01", "item": "Organisational boundary defined (operational/equity)", "iso": "14064-1 cl.5.1"},
    {"id": "AC-02", "item": "Operational boundary covers all material sources", "iso": "14064-1 cl.5.2"},
    {"id": "AC-03", "item": "Base year emissions quantified and documented", "iso": "14064-1 cl.5.3"},
    {"id": "AC-04", "item": "Emission factors sourced and version-controlled", "iso": "14064-1 cl.6.3"},
    {"id": "AC-05", "item": "Activity data complete with source references", "iso": "14064-1 cl.6.2"},
    {"id": "AC-06", "item": "Calculation methodology documented step-by-step", "iso": "14064-1 cl.6.1"},
    {"id": "AC-07", "item": "Uncertainty assessment performed (quantitative)", "iso": "14064-3 cl.5.4"},
    {"id": "AC-08", "item": "Scope 1 direct emissions independently verifiable", "iso": "14064-3 cl.5.2"},
    {"id": "AC-09", "item": "Scope 2 dual reporting (location + market)", "iso": "14064-1 cl.6.5"},
    {"id": "AC-10", "item": "Scope 3 categories screened and justified", "iso": "14064-1 cl.6.6"},
    {"id": "AC-11", "item": "Data quality indicators assessed per source", "iso": "14064-3 cl.5.3"},
    {"id": "AC-12", "item": "Internal controls documented and tested", "iso": "14064-3 cl.5.5"},
    {"id": "AC-13", "item": "Materiality threshold defined and applied", "iso": "14064-3 cl.5.1"},
    {"id": "AC-14", "item": "Restatement policy documented with triggers", "iso": "14064-1 cl.7"},
    {"id": "AC-15", "item": "Provenance hashing applied to all outputs", "iso": "14064-3 cl.6"},
]

ISO_14064_WORKPAPERS: List[Dict[str, str]] = [
    {"wp": "WP-100", "title": "Verification Plan", "scope": "Engagement planning, objectives, scope, criteria"},
    {"wp": "WP-200", "title": "Strategic Analysis", "scope": "Business context, materiality, risk assessment"},
    {"wp": "WP-300", "title": "Process Analysis", "scope": "Data flow mapping, control evaluation"},
    {"wp": "WP-400", "title": "Substantive Testing", "scope": "Recalculation, analytical review, cross-checks"},
    {"wp": "WP-500", "title": "Findings & Conclusions", "scope": "Non-conformities, observations, opinion"},
]


class AssuranceEvidenceTemplate:
    """Assurance evidence package template for PACK-030. Supports MD, HTML, JSON, PDF."""

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
            self._md_hash_chain(data), self._md_lineage(data),
            self._md_methodology(data), self._md_control_matrix(data),
            self._md_calculation_trail(data), self._md_evidence_hierarchy(data),
            self._md_materiality(data), self._md_data_quality(data),
            self._md_checklist(data), self._md_workpapers(data),
            self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_executive_summary(data),
            self._html_hash_chain(data), self._html_evidence_hierarchy(data),
            self._html_checklist(data), self._html_workpapers(data),
            self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Assurance Evidence - {data.get("org_name","")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n'
            f'<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {_compute_hash(body)} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        readiness = self._calculate_readiness(data)
        result = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION, "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(), "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""), "reporting_year": data.get("reporting_year", ""),
            "stakeholder": "assurance_provider",
            "evidence_hierarchy": EVIDENCE_TIERS,
            "hash_chain": data.get("hash_chain", []),
            "lineage": data.get("lineage", {}),
            "methodology": data.get("methodology", {}),
            "controls": data.get("controls", []),
            "assurance_readiness": readiness,
            "workpapers": ISO_14064_WORKPAPERS,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "format": "pdf",
            "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {
                "title": f"Assurance Evidence - {data.get('org_name', '')}",
                "author": "GreenLang PACK-030",
            },
        }

    # -----------------------------------------------------------------------
    # Readiness Calculator
    # -----------------------------------------------------------------------

    def _calculate_readiness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        checklist_data = data.get("assurance_checklist", {})
        checks = {}
        for item in ASSURANCE_CHECKLIST:
            checks[item["id"]] = bool(checklist_data.get(item["id"])) or bool(
                checklist_data.get(item["id"].lower().replace("-", "_"))
            )
        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        return {
            "checks": checks, "passed": passed, "total": total,
            "score": round(passed / total * 100, 1) if total else 0,
            "level": "Ready" if passed == total else ("Mostly Ready" if passed >= total * 0.8 else "Not Ready"),
        }

    # -----------------------------------------------------------------------
    # Markdown Section Renderers
    # -----------------------------------------------------------------------

    def _md_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Assurance Evidence Package\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Audience:** Third-Party Assurance Providers  \n"
            f"**Standard:** ISO 14064-3  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-030 v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data):
        readiness = self._calculate_readiness(data)
        lines = [
            "## 1. Executive Summary\n",
            "| Metric | Value |", "|--------|-------|",
            f"| Assurance Readiness | {readiness['score']}% ({readiness['passed']}/{readiness['total']}) |",
            f"| Readiness Level | {readiness['level']} |",
            f"| Assurance Level | {data.get('assurance_level', 'Limited')} |",
            f"| Standard | ISO 14064-3:2019 |",
            f"| Evidence Artifacts | {len(data.get('hash_chain', []))} |",
            f"| Controls Documented | {len(data.get('controls', []))} |",
        ]
        return "\n".join(lines)

    def _md_hash_chain(self, data):
        chain = data.get("hash_chain", [])
        lines = [
            "## 2. Provenance Hash Chain\n",
            "Each data artifact is hashed with SHA-256 for tamper-evident provenance.\n",
            "| # | Artifact | Type | Hash | Timestamp |",
            "|---|----------|------|------|-----------|",
        ]
        for i, item in enumerate(chain, 1):
            h = item.get("hash", _compute_hash(item))
            lines.append(
                f"| {i} | {item.get('artifact', '')} | {item.get('type', '')} | "
                f"`{h[:16]}...` | {item.get('timestamp', '')} |"
            )
        if not chain:
            lines.append("| - | _Hash chain will be populated during verification_ | - | - | - |")
        return "\n".join(lines)

    def _md_lineage(self, data):
        lineage = data.get("lineage", {})
        stages = lineage.get("stages", [])
        lines = [
            "## 3. Data Lineage Diagram\n",
            "```",
            "Source Data --> Extraction --> Normalization --> Calculation --> Validation --> Reporting",
            "   |               |              |                |               |             |",
            "   v               v              v                v               v             v",
            " Raw Files    Structured     Standardized    Emissions       Quality-       Final",
            " Invoices     Activity       Units/Scopes    Calculated      Checked        Reports",
            "```\n",
            "### Lineage Stages\n",
            "| # | Stage | Input | Output | Transform | Owner |",
            "|---|-------|-------|--------|-----------|-------|",
        ]
        for i, s in enumerate(stages, 1):
            lines.append(
                f"| {i} | {s.get('stage', '')} | {s.get('input', '')} | "
                f"{s.get('output', '')} | {s.get('transform', '')} | "
                f"{s.get('owner', '')} |"
            )
        if not stages:
            lines.extend([
                "| 1 | Extraction | Raw files, invoices | Structured records | OCR/parsing | Data Team |",
                "| 2 | Normalization | Structured records | Standardized data | Unit conversion | Data Team |",
                "| 3 | Calculation | Activity data + EFs | Emissions values | GHG Protocol | MRV Engine |",
                "| 4 | Validation | Emissions values | Validated data | Quality checks | QA Team |",
                "| 5 | Reporting | Validated data | Disclosure reports | Template rendering | Reporting |",
            ])
        return "\n".join(lines)

    def _md_methodology(self, data):
        meth = data.get("methodology", {})
        lines = [
            "## 4. Methodology Documentation\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| GHG Protocol | {meth.get('ghg_protocol', 'Corporate Standard (Revised)')} |",
            f"| Consolidation Approach | {meth.get('consolidation', 'Operational Control')} |",
            f"| Emission Factor Sources | {meth.get('ef_sources', 'IPCC AR6, IEA, EPA, DEFRA')} |",
            f"| GWP Values | {meth.get('gwp', 'IPCC AR6 100-year')} |",
            f"| Scope 2 Method | {meth.get('scope2', 'Dual reporting (location + market)')} |",
            f"| Scope 3 Screening | {meth.get('scope3_screening', 'All 15 categories screened')} |",
            f"| Uncertainty Method | {meth.get('uncertainty', 'Monte Carlo simulation + IPCC Tier 2')} |",
            f"| Base Year | {meth.get('base_year', '')} |",
            f"| Restatement Threshold | {meth.get('restatement_threshold', '5% cumulative change')} |",
        ]
        exclusions = meth.get("exclusions", [])
        if exclusions:
            lines.append("\n### Exclusions\n")
            for ex in exclusions:
                lines.append(f"- **{ex.get('item', '')}**: {ex.get('justification', '')}")
        return "\n".join(lines)

    def _md_control_matrix(self, data):
        controls = data.get("controls", [])
        lines = [
            "## 5. Control Matrix\n",
            "| # | Control | Category | Type | Frequency | Owner | Status |",
            "|---|---------|----------|------|-----------|-------|--------|",
        ]
        for i, c in enumerate(controls, 1):
            lines.append(
                f"| {i} | {c.get('name', '')} | {c.get('category', '')} | "
                f"{c.get('type', 'Preventive')} | {c.get('frequency', '')} | "
                f"{c.get('owner', '')} | {c.get('status', 'Active')} |"
            )
        if not controls:
            for i, cat in enumerate(CONTROL_CATEGORIES, 1):
                lines.append(f"| {i} | {cat} controls | {cat} | - | - | - | Pending |")
        return "\n".join(lines)

    def _md_calculation_trail(self, data):
        trails = data.get("calculation_trails", [])
        lines = [
            "## 6. Calculation Trail\n",
            "Step-by-step calculation audit trail from activity data to reported emissions.\n",
            "| # | Step | Activity Data | Emission Factor | Result (tCO2e) | Source |",
            "|---|------|:------------:|:--------------:|---------------:|--------|",
        ]
        for i, t in enumerate(trails, 1):
            lines.append(
                f"| {i} | {t.get('step', '')} | {t.get('activity', '')} | "
                f"{t.get('ef', '')} | {_dec_comma(t.get('result', 0), 2)} | "
                f"{t.get('source', '')} |"
            )
        if not trails:
            lines.append(
                "| - | _Populate calculation_trails to show step-by-step audit trail_ | - | - | - | - |"
            )
        return "\n".join(lines)

    def _md_evidence_hierarchy(self, data):
        distribution = data.get("evidence_distribution", {})
        lines = [
            "## 7. Evidence Hierarchy\n",
            "Data quality tiers from ISO 14064-1 / GHG Protocol.\n",
            "| Tier | Name | Description | Quality | Coverage (%) |",
            "|:----:|------|-------------|---------|:------------:|",
        ]
        for tier in EVIDENCE_TIERS:
            coverage = distribution.get(f"tier_{tier['tier']}", distribution.get(tier["name"].lower(), "N/A"))
            lines.append(
                f"| {tier['tier']} | {tier['name']} | {tier['description']} | "
                f"{tier['quality']} | {coverage}% |"
            )
        return "\n".join(lines)

    def _md_materiality(self, data):
        mat = data.get("materiality", {})
        lines = [
            "## 8. Materiality Assessment\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Threshold | {mat.get('threshold', '5% of total emissions')} |",
            f"| Method | {mat.get('method', 'Quantitative + qualitative')} |",
            f"| Total Emissions | {_dec_comma(mat.get('total_tco2e', 0), 0)} tCO2e |",
            f"| Material Sources | {mat.get('material_count', 'N/A')} |",
            f"| Excluded Sources | {mat.get('excluded_count', 'N/A')} |",
            f"| Excluded Emissions | {_dec_comma(mat.get('excluded_tco2e', 0), 0)} tCO2e ({_dec(mat.get('excluded_pct', 0))}%) |",
        ]
        return "\n".join(lines)

    def _md_data_quality(self, data):
        dq = data.get("data_quality", {})
        indicators = dq.get("indicators", [])
        lines = [
            "## 9. Data Quality Score\n",
            f"**Overall Score:** {_dec(dq.get('score', 0))}%  \n"
            f"**Assessment:** {dq.get('assessment', 'Reasonable')}\n",
            "| # | Indicator | Score | Weight | Weighted |",
            "|---|-----------|:-----:|:------:|:--------:|",
        ]
        for i, ind in enumerate(indicators, 1):
            score = float(ind.get("score", 0))
            weight = float(ind.get("weight", 1))
            lines.append(
                f"| {i} | {ind.get('name', '')} | {_dec(score)}% | "
                f"{_dec(weight)} | {_dec(score * weight)}% |"
            )
        if not indicators:
            lines.extend([
                "| 1 | Completeness | N/A | 0.30 | - |",
                "| 2 | Accuracy | N/A | 0.25 | - |",
                "| 3 | Consistency | N/A | 0.20 | - |",
                "| 4 | Transparency | N/A | 0.15 | - |",
                "| 5 | Timeliness | N/A | 0.10 | - |",
            ])
        return "\n".join(lines)

    def _md_checklist(self, data):
        readiness = self._calculate_readiness(data)
        lines = [
            "## 10. Assurance Readiness Checklist\n",
            f"**Score:** {readiness['score']}% ({readiness['passed']}/{readiness['total']})  \n"
            f"**Level:** {readiness['level']}\n",
            "| ID | Item | ISO Ref | Status |",
            "|----|------|---------|--------|",
        ]
        for item in ASSURANCE_CHECKLIST:
            status = "Complete" if readiness["checks"].get(item["id"]) else "Pending"
            lines.append(f"| {item['id']} | {item['item']} | {item['iso']} | {status} |")
        return "\n".join(lines)

    def _md_workpapers(self, data):
        wp_data = data.get("workpapers", {})
        lines = [
            "## 11. ISO 14064-3 Workpaper Summary\n",
            "| WP | Title | Scope | Status |",
            "|----|-------|-------|--------|",
        ]
        for wp in ISO_14064_WORKPAPERS:
            status = wp_data.get(wp["wp"], {}).get("status", "Prepared")
            lines.append(f"| {wp['wp']} | {wp['title']} | {wp['scope']} | {status} |")
        return "\n".join(lines)

    def _md_audit(self, data):
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            f"## 12. Audit Trail & Provenance\n\n"
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
            f"*Assurance evidence package - ISO 14064-3 aligned.*"
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
            f"th,td{{border:1px solid #ce93d8;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f3e5f5;}}"
            f".status-ok{{color:#2e7d32;font-weight:600;}}"
            f".status-pending{{color:#f57f17;font-weight:600;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));"
            f"gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});"
            f"border-radius:10px;padding:18px;text-align:center;"
            f"border-left:4px solid {_ACCENT};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.4em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f"code{{background:#f3e5f5;padding:2px 6px;border-radius:3px;font-size:0.85em;}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};"
            f"color:{_SECONDARY};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Assurance Evidence Package</h1>\n'
            f'<p><strong>{data.get("org_name", "")}</strong> | '
            f'{data.get("reporting_year", "")} | ISO 14064-3 | {ts}</p>'
        )

    def _html_executive_summary(self, data):
        readiness = self._calculate_readiness(data)
        return (
            f'<h2>1. Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Readiness</div>'
            f'<div class="card-value">{readiness["score"]}%</div></div>\n'
            f'<div class="card"><div class="card-label">Level</div>'
            f'<div class="card-value">{readiness["level"]}</div></div>\n'
            f'<div class="card"><div class="card-label">Checklist</div>'
            f'<div class="card-value">{readiness["passed"]}/{readiness["total"]}</div></div>\n'
            f'<div class="card"><div class="card-label">Assurance</div>'
            f'<div class="card-value">{data.get("assurance_level", "Limited")}</div></div>\n</div>'
        )

    def _html_hash_chain(self, data):
        chain = data.get("hash_chain", [])
        rows = ""
        for i, item in enumerate(chain, 1):
            h = item.get("hash", _compute_hash(item))
            rows += (
                f'<tr><td>{i}</td><td>{item.get("artifact", "")}</td>'
                f'<td>{item.get("type", "")}</td>'
                f'<td><code>{h[:16]}...</code></td></tr>\n'
            )
        return (
            f'<h2>2. Hash Chain</h2>\n<table>\n'
            f'<tr><th>#</th><th>Artifact</th><th>Type</th><th>Hash</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_evidence_hierarchy(self, data):
        distribution = data.get("evidence_distribution", {})
        rows = ""
        for tier in EVIDENCE_TIERS:
            coverage = distribution.get(f"tier_{tier['tier']}", "N/A")
            rows += (
                f'<tr><td>{tier["tier"]}</td><td>{tier["name"]}</td>'
                f'<td>{tier["quality"]}</td><td>{coverage}%</td></tr>\n'
            )
        return (
            f'<h2>3. Evidence Hierarchy</h2>\n<table>\n'
            f'<tr><th>Tier</th><th>Name</th><th>Quality</th><th>Coverage</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_checklist(self, data):
        readiness = self._calculate_readiness(data)
        rows = ""
        for item in ASSURANCE_CHECKLIST:
            complete = readiness["checks"].get(item["id"])
            cls = "status-ok" if complete else "status-pending"
            label = "Complete" if complete else "Pending"
            rows += (
                f'<tr><td>{item["id"]}</td><td>{item["item"]}</td>'
                f'<td>{item["iso"]}</td><td class="{cls}">{label}</td></tr>\n'
            )
        return (
            f'<h2>4. Assurance Checklist</h2>\n'
            f'<p>Score: {readiness["score"]}% - Level: {readiness["level"]}</p>\n<table>\n'
            f'<tr><th>ID</th><th>Item</th><th>ISO Ref</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_workpapers(self, data):
        wp_data = data.get("workpapers", {})
        rows = ""
        for wp in ISO_14064_WORKPAPERS:
            status = wp_data.get(wp["wp"], {}).get("status", "Prepared")
            rows += f'<tr><td>{wp["wp"]}</td><td>{wp["title"]}</td><td>{status}</td></tr>\n'
        return (
            f'<h2>5. Workpapers</h2>\n<table>\n'
            f'<tr><th>WP</th><th>Title</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_audit(self, data):
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            f'<h2>6. Audit</h2>\n<table>\n'
            f'<tr><th>Param</th><th>Value</th></tr>\n'
            f'<tr><td>ID</td><td><code>{rid}</code></td></tr>\n'
            f'<tr><td>Generated</td><td>{ts}</td></tr>\n'
            f'<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'
        )

    def _html_footer(self, data):
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-030 on {ts} '
            f'- Assurance Evidence Package</div>'
        )
