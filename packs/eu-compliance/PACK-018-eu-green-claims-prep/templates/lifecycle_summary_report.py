# -*- coding: utf-8 -*-
"""
LifecycleSummaryReportTemplate - EU Green Claims Lifecycle Assessment Summary

Renders lifecycle assessment (LCA) and Product Environmental Footprint (PEF)
results into a structured summary report. Covers system boundary definition,
impact category results, hotspot analysis, PEF scoring, and data quality
evaluation to substantiate lifecycle-based environmental claims.

Sections:
    1. Product Overview - Product identification and scope
    2. System Boundary - LCA boundary definition and cut-offs
    3. Impact Assessment - LCIA results by category
    4. Hotspot Analysis - Key lifecycle stage contributors
    5. PEF Score - Product Environmental Footprint single score
    6. Data Quality - DQR assessment per dataset
    7. Provenance - Data lineage and hash chain

PACK Reference: PACK-018 EU Green Claims Prep Pack
Author: GreenLang Team
Version: 18.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_SECTIONS: List[Dict[str, Any]] = [
    {"id": "product_overview", "title": "Product Overview", "order": 1},
    {"id": "system_boundary", "title": "System Boundary", "order": 2},
    {"id": "impact_assessment", "title": "Impact Assessment", "order": 3},
    {"id": "hotspot_analysis", "title": "Hotspot Analysis", "order": 4},
    {"id": "pef_score", "title": "PEF Score", "order": 5},
    {"id": "data_quality", "title": "Data Quality", "order": 6},
    {"id": "provenance", "title": "Provenance", "order": 7},
]

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

class LifecycleSummaryReportTemplate:
    """
    EU Green Claims Directive - Lifecycle Assessment Summary Report.

    Renders LCA/PEF results into a structured report covering system
    boundary, impact categories (EF 3.1 method), lifecycle hotspots,
    single-score PEF results, and data quality ratings. Supports
    substantiation of lifecycle-based environmental claims.

    Example:
        >>> tpl = LifecycleSummaryReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize LifecycleSummaryReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render lifecycle summary report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_product_overview(data),
            self._md_system_boundary(data),
            self._md_impact_assessment(data),
            self._md_hotspot_analysis(data),
            self._md_pef_score(data),
            self._md_data_quality(data),
            self._md_provenance(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render lifecycle summary report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_product_overview(data),
            self._html_impact_assessment(data),
            self._html_hotspot_analysis(data),
            self._html_pef_score(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Lifecycle Summary Report - EU Green Claims</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render lifecycle summary report as structured JSON."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "lifecycle_summary_report",
            "directive_reference": "EU Green Claims Directive 2023/0085",
            "version": "18.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "product_name": data.get("product_name", ""),
            "product_overview": self._section_product_overview(data),
            "system_boundary": self._section_system_boundary(data),
            "impact_assessment": self._section_impact_assessment(data),
            "hotspot_analysis": self._section_hotspot_analysis(data),
            "pef_score": self._section_pef_score(data),
            "data_quality": self._section_data_quality(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def get_sections(self) -> List[Dict[str, Any]]:
        """Return list of available section definitions."""
        return list(_SECTIONS)

    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data and return errors/warnings."""
        errors: List[str] = []
        warnings: List[str] = []
        if not data.get("product_name"):
            errors.append("product_name is required")
        if not data.get("impact_categories"):
            errors.append("impact_categories list is required for LCA results")
        if not data.get("functional_unit"):
            warnings.append("functional_unit missing; will default to empty")
        if not data.get("lifecycle_stages"):
            warnings.append("lifecycle_stages missing; hotspot analysis will be limited")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    # ------------------------------------------------------------------
    # Section builders (dict)
    # ------------------------------------------------------------------

    def _section_product_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build product overview section."""
        return {
            "title": "Product Overview",
            "product_name": data.get("product_name", ""),
            "product_category": data.get("product_category", ""),
            "functional_unit": data.get("functional_unit", ""),
            "reference_flow": data.get("reference_flow", ""),
            "pefcr_applicable": data.get("pefcr_applicable", False),
            "pefcr_name": data.get("pefcr_name", ""),
            "lca_software": data.get("lca_software", ""),
            "lcia_method": data.get("lcia_method", "EF 3.1"),
        }

    def _section_system_boundary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build system boundary section."""
        stages = data.get("lifecycle_stages", [])
        return {
            "title": "System Boundary",
            "cradle_to": data.get("cradle_to", "gate"),
            "included_stages": [s.get("name", "") for s in stages if s.get("included", True)],
            "excluded_stages": [s.get("name", "") for s in stages if not s.get("included", True)],
            "cut_off_criteria": data.get("cut_off_criteria", "1% mass and energy"),
            "temporal_scope": data.get("temporal_scope", ""),
            "geographical_scope": data.get("geographical_scope", ""),
            "technology_scope": data.get("technology_scope", ""),
            "allocation_method": data.get("allocation_method", ""),
        }

    def _section_impact_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build impact assessment section."""
        categories = data.get("impact_categories", [])
        return {
            "title": "Impact Assessment (LCIA Results)",
            "method": data.get("lcia_method", "EF 3.1"),
            "total_categories": len(categories),
            "categories": [
                {
                    "name": c.get("name", ""),
                    "abbreviation": c.get("abbreviation", ""),
                    "value": c.get("value", 0.0),
                    "unit": c.get("unit", ""),
                    "normalised_value": c.get("normalised_value", 0.0),
                    "weighted_value": c.get("weighted_value", 0.0),
                    "contribution_pct": c.get("contribution_pct", 0.0),
                }
                for c in categories
            ],
        }

    def _section_hotspot_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build hotspot analysis section."""
        hotspots = data.get("hotspots", [])
        return {
            "title": "Hotspot Analysis",
            "total_hotspots": len(hotspots),
            "hotspots": [
                {
                    "lifecycle_stage": h.get("lifecycle_stage", ""),
                    "impact_category": h.get("impact_category", ""),
                    "contribution_pct": round(h.get("contribution_pct", 0.0), 1),
                    "process": h.get("process", ""),
                    "improvement_potential": h.get("improvement_potential", ""),
                }
                for h in sorted(hotspots, key=lambda x: x.get("contribution_pct", 0), reverse=True)
            ],
        }

    def _section_pef_score(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build PEF score section."""
        return {
            "title": "Product Environmental Footprint (PEF) Score",
            "single_score": data.get("pef_single_score", 0.0),
            "single_score_unit": data.get("pef_unit", "mPt"),
            "performance_class": data.get("pef_performance_class", ""),
            "benchmark_comparison": data.get("benchmark_comparison", ""),
            "top_contributors": data.get("pef_top_contributors", []),
            "normalisation_method": data.get("normalisation_method", "EF 3.1 person-equivalent"),
            "weighting_method": data.get("weighting_method", "EF 3.1 default weighting"),
        }

    def _section_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build data quality section."""
        dq_entries = data.get("data_quality_entries", [])
        overall_dqr = data.get("overall_dqr", 0.0)
        return {
            "title": "Data Quality Assessment",
            "overall_dqr": round(overall_dqr, 2),
            "dqr_rating": self._get_dqr_rating(overall_dqr),
            "primary_data_pct": data.get("primary_data_pct", 0.0),
            "secondary_data_pct": data.get("secondary_data_pct", 0.0),
            "entries": [
                {
                    "dataset": e.get("dataset", ""),
                    "ter": e.get("ter", 0.0),
                    "ger": e.get("ger", 0.0),
                    "tir": e.get("tir", 0.0),
                    "completeness": e.get("completeness", 0.0),
                    "dqr": round(e.get("dqr", 0.0), 2),
                }
                for e in dq_entries
            ],
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Lifecycle Assessment Summary - EU Green Claims\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Product:** {data.get('product_name', '')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Methodology:** {data.get('lcia_method', 'EF 3.1')}"
        )

    def _md_product_overview(self, data: Dict[str, Any]) -> str:
        """Render product overview as markdown."""
        sec = self._section_product_overview(data)
        return (
            f"## {sec['title']}\n\n"
            f"- **Product:** {sec['product_name']}\n"
            f"- **Category:** {sec['product_category']}\n"
            f"- **Functional Unit:** {sec['functional_unit']}\n"
            f"- **Reference Flow:** {sec['reference_flow']}\n"
            f"- **PEFCR Applicable:** {'Yes' if sec['pefcr_applicable'] else 'No'}\n"
            f"- **LCIA Method:** {sec['lcia_method']}"
        )

    def _md_system_boundary(self, data: Dict[str, Any]) -> str:
        """Render system boundary as markdown."""
        sec = self._section_system_boundary(data)
        included = ", ".join(sec["included_stages"]) if sec["included_stages"] else "N/A"
        excluded = ", ".join(sec["excluded_stages"]) if sec["excluded_stages"] else "None"
        return (
            f"## {sec['title']}\n\n"
            f"- **Scope:** Cradle-to-{sec['cradle_to']}\n"
            f"- **Included Stages:** {included}\n"
            f"- **Excluded Stages:** {excluded}\n"
            f"- **Cut-off Criteria:** {sec['cut_off_criteria']}\n"
            f"- **Temporal Scope:** {sec['temporal_scope']}\n"
            f"- **Geographical Scope:** {sec['geographical_scope']}\n"
            f"- **Allocation Method:** {sec['allocation_method']}"
        )

    def _md_impact_assessment(self, data: Dict[str, Any]) -> str:
        """Render impact assessment as markdown."""
        sec = self._section_impact_assessment(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Method:** {sec['method']}  \n**Categories:** {sec['total_categories']}\n",
            "| Category | Value | Unit | Weighted | Contribution |",
            "|----------|------:|------|--------:|-----------:|",
        ]
        for c in sec["categories"]:
            lines.append(
                f"| {c['name']} | {c['value']:.4g} | {c['unit']} "
                f"| {c['weighted_value']:.4g} | {c['contribution_pct']:.1f}% |"
            )
        return "\n".join(lines)

    def _md_hotspot_analysis(self, data: Dict[str, Any]) -> str:
        """Render hotspot analysis as markdown."""
        sec = self._section_hotspot_analysis(data)
        lines = [
            f"## {sec['title']}\n",
            "| Stage | Impact Category | Contribution | Process |",
            "|-------|----------------|------------:|---------|",
        ]
        for h in sec["hotspots"][:10]:
            lines.append(
                f"| {h['lifecycle_stage']} | {h['impact_category']} "
                f"| {h['contribution_pct']:.1f}% | {h['process']} |"
            )
        return "\n".join(lines)

    def _md_pef_score(self, data: Dict[str, Any]) -> str:
        """Render PEF score as markdown."""
        sec = self._section_pef_score(data)
        return (
            f"## {sec['title']}\n\n"
            f"**Single Score:** {sec['single_score']:.2f} {sec['single_score_unit']}  \n"
            f"**Performance Class:** {sec['performance_class']}  \n"
            f"**Benchmark:** {sec['benchmark_comparison']}\n\n"
            f"- Normalisation: {sec['normalisation_method']}\n"
            f"- Weighting: {sec['weighting_method']}"
        )

    def _md_data_quality(self, data: Dict[str, Any]) -> str:
        """Render data quality as markdown."""
        sec = self._section_data_quality(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Overall DQR:** {sec['overall_dqr']:.2f} ({sec['dqr_rating']})  \n"
            f"**Primary Data:** {sec['primary_data_pct']:.1f}%  \n"
            f"**Secondary Data:** {sec['secondary_data_pct']:.1f}%\n",
            "| Dataset | TeR | GeR | TiR | Completeness | DQR |",
            "|---------|----:|----:|----:|-----------:|----:|",
        ]
        for e in sec["entries"]:
            lines.append(
                f"| {e['dataset']} | {e['ter']:.1f} | {e['ger']:.1f} "
                f"| {e['tir']:.1f} | {e['completeness']:.1f}% | {e['dqr']:.2f} |"
            )
        return "\n".join(lines)

    def _md_provenance(self, data: Dict[str, Any]) -> str:
        """Render provenance section as markdown."""
        prov = _compute_hash(data)
        return (
            f"## Provenance\n\n"
            f"**Input Data Hash:** `{prov}`  \n"
            f"**Template Version:** 18.0.0  \n"
            f"**Generated At:** {self.generated_at.isoformat() if self.generated_at else ''}"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Report generated by PACK-018 EU Green Claims Prep Pack on {ts}*"

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:1100px;margin:auto}"
            "h1{color:#1b5e20;border-bottom:2px solid #1b5e20;padding-bottom:.3em}"
            "h2{color:#2e7d32;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e8f5e9}"
            ".hotspot{color:#e65100;font-weight:bold}"
            ".good{color:#2e7d32;font-weight:bold}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Lifecycle Assessment Summary - EU Green Claims</h1>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> "
            f"| {data.get('product_name', '')}</p>"
        )

    def _html_product_overview(self, data: Dict[str, Any]) -> str:
        """Render product overview HTML."""
        sec = self._section_product_overview(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<ul><li>Product: {sec['product_name']}</li>"
            f"<li>Functional Unit: {sec['functional_unit']}</li>"
            f"<li>LCIA Method: {sec['lcia_method']}</li></ul>"
        )

    def _html_impact_assessment(self, data: Dict[str, Any]) -> str:
        """Render impact assessment HTML."""
        sec = self._section_impact_assessment(data)
        rows = "".join(
            f"<tr><td>{c['name']}</td><td>{c['value']:.4g}</td>"
            f"<td>{c['unit']}</td><td>{c['contribution_pct']:.1f}%</td></tr>"
            for c in sec["categories"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Category</th><th>Value</th><th>Unit</th>"
            f"<th>Contribution</th></tr>{rows}</table>"
        )

    def _html_hotspot_analysis(self, data: Dict[str, Any]) -> str:
        """Render hotspot analysis HTML."""
        sec = self._section_hotspot_analysis(data)
        rows = "".join(
            f"<tr><td>{h['lifecycle_stage']}</td><td>{h['impact_category']}</td>"
            f"<td>{h['contribution_pct']:.1f}%</td></tr>"
            for h in sec["hotspots"][:10]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Stage</th><th>Impact</th><th>Contribution</th></tr>"
            f"{rows}</table>"
        )

    def _html_pef_score(self, data: Dict[str, Any]) -> str:
        """Render PEF score HTML."""
        sec = self._section_pef_score(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p class='good'>Single Score: {sec['single_score']:.2f} "
            f"{sec['single_score_unit']}</p>\n"
            f"<p>Performance Class: {sec['performance_class']}</p>"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_dqr_rating(self, dqr: float) -> str:
        """Determine data quality rating from DQR score."""
        if dqr <= 1.5:
            return "Excellent"
        elif dqr <= 2.0:
            return "Very Good"
        elif dqr <= 3.0:
            return "Good"
        elif dqr <= 4.0:
            return "Fair"
        else:
            return "Poor"
