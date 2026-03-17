# -*- coding: utf-8 -*-
"""
ESRSAnnualStatementTemplate - ESRS Annual Sustainability Statement

Generates the complete annual sustainability statement combining all ESRS
standards into a single publishable document with basis of preparation,
general disclosures, environmental chapter, social chapter, governance
chapter, cross-references, methodology appendix, data tables, and
digital signature per ESRS requirements.

Sections:
    1. Basis of Preparation
    2. General Disclosures
    3. Environmental Chapter (E1-E5 summaries)
    4. Social Chapter (S1-S4 summaries)
    5. Governance Chapter (G1 summary)
    6. Cross References
    7. Appendix Methodology
    8. Appendix Data Tables
    9. Digital Signature

Author: GreenLang Team
Version: 17.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SECTIONS: List[str] = [
    "basis_of_preparation", "general_disclosures", "environmental_chapter",
    "social_chapter", "governance_chapter", "cross_references",
    "appendix_methodology", "appendix_data_tables", "digital_signature",
]


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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


class ESRSAnnualStatementTemplate:
    """
    ESRS Annual Sustainability Statement template.

    Combines all 12 ESRS topical and cross-cutting standards into a
    single structured annual sustainability statement suitable for
    regulatory filing, including basis of preparation, all environmental
    (E1-E5), social (S1-S4), and governance (G1) chapters, cross-reference
    tables, methodology appendix, data tables, and digital signature.

    Example:
        >>> tpl = ESRSAnnualStatementTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ESRSAnnualStatementTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {}
        for section in _SECTIONS:
            result[section] = self.render_section(section, data)
        result["provenance_hash"] = _compute_hash(result)
        result["generated_at"] = self.generated_at.isoformat()
        return result

    def render_section(self, name: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render a single section by name."""
        handler = getattr(self, f"_section_{name}", None)
        if handler is None:
            raise ValueError(f"Unknown section: {name}")
        return handler(data)

    def get_sections(self) -> List[str]:
        return list(_SECTIONS)

    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        errors: List[str] = []
        warnings: List[str] = []
        if not data.get("reporting_year"):
            errors.append("reporting_year is required")
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if "environmental_data" not in data:
            warnings.append("environmental_data missing; will default to empty")
        if "social_data" not in data:
            warnings.append("social_data missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render annual sustainability statement as Markdown."""
        self.generated_at = _utcnow()
        sections = [self._md_header(data), self._md_basis(data), self._md_general(data),
                     self._md_environmental(data), self._md_social(data), self._md_governance(data),
                     self._md_cross_refs(data), self._md_methodology(data), self._md_data_tables(data),
                     self._md_signature(data), self._md_footer(data)]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render annual sustainability statement as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([self._html_header(data), self._html_basis(data),
                          self._html_environmental(data), self._html_social(data),
                          self._html_signature(data)])
        prov = _compute_hash(body)
        return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
                f'<title>Annual Sustainability Statement</title>\n<style>\n{css}\n</style>\n'
                f'</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
                f'<!-- Provenance: {prov} -->\n</body>\n</html>')

    def render_json(self, data: Dict[str, Any]) -> str:
        """Render annual sustainability statement as JSON string."""
        self.generated_at = _utcnow()
        result = {"template": "esrs_annual_statement", "esrs_reference": "ESRS Full Set", "version": "17.0.0",
                  "generated_at": self.generated_at.isoformat(), "entity_name": data.get("entity_name", ""),
                  "reporting_year": data.get("reporting_year", ""),
                  "standards_covered": data.get("standards_covered", []),
                  "total_datapoints": data.get("total_datapoints", 0),
                  "assurance_level": data.get("assurance_level", "limited")}
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # -- Section renderers --

    def _section_basis_of_preparation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"title": "Basis of Preparation",
                "consolidation_scope": data.get("consolidation_scope", ""),
                "reporting_period": data.get("reporting_year", ""),
                "reporting_boundary": data.get("reporting_boundary", ""),
                "value_chain_estimation": data.get("value_chain_estimation", ""),
                "materiality_methodology": data.get("materiality_methodology", ""),
                "standards_applied": data.get("standards_applied", []),
                "comparative_information": data.get("comparative_information", False),
                "restatements": data.get("restatements", [])}

    def _section_general_disclosures(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"title": "General Disclosures (ESRS 2)",
                "governance_summary": data.get("governance_summary", ""),
                "strategy_summary": data.get("strategy_summary", ""),
                "iro_summary": data.get("iro_summary", ""),
                "material_topics": data.get("material_topics", []),
                "esrs2_datapoints_reported": data.get("esrs2_datapoints_reported", 0)}

    def _section_environmental_chapter(self, data: Dict[str, Any]) -> Dict[str, Any]:
        env = data.get("environmental_data", {})
        return {"title": "Environmental Disclosures",
                "e1_climate": {"included": env.get("e1_included", False),
                               "total_ghg_tco2e": env.get("e1_total_ghg_tco2e", 0.0),
                               "key_metric": env.get("e1_key_metric", "")},
                "e2_pollution": {"included": env.get("e2_included", False),
                                 "air_emissions_t": env.get("e2_air_emissions_t", 0.0),
                                 "key_metric": env.get("e2_key_metric", "")},
                "e3_water": {"included": env.get("e3_included", False),
                             "consumption_m3": env.get("e3_consumption_m3", 0.0),
                             "key_metric": env.get("e3_key_metric", "")},
                "e4_biodiversity": {"included": env.get("e4_included", False),
                                    "sites_assessed": env.get("e4_sites_assessed", 0),
                                    "key_metric": env.get("e4_key_metric", "")},
                "e5_circular": {"included": env.get("e5_included", False),
                                "circular_rate_pct": env.get("e5_circular_rate_pct", 0.0),
                                "key_metric": env.get("e5_key_metric", "")}}

    def _section_social_chapter(self, data: Dict[str, Any]) -> Dict[str, Any]:
        soc = data.get("social_data", {})
        return {"title": "Social Disclosures",
                "s1_workforce": {"included": soc.get("s1_included", False),
                                 "headcount": soc.get("s1_headcount", 0),
                                 "key_metric": soc.get("s1_key_metric", "")},
                "s2_value_chain": {"included": soc.get("s2_included", False),
                                   "suppliers_assessed": soc.get("s2_suppliers_assessed", 0),
                                   "key_metric": soc.get("s2_key_metric", "")},
                "s3_communities": {"included": soc.get("s3_included", False),
                                   "communities_assessed": soc.get("s3_communities_assessed", 0),
                                   "key_metric": soc.get("s3_key_metric", "")},
                "s4_consumers": {"included": soc.get("s4_included", False),
                                 "complaints": soc.get("s4_complaints", 0),
                                 "key_metric": soc.get("s4_key_metric", "")}}

    def _section_governance_chapter(self, data: Dict[str, Any]) -> Dict[str, Any]:
        gov = data.get("governance_data", {})
        return {"title": "Governance Disclosures",
                "g1_conduct": {"included": gov.get("g1_included", False),
                               "corruption_incidents": gov.get("g1_corruption_incidents", 0),
                               "lobbying_eur": gov.get("g1_lobbying_eur", 0.0),
                               "key_metric": gov.get("g1_key_metric", "")}}

    def _section_cross_references(self, data: Dict[str, Any]) -> Dict[str, Any]:
        refs = data.get("cross_references", [])
        return {"title": "Cross-Reference Table", "reference_count": len(refs),
                "references": [{"esrs_dr": r.get("esrs_dr", ""), "annual_report_page": r.get("page", ""),
                                "location": r.get("location", "")} for r in refs]}

    def _section_appendix_methodology(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"title": "Appendix: Methodology Notes",
                "emission_factor_sources": data.get("emission_factor_sources", []),
                "calculation_methods": data.get("calculation_methods", []),
                "data_quality_assessment": data.get("data_quality_assessment", ""),
                "estimation_techniques": data.get("estimation_techniques", []),
                "external_assurance": data.get("external_assurance", "")}

    def _section_appendix_data_tables(self, data: Dict[str, Any]) -> Dict[str, Any]:
        tables = data.get("data_tables", [])
        return {"title": "Appendix: Data Tables", "table_count": len(tables),
                "tables": [{"name": t.get("name", ""), "standard": t.get("standard", ""),
                            "row_count": t.get("row_count", 0)} for t in tables]}

    def _section_digital_signature(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"title": "Digital Signature and Attestation",
                "signatory_name": data.get("signatory_name", ""),
                "signatory_role": data.get("signatory_role", ""),
                "signature_date": data.get("signature_date", ""),
                "statement_hash": _compute_hash(data),
                "assurance_provider": data.get("assurance_provider", ""),
                "assurance_opinion": data.get("assurance_opinion", "")}

    # -- Markdown helpers --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (f"# Annual Sustainability Statement\n\n**Entity:** {data.get('entity_name', '')}  \n"
                f"**Reporting Year:** {data.get('reporting_year', '')}  \n**Generated:** {ts}  \n"
                f"**Framework:** European Sustainability Reporting Standards (ESRS)")

    def _md_basis(self, d: Dict[str, Any]) -> str:
        sec = self._section_basis_of_preparation(d)
        comp = "Yes" if sec["comparative_information"] else "No"
        return (f"## {sec['title']}\n\n**Scope:** {sec['consolidation_scope']}  \n"
                f"**Boundary:** {sec['reporting_boundary']}  \n"
                f"**Materiality:** {sec['materiality_methodology']}  \n"
                f"**Comparative Info:** {comp}")

    def _md_general(self, d: Dict[str, Any]) -> str:
        sec = self._section_general_disclosures(d)
        lines = [f"## {sec['title']}\n", f"**Datapoints Reported:** {sec['esrs2_datapoints_reported']}\n",
                 f"{sec['governance_summary']}\n", f"{sec['strategy_summary']}"]
        return "\n".join(lines)

    def _md_environmental(self, d: Dict[str, Any]) -> str:
        sec = self._section_environmental_chapter(d)
        lines = [f"## {sec['title']}\n",
                 "| Standard | Included | Key Metric |", "|----------|:--------:|------------|"]
        for key, label in [("e1_climate", "E1 Climate"), ("e2_pollution", "E2 Pollution"),
                           ("e3_water", "E3 Water"), ("e4_biodiversity", "E4 Biodiversity"),
                           ("e5_circular", "E5 Circular")]:
            s = sec[key]
            inc = "Yes" if s["included"] else "No"
            lines.append(f"| {label} | {inc} | {s['key_metric']} |")
        return "\n".join(lines)

    def _md_social(self, d: Dict[str, Any]) -> str:
        sec = self._section_social_chapter(d)
        lines = [f"## {sec['title']}\n", "| Standard | Included | Key Metric |", "|----------|:--------:|------------|"]
        for key, label in [("s1_workforce", "S1 Workforce"), ("s2_value_chain", "S2 Value Chain"),
                           ("s3_communities", "S3 Communities"), ("s4_consumers", "S4 Consumers")]:
            s = sec[key]
            inc = "Yes" if s["included"] else "No"
            lines.append(f"| {label} | {inc} | {s['key_metric']} |")
        return "\n".join(lines)

    def _md_governance(self, d: Dict[str, Any]) -> str:
        sec = self._section_governance_chapter(d)
        g1 = sec["g1_conduct"]
        inc = "Yes" if g1["included"] else "No"
        return (f"## {sec['title']}\n\n**G1 Included:** {inc}  \n"
                f"**Corruption Incidents:** {g1['corruption_incidents']}  \n"
                f"**Lobbying:** EUR {g1['lobbying_eur']:,.2f}")

    def _md_cross_refs(self, d: Dict[str, Any]) -> str:
        sec = self._section_cross_references(d)
        lines = [f"## {sec['title']}\n"]
        if sec["references"]:
            lines.extend(["| ESRS DR | Location | Page |", "|---------|----------|------|"])
            for r in sec["references"]:
                lines.append(f"| {r['esrs_dr']} | {r['location']} | {r['annual_report_page']} |")
        return "\n".join(lines)

    def _md_methodology(self, d: Dict[str, Any]) -> str:
        sec = self._section_appendix_methodology(d)
        lines = [f"## {sec['title']}\n", f"**Data Quality:** {sec['data_quality_assessment']}  \n"
                 f"**External Assurance:** {sec['external_assurance']}\n"]
        for m in sec["calculation_methods"]:
            lines.append(f"- {m}")
        return "\n".join(lines)

    def _md_data_tables(self, d: Dict[str, Any]) -> str:
        sec = self._section_appendix_data_tables(d)
        lines = [f"## {sec['title']}\n", f"**Tables:** {sec['table_count']}\n"]
        if sec["tables"]:
            lines.extend(["| Table | Standard | Rows |", "|-------|----------|-----:|"])
            for t in sec["tables"]:
                lines.append(f"| {t['name']} | {t['standard']} | {t['row_count']} |")
        return "\n".join(lines)

    def _md_signature(self, d: Dict[str, Any]) -> str:
        sec = self._section_digital_signature(d)
        return (f"## {sec['title']}\n\n**Signatory:** {sec['signatory_name']}  \n"
                f"**Role:** {sec['signatory_role']}  \n**Date:** {sec['signature_date']}  \n"
                f"**Statement Hash:** `{sec['statement_hash'][:16]}...`  \n"
                f"**Assurance Provider:** {sec['assurance_provider']}  \n"
                f"**Opinion:** {sec['assurance_opinion']}")

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Annual Statement generated by PACK-017 ESRS Full Coverage Pack on {ts}*"

    def _css(self) -> str:
        return ("body{font-family:Arial,sans-serif;margin:2em;color:#333}.report{max-width:960px;margin:auto}"
                "h1{color:#1a5632;border-bottom:2px solid #1a5632;padding-bottom:.3em}h2{color:#2d7a4f;margin-top:1.5em}"
                "table{border-collapse:collapse;width:100%;margin:1em 0}th,td{border:1px solid #ccc;padding:8px;text-align:left}"
                "th{background:#f0f7f3}.signature{border-top:2px solid #333;padding-top:1em;margin-top:2em}")

    def _html_header(self, data: Dict[str, Any]) -> str:
        return (f"<h1>Annual Sustainability Statement</h1>\n"
                f"<p><strong>{data.get('entity_name', '')}</strong> | {data.get('reporting_year', '')}</p>")

    def _html_basis(self, data: Dict[str, Any]) -> str:
        sec = self._section_basis_of_preparation(data)
        return (f"<h2>{sec['title']}</h2>\n<p>Scope: {sec['consolidation_scope']}</p>\n"
                f"<p>Boundary: {sec['reporting_boundary']}</p>")

    def _html_environmental(self, data: Dict[str, Any]) -> str:
        sec = self._section_environmental_chapter(data)
        rows = ""
        for key, label in [("e1_climate", "E1"), ("e2_pollution", "E2"), ("e3_water", "E3"),
                           ("e4_biodiversity", "E4"), ("e5_circular", "E5")]:
            s = sec[key]
            rows += f"<tr><td>{label}</td><td>{'Yes' if s['included'] else 'No'}</td><td>{s['key_metric']}</td></tr>"
        return (f"<h2>{sec['title']}</h2>\n"
                f"<table><tr><th>Standard</th><th>Included</th><th>Key Metric</th></tr>{rows}</table>")

    def _html_social(self, data: Dict[str, Any]) -> str:
        sec = self._section_social_chapter(data)
        rows = ""
        for key, label in [("s1_workforce", "S1"), ("s2_value_chain", "S2"),
                           ("s3_communities", "S3"), ("s4_consumers", "S4")]:
            s = sec[key]
            rows += f"<tr><td>{label}</td><td>{'Yes' if s['included'] else 'No'}</td><td>{s['key_metric']}</td></tr>"
        return (f"<h2>{sec['title']}</h2>\n"
                f"<table><tr><th>Standard</th><th>Included</th><th>Key Metric</th></tr>{rows}</table>")

    def _html_signature(self, data: Dict[str, Any]) -> str:
        sec = self._section_digital_signature(data)
        return (f"<div class='signature'>\n<h2>{sec['title']}</h2>\n"
                f"<p><strong>{sec['signatory_name']}</strong>, {sec['signatory_role']}</p>\n"
                f"<p>Date: {sec['signature_date']}</p>\n"
                f"<p>Hash: <code>{sec['statement_hash'][:16]}...</code></p>\n</div>")


# Alias for backward compatibility with templates/__init__.py
ESRSAnnualStatement = ESRSAnnualStatementTemplate
ESRSAnnualReport = ESRSAnnualStatementTemplate
