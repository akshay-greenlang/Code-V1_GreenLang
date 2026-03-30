# -*- coding: utf-8 -*-
"""
ValueChainRiskMapTemplate - CSDDD Value Chain Risk Mapping Report

Renders a comprehensive value chain visualization with risk overlay showing
tier-by-tier risk assessment, country-level risk mapping, sector risk
distribution, hotspot identification, and remediation recommendations per
the CSDDD (Directive (EU) 2024/1760) due diligence requirements.

Regulatory References:
    - Directive (EU) 2024/1760, Article 6 (Identifying Adverse Impacts)
    - Directive (EU) 2024/1760, Article 7 (Prioritisation of Impacts)
    - Directive (EU) 2024/1760, Article 8 (Prevention of Potential Impacts)
    - OECD Due Diligence Guidance for Responsible Business Conduct (2018)
    - UNGPs on Business and Human Rights (2011)

Sections:
    1. Chain Overview - End-to-end value chain summary
    2. Tier Breakdown - Risk assessment per supply chain tier
    3. Country Risk Map - Geographic risk distribution
    4. Sector Risk Distribution - Sectoral risk analysis
    5. Hotspot Analysis - High-risk node identification
    6. Recommendations - Risk mitigation priorities

Author: GreenLang Team
Version: 19.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"

_SECTIONS: List[str] = [
    "chain_overview",
    "tier_breakdown",
    "country_risk_map",
    "sector_risk_distribution",
    "hotspot_analysis",
    "recommendations",
]

# Standard risk categories under CSDDD
_RISK_CATEGORIES: List[str] = [
    "forced_labour",
    "child_labour",
    "workplace_safety",
    "freedom_of_association",
    "discrimination",
    "living_wage",
    "land_rights",
    "pollution",
    "biodiversity_loss",
    "water_stress",
    "climate_impact",
    "deforestation",
]

_RISK_LEVELS: Dict[str, int] = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "negligible": 0,
}

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

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

class ValueChainRiskMapTemplate:
    """
    CSDDD Value Chain Risk Mapping Report.

    Renders a multi-tier value chain risk map covering supply chain tiers
    (raw materials, components, manufacturing, distribution), geographic
    risk analysis by country, sector-level risk distribution, and hotspot
    identification with prioritized mitigation recommendations.

    Example:
        >>> tpl = ValueChainRiskMapTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ValueChainRiskMapTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = utcnow()
        report_id = _new_uuid()
        result: Dict[str, Any] = {"report_id": report_id}
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
        """Return list of available section names."""
        return list(_SECTIONS)

    def validate_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data and return errors/warnings."""
        errors: List[str] = []
        warnings: List[str] = []
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if not data.get("reporting_year"):
            errors.append("reporting_year is required")
        if "value_chain" not in data:
            errors.append("value_chain is required for risk mapping")
        if "country_risks" not in data:
            warnings.append("country_risks missing; country map will be empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render value chain risk map as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data),
            self._md_chain_overview(data),
            self._md_tier_breakdown(data),
            self._md_country_risk_map(data),
            self._md_sector_risk(data),
            self._md_hotspot_analysis(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render value chain risk map as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_chain_overview(data),
            self._html_tier_breakdown(data),
            self._html_country_risk(data),
            self._html_hotspots(data),
            self._html_recommendations(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Value Chain Risk Map - CSDDD</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render value chain risk map as JSON."""
        self.generated_at = utcnow()
        result: Dict[str, Any] = {
            "template": "value_chain_risk_map",
            "directive_reference": "Directive (EU) 2024/1760, Art 6-8",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "chain_overview": self._section_chain_overview(data),
            "tier_breakdown": self._section_tier_breakdown(data),
            "country_risk_map": self._section_country_risk_map(data),
            "sector_risk_distribution": self._section_sector_risk_distribution(data),
            "hotspot_analysis": self._section_hotspot_analysis(data),
            "recommendations": self._section_recommendations(data),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _section_chain_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build value chain overview section."""
        chain = data.get("value_chain", {})
        tiers = chain.get("tiers", [])
        suppliers = chain.get("suppliers", [])
        total_suppliers = len(suppliers)
        total_countries = len(set(s.get("country", "") for s in suppliers if s.get("country")))
        risk_scores = [s.get("risk_score", 0.0) for s in suppliers]
        avg_risk = round(sum(risk_scores) / len(risk_scores), 2) if risk_scores else 0.0
        return {
            "title": "Value Chain Overview",
            "total_tiers": len(tiers),
            "total_suppliers": total_suppliers,
            "total_countries": total_countries,
            "average_risk_score": avg_risk,
            "chain_depth": chain.get("chain_depth", len(tiers)),
            "direct_suppliers": sum(1 for s in suppliers if s.get("tier") == 1),
            "indirect_suppliers": sum(1 for s in suppliers if s.get("tier", 0) > 1),
            "coverage_pct": round(chain.get("coverage_pct", 0.0), 1),
            "last_assessment_date": chain.get("last_assessment_date", ""),
        }

    def _section_tier_breakdown(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build tier-by-tier risk breakdown section."""
        chain = data.get("value_chain", {})
        suppliers = chain.get("suppliers", [])
        tier_map: Dict[int, List[Dict[str, Any]]] = {}
        for s in suppliers:
            tier = s.get("tier", 0)
            if tier not in tier_map:
                tier_map[tier] = []
            tier_map[tier].append(s)
        tier_summaries: List[Dict[str, Any]] = []
        for tier_num in sorted(tier_map.keys()):
            tier_suppliers = tier_map[tier_num]
            scores = [s.get("risk_score", 0.0) for s in tier_suppliers]
            avg = round(sum(scores) / len(scores), 2) if scores else 0.0
            high_risk_count = sum(1 for s in scores if s >= 3.0)
            tier_summaries.append({
                "tier": tier_num,
                "tier_label": self._get_tier_label(tier_num),
                "supplier_count": len(tier_suppliers),
                "average_risk_score": avg,
                "max_risk_score": round(max(scores), 2) if scores else 0.0,
                "high_risk_suppliers": high_risk_count,
                "countries": list(set(
                    s.get("country", "") for s in tier_suppliers if s.get("country")
                )),
            })
        return {
            "title": "Tier-by-Tier Risk Breakdown",
            "tiers": tier_summaries,
            "total_tiers": len(tier_summaries),
        }

    def _section_country_risk_map(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build country-level risk map section."""
        country_risks = data.get("country_risks", [])
        chain = data.get("value_chain", {})
        suppliers = chain.get("suppliers", [])
        country_supplier_count: Dict[str, int] = {}
        for s in suppliers:
            country = s.get("country", "Unknown")
            country_supplier_count[country] = country_supplier_count.get(country, 0) + 1
        country_entries: List[Dict[str, Any]] = []
        for cr in country_risks:
            country_code = cr.get("country_code", "")
            country_entries.append({
                "country_code": country_code,
                "country_name": cr.get("country_name", ""),
                "overall_risk_score": round(cr.get("overall_risk_score", 0.0), 2),
                "human_rights_risk": round(cr.get("human_rights_risk", 0.0), 2),
                "environmental_risk": round(cr.get("environmental_risk", 0.0), 2),
                "governance_risk": round(cr.get("governance_risk", 0.0), 2),
                "supplier_count": country_supplier_count.get(country_code, 0),
                "risk_level": self._score_to_level(
                    cr.get("overall_risk_score", 0.0)
                ),
            })
        country_entries.sort(
            key=lambda x: x["overall_risk_score"], reverse=True
        )
        return {
            "title": "Country Risk Map",
            "total_countries": len(country_entries),
            "high_risk_countries": sum(
                1 for c in country_entries if c["risk_level"] in ("critical", "high")
            ),
            "countries": country_entries,
        }

    def _section_sector_risk_distribution(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build sector risk distribution section."""
        sectors = data.get("sector_risks", [])
        sector_entries: List[Dict[str, Any]] = []
        for sec in sectors:
            sector_entries.append({
                "sector_code": sec.get("sector_code", ""),
                "sector_name": sec.get("sector_name", ""),
                "risk_score": round(sec.get("risk_score", 0.0), 2),
                "risk_level": self._score_to_level(sec.get("risk_score", 0.0)),
                "supplier_count": sec.get("supplier_count", 0),
                "key_risks": sec.get("key_risks", []),
                "high_risk_activities": sec.get("high_risk_activities", []),
            })
        sector_entries.sort(key=lambda x: x["risk_score"], reverse=True)
        return {
            "title": "Sector Risk Distribution",
            "total_sectors": len(sector_entries),
            "high_risk_sectors": sum(
                1 for s in sector_entries if s["risk_level"] in ("critical", "high")
            ),
            "sectors": sector_entries,
        }

    def _section_hotspot_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build hotspot analysis section."""
        chain = data.get("value_chain", {})
        suppliers = chain.get("suppliers", [])
        hotspots: List[Dict[str, Any]] = []
        for s in suppliers:
            if s.get("risk_score", 0.0) >= 3.0:
                hotspots.append({
                    "supplier_id": s.get("supplier_id", ""),
                    "supplier_name": s.get("supplier_name", ""),
                    "tier": s.get("tier", 0),
                    "country": s.get("country", ""),
                    "sector": s.get("sector", ""),
                    "risk_score": round(s.get("risk_score", 0.0), 2),
                    "risk_level": self._score_to_level(s.get("risk_score", 0.0)),
                    "risk_categories": s.get("risk_categories", []),
                    "adverse_impacts_identified": s.get("adverse_impacts", []),
                    "last_audit_date": s.get("last_audit_date", ""),
                    "corrective_actions": s.get("corrective_actions", []),
                })
        hotspots.sort(key=lambda x: x["risk_score"], reverse=True)
        custom_hotspots = data.get("hotspots", [])
        if custom_hotspots:
            hotspots.extend(custom_hotspots)
        return {
            "title": "Hotspot Analysis",
            "total_hotspots": len(hotspots),
            "critical_hotspots": sum(
                1 for h in hotspots if h.get("risk_level") == "critical"
            ),
            "high_hotspots": sum(
                1 for h in hotspots if h.get("risk_level") == "high"
            ),
            "hotspots": hotspots,
        }

    def _section_recommendations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build recommendations section."""
        recommendations = data.get("vc_recommendations", [])
        if not recommendations:
            recommendations = self._generate_default_recommendations(data)
        return {
            "title": "Risk Mitigation Recommendations",
            "total_recommendations": len(recommendations),
            "recommendations": [
                {
                    "priority": r.get("priority", "medium"),
                    "category": r.get("category", ""),
                    "action": r.get("action", ""),
                    "target_tier": r.get("target_tier", "all"),
                    "target_countries": r.get("target_countries", []),
                    "expected_risk_reduction": round(
                        r.get("expected_risk_reduction", 0.0), 1
                    ),
                    "timeline": r.get("timeline", ""),
                }
                for r in recommendations
            ],
        }

    # ------------------------------------------------------------------
    # Calculation helpers
    # ------------------------------------------------------------------

    def _get_tier_label(self, tier: int) -> str:
        """Map tier number to a descriptive label."""
        labels = {
            0: "Own Operations",
            1: "Direct Suppliers (Tier 1)",
            2: "Sub-Suppliers (Tier 2)",
            3: "Raw Material Suppliers (Tier 3)",
            4: "Deep Supply Chain (Tier 4+)",
        }
        return labels.get(tier, f"Tier {tier}")

    def _score_to_level(self, score: float) -> str:
        """Convert numeric risk score to risk level string."""
        if score >= 4.0:
            return "critical"
        elif score >= 3.0:
            return "high"
        elif score >= 2.0:
            return "medium"
        elif score >= 1.0:
            return "low"
        else:
            return "negligible"

    def _generate_default_recommendations(
        self, data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate default recommendations from hotspot analysis."""
        hotspot_sec = self._section_hotspot_analysis(data)
        recs: List[Dict[str, Any]] = []
        if hotspot_sec["critical_hotspots"] > 0:
            recs.append({
                "priority": "critical",
                "category": "immediate_action",
                "action": (
                    "Conduct on-site audits for all critical-risk suppliers "
                    "and establish corrective action plans within 90 days"
                ),
                "target_tier": "all",
                "target_countries": [],
                "expected_risk_reduction": 25.0,
                "timeline": "0-3 months",
            })
        if hotspot_sec["high_hotspots"] > 0:
            recs.append({
                "priority": "high",
                "category": "enhanced_monitoring",
                "action": (
                    "Implement enhanced monitoring for high-risk suppliers "
                    "including quarterly assessments and KPI tracking"
                ),
                "target_tier": "1",
                "target_countries": [],
                "expected_risk_reduction": 15.0,
                "timeline": "3-6 months",
            })
        country_sec = self._section_country_risk_map(data)
        high_risk_countries = [
            c["country_name"] for c in country_sec["countries"]
            if c["risk_level"] in ("critical", "high")
        ]
        if high_risk_countries:
            recs.append({
                "priority": "high",
                "category": "country_risk",
                "action": (
                    f"Develop country-specific risk mitigation plans for "
                    f"high-risk jurisdictions: {', '.join(high_risk_countries[:5])}"
                ),
                "target_tier": "all",
                "target_countries": high_risk_countries[:5],
                "expected_risk_reduction": 20.0,
                "timeline": "3-6 months",
            })
        recs.append({
            "priority": "medium",
            "category": "contractual",
            "action": (
                "Update supplier contracts to include CSDDD-aligned due "
                "diligence clauses with cascading requirements (Art 14)"
            ),
            "target_tier": "1",
            "target_countries": [],
            "expected_risk_reduction": 10.0,
            "timeline": "6-12 months",
        })
        recs.append({
            "priority": "medium",
            "category": "capacity_building",
            "action": (
                "Establish supplier capacity building programme focusing "
                "on human rights and environmental standards"
            ),
            "target_tier": "1-2",
            "target_countries": [],
            "expected_risk_reduction": 15.0,
            "timeline": "6-18 months",
        })
        return recs

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Value Chain Risk Map - CSDDD\n\n"
            f"**Entity:** {data.get('entity_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Generated:** {ts}  \n"
            f"**Reference:** Directive (EU) 2024/1760, Art 6-8"
        )

    def _md_chain_overview(self, data: Dict[str, Any]) -> str:
        """Render chain overview as markdown."""
        sec = self._section_chain_overview(data)
        return (
            f"## {sec['title']}\n\n"
            f"| Metric | Value |\n|--------|------:|\n"
            f"| Total Tiers | {sec['total_tiers']} |\n"
            f"| Total Suppliers | {sec['total_suppliers']:,} |\n"
            f"| Direct Suppliers | {sec['direct_suppliers']:,} |\n"
            f"| Indirect Suppliers | {sec['indirect_suppliers']:,} |\n"
            f"| Countries | {sec['total_countries']} |\n"
            f"| Average Risk Score | {sec['average_risk_score']:.2f} / 5.0 |\n"
            f"| Coverage | {sec['coverage_pct']:.1f}% |"
        )

    def _md_tier_breakdown(self, data: Dict[str, Any]) -> str:
        """Render tier breakdown as markdown."""
        sec = self._section_tier_breakdown(data)
        lines = [
            f"## {sec['title']}\n",
            "| Tier | Suppliers | Avg Risk | Max Risk | High Risk |",
            "|------|--------:|--------:|--------:|---------:|",
        ]
        for t in sec["tiers"]:
            lines.append(
                f"| {t['tier_label']} | {t['supplier_count']:,} | "
                f"{t['average_risk_score']:.2f} | {t['max_risk_score']:.2f} | "
                f"{t['high_risk_suppliers']} |"
            )
        return "\n".join(lines)

    def _md_country_risk_map(self, data: Dict[str, Any]) -> str:
        """Render country risk map as markdown."""
        sec = self._section_country_risk_map(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Countries Assessed:** {sec['total_countries']}  \n"
            f"**High/Critical Risk Countries:** {sec['high_risk_countries']}\n",
            "| Country | Code | HR Risk | Env Risk | Gov Risk | Overall | Suppliers |",
            "|---------|------|-------:|--------:|--------:|-------:|---------:|",
        ]
        for c in sec["countries"]:
            lines.append(
                f"| {c['country_name']} | {c['country_code']} | "
                f"{c['human_rights_risk']:.2f} | {c['environmental_risk']:.2f} | "
                f"{c['governance_risk']:.2f} | {c['overall_risk_score']:.2f} | "
                f"{c['supplier_count']} |"
            )
        return "\n".join(lines)

    def _md_sector_risk(self, data: Dict[str, Any]) -> str:
        """Render sector risk distribution as markdown."""
        sec = self._section_sector_risk_distribution(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Sectors Assessed:** {sec['total_sectors']}  \n"
            f"**High/Critical Risk Sectors:** {sec['high_risk_sectors']}\n",
            "| Sector | Risk Score | Level | Suppliers | Key Risks |",
            "|--------|----------:|-------|--------:|-----------|",
        ]
        for s in sec["sectors"]:
            key_risks = ", ".join(s["key_risks"][:3]) if s["key_risks"] else "N/A"
            lines.append(
                f"| {s['sector_name']} | {s['risk_score']:.2f} | "
                f"{s['risk_level'].title()} | {s['supplier_count']} | {key_risks} |"
            )
        return "\n".join(lines)

    def _md_hotspot_analysis(self, data: Dict[str, Any]) -> str:
        """Render hotspot analysis as markdown."""
        sec = self._section_hotspot_analysis(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Hotspots:** {sec['total_hotspots']}  \n"
            f"**Critical:** {sec['critical_hotspots']} | "
            f"**High:** {sec['high_hotspots']}\n",
        ]
        if sec["hotspots"]:
            lines.append("| Supplier | Tier | Country | Sector | Risk | Level |")
            lines.append("|----------|-----:|---------|--------|-----:|-------|")
            for h in sec["hotspots"][:20]:
                lines.append(
                    f"| {h['supplier_name']} | {h['tier']} | "
                    f"{h['country']} | {h['sector']} | "
                    f"{h['risk_score']:.2f} | {h['risk_level'].title()} |"
                )
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations as markdown."""
        sec = self._section_recommendations(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Recommendations:** {sec['total_recommendations']}\n",
            "| Priority | Category | Action | Timeline | Risk Reduction |",
            "|----------|----------|--------|----------|---------------:|",
        ]
        for r in sec["recommendations"]:
            lines.append(
                f"| {r['priority'].upper()} | {r['category']} | "
                f"{r['action'][:60]}... | {r['timeline']} | "
                f"{r['expected_risk_reduction']:.1f}% |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Report generated by PACK-019 CSDDD Readiness Pack on {ts}*"

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:1200px;margin:auto}"
            "h1{color:#1a237e;border-bottom:2px solid #1a237e;padding-bottom:.3em}"
            "h2{color:#283593;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#e8eaf6}"
            ".critical{background:#ffcdd2;color:#b71c1c}"
            ".high{background:#ffe0b2;color:#e65100}"
            ".medium{background:#fff9c4;color:#f57f17}"
            ".low{background:#c8e6c9;color:#1b5e20}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        return (
            f"<h1>Value Chain Risk Map - CSDDD</h1>\n"
            f"<p><strong>{data.get('entity_name', '')}</strong> | "
            f"{data.get('reporting_year', '')}</p>"
        )

    def _html_chain_overview(self, data: Dict[str, Any]) -> str:
        """Render chain overview HTML."""
        sec = self._section_chain_overview(data)
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Total Suppliers</td><td>{sec['total_suppliers']:,}</td></tr>"
            f"<tr><td>Countries</td><td>{sec['total_countries']}</td></tr>"
            f"<tr><td>Average Risk</td><td>{sec['average_risk_score']:.2f}</td></tr>"
            f"<tr><td>Coverage</td><td>{sec['coverage_pct']:.1f}%</td></tr>"
            f"</table>"
        )

    def _html_tier_breakdown(self, data: Dict[str, Any]) -> str:
        """Render tier breakdown HTML."""
        sec = self._section_tier_breakdown(data)
        rows = "".join(
            f"<tr><td>{t['tier_label']}</td><td>{t['supplier_count']}</td>"
            f"<td>{t['average_risk_score']:.2f}</td>"
            f"<td>{t['high_risk_suppliers']}</td></tr>"
            for t in sec["tiers"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Tier</th><th>Suppliers</th><th>Avg Risk</th>"
            f"<th>High Risk</th></tr>{rows}</table>"
        )

    def _html_country_risk(self, data: Dict[str, Any]) -> str:
        """Render country risk map HTML."""
        sec = self._section_country_risk_map(data)
        rows = ""
        for c in sec["countries"][:20]:
            css_class = c["risk_level"]
            rows += (
                f"<tr class='{css_class}'><td>{c['country_name']}</td>"
                f"<td>{c['overall_risk_score']:.2f}</td>"
                f"<td>{c['risk_level'].title()}</td>"
                f"<td>{c['supplier_count']}</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>High/Critical Risk Countries: {sec['high_risk_countries']}</p>\n"
            f"<table><tr><th>Country</th><th>Risk Score</th><th>Level</th>"
            f"<th>Suppliers</th></tr>{rows}</table>"
        )

    def _html_hotspots(self, data: Dict[str, Any]) -> str:
        """Render hotspot analysis HTML."""
        sec = self._section_hotspot_analysis(data)
        rows = ""
        for h in sec["hotspots"][:15]:
            css_class = h["risk_level"]
            rows += (
                f"<tr class='{css_class}'><td>{h['supplier_name']}</td>"
                f"<td>{h['tier']}</td><td>{h['country']}</td>"
                f"<td>{h['risk_score']:.2f}</td></tr>"
            )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Total Hotspots: {sec['total_hotspots']} "
            f"(Critical: {sec['critical_hotspots']}, "
            f"High: {sec['high_hotspots']})</p>\n"
            f"<table><tr><th>Supplier</th><th>Tier</th><th>Country</th>"
            f"<th>Risk</th></tr>{rows}</table>"
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        """Render recommendations HTML."""
        sec = self._section_recommendations(data)
        rows = "".join(
            f"<tr><td>{r['priority'].upper()}</td><td>{r['category']}</td>"
            f"<td>{r['action'][:80]}</td><td>{r['timeline']}</td></tr>"
            for r in sec["recommendations"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Priority</th><th>Category</th><th>Action</th>"
            f"<th>Timeline</th></tr>{rows}</table>"
        )
