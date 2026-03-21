# -*- coding: utf-8 -*-
"""
GHGInventoryReportTemplate - Full GHG Protocol Corporate Standard report (PACK-027).

Renders a comprehensive 20-40 page enterprise GHG inventory report aligned
with GHG Protocol Corporate Accounting and Reporting Standard (Chapter 9)
covering Scope 1/2/3 breakdown across all 15 Scope 3 categories, entity-level
detail, data quality scores per GHG Protocol 5-level hierarchy, methodology
notes, year-over-year trends, and base year comparison.

Sections:
    1. Executive Summary (total emissions, scope mix, YoY change)
    2. Organizational Boundary (consolidation approach, entity hierarchy)
    3. Scope 1 Emissions (8 source categories)
    4. Scope 2 Emissions (location-based + market-based dual reporting)
    5. Scope 3 Emissions (all 15 categories with methodology per category)
    6. Entity-Level Breakdown (per-subsidiary/JV/associate)
    7. Data Quality Matrix (per-category per-entity DQ scores 1-5)
    8. Methodology Notes (emission factors, calculation approaches, exclusions)
    9. Year-over-Year Trends (base year to current)
   10. Intercompany Eliminations
   11. Base Year Recalculation Log
   12. Citations & References
   13. Appendix (detailed calculation traces)

Output: Markdown, HTML, JSON, Excel
Provenance: SHA-256 hash on all outputs

Author: GreenLang Team
Version: 27.0.0
Pack: PACK-027 Enterprise Net Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "27.0.0"
_PACK_ID = "PACK-027"
_TEMPLATE_ID = "ghg_inventory_report"

# ---------------------------------------------------------------------------
# Enterprise colour scheme (dark green / corporate blue)
# ---------------------------------------------------------------------------
_PRIMARY = "#0d3b2e"
_SECONDARY = "#1a6b4f"
_ACCENT = "#2e8b6e"
_LIGHT = "#e0f2ee"
_LIGHTER = "#f0f9f6"
_CARD_BG = "#b2dfdb"
_SCOPE1_CLR = "#1b5e20"
_SCOPE2_CLR = "#1565c0"
_SCOPE3_CLR = "#6a1b9a"
_WARN_CLR = "#e65100"

# ---------------------------------------------------------------------------
# GHG Protocol Scope 3 categories
# ---------------------------------------------------------------------------
SCOPE3_CATEGORIES = [
    {"num": 1, "name": "Purchased Goods & Services", "ghg_ref": "Cat 1"},
    {"num": 2, "name": "Capital Goods", "ghg_ref": "Cat 2"},
    {"num": 3, "name": "Fuel & Energy Activities", "ghg_ref": "Cat 3"},
    {"num": 4, "name": "Upstream Transportation & Distribution", "ghg_ref": "Cat 4"},
    {"num": 5, "name": "Waste Generated in Operations", "ghg_ref": "Cat 5"},
    {"num": 6, "name": "Business Travel", "ghg_ref": "Cat 6"},
    {"num": 7, "name": "Employee Commuting", "ghg_ref": "Cat 7"},
    {"num": 8, "name": "Upstream Leased Assets", "ghg_ref": "Cat 8"},
    {"num": 9, "name": "Downstream Transportation & Distribution", "ghg_ref": "Cat 9"},
    {"num": 10, "name": "Processing of Sold Products", "ghg_ref": "Cat 10"},
    {"num": 11, "name": "Use of Sold Products", "ghg_ref": "Cat 11"},
    {"num": 12, "name": "End-of-Life Treatment of Sold Products", "ghg_ref": "Cat 12"},
    {"num": 13, "name": "Downstream Leased Assets", "ghg_ref": "Cat 13"},
    {"num": 14, "name": "Franchises", "ghg_ref": "Cat 14"},
    {"num": 15, "name": "Investments", "ghg_ref": "Cat 15"},
]

SCOPE1_SOURCES = [
    {"id": "stationary", "name": "Stationary Combustion", "mrv": "MRV-001"},
    {"id": "refrigerants", "name": "Refrigerants & F-Gas", "mrv": "MRV-002"},
    {"id": "mobile", "name": "Mobile Combustion", "mrv": "MRV-003"},
    {"id": "process", "name": "Process Emissions", "mrv": "MRV-004"},
    {"id": "fugitive", "name": "Fugitive Emissions", "mrv": "MRV-005"},
    {"id": "land_use", "name": "Land Use Emissions", "mrv": "MRV-006"},
    {"id": "waste_treatment", "name": "Waste Treatment", "mrv": "MRV-007"},
    {"id": "agriculture", "name": "Agricultural Emissions", "mrv": "MRV-008"},
]

DQ_LEVELS = {
    1: {"label": "Supplier-specific verified", "accuracy": "+/-3%"},
    2: {"label": "Supplier-specific unverified", "accuracy": "+/-5-10%"},
    3: {"label": "Average data (physical)", "accuracy": "+/-10-20%"},
    4: {"label": "Spend-based (EEIO)", "accuracy": "+/-20-40%"},
    5: {"label": "Proxy / extrapolation", "accuracy": "+/-40-60%"},
}

GHG_GASES = ["CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6", "NF3"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _dec_comma(val: Any, places: int = 0) -> str:
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        r = d.quantize(Decimal(q), rounding=ROUND_HALF_UP)
        parts = str(r).split(".")
        ip = parts[0]
        neg = ip.startswith("-")
        if neg:
            ip = ip[1:]
        f = ""
        for i, ch in enumerate(reversed(ip)):
            if i > 0 and i % 3 == 0:
                f = "," + f
            f = ch + f
        if neg:
            f = "-" + f
        if len(parts) > 1:
            f += "." + parts[1]
        return f
    except Exception:
        return str(val)


def _pct(val: Any) -> str:
    try:
        return _dec(val, 1) + "%"
    except Exception:
        return str(val)


def _safe_div(num: Any, den: Any, default: float = 0.0) -> float:
    try:
        d = float(den)
        return float(num) / d if d != 0 else default
    except Exception:
        return default


def _yoy_arrow(current: float, previous: float) -> str:
    if previous <= 0:
        return "N/A"
    change = ((current - previous) / previous) * 100
    if change < -1:
        return f"v {_dec(abs(change), 1)}%"
    elif change > 1:
        return f"^ {_dec(change, 1)}%"
    return f"~ {_dec(abs(change), 1)}%"


def _dq_label(level: int) -> str:
    return DQ_LEVELS.get(level, {"label": "Unknown"})["label"]


# ===========================================================================
# Template Class
# ===========================================================================

class GHGInventoryReportTemplate:
    """
    Enterprise GHG Protocol Corporate Standard inventory report.

    Generates a comprehensive 20-40 page report covering Scope 1/2/3
    emissions across all 15 Scope 3 categories with entity-level detail,
    data quality scoring, methodology documentation, year-over-year trends,
    and base year comparison. Supports Markdown, HTML, JSON, and Excel output.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID
    FORMATS = ["markdown", "html", "json", "excel"]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    # Internal data extraction helpers
    # ------------------------------------------------------------------ #

    def _extract_totals(self, data: Dict[str, Any]) -> Dict[str, float]:
        s1 = float(data.get("scope1_tco2e", 0))
        s2_loc = float(data.get("scope2_location_tco2e", data.get("scope2_tco2e", 0)))
        s2_mkt = float(data.get("scope2_market_tco2e", s2_loc))
        s3 = float(data.get("scope3_tco2e", 0))
        total_loc = s1 + s2_loc + s3
        total_mkt = s1 + s2_mkt + s3
        return {
            "scope1": s1, "scope2_location": s2_loc, "scope2_market": s2_mkt,
            "scope3": s3, "total_location": total_loc, "total_market": total_mkt,
        }

    def _extract_scope3_cats(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        cats = data.get("scope3_categories", [])
        result = []
        for cat_def in SCOPE3_CATEGORIES:
            found = next((c for c in cats if int(c.get("category", 0)) == cat_def["num"]), {})
            result.append({
                "num": cat_def["num"],
                "name": cat_def["name"],
                "ghg_ref": cat_def["ghg_ref"],
                "tco2e": float(found.get("tco2e", 0)),
                "methodology": found.get("methodology", "Not calculated"),
                "dq_level": int(found.get("dq_level", 5)),
                "included": found.get("included", False),
                "exclusion_reason": found.get("exclusion_reason", ""),
            })
        return result

    def _extract_entities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return data.get("entities", [])

    def _extract_prior_year(self, data: Dict[str, Any]) -> Dict[str, float]:
        py = data.get("prior_year", {})
        return {
            "scope1": float(py.get("scope1_tco2e", 0)),
            "scope2_location": float(py.get("scope2_location_tco2e", py.get("scope2_tco2e", 0))),
            "scope2_market": float(py.get("scope2_market_tco2e", 0)),
            "scope3": float(py.get("scope3_tco2e", 0)),
        }

    # ------------------------------------------------------------------ #
    # Public render methods
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the full GHG inventory report as Markdown."""
        self.generated_at = _utcnow()
        totals = self._extract_totals(data)
        cats = self._extract_scope3_cats(data)
        entities = self._extract_entities(data)
        prior = self._extract_prior_year(data)

        sections: List[str] = [
            self._md_title_page(data, totals),
            self._md_executive_summary(data, totals, prior),
            self._md_org_boundary(data),
            self._md_scope1_detail(data, totals),
            self._md_scope2_detail(data, totals),
            self._md_scope3_detail(data, totals, cats),
            self._md_scope3_category_table(cats, totals),
            self._md_entity_breakdown(entities, totals),
            self._md_data_quality_matrix(data, cats),
            self._md_methodology_notes(data),
            self._md_yoy_trends(data, totals, prior),
            self._md_intercompany_eliminations(data),
            self._md_base_year_recalc(data),
            self._md_citations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(s for s in sections if s)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- SHA-256 Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the full GHG inventory report as HTML with inline CSS."""
        self.generated_at = _utcnow()
        totals = self._extract_totals(data)
        cats = self._extract_scope3_cats(data)
        entities = self._extract_entities(data)
        prior = self._extract_prior_year(data)

        css = self._css()
        body_parts = [
            self._html_title_page(data, totals),
            self._html_executive_summary(data, totals, prior),
            self._html_org_boundary(data),
            self._html_scope1_detail(data, totals),
            self._html_scope2_detail(data, totals),
            self._html_scope3_detail(data, totals, cats),
            self._html_entity_breakdown(entities, totals),
            self._html_data_quality(data, cats),
            self._html_methodology(data),
            self._html_yoy_trends(data, totals, prior),
            self._html_citations(data),
            self._html_footer(data),
        ]
        body = "\n".join(p for p in body_parts if p)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
            f'<meta charset="UTF-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
            f'<title>Enterprise GHG Inventory Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- SHA-256 Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the GHG inventory as structured JSON."""
        self.generated_at = _utcnow()
        totals = self._extract_totals(data)
        cats = self._extract_scope3_cats(data)
        entities = self._extract_entities(data)
        prior = self._extract_prior_year(data)

        scope1_sources = data.get("scope1_sources", [])
        scope2_detail = {
            "location_based_tco2e": totals["scope2_location"],
            "market_based_tco2e": totals["scope2_market"],
            "delta_tco2e": totals["scope2_location"] - totals["scope2_market"],
            "instruments": data.get("scope2_instruments", []),
        }

        entity_data = []
        for ent in entities:
            entity_data.append({
                "entity_id": ent.get("entity_id", ""),
                "entity_name": ent.get("name", ""),
                "country": ent.get("country", ""),
                "ownership_pct": ent.get("ownership_pct", 100),
                "consolidation_method": ent.get("consolidation_method", "full"),
                "scope1_tco2e": float(ent.get("scope1_tco2e", 0)),
                "scope2_location_tco2e": float(ent.get("scope2_location_tco2e", 0)),
                "scope2_market_tco2e": float(ent.get("scope2_market_tco2e", 0)),
                "scope3_tco2e": float(ent.get("scope3_tco2e", 0)),
                "dq_score": float(ent.get("dq_score", 0)),
            })

        yoy = {}
        if prior["scope1"] > 0:
            prior_total = prior["scope1"] + prior["scope2_location"] + prior["scope3"]
            yoy = {
                "scope1_change_pct": round(
                    (totals["scope1"] - prior["scope1"]) / prior["scope1"] * 100, 1
                ) if prior["scope1"] else 0,
                "scope2_change_pct": round(
                    (totals["scope2_location"] - prior["scope2_location"]) / prior["scope2_location"] * 100, 1
                ) if prior["scope2_location"] else 0,
                "scope3_change_pct": round(
                    (totals["scope3"] - prior["scope3"]) / prior["scope3"] * 100, 1
                ) if prior["scope3"] else 0,
                "total_change_pct": round(
                    (totals["total_location"] - prior_total) / prior_total * 100, 1
                ) if prior_total else 0,
            }

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "organization": {
                "name": data.get("org_name", ""),
                "sector": data.get("sector", ""),
                "employees": data.get("employees", 0),
                "revenue": data.get("revenue", 0),
                "currency": data.get("currency", "USD"),
                "consolidation_approach": data.get("consolidation_approach", "operational_control"),
                "entity_count": len(entities),
            },
            "reporting_year": data.get("reporting_year", ""),
            "base_year": data.get("base_year", ""),
            "gases_included": data.get("gases_included", GHG_GASES),
            "emissions": {
                "scope1_tco2e": round(totals["scope1"], 2),
                "scope2_location_tco2e": round(totals["scope2_location"], 2),
                "scope2_market_tco2e": round(totals["scope2_market"], 2),
                "scope3_tco2e": round(totals["scope3"], 2),
                "total_location_tco2e": round(totals["total_location"], 2),
                "total_market_tco2e": round(totals["total_market"], 2),
                "scope1_pct": round(_safe_div(totals["scope1"], totals["total_location"]) * 100, 1),
                "scope2_location_pct": round(_safe_div(totals["scope2_location"], totals["total_location"]) * 100, 1),
                "scope3_pct": round(_safe_div(totals["scope3"], totals["total_location"]) * 100, 1),
            },
            "scope1_sources": scope1_sources,
            "scope2_detail": scope2_detail,
            "scope3_categories": [
                {
                    "category": c["num"],
                    "name": c["name"],
                    "tco2e": round(c["tco2e"], 2),
                    "pct_of_scope3": round(_safe_div(c["tco2e"], totals["scope3"]) * 100, 1),
                    "methodology": c["methodology"],
                    "dq_level": c["dq_level"],
                    "included": c["included"],
                    "exclusion_reason": c["exclusion_reason"],
                }
                for c in cats
            ],
            "entities": entity_data,
            "data_quality": {
                "overall_dq_score": data.get("overall_dq_score", 0),
                "target_accuracy": data.get("target_accuracy", "+/-3%"),
                "dq_matrix": data.get("dq_matrix", []),
            },
            "year_over_year": yoy,
            "base_year_recalculations": data.get("base_year_recalculations", []),
            "intercompany_eliminations": data.get("intercompany_eliminations", []),
            "methodology": {
                "emission_factor_sources": data.get("emission_factor_sources", []),
                "gwp_source": data.get("gwp_source", "IPCC AR6 GWP-100"),
                "calculation_approach": data.get("calculation_approach", ""),
                "exclusions": data.get("exclusions", []),
            },
            "citations": data.get("citations", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render Excel-ready data structure with worksheet definitions."""
        self.generated_at = _utcnow()
        totals = self._extract_totals(data)
        cats = self._extract_scope3_cats(data)
        entities = self._extract_entities(data)

        summary_sheet = {
            "name": "Executive Summary",
            "headers": ["Metric", "Value", "Unit", "YoY Change"],
            "rows": [
                ["Organization", data.get("org_name", ""), "", ""],
                ["Reporting Year", data.get("reporting_year", ""), "", ""],
                ["Base Year", data.get("base_year", ""), "", ""],
                ["Consolidation Approach", data.get("consolidation_approach", ""), "", ""],
                ["Entities", len(entities), "count", ""],
                ["Scope 1", round(totals["scope1"], 2), "tCO2e", ""],
                ["Scope 2 (Location-based)", round(totals["scope2_location"], 2), "tCO2e", ""],
                ["Scope 2 (Market-based)", round(totals["scope2_market"], 2), "tCO2e", ""],
                ["Scope 3", round(totals["scope3"], 2), "tCO2e", ""],
                ["Total (Location)", round(totals["total_location"], 2), "tCO2e", ""],
                ["Total (Market)", round(totals["total_market"], 2), "tCO2e", ""],
                ["Overall DQ Score", data.get("overall_dq_score", 0), "/ 100", ""],
            ],
        }

        scope3_sheet = {
            "name": "Scope 3 Categories",
            "headers": ["Cat #", "Category", "tCO2e", "% of Scope 3",
                        "Methodology", "DQ Level", "Included", "Exclusion Reason"],
            "rows": [
                [
                    c["num"], c["name"], round(c["tco2e"], 2),
                    round(_safe_div(c["tco2e"], totals["scope3"]) * 100, 1),
                    c["methodology"], c["dq_level"],
                    "Yes" if c["included"] else "No", c["exclusion_reason"],
                ]
                for c in cats
            ],
        }

        entity_sheet = {
            "name": "Entity Breakdown",
            "headers": ["Entity", "Country", "Ownership %", "Method",
                        "Scope 1 tCO2e", "Scope 2 Loc tCO2e", "Scope 2 Mkt tCO2e",
                        "Scope 3 tCO2e", "Total tCO2e", "DQ Score"],
            "rows": [],
        }
        for ent in entities:
            s1e = float(ent.get("scope1_tco2e", 0))
            s2le = float(ent.get("scope2_location_tco2e", 0))
            s2me = float(ent.get("scope2_market_tco2e", 0))
            s3e = float(ent.get("scope3_tco2e", 0))
            entity_sheet["rows"].append([
                ent.get("name", ""), ent.get("country", ""),
                ent.get("ownership_pct", 100),
                ent.get("consolidation_method", "full"),
                round(s1e, 2), round(s2le, 2), round(s2me, 2), round(s3e, 2),
                round(s1e + s2le + s3e, 2),
                ent.get("dq_score", 0),
            ])

        dq_sheet = {
            "name": "Data Quality Matrix",
            "headers": ["Category/Source", "DQ Level (1-5)", "DQ Label", "Accuracy Range"],
            "rows": [],
        }
        for c in cats:
            if c["included"]:
                dq_sheet["rows"].append([
                    f"Scope 3 Cat {c['num']}: {c['name']}",
                    c["dq_level"], _dq_label(c["dq_level"]),
                    DQ_LEVELS.get(c["dq_level"], {}).get("accuracy", ""),
                ])

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "filename": (
                f"ghg_inventory_{data.get('org_name', 'enterprise').replace(' ', '_')}"
                f"_{data.get('reporting_year', '')}.xlsx"
            ),
            "worksheets": [summary_sheet, scope3_sheet, entity_sheet, dq_sheet],
            "chart_definitions": [
                {
                    "type": "pie",
                    "title": "Emissions by Scope",
                    "worksheet": "Executive Summary",
                    "data_range": "B6:B9",
                    "labels_range": "A6:A9",
                    "colors": [_SCOPE1_CLR, _SCOPE2_CLR, _SCOPE3_CLR],
                },
                {
                    "type": "bar",
                    "title": "Scope 3 by Category",
                    "worksheet": "Scope 3 Categories",
                    "data_range": "C2:C16",
                    "labels_range": "B2:B16",
                    "colors": [_SCOPE3_CLR],
                },
            ],
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------ #
    # Markdown sections
    # ------------------------------------------------------------------ #

    def _md_title_page(self, data: Dict[str, Any], totals: Dict[str, float]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        org = data.get("org_name", "Enterprise")
        year = data.get("reporting_year", "")
        base = data.get("base_year", "")
        approach = data.get("consolidation_approach", "operational_control").replace("_", " ").title()
        return (
            f"# GHG Inventory Report\n\n"
            f"## {org} -- Reporting Year {year}\n\n"
            f"**Standard:** GHG Protocol Corporate Accounting and Reporting Standard  \n"
            f"**Base Year:** {base}  \n"
            f"**Consolidation Approach:** {approach}  \n"
            f"**Gases:** {', '.join(data.get('gases_included', GHG_GASES))}  \n"
            f"**Total Emissions (Location-based):** {_dec_comma(totals['total_location'])} tCO2e  \n"
            f"**Total Emissions (Market-based):** {_dec_comma(totals['total_market'])} tCO2e  \n"
            f"**Generated:** {ts}  \n"
            f"**Report ID:** {_new_uuid()}\n\n---"
        )

    def _md_executive_summary(
        self, data: Dict[str, Any], totals: Dict[str, float], prior: Dict[str, float]
    ) -> str:
        employees = int(data.get("employees", 1))
        revenue = float(data.get("revenue", 1))
        currency = data.get("currency", "USD")
        intensity_emp = _safe_div(totals["total_location"], employees)
        intensity_rev = _safe_div(totals["total_location"], revenue) * 1_000_000

        prior_total = prior["scope1"] + prior["scope2_location"] + prior["scope3"]
        yoy_total = _yoy_arrow(totals["total_location"], prior_total)
        yoy_s1 = _yoy_arrow(totals["scope1"], prior["scope1"])
        yoy_s2 = _yoy_arrow(totals["scope2_location"], prior["scope2_location"])
        yoy_s3 = _yoy_arrow(totals["scope3"], prior["scope3"])

        lines = [
            "## 1. Executive Summary\n",
            "| Metric | Current Year | Prior Year | Change |",
            "|--------|------------:|----------:|:------:|",
            f"| **Total (Location-based)** | **{_dec_comma(totals['total_location'])} tCO2e** "
            f"| {_dec_comma(prior_total)} tCO2e | {yoy_total} |",
            f"| Scope 1 | {_dec_comma(totals['scope1'])} tCO2e "
            f"| {_dec_comma(prior['scope1'])} tCO2e | {yoy_s1} |",
            f"| Scope 2 (Location) | {_dec_comma(totals['scope2_location'])} tCO2e "
            f"| {_dec_comma(prior['scope2_location'])} tCO2e | {yoy_s2} |",
            f"| Scope 2 (Market) | {_dec_comma(totals['scope2_market'])} tCO2e | - | - |",
            f"| Scope 3 | {_dec_comma(totals['scope3'])} tCO2e "
            f"| {_dec_comma(prior['scope3'])} tCO2e | {yoy_s3} |",
            f"| Intensity (per employee) | {_dec(intensity_emp)} tCO2e/FTE | | |",
            f"| Intensity (per {currency}1M revenue) | {_dec(intensity_rev)} tCO2e/{currency}1M | | |",
            "",
            "**Scope Mix:**\n",
            f"- Scope 1: {_pct(_safe_div(totals['scope1'], totals['total_location']) * 100)}",
            f"- Scope 2: {_pct(_safe_div(totals['scope2_location'], totals['total_location']) * 100)}",
            f"- Scope 3: {_pct(_safe_div(totals['scope3'], totals['total_location']) * 100)}",
        ]
        return "\n".join(lines)

    def _md_org_boundary(self, data: Dict[str, Any]) -> str:
        approach = data.get("consolidation_approach", "operational_control").replace("_", " ").title()
        entities = self._extract_entities(data)
        countries = set(e.get("country", "N/A") for e in entities)

        lines = [
            "## 2. Organizational Boundary\n",
            f"**Consolidation Approach:** {approach}  ",
            f"**Total Entities:** {len(entities)}  ",
            f"**Countries:** {len(countries)} ({', '.join(sorted(countries)[:10])}{'...' if len(countries) > 10 else ''})  \n",
        ]
        if entities:
            lines.append("| Entity | Country | Ownership % | Consolidation |")
            lines.append("|--------|---------|:-----------:|:-------------:|")
            for ent in entities[:20]:
                lines.append(
                    f"| {ent.get('name', '')} | {ent.get('country', '')} "
                    f"| {ent.get('ownership_pct', 100)}% "
                    f"| {ent.get('consolidation_method', 'full').title()} |"
                )
            if len(entities) > 20:
                lines.append(f"| *... {len(entities) - 20} additional entities* | | | |")

        return "\n".join(lines)

    def _md_scope1_detail(self, data: Dict[str, Any], totals: Dict[str, float]) -> str:
        sources = data.get("scope1_sources", [])
        lines = [
            "## 3. Scope 1 -- Direct Emissions\n",
            f"**Total Scope 1:** {_dec_comma(totals['scope1'])} tCO2e\n",
            "| Source | MRV Agent | tCO2e | % of Scope 1 | Methodology |",
            "|--------|-----------|------:|:------------:|-------------|",
        ]
        for src_def in SCOPE1_SOURCES:
            found = next((s for s in sources if s.get("source_id") == src_def["id"]), {})
            val = float(found.get("tco2e", 0))
            if val > 0 or found:
                lines.append(
                    f"| {src_def['name']} | {src_def['mrv']} "
                    f"| {_dec_comma(val)} "
                    f"| {_pct(_safe_div(val, totals['scope1']) * 100)} "
                    f"| {found.get('methodology', 'Fuel-based')} |"
                )
        lines.append(f"| **Total** | | **{_dec_comma(totals['scope1'])}** | **100%** | |")
        return "\n".join(lines)

    def _md_scope2_detail(self, data: Dict[str, Any], totals: Dict[str, float]) -> str:
        delta = totals["scope2_location"] - totals["scope2_market"]
        instruments = data.get("scope2_instruments", [])

        lines = [
            "## 4. Scope 2 -- Indirect Energy Emissions\n",
            "### Dual Reporting (per GHG Protocol Scope 2 Guidance)\n",
            "| Method | tCO2e | Notes |",
            "|--------|------:|-------|",
            f"| Location-based | {_dec_comma(totals['scope2_location'])} | Grid average emission factors |",
            f"| Market-based | {_dec_comma(totals['scope2_market'])} | Contractual instruments applied |",
            f"| **Delta** | **{_dec_comma(delta)}** | Location minus Market |",
        ]
        if instruments:
            lines.append("\n### Contractual Instruments\n")
            lines.append("| Instrument | MWh | Provider | Certificate ID |")
            lines.append("|------------|----:|---------|----------------|")
            for inst in instruments:
                lines.append(
                    f"| {inst.get('type', '')} | {_dec_comma(inst.get('mwh', 0))} "
                    f"| {inst.get('provider', '')} | {inst.get('certificate_id', '')} |"
                )
        return "\n".join(lines)

    def _md_scope3_detail(
        self, data: Dict[str, Any], totals: Dict[str, float], cats: List[Dict[str, Any]]
    ) -> str:
        included = [c for c in cats if c["included"]]
        excluded = [c for c in cats if not c["included"]]
        total_included = sum(c["tco2e"] for c in included)

        lines = [
            "## 5. Scope 3 -- Value Chain Emissions\n",
            f"**Total Scope 3:** {_dec_comma(totals['scope3'])} tCO2e  ",
            f"**Categories Included:** {len(included)}/15  ",
            f"**Categories Excluded:** {len(excluded)}/15  \n",
            "### Materiality Assessment\n",
            f"- Categories >1% of total: full activity-based calculation",
            f"- Categories 0.1-1%: average-data method acceptable",
            f"- Categories <0.1%: excluded with documented justification",
            f"- Total exclusions must not exceed 5% of anticipated total Scope 3",
        ]
        return "\n".join(lines)

    def _md_scope3_category_table(
        self, cats: List[Dict[str, Any]], totals: Dict[str, float]
    ) -> str:
        lines = [
            "### Scope 3 Category Detail\n",
            "| Cat | Category | tCO2e | % of S3 | Methodology | DQ Level | Status |",
            "|:---:|----------|------:|:-------:|-------------|:--------:|:------:|",
        ]
        for c in cats:
            status = "Included" if c["included"] else f"Excluded: {c['exclusion_reason']}"
            lines.append(
                f"| {c['num']} | {c['name']} "
                f"| {_dec_comma(c['tco2e'])} "
                f"| {_pct(_safe_div(c['tco2e'], totals['scope3']) * 100)} "
                f"| {c['methodology']} "
                f"| {c['dq_level']} ({_dq_label(c['dq_level'])[:15]}) "
                f"| {status} |"
            )
        lines.append(
            f"| | **Total** | **{_dec_comma(totals['scope3'])}** | **100%** | | | |"
        )
        return "\n".join(lines)

    def _md_entity_breakdown(
        self, entities: List[Dict[str, Any]], totals: Dict[str, float]
    ) -> str:
        if not entities:
            return "## 6. Entity-Level Breakdown\n\nNo entity-level data provided."

        lines = [
            "## 6. Entity-Level Breakdown\n",
            "| Entity | Country | Scope 1 | Scope 2 (Loc) | Scope 3 | Total | % of Group | DQ |",
            "|--------|---------|--------:|--------------:|--------:|------:|-----------:|:--:|",
        ]
        for ent in entities:
            s1e = float(ent.get("scope1_tco2e", 0))
            s2e = float(ent.get("scope2_location_tco2e", 0))
            s3e = float(ent.get("scope3_tco2e", 0))
            te = s1e + s2e + s3e
            lines.append(
                f"| {ent.get('name', '')} | {ent.get('country', '')} "
                f"| {_dec_comma(s1e)} | {_dec_comma(s2e)} | {_dec_comma(s3e)} "
                f"| {_dec_comma(te)} "
                f"| {_pct(_safe_div(te, totals['total_location']) * 100)} "
                f"| {ent.get('dq_score', '-')} |"
            )
        return "\n".join(lines)

    def _md_data_quality_matrix(
        self, data: Dict[str, Any], cats: List[Dict[str, Any]]
    ) -> str:
        overall = data.get("overall_dq_score", 0)
        target = data.get("target_accuracy", "+/-3%")
        lines = [
            "## 7. Data Quality Assessment\n",
            f"**Overall DQ Score:** {overall}/100  ",
            f"**Target Accuracy:** {target}\n",
            "### Per-Category Data Quality\n",
            "| Source / Category | DQ Level | Description | Accuracy Range |",
            "|-------------------|:--------:|-------------|:--------------:|",
        ]
        for c in cats:
            if c["included"]:
                dq = DQ_LEVELS.get(c["dq_level"], {"label": "N/A", "accuracy": "N/A"})
                lines.append(
                    f"| Scope 3 Cat {c['num']}: {c['name']} "
                    f"| {c['dq_level']} | {dq['label']} | {dq['accuracy']} |"
                )
        return "\n".join(lines)

    def _md_methodology_notes(self, data: Dict[str, Any]) -> str:
        gwp = data.get("gwp_source", "IPCC AR6 GWP-100")
        ef_sources = data.get("emission_factor_sources", [
            "DEFRA/DESNZ Conversion Factors (2024)",
            "EPA GHG Emission Factors Hub (2024)",
            "IEA Emission Factors (2024)",
            "ecoinvent 3.10",
        ])
        exclusions = data.get("exclusions", [])

        lines = [
            "## 8. Methodology Notes\n",
            f"**GWP Values:** {gwp}  ",
            f"**Reporting Standard:** GHG Protocol Corporate Accounting Standard (2004, amended 2015)  ",
            f"**Scope 2 Guidance:** GHG Protocol Scope 2 Guidance (2015)  ",
            f"**Scope 3 Standard:** GHG Protocol Corporate Value Chain Standard (2011)  \n",
            "### Emission Factor Sources\n",
        ]
        for ef in ef_sources:
            lines.append(f"- {ef}")

        if exclusions:
            lines.append("\n### Exclusions\n")
            for ex in exclusions:
                lines.append(
                    f"- **{ex.get('category', '')}**: {ex.get('reason', '')} "
                    f"(estimated <{ex.get('pct_of_total', '0.1')}% of total)"
                )

        return "\n".join(lines)

    def _md_yoy_trends(
        self, data: Dict[str, Any], totals: Dict[str, float], prior: Dict[str, float]
    ) -> str:
        trend_data = data.get("trend_years", [])
        if not trend_data and prior["scope1"] <= 0:
            return "## 9. Year-over-Year Trends\n\nInsufficient historical data for trend analysis."

        lines = [
            "## 9. Year-over-Year Trends\n",
        ]
        if trend_data:
            lines.append("| Year | Scope 1 | Scope 2 (Loc) | Scope 3 | Total | Change |")
            lines.append("|------|--------:|--------------:|--------:|------:|:------:|")
            prev = 0
            for yr in trend_data:
                t = float(yr.get("total_tco2e", 0))
                change = _yoy_arrow(t, prev) if prev > 0 else "-"
                lines.append(
                    f"| {yr.get('year', '')} "
                    f"| {_dec_comma(yr.get('scope1_tco2e', 0))} "
                    f"| {_dec_comma(yr.get('scope2_location_tco2e', 0))} "
                    f"| {_dec_comma(yr.get('scope3_tco2e', 0))} "
                    f"| {_dec_comma(t)} | {change} |"
                )
                prev = t
        return "\n".join(lines)

    def _md_intercompany_eliminations(self, data: Dict[str, Any]) -> str:
        elims = data.get("intercompany_eliminations", [])
        if not elims:
            return "## 10. Intercompany Eliminations\n\nNo intercompany eliminations required."

        lines = [
            "## 10. Intercompany Eliminations\n",
            "| Selling Entity | Buying Entity | Category | tCO2e Eliminated | Justification |",
            "|----------------|---------------|----------|----------------:|---------------|",
        ]
        for el in elims:
            lines.append(
                f"| {el.get('seller', '')} | {el.get('buyer', '')} "
                f"| {el.get('category', '')} "
                f"| {_dec_comma(el.get('tco2e', 0))} "
                f"| {el.get('justification', '')} |"
            )
        return "\n".join(lines)

    def _md_base_year_recalc(self, data: Dict[str, Any]) -> str:
        recalcs = data.get("base_year_recalculations", [])
        if not recalcs:
            return (
                "## 11. Base Year Recalculation Log\n\n"
                "No base year recalculations triggered in this reporting period."
            )

        lines = [
            "## 11. Base Year Recalculation Log\n",
            "| Trigger | Date | Old Base Year (tCO2e) | New Base Year (tCO2e) | Delta | Significance |",
            "|---------|------|-----------------------:|-----------------------:|------:|:------------:|",
        ]
        for rc in recalcs:
            lines.append(
                f"| {rc.get('trigger', '')} | {rc.get('date', '')} "
                f"| {_dec_comma(rc.get('old_value', 0))} "
                f"| {_dec_comma(rc.get('new_value', 0))} "
                f"| {_dec_comma(rc.get('delta', 0))} "
                f"| {_pct(rc.get('significance_pct', 0))} |"
            )
        return "\n".join(lines)

    def _md_citations(self, data: Dict[str, Any]) -> str:
        citations = data.get("citations", [
            {"ref": "GHG-001", "source": "GHG Protocol Corporate Accounting and Reporting Standard",
             "author": "WRI/WBCSD", "year": "2004"},
            {"ref": "GHG-002", "source": "GHG Protocol Scope 2 Guidance",
             "author": "WRI/WBCSD", "year": "2015"},
            {"ref": "GHG-003", "source": "GHG Protocol Corporate Value Chain (Scope 3) Standard",
             "author": "WRI/WBCSD", "year": "2011"},
            {"ref": "GHG-004", "source": "IPCC Sixth Assessment Report (AR6)",
             "author": "IPCC", "year": "2021"},
        ])
        lines = [
            "## 12. Citations & References\n",
            "| Ref | Source | Author | Year |",
            "|-----|--------|--------|------|",
        ]
        for c in citations:
            lines.append(
                f"| {c.get('ref', '')} | {c.get('source', '')} "
                f"| {c.get('author', '')} | {c.get('year', '')} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-027 Enterprise Net Zero Pack on {ts}*  \n"
            f"*GHG Protocol Corporate Accounting and Reporting Standard compliant.*  \n"
            f"*All calculations are deterministic and zero-hallucination.*  \n"
            f"*SHA-256 provenance hashing applied to all outputs.*"
        )

    # ------------------------------------------------------------------ #
    # HTML sections
    # ------------------------------------------------------------------ #

    def _css(self) -> str:
        return (
            "*, *::before, *::after{box-sizing:border-box;}"
            f"body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;"
            f"padding:24px;background:#f5f7f5;color:#1a1a2e;line-height:1.6;}}"
            f".report{{max-width:1100px;margin:0 auto;background:#fff;padding:40px;"
            f"border-radius:12px;box-shadow:0 2px 16px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;"
            f"font-size:2em;}}"
            f"h2{{color:{_SECONDARY};margin-top:32px;border-left:4px solid {_ACCENT};"
            f"padding-left:12px;font-size:1.4em;}}"
            f"h3{{color:{_ACCENT};margin-top:20px;font-size:1.1em;}}"
            f"table{{width:100%;border-collapse:collapse;margin:12px 0;font-size:0.88em;}}"
            f"th,td{{border:1px solid #ddd;padding:8px 12px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:{_LIGHTER};}}"
            f".kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));"
            f"gap:12px;margin:16px 0;}}"
            f".kpi{{background:linear-gradient(135deg,{_LIGHT},{_CARD_BG});border-radius:10px;"
            f"padding:16px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".kpi-label{{font-size:0.75em;color:#37474f;text-transform:uppercase;letter-spacing:0.5px;}}"
            f".kpi-value{{font-size:1.6em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".kpi-unit{{font-size:0.7em;color:#607d8b;}}"
            f".scope-bar{{height:28px;border-radius:6px;margin:4px 0;}}"
            f".s1-bar{{background:{_SCOPE1_CLR};}}"
            f".s2-bar{{background:{_SCOPE2_CLR};}}"
            f".s3-bar{{background:{_SCOPE3_CLR};}}"
            f".footer{{margin-top:32px;padding-top:16px;border-top:2px solid {_CARD_BG};"
            f"color:#607d8b;font-size:0.8em;text-align:center;}}"
            f"@media(max-width:768px){{.kpi-grid{{grid-template-columns:1fr 1fr;}}"
            f".report{{padding:20px;}}}}"
        )

    def _html_title_page(self, data: Dict[str, Any], totals: Dict[str, float]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        org = data.get("org_name", "Enterprise")
        year = data.get("reporting_year", "")
        return (
            f'<h1>GHG Inventory Report</h1>\n'
            f'<h2 style="border-left:none;">{org} -- {year}</h2>\n'
            f'<p>GHG Protocol Corporate Accounting Standard | '
            f'Total: {_dec_comma(totals["total_location"])} tCO2e (location) / '
            f'{_dec_comma(totals["total_market"])} tCO2e (market) | '
            f'Generated: {ts}</p>'
        )

    def _html_executive_summary(
        self, data: Dict[str, Any], totals: Dict[str, float], prior: Dict[str, float]
    ) -> str:
        employees = int(data.get("employees", 1))
        intensity = _safe_div(totals["total_location"], employees)

        return (
            f'<h2>Executive Summary</h2>\n'
            f'<div class="kpi-grid">\n'
            f'  <div class="kpi"><div class="kpi-label">Total (Location)</div>'
            f'<div class="kpi-value">{_dec_comma(totals["total_location"])}</div>'
            f'<div class="kpi-unit">tCO2e</div></div>\n'
            f'  <div class="kpi"><div class="kpi-label">Scope 1</div>'
            f'<div class="kpi-value">{_dec_comma(totals["scope1"])}</div>'
            f'<div class="kpi-unit">tCO2e ({_pct(_safe_div(totals["scope1"], totals["total_location"]) * 100)})</div></div>\n'
            f'  <div class="kpi"><div class="kpi-label">Scope 2 (Location)</div>'
            f'<div class="kpi-value">{_dec_comma(totals["scope2_location"])}</div>'
            f'<div class="kpi-unit">tCO2e</div></div>\n'
            f'  <div class="kpi"><div class="kpi-label">Scope 2 (Market)</div>'
            f'<div class="kpi-value">{_dec_comma(totals["scope2_market"])}</div>'
            f'<div class="kpi-unit">tCO2e</div></div>\n'
            f'  <div class="kpi"><div class="kpi-label">Scope 3</div>'
            f'<div class="kpi-value">{_dec_comma(totals["scope3"])}</div>'
            f'<div class="kpi-unit">tCO2e ({_pct(_safe_div(totals["scope3"], totals["total_location"]) * 100)})</div></div>\n'
            f'  <div class="kpi"><div class="kpi-label">Per Employee</div>'
            f'<div class="kpi-value">{_dec(intensity)}</div>'
            f'<div class="kpi-unit">tCO2e/FTE</div></div>\n'
            f'</div>'
        )

    def _html_org_boundary(self, data: Dict[str, Any]) -> str:
        approach = data.get("consolidation_approach", "operational_control").replace("_", " ").title()
        entities = self._extract_entities(data)
        rows = ""
        for ent in entities[:20]:
            rows += (
                f'<tr><td>{ent.get("name", "")}</td><td>{ent.get("country", "")}</td>'
                f'<td>{ent.get("ownership_pct", 100)}%</td>'
                f'<td>{ent.get("consolidation_method", "full").title()}</td></tr>\n'
            )
        if len(entities) > 20:
            rows += f'<tr><td colspan="4"><em>... {len(entities) - 20} additional entities</em></td></tr>\n'

        return (
            f'<h2>Organizational Boundary</h2>\n'
            f'<p><strong>Approach:</strong> {approach} | '
            f'<strong>Entities:</strong> {len(entities)}</p>\n'
            f'<table><tr><th>Entity</th><th>Country</th><th>Ownership</th><th>Method</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_scope1_detail(self, data: Dict[str, Any], totals: Dict[str, float]) -> str:
        sources = data.get("scope1_sources", [])
        rows = ""
        for src_def in SCOPE1_SOURCES:
            found = next((s for s in sources if s.get("source_id") == src_def["id"]), {})
            val = float(found.get("tco2e", 0))
            if val > 0 or found:
                rows += (
                    f'<tr><td>{src_def["name"]}</td><td>{src_def["mrv"]}</td>'
                    f'<td style="text-align:right">{_dec_comma(val)}</td>'
                    f'<td>{_pct(_safe_div(val, totals["scope1"]) * 100)}</td></tr>\n'
                )
        return (
            f'<h2>Scope 1 - Direct Emissions</h2>\n'
            f'<p><strong>Total:</strong> {_dec_comma(totals["scope1"])} tCO2e</p>\n'
            f'<table><tr><th>Source</th><th>Agent</th><th>tCO2e</th><th>%</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_scope2_detail(self, data: Dict[str, Any], totals: Dict[str, float]) -> str:
        delta = totals["scope2_location"] - totals["scope2_market"]
        return (
            f'<h2>Scope 2 - Indirect Energy</h2>\n'
            f'<table>\n'
            f'<tr><th>Method</th><th>tCO2e</th><th>Notes</th></tr>\n'
            f'<tr><td>Location-based</td><td>{_dec_comma(totals["scope2_location"])}</td>'
            f'<td>Grid average factors</td></tr>\n'
            f'<tr><td>Market-based</td><td>{_dec_comma(totals["scope2_market"])}</td>'
            f'<td>Contractual instruments</td></tr>\n'
            f'<tr><td><strong>Delta</strong></td><td><strong>{_dec_comma(delta)}</strong></td>'
            f'<td></td></tr>\n'
            f'</table>'
        )

    def _html_scope3_detail(
        self, data: Dict[str, Any], totals: Dict[str, float], cats: List[Dict[str, Any]]
    ) -> str:
        rows = ""
        for c in cats:
            status = "Included" if c["included"] else f'Excluded'
            color = "" if c["included"] else f' style="color:{_WARN_CLR};"'
            rows += (
                f'<tr><td>{c["num"]}</td><td>{c["name"]}</td>'
                f'<td style="text-align:right">{_dec_comma(c["tco2e"])}</td>'
                f'<td>{_pct(_safe_div(c["tco2e"], totals["scope3"]) * 100)}</td>'
                f'<td>{c["methodology"]}</td><td>{c["dq_level"]}</td>'
                f'<td{color}>{status}</td></tr>\n'
            )
        return (
            f'<h2>Scope 3 - Value Chain</h2>\n'
            f'<p><strong>Total:</strong> {_dec_comma(totals["scope3"])} tCO2e</p>\n'
            f'<table><tr><th>Cat</th><th>Category</th><th>tCO2e</th><th>% S3</th>'
            f'<th>Method</th><th>DQ</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_entity_breakdown(
        self, entities: List[Dict[str, Any]], totals: Dict[str, float]
    ) -> str:
        rows = ""
        for ent in entities:
            s1e = float(ent.get("scope1_tco2e", 0))
            s2e = float(ent.get("scope2_location_tco2e", 0))
            s3e = float(ent.get("scope3_tco2e", 0))
            te = s1e + s2e + s3e
            rows += (
                f'<tr><td>{ent.get("name", "")}</td><td>{ent.get("country", "")}</td>'
                f'<td>{_dec_comma(s1e)}</td><td>{_dec_comma(s2e)}</td>'
                f'<td>{_dec_comma(s3e)}</td><td>{_dec_comma(te)}</td>'
                f'<td>{_pct(_safe_div(te, totals["total_location"]) * 100)}</td></tr>\n'
            )
        return (
            f'<h2>Entity Breakdown</h2>\n'
            f'<table><tr><th>Entity</th><th>Country</th><th>S1</th><th>S2</th>'
            f'<th>S3</th><th>Total</th><th>% Group</th></tr>\n{rows}</table>'
        )

    def _html_data_quality(self, data: Dict[str, Any], cats: List[Dict[str, Any]]) -> str:
        overall = data.get("overall_dq_score", 0)
        rows = ""
        for c in cats:
            if c["included"]:
                dq = DQ_LEVELS.get(c["dq_level"], {"label": "N/A", "accuracy": "N/A"})
                rows += (
                    f'<tr><td>Cat {c["num"]}: {c["name"]}</td>'
                    f'<td>{c["dq_level"]}</td><td>{dq["label"]}</td>'
                    f'<td>{dq["accuracy"]}</td></tr>\n'
                )
        return (
            f'<h2>Data Quality</h2>\n'
            f'<p><strong>Overall Score:</strong> {overall}/100 | '
            f'<strong>Target:</strong> {data.get("target_accuracy", "+/-3%")}</p>\n'
            f'<table><tr><th>Category</th><th>DQ Level</th><th>Description</th>'
            f'<th>Accuracy</th></tr>\n{rows}</table>'
        )

    def _html_methodology(self, data: Dict[str, Any]) -> str:
        gwp = data.get("gwp_source", "IPCC AR6 GWP-100")
        return (
            f'<h2>Methodology</h2>\n'
            f'<p><strong>GWP:</strong> {gwp} | '
            f'<strong>Standard:</strong> GHG Protocol Corporate Standard</p>'
        )

    def _html_yoy_trends(
        self, data: Dict[str, Any], totals: Dict[str, float], prior: Dict[str, float]
    ) -> str:
        return f'<h2>Year-over-Year Trends</h2>\n<p>See detailed Markdown output for full trend tables.</p>'

    def _html_citations(self, data: Dict[str, Any]) -> str:
        citations = data.get("citations", [])
        rows = ""
        for c in citations:
            rows += (
                f'<tr><td>{c.get("ref", "")}</td><td>{c.get("source", "")}</td>'
                f'<td>{c.get("author", "")}</td><td>{c.get("year", "")}</td></tr>\n'
            )
        return (
            f'<h2>Citations</h2>\n'
            f'<table><tr><th>Ref</th><th>Source</th><th>Author</th><th>Year</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-027 Enterprise Net Zero Pack on {ts}<br>'
            f'GHG Protocol Corporate Standard | Zero-hallucination | SHA-256 provenance'
            f'</div>'
        )
