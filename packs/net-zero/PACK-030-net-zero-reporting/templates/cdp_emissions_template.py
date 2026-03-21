# -*- coding: utf-8 -*-
"""
CDPEmissionsTemplate - CDP C4-C7 Emissions Template for PACK-030.

Renders CDP Climate Change questionnaire modules C4 (Targets), C5
(Emissions Methodology), C6 (Scope 1 & 2 Emissions), and C7 (Scope 3
Emissions) with structured data tables, methodology disclosures,
emission factor documentation, and completeness scoring. Multi-format
output (MD, HTML, JSON, PDF) with SHA-256 provenance hashing.

Sections:
    1.  Executive Summary
    2.  C4 - Targets & Performance
    3.  C4.1 - Emissions Reduction Targets
    4.  C4.2 - Target Progress Table
    5.  C5 - Emissions Methodology
    6.  C5.1 - Consolidation Approach & Standards
    7.  C6 - Scope 1 & Scope 2 Emissions
    8.  C6.1 - Scope 1 Breakdown by GHG
    9.  C6.3 - Scope 2 Location vs Market
    10. C7 - Scope 3 Emissions by Category
    11. C7.1 - Scope 3 Category Details
    12. Completeness Scoring
    13. XBRL Tagging Summary
    14. Audit Trail & Provenance

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
_TEMPLATE_ID = "cdp_emissions"

_PRIMARY = "#0d3b66"
_SECONDARY = "#1a6b8a"
_ACCENT = "#28a745"
_LIGHT = "#e3f0f7"
_LIGHTER = "#f4f9fc"
_SUCCESS = "#2e7d32"
_WARN = "#ef6c00"
_DANGER = "#c62828"

SCOPE3_CATEGORIES = [
    {"cat": 1, "name": "Purchased goods & services", "direction": "Upstream"},
    {"cat": 2, "name": "Capital goods", "direction": "Upstream"},
    {"cat": 3, "name": "Fuel- and energy-related activities", "direction": "Upstream"},
    {"cat": 4, "name": "Upstream transportation & distribution", "direction": "Upstream"},
    {"cat": 5, "name": "Waste generated in operations", "direction": "Upstream"},
    {"cat": 6, "name": "Business travel", "direction": "Upstream"},
    {"cat": 7, "name": "Employee commuting", "direction": "Upstream"},
    {"cat": 8, "name": "Upstream leased assets", "direction": "Upstream"},
    {"cat": 9, "name": "Downstream transportation & distribution", "direction": "Downstream"},
    {"cat": 10, "name": "Processing of sold products", "direction": "Downstream"},
    {"cat": 11, "name": "Use of sold products", "direction": "Downstream"},
    {"cat": 12, "name": "End-of-life treatment of sold products", "direction": "Downstream"},
    {"cat": 13, "name": "Downstream leased assets", "direction": "Downstream"},
    {"cat": 14, "name": "Franchises", "direction": "Downstream"},
    {"cat": 15, "name": "Investments", "direction": "Downstream"},
]

GHG_GASES = ["CO2", "CH4", "N2O", "HFCs", "PFCs", "SF6", "NF3"]

XBRL_TAGS: Dict[str, str] = {
    "scope1_total": "gl:CDPScope1Total",
    "scope2_location": "gl:CDPScope2LocationBased",
    "scope2_market": "gl:CDPScope2MarketBased",
    "scope3_total": "gl:CDPScope3Total",
    "total_emissions": "gl:CDPTotalEmissions",
    "emissions_intensity": "gl:CDPEmissionsIntensity",
    "target_reduction_pct": "gl:CDPTargetReductionPct",
    "methodology": "gl:CDPEmissionsMethodology",
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

def _pct_of_total(part: float, total: float) -> str:
    if total == 0:
        return "0.00"
    return _dec(part / total * 100)


class CDPEmissionsTemplate:
    """
    CDP C4-C7 Emissions template for PACK-030 Net Zero Reporting Pack.

    Generates CDP questionnaire responses for emissions modules covering
    targets (C4), methodology (C5), Scope 1/2 (C6), and Scope 3 (C7).
    Supports MD, HTML, JSON, PDF.

    Example:
        >>> template = CDPEmissionsTemplate()
        >>> data = {
        ...     "org_name": "GreenCorp", "cdp_year": 2025,
        ...     "scope1": {"total": 25000, "by_ghg": {"CO2": 23000, "CH4": 1500, "N2O": 500}},
        ...     "scope2": {"location": 18000, "market": 12000},
        ...     "scope3": {"categories": {1: 45000, 6: 2000, 7: 3000}},
        ...     "targets": [{"scope": "1+2", "base_year": 2020, "target_year": 2030, "reduction_pct": 46.2}],
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render CDP emissions report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_c4_targets(data), self._md_c4_1_targets_detail(data),
            self._md_c4_2_progress(data), self._md_c5_methodology(data),
            self._md_c5_1_standards(data), self._md_c6_scope12(data),
            self._md_c6_1_ghg_breakdown(data), self._md_c6_3_scope2(data),
            self._md_c7_scope3(data), self._md_c7_1_categories(data),
            self._md_completeness(data), self._md_xbrl(data),
            self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render CDP emissions report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_executive_summary(data),
            self._html_c4_targets(data), self._html_c6_scope12(data),
            self._html_c6_1_ghg_breakdown(data), self._html_c6_3_scope2(data),
            self._html_c7_scope3(data), self._html_completeness(data),
            self._html_xbrl(data), self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>CDP Emissions - {data.get("org_name", "")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render as structured JSON."""
        self.generated_at = _utcnow()
        s1 = data.get("scope1", {})
        s2 = data.get("scope2", {})
        s3 = data.get("scope3", {})
        s1_total = float(s1.get("total", 0))
        s2_loc = float(s2.get("location", 0))
        s2_mkt = float(s2.get("market", 0))
        s3_cats = s3.get("categories", {})
        s3_total = sum(float(v) for v in s3_cats.values())
        total_loc = s1_total + s2_loc + s3_total
        total_mkt = s1_total + s2_mkt + s3_total

        completeness = self._calculate_completeness(data)

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION,
            "pack_id": _PACK_ID, "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(), "org_name": data.get("org_name", ""),
            "cdp_year": data.get("cdp_year", ""),
            "scope1": {"total": str(s1_total), "by_ghg": s1.get("by_ghg", {})},
            "scope2": {"location_based": str(s2_loc), "market_based": str(s2_mkt)},
            "scope3": {
                "total": str(s3_total),
                "categories_reported": len(s3_cats),
                "categories": {str(k): str(v) for k, v in s3_cats.items()},
            },
            "totals": {
                "location_based": str(total_loc),
                "market_based": str(total_mkt),
            },
            "targets": data.get("targets", []),
            "completeness": completeness,
            "methodology": data.get("methodology", {}),
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return PDF-ready data."""
        return {
            "format": "pdf", "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {"title": f"CDP Emissions - {data.get('org_name', '')}", "author": "GreenLang PACK-030", "framework": "CDP"},
        }

    def _calculate_completeness(self, data: Dict[str, Any]) -> Dict[str, Any]:
        checks = {
            "scope1_total": data.get("scope1", {}).get("total", 0) > 0,
            "scope1_ghg_breakdown": len(data.get("scope1", {}).get("by_ghg", {})) > 0,
            "scope2_location": data.get("scope2", {}).get("location", 0) > 0,
            "scope2_market": data.get("scope2", {}).get("market", 0) > 0,
            "scope3_reported": len(data.get("scope3", {}).get("categories", {})) > 0,
            "targets_set": len(data.get("targets", [])) > 0,
            "methodology_disclosed": bool(data.get("methodology")),
            "emission_factors_documented": bool(data.get("emission_factors")),
        }
        passed = sum(1 for v in checks.values() if v)
        total = len(checks)
        return {"checks": checks, "passed": passed, "total": total, "score": round(passed / total * 100, 1)}

    # -- Markdown sections --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# CDP Climate Change - Emissions Report (C4-C7)\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**CDP Year:** {data.get('cdp_year', '')}  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-030 v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1", {}).get("total", 0))
        s2_loc = float(data.get("scope2", {}).get("location", 0))
        s2_mkt = float(data.get("scope2", {}).get("market", 0))
        s3_cats = data.get("scope3", {}).get("categories", {})
        s3 = sum(float(v) for v in s3_cats.values())
        total = s1 + s2_loc + s3
        lines = [
            "## 1. Executive Summary\n",
            "| Metric | Value |", "|--------|-------|",
            f"| Scope 1 | {_dec_comma(s1, 0)} tCO2e |",
            f"| Scope 2 (Location) | {_dec_comma(s2_loc, 0)} tCO2e |",
            f"| Scope 2 (Market) | {_dec_comma(s2_mkt, 0)} tCO2e |",
            f"| Scope 3 | {_dec_comma(s3, 0)} tCO2e |",
            f"| Total (Location) | {_dec_comma(total, 0)} tCO2e |",
            f"| Scope 3 Categories Reported | {len(s3_cats)} of 15 |",
            f"| Targets Set | {len(data.get('targets', []))} |",
        ]
        return "\n".join(lines)

    def _md_c4_targets(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        lines = [
            "## 2. C4 - Targets & Performance\n",
            f"**Number of Active Targets:** {len(targets)}\n",
        ]
        return "\n".join(lines)

    def _md_c4_1_targets_detail(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        lines = [
            "## 3. C4.1 - Emissions Reduction Targets\n",
            "| # | Scope | Type | Base Year | Target Year | Reduction (%) | SBTi Validated |",
            "|---|-------|------|:---------:|:-----------:|--------------:|:--------------:|",
        ]
        for i, t in enumerate(targets, 1):
            lines.append(
                f"| {i} | {t.get('scope', '')} | {t.get('type', 'Absolute')} "
                f"| {t.get('base_year', '')} | {t.get('target_year', '')} "
                f"| {_dec(t.get('reduction_pct', 0))}% | {t.get('sbti_validated', 'No')} |"
            )
        if not targets:
            lines.append("| - | _No targets set_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_c4_2_progress(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        lines = [
            "## 4. C4.2 - Target Progress Table\n",
            "| # | Scope | Base Emissions (tCO2e) | Current (tCO2e) | Target (tCO2e) | Progress (%) |",
            "|---|-------|-----------------------:|----------------:|---------------:|-------------:|",
        ]
        for i, t in enumerate(targets, 1):
            base = float(t.get("base_emissions", 0))
            current = float(t.get("current_emissions", 0))
            reduction = float(t.get("reduction_pct", 0))
            target_em = base * (1 - reduction / 100)
            total_needed = base - target_em
            actual_red = base - current
            progress = (actual_red / total_needed * 100) if total_needed > 0 else 0
            lines.append(
                f"| {i} | {t.get('scope', '')} | {_dec_comma(base, 0)} "
                f"| {_dec_comma(current, 0)} | {_dec_comma(target_em, 0)} "
                f"| {_dec(min(progress, 100))}% |"
            )
        return "\n".join(lines)

    def _md_c5_methodology(self, data: Dict[str, Any]) -> str:
        meth = data.get("methodology", {})
        lines = [
            "## 5. C5 - Emissions Methodology\n",
            "| Parameter | Value |", "|-----------|-------|",
            f"| Standard Used | {meth.get('standard', 'GHG Protocol Corporate Standard')} |",
            f"| Consolidation Approach | {meth.get('consolidation', 'Operational control')} |",
            f"| GWP Source | {meth.get('gwp_source', 'IPCC AR6')} |",
            f"| GWP Timeframe | {meth.get('gwp_timeframe', '100-year')} |",
            f"| Emission Factor Sources | {meth.get('ef_sources', 'IPCC, IEA, DEFRA, EPA')} |",
            f"| Base Year | {meth.get('base_year', '')} |",
            f"| Recalculation Policy | {meth.get('recalculation', 'Recalculate for structural changes >5%')} |",
        ]
        return "\n".join(lines)

    def _md_c5_1_standards(self, data: Dict[str, Any]) -> str:
        standards = data.get("standards_applied", [])
        lines = [
            "## 6. C5.1 - Consolidation Approach & Standards\n",
            "| # | Standard | Application | Scope |",
            "|---|----------|-------------|-------|",
        ]
        default_standards = [
            {"name": "GHG Protocol Corporate Standard", "application": "Scope 1+2 accounting", "scope": "All operations"},
            {"name": "GHG Protocol Scope 3 Standard", "application": "Value chain emissions", "scope": "15 categories"},
            {"name": "GHG Protocol Scope 2 Guidance", "application": "Location + Market methods", "scope": "Purchased electricity"},
        ]
        for i, s in enumerate(standards or default_standards, 1):
            lines.append(f"| {i} | {s.get('name', '')} | {s.get('application', '')} | {s.get('scope', '')} |")
        return "\n".join(lines)

    def _md_c6_scope12(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1", {}).get("total", 0))
        s2_loc = float(data.get("scope2", {}).get("location", 0))
        s2_mkt = float(data.get("scope2", {}).get("market", 0))
        total_loc = s1 + s2_loc
        total_mkt = s1 + s2_mkt
        lines = [
            "## 7. C6 - Scope 1 & Scope 2 Emissions\n",
            "| Scope | Method | Emissions (tCO2e) | Share of S1+S2 (%) |",
            "|-------|--------|------------------:|-------------------:|",
            f"| Scope 1 | Direct | {_dec_comma(s1, 0)} | {_pct_of_total(s1, total_loc)}% |",
            f"| Scope 2 | Location-based | {_dec_comma(s2_loc, 0)} | {_pct_of_total(s2_loc, total_loc)}% |",
            f"| Scope 2 | Market-based | {_dec_comma(s2_mkt, 0)} | - |",
            f"| **Total (Location)** | - | **{_dec_comma(total_loc, 0)}** | **100%** |",
            f"| **Total (Market)** | - | **{_dec_comma(total_mkt, 0)}** | - |",
        ]
        return "\n".join(lines)

    def _md_c6_1_ghg_breakdown(self, data: Dict[str, Any]) -> str:
        by_ghg = data.get("scope1", {}).get("by_ghg", {})
        total = sum(float(v) for v in by_ghg.values())
        lines = [
            "## 8. C6.1 - Scope 1 Breakdown by GHG\n",
            "| GHG | Emissions (tCO2e) | Share (%) |",
            "|-----|------------------:|----------:|",
        ]
        for gas in GHG_GASES:
            em = float(by_ghg.get(gas, 0))
            if em > 0:
                lines.append(f"| {gas} | {_dec_comma(em, 0)} | {_pct_of_total(em, total)}% |")
        lines.append(f"| **Total** | **{_dec_comma(total, 0)}** | **100%** |")
        return "\n".join(lines)

    def _md_c6_3_scope2(self, data: Dict[str, Any]) -> str:
        s2 = data.get("scope2", {})
        s2_loc = float(s2.get("location", 0))
        s2_mkt = float(s2.get("market", 0))
        diff = s2_loc - s2_mkt
        instruments = data.get("scope2_instruments", [])
        lines = [
            "## 9. C6.3 - Scope 2 Location vs Market\n",
            "| Method | Emissions (tCO2e) |",
            "|--------|------------------:|",
            f"| Location-based | {_dec_comma(s2_loc, 0)} |",
            f"| Market-based | {_dec_comma(s2_mkt, 0)} |",
            f"| **Difference** | **{_dec_comma(diff, 0)}** |",
        ]
        if instruments:
            lines.extend([
                "\n### Contractual Instruments\n",
                "| # | Instrument | Volume (MWh) | Impact (tCO2e) |",
                "|---|-----------|-------------:|---------------:|",
            ])
            for i, inst in enumerate(instruments, 1):
                lines.append(
                    f"| {i} | {inst.get('type', '')} | {_dec_comma(inst.get('volume_mwh', 0), 0)} "
                    f"| {_dec_comma(inst.get('impact_tco2e', 0), 0)} |"
                )
        return "\n".join(lines)

    def _md_c7_scope3(self, data: Dict[str, Any]) -> str:
        s3_cats = data.get("scope3", {}).get("categories", {})
        s3_total = sum(float(v) for v in s3_cats.values())
        lines = [
            "## 10. C7 - Scope 3 Emissions by Category\n",
            "| Cat | Name | Direction | Emissions (tCO2e) | Share (%) | Reported |",
            "|:---:|------|-----------|------------------:|----------:|:--------:|",
        ]
        for cat_info in SCOPE3_CATEGORIES:
            cat_num = cat_info["cat"]
            em = float(s3_cats.get(cat_num, s3_cats.get(str(cat_num), 0)))
            reported = em > 0
            lines.append(
                f"| {cat_num} | {cat_info['name']} | {cat_info['direction']} "
                f"| {_dec_comma(em, 0) if reported else '-'} "
                f"| {_pct_of_total(em, s3_total) + '%' if reported else '-'} "
                f"| {'Yes' if reported else 'No'} |"
            )
        lines.append(f"\n**Total Scope 3:** {_dec_comma(s3_total, 0)} tCO2e ({len(s3_cats)} categories reported)")
        return "\n".join(lines)

    def _md_c7_1_categories(self, data: Dict[str, Any]) -> str:
        s3_detail = data.get("scope3_detail", {})
        lines = ["## 11. C7.1 - Scope 3 Category Details\n"]
        if s3_detail:
            for cat_num, detail in sorted(s3_detail.items(), key=lambda x: int(x[0])):
                cat_name = next((c["name"] for c in SCOPE3_CATEGORIES if c["cat"] == int(cat_num)), f"Category {cat_num}")
                lines.extend([
                    f"### Category {cat_num}: {cat_name}\n",
                    "| Field | Value |", "|-------|-------|",
                    f"| Emissions | {_dec_comma(detail.get('emissions', 0), 0)} tCO2e |",
                    f"| Method | {detail.get('method', 'Spend-based')} |",
                    f"| Data Quality | {detail.get('data_quality', 'Medium')} |",
                    f"| Boundary | {detail.get('boundary', 'All relevant activities')} |",
                    "",
                ])
        else:
            lines.append("*No detailed Scope 3 category data provided.*")
        return "\n".join(lines)

    def _md_completeness(self, data: Dict[str, Any]) -> str:
        comp = self._calculate_completeness(data)
        lines = [
            "## 12. Completeness Scoring\n",
            f"**Overall Score:** {comp['score']}% ({comp['passed']}/{comp['total']} checks)\n",
            "| Check | Status |", "|-------|--------|",
        ]
        for check, passed in comp["checks"].items():
            lines.append(f"| {check.replace('_', ' ').title()} | {'PASS' if passed else 'MISSING'} |")
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1", {}).get("total", 0))
        s2_loc = float(data.get("scope2", {}).get("location", 0))
        s3_cats = data.get("scope3", {}).get("categories", {})
        s3 = sum(float(v) for v in s3_cats.values())
        lines = [
            "## 13. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |", "|------------|----------|-------|",
            f"| Scope 1 Total | {XBRL_TAGS['scope1_total']} | {_dec_comma(s1, 0)} tCO2e |",
            f"| Scope 2 (Location) | {XBRL_TAGS['scope2_location']} | {_dec_comma(s2_loc, 0)} tCO2e |",
            f"| Scope 3 Total | {XBRL_TAGS['scope3_total']} | {_dec_comma(s3, 0)} tCO2e |",
            f"| Total Emissions | {XBRL_TAGS['total_emissions']} | {_dec_comma(s1 + s2_loc + s3, 0)} tCO2e |",
        ]
        return "\n".join(lines)

    def _md_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return (
            "## 14. Audit Trail & Provenance\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Report ID | `{rid}` |\n| Generated | {ts} |\n"
            f"| Template | {_TEMPLATE_ID} |\n| Version | {_MODULE_VERSION} |\n"
            f"| Pack | {_PACK_ID} |\n| Data Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-030 on {ts}*  \n*CDP C4-C7 emissions disclosure.*"

    # -- HTML sections --

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"h3{{color:{_ACCENT};}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #b3d4e6;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f0f7fb;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.4em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.75em;color:{_ACCENT};}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>CDP Climate Change - Emissions (C4-C7)</h1>\n<p><strong>{data.get("org_name", "")}</strong> | CDP {data.get("cdp_year", "")} | {ts}</p>'

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1", {}).get("total", 0))
        s2_loc = float(data.get("scope2", {}).get("location", 0))
        s3_cats = data.get("scope3", {}).get("categories", {})
        s3 = sum(float(v) for v in s3_cats.values())
        return (
            f'<h2>1. Executive Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Scope 1</div><div class="card-value">{_dec_comma(s1, 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Scope 2</div><div class="card-value">{_dec_comma(s2_loc, 0)}</div><div class="card-unit">tCO2e (Location)</div></div>\n'
            f'<div class="card"><div class="card-label">Scope 3</div><div class="card-value">{_dec_comma(s3, 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Total</div><div class="card-value">{_dec_comma(s1 + s2_loc + s3, 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'</div>'
        )

    def _html_c4_targets(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        rows = ""
        for i, t in enumerate(targets, 1):
            rows += f'<tr><td>{i}</td><td>{t.get("scope", "")}</td><td>{t.get("base_year", "")}</td><td>{t.get("target_year", "")}</td><td>{_dec(t.get("reduction_pct", 0))}%</td></tr>\n'
        return f'<h2>2. C4 - Targets</h2>\n<table>\n<tr><th>#</th><th>Scope</th><th>Base</th><th>Target</th><th>Reduction</th></tr>\n{rows}</table>'

    def _html_c6_scope12(self, data: Dict[str, Any]) -> str:
        s1 = float(data.get("scope1", {}).get("total", 0))
        s2_loc = float(data.get("scope2", {}).get("location", 0))
        s2_mkt = float(data.get("scope2", {}).get("market", 0))
        return (
            f'<h2>3. C6 - Scope 1 & 2</h2>\n<table>\n<tr><th>Scope</th><th>Method</th><th>Emissions (tCO2e)</th></tr>\n'
            f'<tr><td>Scope 1</td><td>Direct</td><td>{_dec_comma(s1, 0)}</td></tr>\n'
            f'<tr><td>Scope 2</td><td>Location</td><td>{_dec_comma(s2_loc, 0)}</td></tr>\n'
            f'<tr><td>Scope 2</td><td>Market</td><td>{_dec_comma(s2_mkt, 0)}</td></tr>\n'
            f'<tr><td><strong>Total (Loc)</strong></td><td>-</td><td><strong>{_dec_comma(s1 + s2_loc, 0)}</strong></td></tr>\n'
            f'</table>'
        )

    def _html_c6_1_ghg_breakdown(self, data: Dict[str, Any]) -> str:
        by_ghg = data.get("scope1", {}).get("by_ghg", {})
        rows = ""
        for gas in GHG_GASES:
            em = float(by_ghg.get(gas, 0))
            if em > 0:
                rows += f'<tr><td>{gas}</td><td>{_dec_comma(em, 0)}</td></tr>\n'
        return f'<h2>4. Scope 1 by GHG</h2>\n<table>\n<tr><th>GHG</th><th>Emissions (tCO2e)</th></tr>\n{rows}</table>'

    def _html_c6_3_scope2(self, data: Dict[str, Any]) -> str:
        s2_loc = float(data.get("scope2", {}).get("location", 0))
        s2_mkt = float(data.get("scope2", {}).get("market", 0))
        return (
            f'<h2>5. Scope 2 Comparison</h2>\n<table>\n<tr><th>Method</th><th>Emissions (tCO2e)</th></tr>\n'
            f'<tr><td>Location</td><td>{_dec_comma(s2_loc, 0)}</td></tr>\n'
            f'<tr><td>Market</td><td>{_dec_comma(s2_mkt, 0)}</td></tr>\n'
            f'<tr><td><strong>Difference</strong></td><td><strong>{_dec_comma(s2_loc - s2_mkt, 0)}</strong></td></tr>\n'
            f'</table>'
        )

    def _html_c7_scope3(self, data: Dict[str, Any]) -> str:
        s3_cats = data.get("scope3", {}).get("categories", {})
        rows = ""
        for cat_info in SCOPE3_CATEGORIES:
            cat_num = cat_info["cat"]
            em = float(s3_cats.get(cat_num, s3_cats.get(str(cat_num), 0)))
            rows += f'<tr><td>{cat_num}</td><td>{cat_info["name"]}</td><td>{_dec_comma(em, 0) if em > 0 else "-"}</td></tr>\n'
        return f'<h2>6. C7 - Scope 3</h2>\n<table>\n<tr><th>Cat</th><th>Name</th><th>Emissions (tCO2e)</th></tr>\n{rows}</table>'

    def _html_completeness(self, data: Dict[str, Any]) -> str:
        comp = self._calculate_completeness(data)
        return f'<h2>7. Completeness</h2>\n<p>Score: {comp["score"]}% ({comp["passed"]}/{comp["total"]})</p>'

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        rows = ""
        for key, tag in XBRL_TAGS.items():
            rows += f'<tr><td>{key.replace("_", " ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>8. XBRL Tags</h2>\n<table>\n<tr><th>Data Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>9. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-030 on {ts} - CDP Emissions</div>'
