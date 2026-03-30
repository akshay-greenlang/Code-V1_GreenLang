# -*- coding: utf-8 -*-
"""
VarianceAnalysisReportTemplate - LMDI decomposition & variance analysis for PACK-029.

Renders a comprehensive variance analysis report with LMDI (Logarithmic Mean
Divisia Index) decomposition, Kaya identity waterfall chart data, scope-level
and category-level attribution, root cause classification, year-over-year
variance trends, and corrective action recommendations.
Multi-format output (MD, HTML, JSON, PDF).

Sections:
    1.  Executive Summary
    2.  LMDI Decomposition Table
    3.  Kaya Identity Waterfall
    4.  Scope-Level Attribution
    5.  Category-Level Attribution (Scope 3)
    6.  Root Cause Classification
    7.  Year-over-Year Variance Trends
    8.  Corrective Action Recommendations
    9.  Sensitivity Analysis
    10. XBRL Tagging Summary
    11. Audit Trail & Provenance

Author: GreenLang Team
Version: 29.0.0
Pack: PACK-029 Interim Targets Pack
"""

import hashlib
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "29.0.0"
_PACK_ID = "PACK-029"
_TEMPLATE_ID = "variance_analysis_report"

_PRIMARY = "#0a4a3a"
_SECONDARY = "#167a5b"
_ACCENT = "#1db954"
_LIGHT = "#e0f5ed"
_LIGHTER = "#f0faf5"
_SUCCESS = "#2e7d32"
_WARN = "#ef6c00"
_DANGER = "#c62828"

LMDI_EFFECTS = [
    {"id": "activity", "name": "Activity Effect", "desc": "Change in production/revenue volume"},
    {"id": "structure", "name": "Structural Effect", "desc": "Shift in activity mix across sectors/products"},
    {"id": "intensity", "name": "Intensity Effect", "desc": "Change in emissions intensity per unit activity"},
    {"id": "fuel_mix", "name": "Fuel Mix Effect", "desc": "Shift in energy carrier composition"},
    {"id": "emission_factor", "name": "Emission Factor Effect", "desc": "Changes in grid or supplier emission factors"},
]

KAYA_FACTORS = [
    {"id": "population", "name": "Population/Scale", "unit": "headcount/units"},
    {"id": "gdp_per_capita", "name": "GDP per Capita / Revenue per Employee", "unit": "currency/person"},
    {"id": "energy_intensity", "name": "Energy Intensity", "unit": "MJ/currency"},
    {"id": "carbon_intensity", "name": "Carbon Intensity of Energy", "unit": "tCO2e/MJ"},
]

ROOT_CAUSE_CATEGORIES = [
    {"id": "internal_initiative", "name": "Internal Initiative", "desc": "Planned reduction measures"},
    {"id": "organic_growth", "name": "Organic Growth", "desc": "Business expansion / contraction"},
    {"id": "acquisition", "name": "M&A Activity", "desc": "Mergers, acquisitions, divestitures"},
    {"id": "methodology", "name": "Methodology Change", "desc": "Updated emission factors or boundaries"},
    {"id": "external_factor", "name": "External Factor", "desc": "Grid decarbonization, weather, regulation"},
    {"id": "data_quality", "name": "Data Quality", "desc": "Improved measurement replacing estimates"},
]

XBRL_TAGS: Dict[str, str] = {
    "total_variance": "gl:TotalEmissionsVariance",
    "activity_effect": "gl:LMDIActivityEffect",
    "structure_effect": "gl:LMDIStructuralEffect",
    "intensity_effect": "gl:LMDIIntensityEffect",
    "fuel_mix_effect": "gl:LMDIFuelMixEffect",
    "emission_factor_effect": "gl:LMDIEmissionFactorEffect",
    "decomposition_residual": "gl:LMDIDecompositionResidual",
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

def _pct_change(current: Any, baseline: Any) -> Decimal:
    c = Decimal(str(current))
    b = Decimal(str(baseline))
    if b == 0:
        return Decimal("0.00")
    return ((c - b) / b * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def _logarithmic_mean(a: float, b: float) -> float:
    """Calculate logarithmic mean for LMDI decomposition."""
    if a <= 0 or b <= 0:
        return 0.0
    if abs(a - b) < 1e-10:
        return a
    return (a - b) / math.log(a / b)

class VarianceAnalysisReportTemplate:
    """
    Variance analysis report template with LMDI decomposition for PACK-029.

    Renders LMDI decomposition tables, Kaya identity waterfall data, scope
    and category attribution, root cause classification, YoY trends, and
    corrective action recommendations. Supports MD, HTML, JSON, PDF.

    Example:
        >>> template = VarianceAnalysisReportTemplate()
        >>> data = {
        ...     "org_name": "GreenCorp",
        ...     "reporting_year": 2025,
        ...     "prior_year_emissions": 95000,
        ...     "current_year_emissions": 88000,
        ...     "target_emissions": 85000,
        ...     "lmdi_effects": {"activity": 3000, "intensity": -8000, "structure": -2000},
        ... }
        >>> md = template.render_markdown(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render full variance analysis report as Markdown."""
        self.generated_at = utcnow()
        sections = [
            self._md_header(data), self._md_executive_summary(data),
            self._md_lmdi(data), self._md_kaya(data),
            self._md_scope_attribution(data), self._md_category_attribution(data),
            self._md_root_causes(data), self._md_yoy_trends(data),
            self._md_corrective_actions(data), self._md_sensitivity(data),
            self._md_xbrl(data), self._md_audit(data), self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render full variance analysis report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        parts = [
            self._html_header(data), self._html_executive_summary(data),
            self._html_lmdi(data), self._html_kaya(data),
            self._html_scope_attribution(data), self._html_category_attribution(data),
            self._html_root_causes(data), self._html_yoy_trends(data),
            self._html_corrective_actions(data), self._html_sensitivity(data),
            self._html_xbrl(data), self._html_audit(data), self._html_footer(data),
        ]
        body = "\n".join(parts)
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Variance Analysis - {data.get("org_name","")}</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured JSON."""
        self.generated_at = utcnow()
        prior = float(data.get("prior_year_emissions", 0))
        current = float(data.get("current_year_emissions", 0))
        target = float(data.get("target_emissions", 0))
        total_var = current - prior
        target_var = current - target
        lmdi = data.get("lmdi_effects", {})
        effects_total = sum(float(v) for v in lmdi.values())
        residual = total_var - effects_total

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID, "version": _MODULE_VERSION,
            "pack_id": _PACK_ID, "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(), "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "emissions": {
                "prior_year": str(prior), "current_year": str(current),
                "target": str(target), "total_variance": str(total_var),
                "target_variance": str(target_var),
                "variance_pct": str(round(float(_pct_change(current, prior)), 2)),
            },
            "lmdi_decomposition": {
                "effects": {k: str(v) for k, v in lmdi.items()},
                "effects_total": str(effects_total),
                "residual": str(round(residual, 2)),
                "decomposition_quality": "Good" if abs(residual) < abs(total_var) * 0.05 else "Review needed",
            },
            "kaya_factors": data.get("kaya_factors", {}),
            "scope_attribution": data.get("scope_attribution", {}),
            "category_attribution": data.get("category_attribution", {}),
            "root_causes": data.get("root_causes", []),
            "corrective_actions": data.get("corrective_actions", []),
            "xbrl_tags": XBRL_TAGS,
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_pdf(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return PDF-ready structured data."""
        return {
            "format": "pdf", "html_content": self.render_html(data),
            "structured_data": self.render_json(data),
            "metadata": {"title": f"Variance Analysis - {data.get('org_name','')}", "author": "GreenLang PACK-029"},
        }

    # -- Markdown --

    def _md_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Variance Analysis Report\n\n"
            f"**Organization:** {data.get('org_name', '')}  \n"
            f"**Reporting Year:** {data.get('reporting_year', '')}  \n"
            f"**Report Date:** {ts}  \n"
            f"**Pack:** PACK-029 Interim Targets Pack v{_MODULE_VERSION}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        prior = float(data.get("prior_year_emissions", 0))
        current = float(data.get("current_year_emissions", 0))
        target = float(data.get("target_emissions", 0))
        total_var = current - prior
        target_var = current - target
        var_pct = float(_pct_change(current, prior)) if prior else 0
        on_track = current <= target
        lines = [
            "## 1. Executive Summary\n",
            "| Metric | Value |", "|--------|-------|",
            f"| Prior Year Emissions | {_dec_comma(prior, 0)} tCO2e |",
            f"| Current Year Emissions | {_dec_comma(current, 0)} tCO2e |",
            f"| Year-over-Year Change | {'+' if total_var > 0 else ''}{_dec_comma(total_var, 0)} tCO2e ({'+' if var_pct > 0 else ''}{_dec(var_pct)}%) |",
            f"| Target Emissions | {_dec_comma(target, 0)} tCO2e |",
            f"| Gap to Target | {'+' if target_var > 0 else ''}{_dec_comma(target_var, 0)} tCO2e |",
            f"| On Track | **{'Yes' if on_track else 'No'}** |",
        ]
        return "\n".join(lines)

    def _md_lmdi(self, data: Dict[str, Any]) -> str:
        prior = float(data.get("prior_year_emissions", 0))
        current = float(data.get("current_year_emissions", 0))
        total_var = current - prior
        lmdi = data.get("lmdi_effects", {})
        effects_total = sum(float(v) for v in lmdi.values())
        residual = total_var - effects_total
        lines = [
            "## 2. LMDI Decomposition Table\n",
            "The Logarithmic Mean Divisia Index (LMDI) decomposes the total "
            "emissions change into distinct driving effects.\n",
            "| Effect | Impact (tCO2e) | Share (%) | Direction |",
            "|--------|---------------:|----------:|-----------|",
        ]
        for effect_def in LMDI_EFFECTS:
            val = float(lmdi.get(effect_def["id"], 0))
            share = (val / total_var * 100) if total_var != 0 else 0
            direction = "Increase" if val > 0 else ("Decrease" if val < 0 else "Neutral")
            lines.append(
                f"| {effect_def['name']} | {'+' if val > 0 else ''}{_dec_comma(val, 0)} "
                f"| {_dec(abs(share))}% | {direction} |"
            )
        lines.extend([
            f"| **Subtotal (Explained)** | **{'+' if effects_total > 0 else ''}{_dec_comma(effects_total, 0)}** | - | - |",
            f"| Residual | {'+' if residual > 0 else ''}{_dec_comma(residual, 0)} | - | - |",
            f"| **Total Variance** | **{'+' if total_var > 0 else ''}{_dec_comma(total_var, 0)}** | **100%** | - |",
        ])
        return "\n".join(lines)

    def _md_kaya(self, data: Dict[str, Any]) -> str:
        kaya = data.get("kaya_factors", {})
        lines = [
            "## 3. Kaya Identity Waterfall\n",
            "CO2 = Population x (GDP/Population) x (Energy/GDP) x (CO2/Energy)\n",
            "| Factor | Prior Year | Current Year | Change (%) | Contribution (tCO2e) |",
            "|--------|------------|:------------:|:----------:|---------------------:|",
        ]
        for factor_def in KAYA_FACTORS:
            kf = kaya.get(factor_def["id"], {})
            prior_val = kf.get("prior", "-")
            current_val = kf.get("current", "-")
            change_pct = kf.get("change_pct", 0)
            contribution = kf.get("contribution_tco2e", 0)
            lines.append(
                f"| {factor_def['name']} | {prior_val} | {current_val} "
                f"| {'+' if float(change_pct) > 0 else ''}{_dec(change_pct)}% "
                f"| {'+' if float(contribution) > 0 else ''}{_dec_comma(contribution, 0)} |"
            )
        return "\n".join(lines)

    def _md_scope_attribution(self, data: Dict[str, Any]) -> str:
        scope_attr = data.get("scope_attribution", {})
        lines = [
            "## 4. Scope-Level Attribution\n",
            "| Scope | Prior (tCO2e) | Current (tCO2e) | Change (tCO2e) | Change (%) |",
            "|-------|-------------:|----------------:|---------------:|-----------:|",
        ]
        for scope in ["scope1", "scope2_location", "scope2_market", "scope3"]:
            sa = scope_attr.get(scope, {})
            prior = float(sa.get("prior", 0))
            current = float(sa.get("current", 0))
            change = current - prior
            pct = float(_pct_change(current, prior)) if prior else 0
            lines.append(
                f"| {scope.replace('_', ' ').title()} | {_dec_comma(prior, 0)} | {_dec_comma(current, 0)} "
                f"| {'+' if change > 0 else ''}{_dec_comma(change, 0)} "
                f"| {'+' if pct > 0 else ''}{_dec(pct)}% |"
            )
        return "\n".join(lines)

    def _md_category_attribution(self, data: Dict[str, Any]) -> str:
        cat_attr = data.get("category_attribution", {})
        lines = [
            "## 5. Category-Level Attribution (Scope 3)\n",
            "| Category | Prior (tCO2e) | Current (tCO2e) | Change (tCO2e) | Change (%) | Driver |",
            "|----------|-------------:|----------------:|---------------:|-----------:|--------|",
        ]
        for cat_name, cat_data in sorted(cat_attr.items()):
            prior = float(cat_data.get("prior", 0))
            current = float(cat_data.get("current", 0))
            change = current - prior
            pct = float(_pct_change(current, prior)) if prior else 0
            driver = cat_data.get("driver", "-")
            lines.append(
                f"| {cat_name} | {_dec_comma(prior, 0)} | {_dec_comma(current, 0)} "
                f"| {'+' if change > 0 else ''}{_dec_comma(change, 0)} "
                f"| {'+' if pct > 0 else ''}{_dec(pct)}% | {driver} |"
            )
        if not cat_attr:
            lines.append("| _No category-level data provided_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_root_causes(self, data: Dict[str, Any]) -> str:
        root_causes = data.get("root_causes", [])
        lines = [
            "## 6. Root Cause Classification\n",
            "| # | Root Cause | Category | Impact (tCO2e) | Impact (%) | Controllable |",
            "|---|-----------|----------|---------------:|-----------:|:------------:|",
        ]
        for i, rc in enumerate(root_causes, 1):
            impact = float(rc.get("impact_tco2e", 0))
            lines.append(
                f"| {i} | {rc.get('description', '')} | {rc.get('category', '')} "
                f"| {'+' if impact > 0 else ''}{_dec_comma(impact, 0)} "
                f"| {'+' if impact > 0 else ''}{_dec(rc.get('impact_pct', 0))}% "
                f"| {'Yes' if rc.get('controllable', False) else 'No'} |"
            )
        if not root_causes:
            lines.append("| - | _No root causes provided_ | - | - | - | - |")
        lines.extend(["\n### Root Cause Category Definitions\n"])
        for cat in ROOT_CAUSE_CATEGORIES:
            lines.append(f"- **{cat['name']}:** {cat['desc']}")
        return "\n".join(lines)

    def _md_yoy_trends(self, data: Dict[str, Any]) -> str:
        trends = data.get("variance_trends", [])
        lines = [
            "## 7. Year-over-Year Variance Trends\n",
            "| Year | Emissions (tCO2e) | Target (tCO2e) | Variance (tCO2e) | Variance (%) | Status |",
            "|------|------------------:|---------------:|-----------------:|-------------:|--------|",
        ]
        for t in trends:
            em = float(t.get("emissions", 0))
            tgt = float(t.get("target", 0))
            var = em - tgt
            pct = (var / tgt * 100) if tgt != 0 else 0
            status = "On Track" if em <= tgt else "Off Track"
            lines.append(
                f"| {t.get('year', '')} | {_dec_comma(em, 0)} | {_dec_comma(tgt, 0)} "
                f"| {'+' if var > 0 else ''}{_dec_comma(var, 0)} "
                f"| {'+' if pct > 0 else ''}{_dec(pct)}% | {status} |"
            )
        if not trends:
            lines.append("| - | _No trend data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_corrective_actions(self, data: Dict[str, Any]) -> str:
        actions = data.get("corrective_actions", [])
        lines = [
            "## 8. Corrective Action Recommendations\n",
            "| # | Action | Priority | Expected Impact (tCO2e) | Timeline | Owner |",
            "|---|--------|----------|------------------------:|----------|-------|",
        ]
        for i, a in enumerate(actions, 1):
            lines.append(
                f"| {i} | {a.get('action', '')} | {a.get('priority', 'Medium')} "
                f"| {_dec_comma(a.get('impact_tco2e', 0), 0)} "
                f"| {a.get('timeline', 'TBD')} | {a.get('owner', 'TBD')} |"
            )
        if not actions:
            lines.append("| - | _No corrective actions defined_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_sensitivity(self, data: Dict[str, Any]) -> str:
        sensitivity = data.get("sensitivity_analysis", {})
        scenarios = sensitivity.get("scenarios", [])
        lines = [
            "## 9. Sensitivity Analysis\n",
            "| Scenario | Assumption Change | Impact on Variance (tCO2e) | Impact (%) |",
            "|----------|-------------------|---------------------------:|-----------:|",
        ]
        for s in scenarios:
            lines.append(
                f"| {s.get('scenario', '')} | {s.get('assumption', '')} "
                f"| {'+' if float(s.get('impact_tco2e', 0)) > 0 else ''}{_dec_comma(s.get('impact_tco2e', 0), 0)} "
                f"| {_dec(s.get('impact_pct', 0))}% |"
            )
        if not scenarios:
            lines.append("| - | _No sensitivity scenarios_ | - | - |")
        return "\n".join(lines)

    def _md_xbrl(self, data: Dict[str, Any]) -> str:
        lmdi = data.get("lmdi_effects", {})
        lines = [
            "## 10. XBRL Tagging Summary\n",
            "| Data Point | XBRL Tag | Value |", "|------------|----------|-------|",
        ]
        prior = float(data.get("prior_year_emissions", 0))
        current = float(data.get("current_year_emissions", 0))
        total_var = current - prior
        lines.append(f"| Total Variance | {XBRL_TAGS['total_variance']} | {_dec_comma(total_var, 0)} tCO2e |")
        for effect_def in LMDI_EFFECTS:
            tag_key = f"{effect_def['id']}_effect"
            if tag_key in XBRL_TAGS:
                val = float(lmdi.get(effect_def["id"], 0))
                lines.append(f"| {effect_def['name']} | {XBRL_TAGS[tag_key]} | {_dec_comma(val, 0)} tCO2e |")
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
            f"| Pack | {_PACK_ID} |\n| Hash | `{dh[:16]}...` |"
        )

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n\n*Generated by GreenLang PACK-029 on {ts}*  \n*LMDI decomposition & variance analysis.*"

    # -- HTML --

    def _css(self) -> str:
        return (
            f"body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;background:{_LIGHTER};color:#1a1a2e;}}"
            f".report{{max-width:1200px;margin:0 auto;background:#fff;padding:40px;border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}}"
            f"h1{{color:{_PRIMARY};border-bottom:3px solid {_SECONDARY};padding-bottom:12px;}}"
            f"h2{{color:{_SECONDARY};margin-top:35px;border-left:4px solid {_ACCENT};padding-left:12px;}}"
            f"h3{{color:{_ACCENT};}}"
            f"table{{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}}"
            f"th,td{{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}}"
            f"th{{background:{_LIGHT};font-weight:600;color:{_PRIMARY};}}"
            f"tr:nth-child(even){{background:#f9fbe7;}}"
            f".summary-cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin:20px 0;}}"
            f".card{{background:linear-gradient(135deg,{_LIGHT},{_LIGHTER});border-radius:10px;padding:18px;text-align:center;border-left:4px solid {_SECONDARY};}}"
            f".card-label{{font-size:0.8em;color:{_SECONDARY};text-transform:uppercase;}}"
            f".card-value{{font-size:1.5em;font-weight:700;color:{_PRIMARY};margin-top:4px;}}"
            f".card-unit{{font-size:0.75em;color:{_ACCENT};}}"
            f".waterfall-bar{{display:inline-block;height:16px;border-radius:4px;min-width:4px;}}"
            f".waterfall-positive{{background:{_DANGER};}}"
            f".waterfall-negative{{background:{_SUCCESS};}}"
            f".footer{{margin-top:40px;padding-top:20px;border-top:2px solid {_LIGHT};color:{_ACCENT};font-size:0.85em;text-align:center;}}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<h1>Variance Analysis Report</h1>\n<p><strong>Organization:</strong> {data.get("org_name","")} | <strong>Year:</strong> {data.get("reporting_year","")} | <strong>Generated:</strong> {ts}</p>'

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        prior = float(data.get("prior_year_emissions", 0))
        current = float(data.get("current_year_emissions", 0))
        target = float(data.get("target_emissions", 0))
        total_var = current - prior
        var_pct = float(_pct_change(current, prior)) if prior else 0
        on_track = current <= target
        return (
            f'<h2>1. Executive Summary</h2>\n<div class="summary-cards">\n'
            f'<div class="card"><div class="card-label">Prior Year</div><div class="card-value">{_dec_comma(prior, 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">Current Year</div><div class="card-value">{_dec_comma(current, 0)}</div><div class="card-unit">tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">YoY Change</div><div class="card-value">{_dec(var_pct)}%</div><div class="card-unit">{_dec_comma(total_var, 0)} tCO2e</div></div>\n'
            f'<div class="card"><div class="card-label">On Track</div><div class="card-value">{"Yes" if on_track else "No"}</div></div>\n'
            f'</div>'
        )

    def _html_lmdi(self, data: Dict[str, Any]) -> str:
        prior = float(data.get("prior_year_emissions", 0))
        current = float(data.get("current_year_emissions", 0))
        total_var = current - prior
        lmdi = data.get("lmdi_effects", {})
        max_abs = max((abs(float(v)) for v in lmdi.values()), default=1)
        rows = ""
        for effect_def in LMDI_EFFECTS:
            val = float(lmdi.get(effect_def["id"], 0))
            bar_width = min(100, abs(val) / max(1, max_abs) * 100)
            bar_cls = "waterfall-positive" if val > 0 else "waterfall-negative"
            rows += (
                f'<tr><td>{effect_def["name"]}</td><td>{_dec_comma(val, 0)}</td>'
                f'<td><span class="waterfall-bar {bar_cls}" style="width:{bar_width}%"></span></td>'
                f'<td>{effect_def["desc"]}</td></tr>\n'
            )
        return (
            f'<h2>2. LMDI Decomposition</h2>\n<table>\n'
            f'<tr><th>Effect</th><th>Impact (tCO2e)</th><th>Waterfall</th><th>Description</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_kaya(self, data: Dict[str, Any]) -> str:
        kaya = data.get("kaya_factors", {})
        rows = ""
        for f_def in KAYA_FACTORS:
            kf = kaya.get(f_def["id"], {})
            rows += f'<tr><td>{f_def["name"]}</td><td>{kf.get("prior","-")}</td><td>{kf.get("current","-")}</td><td>{_dec(kf.get("change_pct",0))}%</td></tr>\n'
        return f'<h2>3. Kaya Identity</h2>\n<table>\n<tr><th>Factor</th><th>Prior</th><th>Current</th><th>Change</th></tr>\n{rows}</table>'

    def _html_scope_attribution(self, data: Dict[str, Any]) -> str:
        scope_attr = data.get("scope_attribution", {})
        rows = ""
        for scope in ["scope1", "scope2_location", "scope2_market", "scope3"]:
            sa = scope_attr.get(scope, {})
            prior = float(sa.get("prior", 0))
            current = float(sa.get("current", 0))
            change = current - prior
            rows += f'<tr><td>{scope.replace("_"," ").title()}</td><td>{_dec_comma(prior, 0)}</td><td>{_dec_comma(current, 0)}</td><td>{_dec_comma(change, 0)}</td></tr>\n'
        return f'<h2>4. Scope Attribution</h2>\n<table>\n<tr><th>Scope</th><th>Prior</th><th>Current</th><th>Change</th></tr>\n{rows}</table>'

    def _html_category_attribution(self, data: Dict[str, Any]) -> str:
        cat_attr = data.get("category_attribution", {})
        rows = ""
        for cat_name, cat_data in sorted(cat_attr.items()):
            prior = float(cat_data.get("prior", 0))
            current = float(cat_data.get("current", 0))
            rows += f'<tr><td>{cat_name}</td><td>{_dec_comma(prior, 0)}</td><td>{_dec_comma(current, 0)}</td><td>{_dec_comma(current - prior, 0)}</td></tr>\n'
        return f'<h2>5. Category Attribution</h2>\n<table>\n<tr><th>Category</th><th>Prior</th><th>Current</th><th>Change</th></tr>\n{rows}</table>'

    def _html_root_causes(self, data: Dict[str, Any]) -> str:
        rcs = data.get("root_causes", [])
        rows = ""
        for i, rc in enumerate(rcs, 1):
            rows += f'<tr><td>{i}</td><td>{rc.get("description","")}</td><td>{rc.get("category","")}</td><td>{_dec_comma(rc.get("impact_tco2e",0), 0)}</td></tr>\n'
        return f'<h2>6. Root Causes</h2>\n<table>\n<tr><th>#</th><th>Description</th><th>Category</th><th>Impact (tCO2e)</th></tr>\n{rows}</table>'

    def _html_yoy_trends(self, data: Dict[str, Any]) -> str:
        trends = data.get("variance_trends", [])
        rows = ""
        for t in trends:
            em = float(t.get("emissions", 0))
            tgt = float(t.get("target", 0))
            rows += f'<tr><td>{t.get("year","")}</td><td>{_dec_comma(em, 0)}</td><td>{_dec_comma(tgt, 0)}</td><td>{_dec_comma(em - tgt, 0)}</td></tr>\n'
        return f'<h2>7. YoY Trends</h2>\n<table>\n<tr><th>Year</th><th>Emissions</th><th>Target</th><th>Variance</th></tr>\n{rows}</table>'

    def _html_corrective_actions(self, data: Dict[str, Any]) -> str:
        actions = data.get("corrective_actions", [])
        rows = ""
        for i, a in enumerate(actions, 1):
            rows += f'<tr><td>{i}</td><td>{a.get("action","")}</td><td>{a.get("priority","Medium")}</td><td>{_dec_comma(a.get("impact_tco2e",0), 0)}</td><td>{a.get("timeline","TBD")}</td></tr>\n'
        return f'<h2>8. Corrective Actions</h2>\n<table>\n<tr><th>#</th><th>Action</th><th>Priority</th><th>Impact (tCO2e)</th><th>Timeline</th></tr>\n{rows}</table>'

    def _html_sensitivity(self, data: Dict[str, Any]) -> str:
        scenarios = data.get("sensitivity_analysis", {}).get("scenarios", [])
        rows = ""
        for s in scenarios:
            rows += f'<tr><td>{s.get("scenario","")}</td><td>{s.get("assumption","")}</td><td>{_dec_comma(s.get("impact_tco2e",0), 0)}</td></tr>\n'
        return f'<h2>9. Sensitivity</h2>\n<table>\n<tr><th>Scenario</th><th>Assumption</th><th>Impact (tCO2e)</th></tr>\n{rows}</table>'

    def _html_xbrl(self, data: Dict[str, Any]) -> str:
        rows = ""
        for key, tag in XBRL_TAGS.items():
            rows += f'<tr><td>{key.replace("_"," ").title()}</td><td><code>{tag}</code></td></tr>\n'
        return f'<h2>10. XBRL Tags</h2>\n<table>\n<tr><th>Data Point</th><th>Tag</th></tr>\n{rows}</table>'

    def _html_audit(self, data: Dict[str, Any]) -> str:
        rid = _new_uuid()
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M:%S UTC") if self.generated_at else ""
        dh = _compute_hash(data)
        return f'<h2>11. Audit Trail</h2>\n<table>\n<tr><th>Parameter</th><th>Value</th></tr>\n<tr><td>Report ID</td><td><code>{rid}</code></td></tr>\n<tr><td>Generated</td><td>{ts}</td></tr>\n<tr><td>Template</td><td>{_TEMPLATE_ID}</td></tr>\n<tr><td>Hash</td><td><code>{dh[:16]}...</code></td></tr>\n</table>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f'<div class="footer">Generated by GreenLang PACK-029 on {ts} - LMDI variance analysis</div>'
