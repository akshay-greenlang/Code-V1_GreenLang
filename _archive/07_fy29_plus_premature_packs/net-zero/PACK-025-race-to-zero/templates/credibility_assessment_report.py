# -*- coding: utf-8 -*-
"""
CredibilityAssessmentReportTemplate - Race to Zero credibility for PACK-025.

Renders a comprehensive credibility assessment report aligned with the
UN HLEG 10 recommendations, covering credibility score breakdown across
7 dimensions, science-based ambition assessment, governance maturity,
transparency and verification status, and improvement recommendations.

Sections:
    1. Assessment Overview
    2. HLEG 10 Recommendations Compliance Matrix
    3. Credibility Score Breakdown (7 Dimensions)
    4. Science-Based Ambition Assessment
    5. Governance Maturity Evaluation
    6. Transparency & Verification Status
    7. Offset & Removal Integrity
    8. Improvement Recommendations
    9. Credibility Trend History

Author: GreenLang Team
Version: 25.0.0
Pack: PACK-025 Race to Zero Pack
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

_MODULE_VERSION = "25.0.0"
_PACK_ID = "PACK-025"
_TEMPLATE_ID = "credibility_assessment_report"

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

def _safe_div(n: Any, d: Any) -> float:
    try:
        dv = float(d)
        return float(n) / dv if dv != 0 else 0.0
    except Exception:
        return 0.0

# HLEG 10 Recommendations
HLEG_RECOMMENDATIONS = [
    {"id": "R01", "recommendation": "Announce a net-zero pledge", "category": "Ambition",
     "description": "Publicly commit to reaching net-zero GHG emissions by 2050 at latest"},
    {"id": "R02", "recommendation": "Set interim targets", "category": "Ambition",
     "description": "Near-term targets for 2025-2030 aligned with 1.5C pathways"},
    {"id": "R03", "recommendation": "Use voluntary carbon credits responsibly", "category": "Integrity",
     "description": "Credits beyond value chain only; no use for interim targets"},
    {"id": "R04", "recommendation": "Create a transition plan", "category": "Planning",
     "description": "Comprehensive plan with milestones, governance, and investment"},
    {"id": "R05", "recommendation": "Phase out fossil fuels", "category": "Action",
     "description": "No new fossil fuel capacity; rapid phase-out of existing"},
    {"id": "R06", "recommendation": "Align lobbying and advocacy", "category": "Integrity",
     "description": "All policy engagement consistent with 1.5C pathway"},
    {"id": "R07", "recommendation": "Ensure people-centered transition", "category": "Just Transition",
     "description": "Protect workers and communities in transition"},
    {"id": "R08", "recommendation": "Increase transparency", "category": "Transparency",
     "description": "Annual public reporting on progress with standardized metrics"},
    {"id": "R09", "recommendation": "Invest in just transition", "category": "Just Transition",
     "description": "Financial support for affected communities and workers"},
    {"id": "R10", "recommendation": "Accelerate near-term action", "category": "Action",
     "description": "Immediate emission reductions, not delayed action"},
]

# 7 Credibility Dimensions
CREDIBILITY_DIMENSIONS = [
    {"name": "Ambition", "max_score": 100, "weight": 0.20,
     "description": "Science-based target alignment and scope coverage"},
    {"name": "Integrity", "max_score": 100, "weight": 0.15,
     "description": "Responsible credit use and anti-greenwashing measures"},
    {"name": "Transparency", "max_score": 100, "weight": 0.15,
     "description": "Public disclosure quality and reporting completeness"},
    {"name": "Action", "max_score": 100, "weight": 0.20,
     "description": "Concrete reduction actions and investment deployment"},
    {"name": "Governance", "max_score": 100, "weight": 0.10,
     "description": "Board oversight, policies, and accountability structures"},
    {"name": "Verification", "max_score": 100, "weight": 0.10,
     "description": "Third-party assurance level and verification scope"},
    {"name": "Engagement", "max_score": 100, "weight": 0.10,
     "description": "Supply chain and stakeholder engagement effectiveness"},
]

class CredibilityAssessmentReportTemplate:
    """Race to Zero credibility assessment report template for PACK-025.

    Generates comprehensive credibility assessment reports based on the
    UN HLEG 10 recommendations with 7-dimension scoring, governance
    maturity evaluation, and improvement roadmap.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID
    HLEG = HLEG_RECOMMENDATIONS
    DIMENSIONS = CREDIBILITY_DIMENSIONS

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    #  Public render methods                                               #
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the credibility assessment report as Markdown."""
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_overview(data),
            self._md_hleg_matrix(data),
            self._md_credibility_breakdown(data),
            self._md_ambition_assessment(data),
            self._md_governance_maturity(data),
            self._md_transparency_verification(data),
            self._md_offset_integrity(data),
            self._md_improvements(data),
            self._md_trend_history(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the credibility assessment report as HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_overview(data),
            self._html_hleg_matrix(data),
            self._html_credibility_breakdown(data),
            self._html_ambition(data),
            self._html_governance(data),
            self._html_transparency(data),
            self._html_improvements(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Race to Zero - Credibility Assessment</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the credibility assessment as structured JSON."""
        self.generated_at = utcnow()
        dimensions = data.get("dimensions", [])
        hleg_scores = data.get("hleg_scores", {})

        # Calculate overall weighted score
        overall_score = self._calc_overall_score(dimensions)
        hleg_compliance = self._calc_hleg_compliance(hleg_scores)

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "assessment_date": data.get("assessment_date", ""),
            "overall_score": round(overall_score, 1),
            "rating": self._score_to_rating(overall_score),
            "hleg_compliance": {
                "met": hleg_compliance["met"],
                "total": 10,
                "compliance_pct": round(hleg_compliance["pct"], 1),
            },
            "dimensions": dimensions,
            "hleg_scores": hleg_scores,
            "governance_maturity": data.get("governance_maturity", {}),
            "improvements": data.get("improvements", []),
            "trend_history": data.get("trend_history", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel_data(self, data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Return structured data for Excel/openpyxl export."""
        self.generated_at = utcnow()
        sheets: Dict[str, List[Dict[str, Any]]] = {}

        # Sheet 1: HLEG Matrix
        hleg_scores = data.get("hleg_scores", {})
        hleg_rows: List[Dict[str, Any]] = []
        for rec in HLEG_RECOMMENDATIONS:
            rid = rec["id"]
            score = hleg_scores.get(rid, {})
            hleg_rows.append({
                "ID": rid,
                "Recommendation": rec["recommendation"],
                "Category": rec["category"],
                "Met": "Yes" if score.get("met", False) else "No",
                "Score": score.get("score", 0),
                "Evidence": score.get("evidence", ""),
                "Gap": score.get("gap", ""),
            })
        sheets["HLEG Matrix"] = hleg_rows

        # Sheet 2: Credibility Dimensions
        dimensions = data.get("dimensions", [])
        dim_rows: List[Dict[str, Any]] = []
        for dim in dimensions:
            dim_rows.append({
                "Dimension": dim.get("name", ""),
                "Score": dim.get("score", 0),
                "Max Score": dim.get("max_score", 100),
                "Weight": dim.get("weight", 0),
                "Weighted Score": round(dim.get("score", 0) * dim.get("weight", 0), 1),
                "Trend": dim.get("trend", ""),
                "Key Strength": dim.get("strength", ""),
                "Key Gap": dim.get("gap", ""),
            })
        sheets["Credibility Dimensions"] = dim_rows

        # Sheet 3: Governance Maturity
        gov = data.get("governance_maturity", {})
        gov_areas = gov.get("areas", [])
        gov_rows: List[Dict[str, Any]] = []
        for area in gov_areas:
            gov_rows.append({
                "Area": area.get("area", ""),
                "Maturity Level": area.get("level", ""),
                "Score": area.get("score", 0),
                "Target Level": area.get("target_level", ""),
                "Gap": area.get("gap", ""),
            })
        sheets["Governance Maturity"] = gov_rows

        # Sheet 4: Improvements
        improvements = data.get("improvements", [])
        imp_rows: List[Dict[str, Any]] = []
        for imp in improvements:
            imp_rows.append({
                "Priority": imp.get("priority", ""),
                "Dimension": imp.get("dimension", ""),
                "Recommendation": imp.get("recommendation", ""),
                "Expected Impact": imp.get("impact", ""),
                "Timeline": imp.get("timeline", ""),
                "Owner": imp.get("owner", ""),
            })
        sheets["Improvements"] = imp_rows

        # Sheet 5: Trend History
        trend = data.get("trend_history", [])
        trend_rows: List[Dict[str, Any]] = []
        for t in trend:
            trend_rows.append({
                "Assessment Date": t.get("date", ""),
                "Overall Score": t.get("overall_score", 0),
                "Rating": t.get("rating", ""),
                "HLEG Compliance (%)": t.get("hleg_pct", 0),
                "Key Change": t.get("key_change", ""),
            })
        sheets["Trend History"] = trend_rows

        return sheets

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _calc_overall_score(self, dimensions: List[Dict[str, Any]]) -> float:
        if not dimensions:
            return 0.0
        total_weighted = 0.0
        total_weight = 0.0
        for dim in dimensions:
            w = dim.get("weight", 1.0 / len(dimensions))
            total_weighted += dim.get("score", 0) * w
            total_weight += w
        return total_weighted / max(total_weight, 0.01) if total_weight else 0.0

    def _calc_hleg_compliance(self, hleg_scores: Dict[str, Dict]) -> Dict[str, Any]:
        met = sum(1 for v in hleg_scores.values() if v.get("met", False))
        return {"met": met, "pct": _safe_div(met, 10) * 100}

    def _score_to_rating(self, score: float) -> str:
        if score >= 90:
            return "EXEMPLARY"
        elif score >= 75:
            return "STRONG"
        elif score >= 60:
            return "GOOD"
        elif score >= 40:
            return "DEVELOPING"
        elif score >= 20:
            return "NASCENT"
        return "INSUFFICIENT"

    # ------------------------------------------------------------------ #
    #  Markdown sections                                                   #
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Race to Zero -- Credibility Assessment Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Assessment Date:** {data.get('assessment_date', ts)}  \n"
            f"**Assessor:** {data.get('assessor', 'GreenLang Platform')}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_overview(self, data: Dict[str, Any]) -> str:
        dimensions = data.get("dimensions", [])
        hleg_scores = data.get("hleg_scores", {})
        overall = self._calc_overall_score(dimensions)
        rating = self._score_to_rating(overall)
        hleg = self._calc_hleg_compliance(hleg_scores)

        return (
            f"## 1. Assessment Overview\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| **Overall Credibility Score** | **{_dec(overall, 1)}/100** |\n"
            f"| **Rating** | **{rating}** |\n"
            f"| HLEG Recommendations Met | {hleg['met']}/10 ({_pct(hleg['pct'])}) |\n"
            f"| Dimensions Assessed | {len(dimensions)} |\n"
            f"| Assessment Standard | UN HLEG Net-Zero Recommendations (2022) |\n"
            f"| Scoring Framework | GreenLang 7-Dimension Credibility Model |"
        )

    def _md_hleg_matrix(self, data: Dict[str, Any]) -> str:
        hleg_scores = data.get("hleg_scores", {})
        lines = [
            "## 2. HLEG 10 Recommendations Compliance Matrix\n",
            "| # | Recommendation | Category | Met | Score | Evidence |",
            "|---|---------------|----------|:---:|:-----:|----------|",
        ]
        for rec in HLEG_RECOMMENDATIONS:
            rid = rec["id"]
            score = hleg_scores.get(rid, {})
            met_text = "YES" if score.get("met", False) else "NO"
            lines.append(
                f"| {rid} | {rec['recommendation']} "
                f"| {rec['category']} "
                f"| {met_text} "
                f"| {_dec(score.get('score', 0), 0)}/100 "
                f"| {score.get('evidence', '')} |"
            )

        hleg = self._calc_hleg_compliance(hleg_scores)
        lines.append(
            f"\n**HLEG Compliance:** {hleg['met']}/10 recommendations met ({_pct(hleg['pct'])})"
        )
        return "\n".join(lines)

    def _md_credibility_breakdown(self, data: Dict[str, Any]) -> str:
        dimensions = data.get("dimensions", [])
        overall = self._calc_overall_score(dimensions)

        lines = [
            "## 3. Credibility Score Breakdown (7 Dimensions)\n",
            f"**Overall Weighted Score: {_dec(overall, 1)}/100 ({self._score_to_rating(overall)})**\n",
            "| # | Dimension | Score | Weight | Weighted | Trend | Key Strength | Key Gap |",
            "|---|-----------|:-----:|:------:|:--------:|:-----:|-------------|---------|",
        ]

        for i, dim in enumerate(dimensions, 1):
            score = dim.get("score", 0)
            weight = dim.get("weight", 0)
            weighted = score * weight
            lines.append(
                f"| {i} | {dim.get('name', '-')} "
                f"| {_dec(score, 1)} "
                f"| {_dec(weight, 2)} "
                f"| {_dec(weighted, 1)} "
                f"| {dim.get('trend', '-')} "
                f"| {dim.get('strength', '-')} "
                f"| {dim.get('gap', '-')} |"
            )

        if not dimensions:
            for i, dim_def in enumerate(CREDIBILITY_DIMENSIONS, 1):
                lines.append(
                    f"| {i} | {dim_def['name']} "
                    f"| -- | {_dec(dim_def['weight'], 2)} "
                    f"| -- | -- | -- | -- |"
                )

        return "\n".join(lines)

    def _md_ambition_assessment(self, data: Dict[str, Any]) -> str:
        ambition = data.get("ambition_assessment", {})
        return (
            f"## 4. Science-Based Ambition Assessment\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Target Pathway | {ambition.get('pathway', '1.5C no/limited overshoot')} |\n"
            f"| SBTi Validation | {ambition.get('sbti_validation', 'N/A')} |\n"
            f"| Interim Target Coverage | {ambition.get('interim_coverage', 'S1+S2')} |\n"
            f"| LT Target Coverage | {ambition.get('lt_coverage', 'S1+S2+S3')} |\n"
            f"| S3 Coverage | {_pct(ambition.get('scope3_coverage_pct', 0))} |\n"
            f"| Temperature Alignment | {ambition.get('temperature_alignment', 'N/A')} |\n"
            f"| Halving by 2030 | {ambition.get('halving_by_2030', 'N/A')} |\n"
            f"| Ambition Score | {_dec(ambition.get('score', 0), 1)}/100 |"
        )

    def _md_governance_maturity(self, data: Dict[str, Any]) -> str:
        gov = data.get("governance_maturity", {})
        areas = gov.get("areas", [])
        overall_level = gov.get("overall_level", "N/A")

        lines = [
            "## 5. Governance Maturity Evaluation\n",
            f"**Overall Maturity Level:** {overall_level}\n",
        ]

        if areas:
            lines.extend([
                "| Area | Maturity Level | Score | Target | Gap |",
                "|------|:-------------:|:-----:|:------:|-----|",
            ])
            for area in areas:
                lines.append(
                    f"| {area.get('area', '-')} "
                    f"| {area.get('level', '-')} "
                    f"| {_dec(area.get('score', 0), 0)}/100 "
                    f"| {area.get('target_level', '-')} "
                    f"| {area.get('gap', '-')} |"
                )
        else:
            lines.extend([
                "### Governance Maturity Areas\n",
                "| Area | Description |",
                "|------|-------------|",
                "| Board Oversight | Climate governance at board level |",
                "| Executive Accountability | C-suite climate responsibility |",
                "| Policy Integration | Climate in business strategy |",
                "| Risk Management | Climate risk in ERM framework |",
                "| Incentives | Climate targets in executive compensation |",
            ])

        return "\n".join(lines)

    def _md_transparency_verification(self, data: Dict[str, Any]) -> str:
        tv = data.get("transparency_verification", {})
        return (
            f"## 6. Transparency & Verification Status\n\n"
            f"### Transparency\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Public Reporting | {tv.get('public_reporting', 'N/A')} |\n"
            f"| CDP Disclosure | {tv.get('cdp_disclosure', 'N/A')} |\n"
            f"| TCFD Alignment | {tv.get('tcfd_alignment', 'N/A')} |\n"
            f"| Annual Report Disclosure | {tv.get('annual_report', 'N/A')} |\n"
            f"| Transparency Score | {_dec(tv.get('transparency_score', 0), 1)}/100 |\n\n"
            f"### Verification\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Verification Status | {tv.get('verification_status', 'N/A')} |\n"
            f"| Assurance Level | {tv.get('assurance_level', 'N/A')} |\n"
            f"| Verifier | {tv.get('verifier', 'N/A')} |\n"
            f"| Standard | {tv.get('verification_standard', 'ISO 14064-3:2019')} |\n"
            f"| Verification Score | {_dec(tv.get('verification_score', 0), 1)}/100 |"
        )

    def _md_offset_integrity(self, data: Dict[str, Any]) -> str:
        offset = data.get("offset_integrity", {})
        return (
            f"## 7. Offset & Removal Integrity\n\n"
            f"| Parameter | Value |\n|-----------|-------|\n"
            f"| Offset Usage for Interim Targets | {offset.get('interim_offsets', 'No (compliant)')} |\n"
            f"| Offset Volume (tCO2e) | {_dec_comma(offset.get('offset_volume_tco2e', 0))} |\n"
            f"| Removal Share | {_pct(offset.get('removal_share_pct', 0))} |\n"
            f"| ICVCM CCP Aligned | {_pct(offset.get('ccp_aligned_pct', 0))} |\n"
            f"| Registry Transparency | {offset.get('registry_transparency', 'N/A')} |\n"
            f"| Additionality Verified | {offset.get('additionality', 'N/A')} |\n"
            f"| Permanence Assessment | {offset.get('permanence', 'N/A')} |\n"
            f"| Integrity Score | {_dec(offset.get('score', 0), 1)}/100 |"
        )

    def _md_improvements(self, data: Dict[str, Any]) -> str:
        improvements = data.get("improvements", [])
        lines = ["## 8. Improvement Recommendations\n"]

        if improvements:
            lines.extend([
                "| Priority | Dimension | Recommendation | Impact | Timeline | Owner |",
                "|:--------:|-----------|----------------|--------|:--------:|-------|",
            ])
            for imp in improvements:
                lines.append(
                    f"| {imp.get('priority', '-')} "
                    f"| {imp.get('dimension', '-')} "
                    f"| {imp.get('recommendation', '-')} "
                    f"| {imp.get('impact', '-')} "
                    f"| {imp.get('timeline', '-')} "
                    f"| {imp.get('owner', '-')} |"
                )
        else:
            lines.extend([
                "1. **[HIGH]** Validate targets with SBTi to strengthen ambition score",
                "2. **[HIGH]** Engage third-party verifier for limited assurance",
                "3. **[MEDIUM]** Expand Scope 3 coverage to meet 67% threshold",
                "4. **[MEDIUM]** Strengthen board-level climate governance",
                "5. **[LOW]** Align lobbying activities with stated climate commitments",
            ])

        return "\n".join(lines)

    def _md_trend_history(self, data: Dict[str, Any]) -> str:
        trend = data.get("trend_history", [])
        lines = ["## 9. Credibility Trend History\n"]

        if trend:
            lines.extend([
                "| Date | Overall Score | Rating | HLEG Met | Key Change |",
                "|:----:|:------------:|:------:|:--------:|------------|",
            ])
            for t in trend:
                lines.append(
                    f"| {t.get('date', '-')} "
                    f"| {_dec(t.get('overall_score', 0), 1)} "
                    f"| {t.get('rating', '-')} "
                    f"| {t.get('hleg_met', 0)}/10 "
                    f"| {t.get('key_change', '-')} |"
                )
        else:
            lines.append("_First assessment -- no trend history available._")

        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-025 Race to Zero Pack on {ts}.*  \n"
            f"*Credibility framework aligned with UN HLEG recommendations (2022).*"
        )

    # ------------------------------------------------------------------ #
    #  HTML sections                                                       #
    # ------------------------------------------------------------------ #

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:20px;"
            "background:#f0f4f0;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;padding-left:12px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".gauge{width:140px;height:140px;border-radius:50%;margin:0 auto;"
            "display:flex;align-items:center;justify-content:center;font-size:2em;"
            "font-weight:700;color:#1b5e20;}"
            ".gauge-exemplary{border:8px solid #2e7d32;background:#e8f5e9;}"
            ".gauge-strong{border:8px solid #43a047;background:#e8f5e9;}"
            ".gauge-good{border:8px solid #66bb6a;background:#f1f8e9;}"
            ".gauge-developing{border:8px solid #ff9800;background:#fff3e0;}"
            ".gauge-nascent{border:8px solid #ef5350;background:#ffebee;}"
            ".gauge-insufficient{border:8px solid #d32f2f;background:#ffcdd2;}"
            ".progress-bar{background:#e0e0e0;border-radius:8px;height:20px;overflow:hidden;margin:4px 0;}"
            ".progress-fill{height:100%;border-radius:8px;}"
            ".fill-green{background:linear-gradient(90deg,#43a047,#66bb6a);}"
            ".fill-amber{background:linear-gradient(90deg,#ff9800,#ffb74d);}"
            ".fill-red{background:linear-gradient(90deg,#ef5350,#ef9a9a);}"
            ".badge-met{background:#43a047;color:#fff;padding:2px 10px;border-radius:4px;}"
            ".badge-not-met{background:#ef5350;color:#fff;padding:2px 10px;border-radius:4px;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Race to Zero -- Credibility Assessment</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Date:</strong> {ts}</p>'
        )

    def _html_overview(self, data: Dict[str, Any]) -> str:
        dimensions = data.get("dimensions", [])
        hleg_scores = data.get("hleg_scores", {})
        overall = self._calc_overall_score(dimensions)
        rating = self._score_to_rating(overall)
        hleg = self._calc_hleg_compliance(hleg_scores)
        gauge_class = f"gauge-{rating.lower()}"

        return (
            f'<h2>1. Assessment Overview</h2>\n'
            f'<div class="cards">\n'
            f'  <div class="card"><div class="gauge {gauge_class}">{_dec(overall, 0)}</div>'
            f'<div class="card-label">{rating}</div></div>\n'
            f'  <div class="card"><div class="card-label">HLEG Compliance</div>'
            f'<div class="card-value">{hleg["met"]}/10</div>{_pct(hleg["pct"])}</div>\n'
            f'  <div class="card"><div class="card-label">Dimensions</div>'
            f'<div class="card-value">{len(dimensions)}</div>assessed</div>\n'
            f'</div>'
        )

    def _html_hleg_matrix(self, data: Dict[str, Any]) -> str:
        hleg_scores = data.get("hleg_scores", {})
        rows = ""
        for rec in HLEG_RECOMMENDATIONS:
            rid = rec["id"]
            score = hleg_scores.get(rid, {})
            met = score.get("met", False)
            badge = "badge-met" if met else "badge-not-met"
            label = "MET" if met else "GAP"
            rows += (
                f'<tr><td>{rid}</td><td>{rec["recommendation"]}</td>'
                f'<td>{rec["category"]}</td>'
                f'<td><span class="{badge}">{label}</span></td></tr>\n'
            )
        return (
            f'<h2>2. HLEG Recommendations</h2>\n'
            f'<table><tr><th>ID</th><th>Recommendation</th><th>Category</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_credibility_breakdown(self, data: Dict[str, Any]) -> str:
        dimensions = data.get("dimensions", [])
        rows = ""
        for dim in dimensions:
            score = dim.get("score", 0)
            fill_class = "fill-green" if score >= 70 else ("fill-amber" if score >= 40 else "fill-red")
            rows += (
                f'<tr><td>{dim.get("name", "-")}</td>'
                f'<td><div class="progress-bar"><div class="progress-fill {fill_class}" '
                f'style="width:{score}%"></div></div> {_dec(score, 1)}/100</td>'
                f'<td>{_dec(dim.get("weight", 0), 2)}</td></tr>\n'
            )
        if not rows:
            rows = '<tr><td colspan="3"><em>Dimensions not assessed</em></td></tr>'
        return (
            f'<h2>3. Credibility Breakdown</h2>\n'
            f'<table><tr><th>Dimension</th><th>Score</th><th>Weight</th></tr>\n{rows}</table>'
        )

    def _html_ambition(self, data: Dict[str, Any]) -> str:
        amb = data.get("ambition_assessment", {})
        return (
            f'<h2>4. Ambition Assessment</h2>\n'
            f'<div class="cards">\n'
            f'  <div class="card"><div class="card-label">Pathway</div>'
            f'<div class="card-value">{amb.get("pathway", "1.5C")}</div></div>\n'
            f'  <div class="card"><div class="card-label">SBTi</div>'
            f'<div class="card-value">{amb.get("sbti_validation", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">S3 Coverage</div>'
            f'<div class="card-value">{_pct(amb.get("scope3_coverage_pct", 0))}</div></div>\n'
            f'  <div class="card"><div class="card-label">Score</div>'
            f'<div class="card-value">{_dec(amb.get("score", 0), 1)}</div>/100</div>\n'
            f'</div>'
        )

    def _html_governance(self, data: Dict[str, Any]) -> str:
        gov = data.get("governance_maturity", {})
        areas = gov.get("areas", [])
        rows = ""
        for area in areas:
            score = area.get("score", 0)
            fill_class = "fill-green" if score >= 70 else ("fill-amber" if score >= 40 else "fill-red")
            rows += (
                f'<tr><td>{area.get("area", "-")}</td><td>{area.get("level", "-")}</td>'
                f'<td><div class="progress-bar"><div class="progress-fill {fill_class}" '
                f'style="width:{score}%"></div></div></td></tr>\n'
            )
        if not rows:
            rows = '<tr><td colspan="3"><em>Governance maturity not assessed</em></td></tr>'
        return (
            f'<h2>5. Governance Maturity</h2>\n'
            f'<table><tr><th>Area</th><th>Level</th><th>Score</th></tr>\n{rows}</table>'
        )

    def _html_transparency(self, data: Dict[str, Any]) -> str:
        tv = data.get("transparency_verification", {})
        return (
            f'<h2>6. Transparency & Verification</h2>\n'
            f'<div class="cards">\n'
            f'  <div class="card"><div class="card-label">Transparency</div>'
            f'<div class="card-value">{_dec(tv.get("transparency_score", 0), 1)}</div>/100</div>\n'
            f'  <div class="card"><div class="card-label">Verification</div>'
            f'<div class="card-value">{_dec(tv.get("verification_score", 0), 1)}</div>/100</div>\n'
            f'  <div class="card"><div class="card-label">Assurance</div>'
            f'<div class="card-value">{tv.get("assurance_level", "N/A")}</div></div>\n'
            f'</div>'
        )

    def _html_improvements(self, data: Dict[str, Any]) -> str:
        improvements = data.get("improvements", [])
        items = ""
        for imp in improvements:
            items += (f'<li><strong>[{imp.get("priority", "")}]</strong> '
                      f'{imp.get("recommendation", "")} '
                      f'<em>({imp.get("dimension", "")})</em></li>\n')
        if not items:
            items = '<li><em>Improvement recommendations pending assessment</em></li>'
        return f'<h2>8. Improvements</h2>\n<ol>\n{items}</ol>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-025 Race to Zero Pack on {ts}'
            f'</div>'
        )
