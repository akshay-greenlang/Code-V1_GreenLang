# -*- coding: utf-8 -*-
"""
VCMIClaimsReportTemplate - VCMI Claims Code validation for PACK-022.

Renders a VCMI Claims Code report with claim eligibility, foundational criteria
checklists, tier requirements, credit quality analysis, greenwashing risk flags,
gap analysis, ISO 14068-1 comparison, recommendations, and re-validation schedule.

Sections:
    1. Claim Eligibility Summary
    2. Foundational Criteria Checklist (4 criteria)
    3. Evidence Assessment per Criterion
    4. Tier Requirements (Silver/Gold/Platinum)
    5. Credit Quality Analysis
    6. Greenwashing Risk Flags
    7. Gap to Next Tier
    8. ISO 14068-1 Comparison
    9. Recommendations
   10. Annual Re-validation Schedule

Author: GreenLang Team
Version: 22.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODULE_VERSION = "22.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    if isinstance(data, dict):
        raw = json.dumps(data, sort_keys=True, default=str)
    else:
        raw = str(data)
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


_FOUNDATIONAL_CRITERIA = [
    {"id": "FC1", "name": "Emissions Reduction Target", "description": "Company has set a science-based near-term emissions reduction target (SBTi or equivalent)"},
    {"id": "FC2", "name": "Progress on Target", "description": "Company is making demonstrable progress towards its emissions reduction target"},
    {"id": "FC3", "name": "Public Commitment", "description": "Company has made a public commitment to net-zero with a long-term target"},
    {"id": "FC4", "name": "Transparency & Reporting", "description": "Company publicly discloses emissions, targets, and credit retirement details"},
]


class VCMIClaimsReportTemplate:
    """
    VCMI Claims Code validation and certification report template.

    Validates eligibility for VCMI claims (Silver/Gold/Platinum), checks
    foundational criteria, assesses credit quality, flags greenwashing
    risks, and compares with ISO 14068-1 requirements.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_eligibility_summary(data),
            self._md_foundational_criteria(data),
            self._md_evidence_assessment(data),
            self._md_tier_requirements(data),
            self._md_credit_quality(data),
            self._md_greenwashing_risks(data),
            self._md_gap_analysis(data),
            self._md_iso_comparison(data),
            self._md_recommendations(data),
            self._md_revalidation_schedule(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_eligibility_summary(data),
            self._html_foundational_criteria(data),
            self._html_evidence_assessment(data),
            self._html_tier_requirements(data),
            self._html_credit_quality(data),
            self._html_greenwashing_risks(data),
            self._html_gap_analysis(data),
            self._html_iso_comparison(data),
            self._html_recommendations(data),
            self._html_revalidation_schedule(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>VCMI Claims Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        self.generated_at = _utcnow()
        criteria_results = data.get("criteria_results", {})
        pass_count = sum(
            1 for c in _FOUNDATIONAL_CRITERIA
            if criteria_results.get(c["id"], {}).get("pass", False)
        )

        result: Dict[str, Any] = {
            "template": "vcmi_claims_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "eligibility": {
                "tier_achieved": data.get("tier_achieved", "None"),
                "foundational_criteria_passed": pass_count,
                "foundational_criteria_total": len(_FOUNDATIONAL_CRITERIA),
                "eligible": pass_count == len(_FOUNDATIONAL_CRITERIA),
            },
            "criteria_results": criteria_results,
            "evidence_assessment": data.get("evidence_assessment", []),
            "tier_requirements": data.get("tier_requirements", []),
            "credit_quality": data.get("credit_quality", {}),
            "greenwashing_risks": data.get("greenwashing_risks", []),
            "gap_analysis": data.get("gap_analysis", {}),
            "iso_comparison": data.get("iso_comparison", []),
            "recommendations": data.get("recommendations", []),
            "revalidation_schedule": data.get("revalidation_schedule", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Markdown sections
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# VCMI Claims Code Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_eligibility_summary(self, data: Dict[str, Any]) -> str:
        tier = data.get("tier_achieved", "None")
        criteria_results = data.get("criteria_results", {})
        pass_count = sum(
            1 for c in _FOUNDATIONAL_CRITERIA
            if criteria_results.get(c["id"], {}).get("pass", False)
        )
        eligible = pass_count == len(_FOUNDATIONAL_CRITERIA)
        return (
            "## 1. Claim Eligibility Summary\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            f"| Tier Achieved | **{tier}** |\n"
            f"| Foundational Criteria | {pass_count}/{len(_FOUNDATIONAL_CRITERIA)} passed |\n"
            f"| Eligible for Claim | {'Yes' if eligible else 'No'} |\n"
            f"| Credit Volume Retired | {_dec_comma(data.get('credits_retired_tco2e', 0))} tCO2e |\n"
            f"| Residual Emissions | {_dec_comma(data.get('residual_emissions_tco2e', 0))} tCO2e |\n"
            f"| Offset Ratio | {_dec(data.get('offset_ratio_pct', 0))}% |"
        )

    def _md_foundational_criteria(self, data: Dict[str, Any]) -> str:
        criteria_results = data.get("criteria_results", {})
        lines = [
            "## 2. Foundational Criteria Checklist\n",
            "| ID | Criterion | Description | Status |",
            "|----|-----------|-------------|:------:|",
        ]
        for c in _FOUNDATIONAL_CRITERIA:
            result = criteria_results.get(c["id"], {})
            status = "PASS" if result.get("pass", False) else "FAIL"
            lines.append(
                f"| {c['id']} | {c['name']} | {c['description']} | {status} |"
            )
        return "\n".join(lines)

    def _md_evidence_assessment(self, data: Dict[str, Any]) -> str:
        evidence = data.get("evidence_assessment", [])
        lines = [
            "## 3. Evidence Assessment per Criterion\n",
            "| Criterion | Evidence Provided | Score (0-10) | Confidence | Gaps |",
            "|-----------|------------------|:------------:|:----------:|------|",
        ]
        for ev in evidence:
            lines.append(
                f"| {ev.get('criterion', '-')} | {ev.get('evidence', '-')} "
                f"| {_dec(ev.get('score', 0), 1)} "
                f"| {ev.get('confidence', '-')} "
                f"| {ev.get('gaps', '-')} |"
            )
        if not evidence:
            lines.append("| _No evidence data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_tier_requirements(self, data: Dict[str, Any]) -> str:
        tiers = data.get("tier_requirements", [])
        lines = [
            "## 4. Tier Requirements (Silver/Gold/Platinum)\n",
            "| Tier | Min Offset Ratio (%) | Credit Quality | Additional Requirements | Achieved |",
            "|------|:--------------------:|:--------------:|------------------------|:--------:|",
        ]
        for t in tiers:
            achieved = "Yes" if t.get("achieved", False) else "No"
            lines.append(
                f"| {t.get('tier', '-')} "
                f"| {_dec(t.get('min_offset_ratio_pct', 0))}% "
                f"| {t.get('credit_quality', '-')} "
                f"| {t.get('additional_requirements', '-')} "
                f"| {achieved} |"
            )
        if not tiers:
            lines.append("| _No tier data_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_credit_quality(self, data: Dict[str, Any]) -> str:
        cq = data.get("credit_quality", {})
        credits_list = cq.get("credits", [])
        lines = [
            "## 5. Credit Quality Analysis\n",
            f"**Overall Quality Score:** {_dec(cq.get('overall_score', 0), 1)} / 10.0  \n"
            f"**ICVCM CCP Eligible:** {cq.get('icvcm_eligible', 'N/A')}  \n"
            f"**Total Volume:** {_dec_comma(cq.get('total_volume_tco2e', 0))} tCO2e\n",
            "| Credit Type | Registry | Volume (tCO2e) | Vintage | Quality Score | CCP Eligible |",
            "|-------------|----------|---------------:|---------|:-------------:|:------------:|",
        ]
        for cr in credits_list:
            ccp = "Yes" if cr.get("ccp_eligible", False) else "No"
            lines.append(
                f"| {cr.get('type', '-')} | {cr.get('registry', '-')} "
                f"| {_dec_comma(cr.get('volume_tco2e', 0))} "
                f"| {cr.get('vintage', '-')} "
                f"| {_dec(cr.get('quality_score', 0), 1)} "
                f"| {ccp} |"
            )
        if not credits_list:
            lines.append("| _No credit data_ | - | - | - | - | - |")
        return "\n".join(lines)

    def _md_greenwashing_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("greenwashing_risks", [])
        lines = [
            "## 6. Greenwashing Risk Flags\n",
            "| # | Risk Flag | Severity | Description | Mitigation |",
            "|---|-----------|:--------:|-------------|------------|",
        ]
        for i, r in enumerate(risks, 1):
            lines.append(
                f"| {i} | {r.get('flag', '-')} "
                f"| {r.get('severity', '-')} "
                f"| {r.get('description', '-')} "
                f"| {r.get('mitigation', '-')} |"
            )
        if not risks:
            lines.append("| - | _No greenwashing risks identified_ | - | - | - |")
        return "\n".join(lines)

    def _md_gap_analysis(self, data: Dict[str, Any]) -> str:
        gap = data.get("gap_analysis", {})
        current_tier = gap.get("current_tier", "None")
        next_tier = gap.get("next_tier", "Silver")
        gaps_list = gap.get("gaps", [])
        lines = [
            "## 7. Gap to Next Tier\n",
            f"**Current Tier:** {current_tier}  \n"
            f"**Next Tier:** {next_tier}\n",
            "| # | Gap | Category | Effort | Priority |",
            "|---|-----|----------|:------:|:--------:|",
        ]
        for i, g in enumerate(gaps_list, 1):
            lines.append(
                f"| {i} | {g.get('gap', '-')} "
                f"| {g.get('category', '-')} "
                f"| {g.get('effort', '-')} "
                f"| {g.get('priority', '-')} |"
            )
        if not gaps_list:
            lines.append("| - | _No gaps identified_ | - | - | - |")
        return "\n".join(lines)

    def _md_iso_comparison(self, data: Dict[str, Any]) -> str:
        comparison = data.get("iso_comparison", [])
        lines = [
            "## 8. ISO 14068-1 Comparison\n",
            "| Requirement | VCMI Claims Code | ISO 14068-1 | Status |",
            "|-------------|:----------------:|:-----------:|:------:|",
        ]
        for c in comparison:
            lines.append(
                f"| {c.get('requirement', '-')} "
                f"| {c.get('vcmi', '-')} "
                f"| {c.get('iso', '-')} "
                f"| {c.get('status', '-')} |"
            )
        if not comparison:
            lines.append("| _No comparison data_ | - | - | - |")
        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        lines = ["## 9. Recommendations\n"]
        if recs:
            for i, rec in enumerate(recs, 1):
                priority = rec.get("priority", "MEDIUM")
                lines.append(f"### {i}. [{priority}] {rec.get('title', 'Recommendation')}\n")
                lines.append(f"{rec.get('description', '')}\n")
                actions = rec.get("actions", [])
                for action in actions:
                    lines.append(f"  - {action}")
                lines.append("")
        else:
            lines.append("_No recommendations at this time._")
        return "\n".join(lines)

    def _md_revalidation_schedule(self, data: Dict[str, Any]) -> str:
        schedule = data.get("revalidation_schedule", [])
        lines = [
            "## 10. Annual Re-validation Schedule\n",
            "| Activity | Due Date | Owner | Frequency | Status |",
            "|----------|----------|-------|:---------:|:------:|",
        ]
        for s in schedule:
            lines.append(
                f"| {s.get('activity', '-')} "
                f"| {s.get('due_date', '-')} "
                f"| {s.get('owner', '-')} "
                f"| {s.get('frequency', '-')} "
                f"| {s.get('status', '-')} |"
            )
        if not schedule:
            lines.append("| _No schedule defined_ | - | - | - | - |")
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-022 Net Zero Acceleration Pack on {ts}*  \n"
            f"*VCMI Claims Code of Practice (June 2023) and ISO 14068-1:2023.*"
        )

    # ------------------------------------------------------------------
    # HTML sections
    # ------------------------------------------------------------------

    def _css(self) -> str:
        return (
            "body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;"
            "padding:20px;background:#f0f4f0;color:#1a1a2e;}"
            ".report{max-width:1200px;margin:0 auto;background:#fff;padding:40px;"
            "border-radius:12px;box-shadow:0 2px 12px rgba(0,0,0,0.08);}"
            "h1{color:#1b5e20;border-bottom:3px solid #2e7d32;padding-bottom:12px;"
            "font-size:1.8em;}"
            "h2{color:#2e7d32;margin-top:35px;border-left:4px solid #43a047;"
            "padding-left:12px;font-size:1.3em;}"
            "h3{color:#388e3c;margin-top:20px;}"
            "table{width:100%;border-collapse:collapse;margin:15px 0;font-size:0.9em;}"
            "th,td{border:1px solid #c8e6c9;padding:10px 14px;text-align:left;}"
            "th{background:#e8f5e9;font-weight:600;color:#1b5e20;}"
            "tr:nth-child(even){background:#f9fbe7;}"
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;letter-spacing:0.5px;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".card-unit{font-size:0.75em;color:#689f38;}"
            ".pass{color:#1b5e20;font-weight:700;font-size:1.1em;}"
            ".fail{color:#c62828;font-weight:700;font-size:1.1em;}"
            ".tier-silver{background:linear-gradient(135deg,#e0e0e0,#bdbdbd);}"
            ".tier-gold{background:linear-gradient(135deg,#fff8e1,#ffd54f);}"
            ".tier-platinum{background:linear-gradient(135deg,#e8eaf6,#9fa8da);}"
            ".risk-high{color:#c62828;font-weight:600;}"
            ".risk-medium{color:#e65100;font-weight:600;}"
            ".risk-low{color:#1b5e20;font-weight:600;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>VCMI Claims Code Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_eligibility_summary(self, data: Dict[str, Any]) -> str:
        tier = data.get("tier_achieved", "None")
        criteria_results = data.get("criteria_results", {})
        pass_count = sum(
            1 for c in _FOUNDATIONAL_CRITERIA
            if criteria_results.get(c["id"], {}).get("pass", False)
        )
        eligible = pass_count == len(_FOUNDATIONAL_CRITERIA)
        tier_cls = (
            "tier-platinum" if tier.lower() == "platinum"
            else "tier-gold" if tier.lower() == "gold"
            else "tier-silver" if tier.lower() == "silver"
            else ""
        )
        return (
            f'<h2>1. Claim Eligibility Summary</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card {tier_cls}"><div class="card-label">Tier Achieved</div>'
            f'<div class="card-value">{tier}</div></div>\n'
            f'  <div class="card"><div class="card-label">Criteria Passed</div>'
            f'<div class="card-value">{pass_count}/{len(_FOUNDATIONAL_CRITERIA)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Eligible</div>'
            f'<div class="card-value">{"Yes" if eligible else "No"}</div></div>\n'
            f'  <div class="card"><div class="card-label">Credits Retired</div>'
            f'<div class="card-value">{_dec_comma(data.get("credits_retired_tco2e", 0))}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">Offset Ratio</div>'
            f'<div class="card-value">{_dec(data.get("offset_ratio_pct", 0))}%</div></div>\n'
            f'</div>'
        )

    def _html_foundational_criteria(self, data: Dict[str, Any]) -> str:
        criteria_results = data.get("criteria_results", {})
        rows = ""
        for c in _FOUNDATIONAL_CRITERIA:
            result = criteria_results.get(c["id"], {})
            passed = result.get("pass", False)
            cls = "pass" if passed else "fail"
            icon = "&#10004;" if passed else "&#10008;"
            rows += (
                f'<tr><td>{c["id"]}</td><td>{c["name"]}</td>'
                f'<td>{c["description"]}</td>'
                f'<td class="{cls}">{icon}</td></tr>\n'
            )
        return (
            f'<h2>2. Foundational Criteria Checklist</h2>\n'
            f'<table>\n'
            f'<tr><th>ID</th><th>Criterion</th><th>Description</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_evidence_assessment(self, data: Dict[str, Any]) -> str:
        evidence = data.get("evidence_assessment", [])
        rows = ""
        for ev in evidence:
            score = float(Decimal(str(ev.get("score", 0))))
            score_cls = "pass" if score >= 7 else "fail" if score < 5 else ""
            rows += (
                f'<tr><td>{ev.get("criterion", "-")}</td>'
                f'<td>{ev.get("evidence", "-")}</td>'
                f'<td class="{score_cls}">{_dec(score, 1)}</td>'
                f'<td>{ev.get("confidence", "-")}</td>'
                f'<td>{ev.get("gaps", "-")}</td></tr>\n'
            )
        return (
            f'<h2>3. Evidence Assessment</h2>\n'
            f'<table>\n'
            f'<tr><th>Criterion</th><th>Evidence</th><th>Score</th>'
            f'<th>Confidence</th><th>Gaps</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_tier_requirements(self, data: Dict[str, Any]) -> str:
        tiers = data.get("tier_requirements", [])
        rows = ""
        for t in tiers:
            achieved = t.get("achieved", False)
            cls = "pass" if achieved else "fail"
            icon = "&#10004;" if achieved else "&#10008;"
            tier_name = t.get("tier", "")
            tier_cls = (
                "tier-platinum" if "platinum" in tier_name.lower()
                else "tier-gold" if "gold" in tier_name.lower()
                else "tier-silver" if "silver" in tier_name.lower()
                else ""
            )
            rows += (
                f'<tr class="{tier_cls}"><td><strong>{tier_name}</strong></td>'
                f'<td>{_dec(t.get("min_offset_ratio_pct", 0))}%</td>'
                f'<td>{t.get("credit_quality", "-")}</td>'
                f'<td>{t.get("additional_requirements", "-")}</td>'
                f'<td class="{cls}">{icon}</td></tr>\n'
            )
        return (
            f'<h2>4. Tier Requirements</h2>\n'
            f'<table>\n'
            f'<tr><th>Tier</th><th>Min Offset Ratio</th><th>Credit Quality</th>'
            f'<th>Additional Requirements</th><th>Achieved</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_credit_quality(self, data: Dict[str, Any]) -> str:
        cq = data.get("credit_quality", {})
        credits_list = cq.get("credits", [])
        rows = ""
        for cr in credits_list:
            ccp = cr.get("ccp_eligible", False)
            ccp_cls = "pass" if ccp else "fail"
            ccp_icon = "&#10004;" if ccp else "&#10008;"
            rows += (
                f'<tr><td>{cr.get("type", "-")}</td><td>{cr.get("registry", "-")}</td>'
                f'<td>{_dec_comma(cr.get("volume_tco2e", 0))}</td>'
                f'<td>{cr.get("vintage", "-")}</td>'
                f'<td>{_dec(cr.get("quality_score", 0), 1)}</td>'
                f'<td class="{ccp_cls}">{ccp_icon}</td></tr>\n'
            )
        return (
            f'<h2>5. Credit Quality Analysis</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Quality Score</div>'
            f'<div class="card-value">{_dec(cq.get("overall_score", 0), 1)}</div>'
            f'<div class="card-unit">/ 10.0</div></div>\n'
            f'  <div class="card"><div class="card-label">ICVCM CCP</div>'
            f'<div class="card-value">{cq.get("icvcm_eligible", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Total Volume</div>'
            f'<div class="card-value">{_dec_comma(cq.get("total_volume_tco2e", 0))}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Type</th><th>Registry</th><th>Volume (tCO2e)</th>'
            f'<th>Vintage</th><th>Quality</th><th>CCP</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_greenwashing_risks(self, data: Dict[str, Any]) -> str:
        risks = data.get("greenwashing_risks", [])
        rows = ""
        for i, r in enumerate(risks, 1):
            severity = r.get("severity", "Medium")
            s_cls = (
                "risk-high" if severity.lower() == "high"
                else "risk-low" if severity.lower() == "low"
                else "risk-medium"
            )
            rows += (
                f'<tr><td>{i}</td><td>{r.get("flag", "-")}</td>'
                f'<td class="{s_cls}">{severity}</td>'
                f'<td>{r.get("description", "-")}</td>'
                f'<td>{r.get("mitigation", "-")}</td></tr>\n'
            )
        return (
            f'<h2>6. Greenwashing Risk Flags</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Risk Flag</th><th>Severity</th>'
            f'<th>Description</th><th>Mitigation</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_gap_analysis(self, data: Dict[str, Any]) -> str:
        gap = data.get("gap_analysis", {})
        gaps_list = gap.get("gaps", [])
        rows = ""
        for i, g in enumerate(gaps_list, 1):
            rows += (
                f'<tr><td>{i}</td><td>{g.get("gap", "-")}</td>'
                f'<td>{g.get("category", "-")}</td>'
                f'<td>{g.get("effort", "-")}</td>'
                f'<td>{g.get("priority", "-")}</td></tr>\n'
            )
        return (
            f'<h2>7. Gap to Next Tier</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Current Tier</div>'
            f'<div class="card-value">{gap.get("current_tier", "None")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Next Tier</div>'
            f'<div class="card-value">{gap.get("next_tier", "Silver")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Gaps to Close</div>'
            f'<div class="card-value">{len(gaps_list)}</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Gap</th><th>Category</th>'
            f'<th>Effort</th><th>Priority</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_iso_comparison(self, data: Dict[str, Any]) -> str:
        comparison = data.get("iso_comparison", [])
        rows = ""
        for c in comparison:
            status = c.get("status", "N/A")
            cls = "pass" if status.lower() in ("aligned", "met", "pass") else "fail" if status.lower() in ("gap", "fail") else ""
            rows += (
                f'<tr><td>{c.get("requirement", "-")}</td>'
                f'<td>{c.get("vcmi", "-")}</td>'
                f'<td>{c.get("iso", "-")}</td>'
                f'<td class="{cls}">{status}</td></tr>\n'
            )
        return (
            f'<h2>8. ISO 14068-1 Comparison</h2>\n'
            f'<table>\n'
            f'<tr><th>Requirement</th><th>VCMI</th><th>ISO 14068-1</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        items = ""
        for i, rec in enumerate(recs, 1):
            priority = rec.get("priority", "MEDIUM")
            pri_cls = "risk-high" if priority == "HIGH" else "risk-low" if priority == "LOW" else "risk-medium"
            actions_html = ""
            if rec.get("actions"):
                actions_html = "<ul>" + "".join(f"<li>{a}</li>" for a in rec["actions"]) + "</ul>"
            items += (
                f'<div style="margin:12px 0;padding:12px;border:1px solid #c8e6c9;border-radius:8px;">'
                f'<strong>{i}. <span class="{pri_cls}">[{priority}]</span> '
                f'{rec.get("title", "")}</strong>'
                f'<p>{rec.get("description", "")}</p>'
                f'{actions_html}</div>\n'
            )
        return f'<h2>9. Recommendations</h2>\n{items}'

    def _html_revalidation_schedule(self, data: Dict[str, Any]) -> str:
        schedule = data.get("revalidation_schedule", [])
        rows = ""
        for s in schedule:
            rows += (
                f'<tr><td>{s.get("activity", "-")}</td>'
                f'<td>{s.get("due_date", "-")}</td>'
                f'<td>{s.get("owner", "-")}</td>'
                f'<td>{s.get("frequency", "-")}</td>'
                f'<td>{s.get("status", "-")}</td></tr>\n'
            )
        return (
            f'<h2>10. Annual Re-validation Schedule</h2>\n'
            f'<table>\n'
            f'<tr><th>Activity</th><th>Due Date</th><th>Owner</th>'
            f'<th>Frequency</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-022 Net Zero '
            f'Acceleration Pack on {ts}<br>'
            f'VCMI Claims Code of Practice (June 2023) and ISO 14068-1:2023.</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
