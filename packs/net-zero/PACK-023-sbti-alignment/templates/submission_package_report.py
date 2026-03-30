# -*- coding: utf-8 -*-
"""
SubmissionPackageReportTemplate - SBTi submission package for PACK-023.

Renders a complete SBTi submission package document formatted per SBTi
submission template requirements, covering target summary, 42-criterion
validation matrix, supporting evidence index, governance documentation,
emissions inventory summary, pathway methodology description, and
overall readiness assessment.

Sections:
    1. Submission Overview (company info, submission type, status)
    2. Target Summary (all targets with parameters)
    3. 42-Criterion Validation Matrix (C1-C28 + NZ-C1 to NZ-C14)
    4. Supporting Evidence Index (document inventory)
    5. Governance Documentation (board approval, roles, policies)
    6. Emissions Inventory Summary (S1/S2/S3 base year & current)
    7. Pathway Methodology (ACA/SDA/FLAG rationale)
    8. Readiness Assessment (submission-ready scoring)

Author: GreenLang Team
Version: 23.0.0
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

_MODULE_VERSION = "23.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if isinstance(data, dict):
        raw = json.dumps(data, sort_keys=True, default=str)
    else:
        raw = str(data)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _dec(val: Any, places: int = 2) -> str:
    """Format a value as a Decimal string with fixed decimal places."""
    try:
        d = Decimal(str(val))
        q = "0." + "0" * places if places > 0 else "0"
        return str(d.quantize(Decimal(q), rounding=ROUND_HALF_UP))
    except Exception:
        return str(val)

def _dec_comma(val: Any, places: int = 2) -> str:
    """Format a Decimal value with thousands separator."""
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

def _pct(val: Any) -> str:
    """Format a value as percentage string."""
    try:
        return _dec(val, 1) + "%"
    except Exception:
        return str(val)

def _status_icon(status: str) -> str:
    """Return a text-based status indicator for markdown."""
    s = str(status).upper()
    if s == "PASS":
        return "PASS"
    elif s == "FAIL":
        return "FAIL"
    elif s in ("WARNING", "WARN"):
        return "WARN"
    elif s in ("NA", "N/A"):
        return "N/A"
    elif s in ("READY", "COMPLETE"):
        return "READY"
    elif s in ("PENDING", "IN_PROGRESS"):
        return "PENDING"
    return status

class SubmissionPackageReportTemplate:
    """
    SBTi submission package report template.

    Renders the complete SBTi submission package document with target
    definitions, 42-criterion validation matrix, supporting evidence
    index, governance documentation, emissions inventory summary,
    pathway methodology, and readiness assessment formatted per SBTi
    submission template requirements.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize SubmissionPackageReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render submission package report as Markdown."""
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_submission_overview(data),
            self._md_target_summary(data),
            self._md_validation_matrix(data),
            self._md_evidence_index(data),
            self._md_governance(data),
            self._md_emissions_inventory(data),
            self._md_pathway_methodology(data),
            self._md_readiness_assessment(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render submission package report as self-contained HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_submission_overview(data),
            self._html_target_summary(data),
            self._html_validation_matrix(data),
            self._html_evidence_index(data),
            self._html_governance(data),
            self._html_emissions_inventory(data),
            self._html_pathway_methodology(data),
            self._html_readiness_assessment(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>SBTi Submission Package</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render submission package report as structured JSON."""
        self.generated_at = utcnow()
        submission = data.get("submission", {})
        targets = data.get("targets", [])
        near_term = data.get("near_term_criteria", [])
        net_zero = data.get("net_zero_criteria", [])
        evidence = data.get("evidence_index", [])
        governance = data.get("governance", {})
        inventory = data.get("emissions_inventory", {})
        pathway = data.get("pathway_methodology", {})
        readiness = data.get("readiness", {})

        all_criteria = near_term + net_zero
        passed = len([c for c in all_criteria if str(c.get("status", "")).upper() == "PASS"])
        total = len(all_criteria)

        result: Dict[str, Any] = {
            "template": "submission_package_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "submission": submission,
            "targets": targets,
            "validation_summary": {
                "total_criteria": total,
                "passed": passed,
                "failed": total - passed,
                "score_pct": round(passed / total * 100, 1) if total > 0 else 0,
            },
            "near_term_criteria": near_term,
            "net_zero_criteria": net_zero,
            "evidence_index": evidence,
            "governance": governance,
            "emissions_inventory": inventory,
            "pathway_methodology": pathway,
            "readiness": readiness,
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
            f"# SBTi Submission Package\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Document Type:** SBTi Target Validation Submission\n\n---"
        )

    def _md_submission_overview(self, data: Dict[str, Any]) -> str:
        sub = data.get("submission", {})
        return (
            f"## 1. Submission Overview\n\n"
            f"| Field | Value |\n|-------|-------|\n"
            f"| Company Name | {sub.get('company_name', data.get('org_name', 'N/A'))} |\n"
            f"| Submission Type | {sub.get('submission_type', 'Initial Validation')} |\n"
            f"| Target Type | {sub.get('target_type', 'Near-term + Net-zero')} |\n"
            f"| Sector | {sub.get('sector', 'N/A')} |\n"
            f"| ISIC Code | {sub.get('isic_code', 'N/A')} |\n"
            f"| Revenue (M USD) | {_dec_comma(sub.get('revenue_musd', 0), 0)} |\n"
            f"| Employees | {_dec_comma(sub.get('employees', 0), 0)} |\n"
            f"| Consolidation Approach | {sub.get('consolidation_approach', 'Operational Control')} |\n"
            f"| Parent Company | {sub.get('parent_company', 'N/A')} |\n"
            f"| Submission Date | {sub.get('submission_date', 'N/A')} |\n"
            f"| Validation Body | {sub.get('validation_body', 'SBTi')} |\n"
            f"| Submission Status | {sub.get('status', 'Draft')} |\n"
            f"| Contact Name | {sub.get('contact_name', 'N/A')} |\n"
            f"| Contact Email | {sub.get('contact_email', 'N/A')} |"
        )

    def _md_target_summary(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        lines = [
            "## 2. Target Summary\n",
            "All targets submitted for SBTi validation.\n",
            "| # | Target Name | Type | Scope | Base Year | Target Year "
            "| Reduction (%) | Pathway | Coverage (%) "
            "| Boundary | Status |",
            "|---|-------------|------|-------|:---------:|:-----------:"
            "|:-------------:|---------|:------------:"
            "|----------|--------|",
        ]
        for i, t in enumerate(targets, 1):
            lines.append(
                f"| {i} | {t.get('name', '-')} "
                f"| {t.get('type', '-')} "
                f"| {t.get('scope', '-')} "
                f"| {t.get('base_year', '-')} "
                f"| {t.get('target_year', '-')} "
                f"| {_pct(t.get('reduction_pct', 0))} "
                f"| {t.get('pathway', '-')} "
                f"| {_pct(t.get('coverage_pct', 0))} "
                f"| {t.get('boundary', '-')} "
                f"| {t.get('status', '-')} |"
            )
        if not targets:
            lines.append(
                "| - | _No targets defined_ | - | - | - | - | - | - | - | - | - |"
            )

        # Target formulation per SBTi template
        formulations = data.get("target_formulations", [])
        if formulations:
            lines.append("")
            lines.append("### Target Formulations (SBTi Template Format)\n")
            for f in formulations:
                lines.append(
                    f"**{f.get('target_id', 'T')}:** "
                    f"_{f.get('formulation', 'N/A')}_\n"
                )

        return "\n".join(lines)

    def _md_validation_matrix(self, data: Dict[str, Any]) -> str:
        near_term = data.get("near_term_criteria", [])
        net_zero = data.get("net_zero_criteria", [])
        all_c = near_term + net_zero
        passed = len([c for c in all_c if str(c.get("status", "")).upper() == "PASS"])
        total = len(all_c)
        score = round(passed / total * 100, 1) if total > 0 else 0

        lines = [
            "## 3. 42-Criterion Validation Matrix\n",
            f"**Validation Score:** {_dec(score, 1)}% ({passed}/{total} criteria passed)\n",
            "### Near-Term Criteria (C1-C28)\n",
            "| Criterion | Description | Status | Evidence Ref | Notes |",
            "|-----------|-------------|:------:|:------------:|-------|",
        ]
        for c in near_term:
            lines.append(
                f"| {c.get('id', '-')} | {c.get('description', '-')} "
                f"| {_status_icon(c.get('status', 'N/A'))} "
                f"| {c.get('evidence_ref', '-')} "
                f"| {c.get('notes', '-')} |"
            )
        if not near_term:
            lines.append("| - | _No near-term criteria_ | - | - | - |")

        lines.append("")
        lines.append("### Net-Zero Criteria (NZ-C1 to NZ-C14)\n")
        lines.append("| Criterion | Description | Status | Evidence Ref | Notes |")
        lines.append("|-----------|-------------|:------:|:------------:|-------|")
        for c in net_zero:
            lines.append(
                f"| {c.get('id', '-')} | {c.get('description', '-')} "
                f"| {_status_icon(c.get('status', 'N/A'))} "
                f"| {c.get('evidence_ref', '-')} "
                f"| {c.get('notes', '-')} |"
            )
        if not net_zero:
            lines.append("| - | _No net-zero criteria_ | - | - | - |")

        # Group summary
        lines.append("")
        lines.append("### Criteria Group Summary\n")
        lines.append("| Group | Range | Passed | Total | Status |")
        lines.append("|-------|:-----:|:------:|:-----:|--------|")
        groups = [
            ("Boundary & Coverage", "C1-C7", near_term[:7]),
            ("Base Year & Inventory", "C8-C14", near_term[7:14]),
            ("Target Ambition", "C15-C21", near_term[14:21]),
            ("Reporting & Disclosure", "C22-C28", near_term[21:28]),
            ("Net-Zero Requirements", "NZ-C1 to NZ-C14", net_zero),
        ]
        for name, rng, criteria in groups:
            g_passed = len([
                c for c in criteria
                if str(c.get("status", "")).upper() == "PASS"
            ])
            g_total = len(criteria)
            g_status = "PASS" if g_passed == g_total and g_total > 0 else "FAIL"
            lines.append(
                f"| {name} | {rng} | {g_passed} | {g_total} | {g_status} |"
            )

        return "\n".join(lines)

    def _md_evidence_index(self, data: Dict[str, Any]) -> str:
        evidence = data.get("evidence_index", [])
        lines = [
            "## 4. Supporting Evidence Index\n",
            "Index of all supporting documents for the submission.\n",
            "| Ref | Document Title | Type | Criteria Covered "
            "| Date | Version | Status |",
            "|:---:|---------------|------|:-----------------:"
            "|:----:|:-------:|--------|",
        ]
        for e in evidence:
            lines.append(
                f"| {e.get('ref', '-')} "
                f"| {e.get('title', '-')} "
                f"| {e.get('type', '-')} "
                f"| {e.get('criteria_covered', '-')} "
                f"| {e.get('date', '-')} "
                f"| {e.get('version', '-')} "
                f"| {_status_icon(e.get('status', 'N/A'))} |"
            )
        if not evidence:
            lines.append(
                "| - | _No evidence documents_ | - | - | - | - | - |"
            )

        # Evidence completeness
        completeness = data.get("evidence_completeness", {})
        if completeness:
            lines.append("")
            lines.append(
                f"**Documents Provided:** {completeness.get('provided', 0)} / "
                f"{completeness.get('required', 0)}  \n"
                f"**Completeness:** {_pct(completeness.get('completeness_pct', 0))}  \n"
                f"**Missing Documents:** {completeness.get('missing', 'None')}"
            )
        return "\n".join(lines)

    def _md_governance(self, data: Dict[str, Any]) -> str:
        gov = data.get("governance", {})
        roles = gov.get("roles", [])
        policies = gov.get("policies", [])
        lines = [
            "## 5. Governance Documentation\n",
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Board Approval | {gov.get('board_approval', 'N/A')} |\n"
            f"| Board Approval Date | {gov.get('board_approval_date', 'N/A')} |\n"
            f"| CEO/Executive Sign-Off | {gov.get('ceo_signoff', 'N/A')} |\n"
            f"| Climate Governance Framework | {gov.get('climate_framework', 'N/A')} |\n"
            f"| Target Review Frequency | {gov.get('review_frequency', 'Annual')} |\n"
            f"| Dedicated Sustainability Team | {gov.get('sustainability_team', 'N/A')} |\n"
            f"| External Verification | {gov.get('external_verification', 'N/A')} |\n"
            f"| Public Commitment | {gov.get('public_commitment', 'N/A')} |",
        ]

        if roles:
            lines.append("")
            lines.append("### Key Roles & Responsibilities\n")
            lines.append("| Role | Name | Responsibility | Authority |")
            lines.append("|------|------|----------------|-----------|")
            for r in roles:
                lines.append(
                    f"| {r.get('role', '-')} "
                    f"| {r.get('name', '-')} "
                    f"| {r.get('responsibility', '-')} "
                    f"| {r.get('authority', '-')} |"
                )

        if policies:
            lines.append("")
            lines.append("### Supporting Policies\n")
            lines.append("| Policy | Status | Last Updated | Owner |")
            lines.append("|--------|:------:|:------------:|-------|")
            for p in policies:
                lines.append(
                    f"| {p.get('policy', '-')} "
                    f"| {_status_icon(p.get('status', 'N/A'))} "
                    f"| {p.get('last_updated', '-')} "
                    f"| {p.get('owner', '-')} |"
                )

        return "\n".join(lines)

    def _md_emissions_inventory(self, data: Dict[str, Any]) -> str:
        inv = data.get("emissions_inventory", {})
        base = inv.get("base_year", {})
        current = inv.get("current_year", {})
        lines = [
            "## 6. Emissions Inventory Summary\n",
            "### Base Year vs Current Year\n",
            "| Scope | Base Year (tCO2e) | Current Year (tCO2e) "
            "| Change (%) | Method |",
            "|-------|------------------:|:--------------------:"
            "|:----------:|--------|",
        ]
        scopes = [
            ("Scope 1", "scope1"),
            ("Scope 2 (Location)", "scope2_location"),
            ("Scope 2 (Market)", "scope2_market"),
            ("Scope 3", "scope3"),
            ("Total", "total"),
        ]
        for label, key in scopes:
            base_val = float(base.get(f"{key}_tco2e", 0))
            curr_val = float(current.get(f"{key}_tco2e", 0))
            change = (
                (curr_val - base_val) / base_val * 100
                if base_val > 0 else 0
            )
            method = inv.get(f"{key}_method", "-")
            lines.append(
                f"| {label} "
                f"| {_dec_comma(base_val, 0)} "
                f"| {_dec_comma(curr_val, 0)} "
                f"| {'+' if change > 0 else ''}{_pct(change)} "
                f"| {method} |"
            )

        # S3 category breakdown
        s3_cats = inv.get("scope3_categories", [])
        if s3_cats:
            lines.append("")
            lines.append("### Scope 3 Category Detail\n")
            lines.append(
                "| Cat # | Category | Base (tCO2e) | Current (tCO2e) "
                "| % of S3 | Included | Method |"
            )
            lines.append(
                "|:-----:|----------|:-----------:|:---------------:"
                "|:-------:|:--------:|--------|"
            )
            for cat in s3_cats:
                lines.append(
                    f"| {cat.get('number', '-')} "
                    f"| {cat.get('name', '-')} "
                    f"| {_dec_comma(cat.get('base_tco2e', 0), 0)} "
                    f"| {_dec_comma(cat.get('current_tco2e', 0), 0)} "
                    f"| {_pct(cat.get('pct_of_s3', 0))} "
                    f"| {cat.get('included', '-')} "
                    f"| {cat.get('method', '-')} |"
                )

        # Verification
        verification = inv.get("verification", {})
        if verification:
            lines.append("")
            lines.append("### Verification Status\n")
            lines.append(
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Verification Body | {verification.get('body', 'N/A')} |\n"
                f"| Assurance Level | {verification.get('assurance_level', 'N/A')} |\n"
                f"| Standard | {verification.get('standard', 'N/A')} |\n"
                f"| Scope Covered | {verification.get('scope_covered', 'N/A')} |\n"
                f"| Report Date | {verification.get('report_date', 'N/A')} |"
            )

        return "\n".join(lines)

    def _md_pathway_methodology(self, data: Dict[str, Any]) -> str:
        pm = data.get("pathway_methodology", {})
        methods = pm.get("methods", [])
        lines = [
            "## 7. Pathway Methodology\n",
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Primary Pathway | {pm.get('primary_pathway', 'N/A')} |\n"
            f"| Secondary Pathway | {pm.get('secondary_pathway', 'N/A')} |\n"
            f"| Scenario | {pm.get('scenario', 'IEA NZE 2050')} |\n"
            f"| Temperature Alignment | {pm.get('temperature', '1.5C')} |\n"
            f"| Cross-Sector Model | {pm.get('cross_sector_model', 'N/A')} |\n"
            f"| Sector-Specific Model | {pm.get('sector_specific_model', 'N/A')} |",
        ]

        if methods:
            lines.append("")
            lines.append("### Method Details\n")
            lines.append(
                "| Scope | Method | Rationale | Benchmark Source "
                "| Annual Rate | Validation |"
            )
            lines.append(
                "|-------|--------|-----------|:-----------------:"
                "|:-----------:|:----------:|"
            )
            for m in methods:
                lines.append(
                    f"| {m.get('scope', '-')} "
                    f"| {m.get('method', '-')} "
                    f"| {m.get('rationale', '-')} "
                    f"| {m.get('benchmark_source', '-')} "
                    f"| {_pct(m.get('annual_rate', 0))} "
                    f"| {m.get('validation', '-')} |"
                )

        # Assumptions
        assumptions = pm.get("assumptions", [])
        if assumptions:
            lines.append("")
            lines.append("### Key Pathway Assumptions\n")
            lines.append("| # | Assumption | Impact | Sensitivity |")
            lines.append("|---|------------|--------|:-----------:|")
            for i, a in enumerate(assumptions, 1):
                lines.append(
                    f"| {i} | {a.get('assumption', '-')} "
                    f"| {a.get('impact', '-')} "
                    f"| {a.get('sensitivity', '-')} |"
                )

        return "\n".join(lines)

    def _md_readiness_assessment(self, data: Dict[str, Any]) -> str:
        readiness = data.get("readiness", {})
        dimensions = readiness.get("dimensions", [])
        overall = float(readiness.get("overall_pct", 0))
        lines = [
            "## 8. Readiness Assessment\n",
            f"**Overall Readiness:** {_pct(overall)}  \n"
            f"**Submission Status:** {readiness.get('status', 'N/A')}  \n"
            f"**Recommended Action:** {readiness.get('recommendation', 'N/A')}\n",
            "| Dimension | Score (%) | Weight | Weighted Score | Status |",
            "|-----------|:---------:|:------:|:--------------:|--------|",
        ]
        for d in dimensions:
            score = float(d.get("score_pct", 0))
            weight = float(d.get("weight", 0))
            weighted = score * weight / 100
            lines.append(
                f"| {d.get('name', '-')} "
                f"| {_pct(score)} "
                f"| {_dec(weight, 0)} "
                f"| {_dec(weighted, 1)} "
                f"| {d.get('status', '-')} |"
            )
        if not dimensions:
            lines.append("| _No dimensions assessed_ | - | - | - | - |")

        # Blockers
        blockers = readiness.get("blockers", [])
        if blockers:
            lines.append("")
            lines.append("### Submission Blockers\n")
            lines.append("| # | Blocker | Severity | Resolution | Owner | ETA |")
            lines.append("|---|---------|:--------:|------------|-------|:---:|")
            for i, b in enumerate(blockers, 1):
                lines.append(
                    f"| {i} | {b.get('blocker', '-')} "
                    f"| {b.get('severity', '-')} "
                    f"| {b.get('resolution', '-')} "
                    f"| {b.get('owner', '-')} "
                    f"| {b.get('eta', '-')} |"
                )

        # Submission timeline
        timeline = readiness.get("timeline", [])
        if timeline:
            lines.append("")
            lines.append("### Submission Timeline\n")
            lines.append("| # | Milestone | Target Date | Status | Owner |")
            lines.append("|---|-----------|:-----------:|--------|-------|")
            for i, t in enumerate(timeline, 1):
                lines.append(
                    f"| {i} | {t.get('milestone', '-')} "
                    f"| {t.get('target_date', '-')} "
                    f"| {_status_icon(t.get('status', 'N/A'))} "
                    f"| {t.get('owner', '-')} |"
                )

        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-023 SBTi Alignment Pack on {ts}*  \n"
            f"*Submission package formatted per SBTi Target Validation Protocol V5.3 "
            f"and Net-Zero Standard V1.3.*  \n"
            f"*This document is for internal preparation. Official submission must be "
            f"made through the SBTi Target Validation Platform.*"
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
            ".summary-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));"
            "gap:16px;margin:20px 0;}"
            ".card{background:linear-gradient(135deg,#e8f5e9,#c8e6c9);border-radius:10px;"
            "padding:18px;text-align:center;border-left:4px solid #2e7d32;}"
            ".card-label{font-size:0.8em;color:#558b2f;text-transform:uppercase;letter-spacing:0.5px;}"
            ".card-value{font-size:1.5em;font-weight:700;color:#1b5e20;margin-top:4px;}"
            ".card-unit{font-size:0.75em;color:#689f38;}"
            ".badge-pass{display:inline-block;background:#43a047;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".badge-fail{display:inline-block;background:#ef5350;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".badge-warn{display:inline-block;background:#ff9800;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".badge-na{display:inline-block;background:#9e9e9e;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".badge-ready{display:inline-block;background:#1b5e20;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".badge-pending{display:inline-block;background:#ff9800;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".severity-high{color:#d32f2f;font-weight:700;}"
            ".severity-medium{color:#f57c00;font-weight:600;}"
            ".severity-low{color:#388e3c;}"
            ".progress-bar{width:100%;height:20px;background:#e0e0e0;border-radius:10px;"
            "overflow:hidden;margin:4px 0;}"
            ".progress-fill{height:100%;border-radius:10px;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_status_badge(self, status: str) -> str:
        """Return an HTML badge for status."""
        s = str(status).upper()
        if s == "PASS":
            return '<span class="badge-pass">PASS</span>'
        elif s == "FAIL":
            return '<span class="badge-fail">FAIL</span>'
        elif s in ("WARNING", "WARN"):
            return '<span class="badge-warn">WARN</span>'
        elif s in ("NA", "N/A"):
            return '<span class="badge-na">N/A</span>'
        elif s in ("READY", "COMPLETE"):
            return '<span class="badge-ready">READY</span>'
        elif s in ("PENDING", "IN_PROGRESS"):
            return '<span class="badge-pending">PENDING</span>'
        return status

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>SBTi Submission Package</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts} | '
            f'<strong>Type:</strong> Target Validation Submission</p>'
        )

    def _html_submission_overview(self, data: Dict[str, Any]) -> str:
        sub = data.get("submission", {})
        return (
            f'<h2>1. Submission Overview</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Submission Type</div>'
            f'<div class="card-value">{sub.get("submission_type", "Initial")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Sector</div>'
            f'<div class="card-value">{sub.get("sector", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Status</div>'
            f'<div class="card-value">{self._html_status_badge(sub.get("status", "Draft"))}</div></div>\n'
            f'  <div class="card"><div class="card-label">Revenue</div>'
            f'<div class="card-value">{_dec_comma(sub.get("revenue_musd", 0), 0)}</div>'
            f'<div class="card-unit">M USD</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Field</th><th>Value</th></tr>\n'
            f'<tr><td>Company Name</td><td>{sub.get("company_name", data.get("org_name", "N/A"))}</td></tr>\n'
            f'<tr><td>ISIC Code</td><td>{sub.get("isic_code", "N/A")}</td></tr>\n'
            f'<tr><td>Consolidation</td><td>{sub.get("consolidation_approach", "Operational Control")}</td></tr>\n'
            f'<tr><td>Submission Date</td><td>{sub.get("submission_date", "N/A")}</td></tr>\n'
            f'<tr><td>Contact</td><td>{sub.get("contact_name", "N/A")} ({sub.get("contact_email", "N/A")})</td></tr>\n'
            f'</table>'
        )

    def _html_target_summary(self, data: Dict[str, Any]) -> str:
        targets = data.get("targets", [])
        rows = ""
        for i, t in enumerate(targets, 1):
            rows += (
                f'<tr><td>{i}</td>'
                f'<td><strong>{t.get("name", "-")}</strong></td>'
                f'<td>{t.get("type", "-")}</td>'
                f'<td>{t.get("scope", "-")}</td>'
                f'<td>{t.get("base_year", "-")}</td>'
                f'<td>{t.get("target_year", "-")}</td>'
                f'<td>{_pct(t.get("reduction_pct", 0))}</td>'
                f'<td>{t.get("pathway", "-")}</td>'
                f'<td>{_pct(t.get("coverage_pct", 0))}</td>'
                f'<td>{t.get("status", "-")}</td></tr>\n'
            )
        return (
            f'<h2>2. Target Summary</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Target</th><th>Type</th><th>Scope</th>'
            f'<th>Base Year</th><th>Target Year</th><th>Reduction</th>'
            f'<th>Pathway</th><th>Coverage</th><th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_validation_matrix(self, data: Dict[str, Any]) -> str:
        near_term = data.get("near_term_criteria", [])
        net_zero = data.get("net_zero_criteria", [])
        all_c = near_term + net_zero
        passed = len([c for c in all_c if str(c.get("status", "")).upper() == "PASS"])
        total = len(all_c)
        score = round(passed / total * 100, 1) if total > 0 else 0
        bar_color = "#43a047" if score >= 80 else "#ff9800" if score >= 50 else "#ef5350"

        nt_rows = ""
        for c in near_term:
            nt_rows += (
                f'<tr><td><strong>{c.get("id", "-")}</strong></td>'
                f'<td>{c.get("description", "-")}</td>'
                f'<td>{self._html_status_badge(c.get("status", "N/A"))}</td>'
                f'<td>{c.get("evidence_ref", "-")}</td>'
                f'<td>{c.get("notes", "-")}</td></tr>\n'
            )

        nz_rows = ""
        for c in net_zero:
            nz_rows += (
                f'<tr><td><strong>{c.get("id", "-")}</strong></td>'
                f'<td>{c.get("description", "-")}</td>'
                f'<td>{self._html_status_badge(c.get("status", "N/A"))}</td>'
                f'<td>{c.get("evidence_ref", "-")}</td>'
                f'<td>{c.get("notes", "-")}</td></tr>\n'
            )

        return (
            f'<h2>3. 42-Criterion Validation Matrix</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Score</div>'
            f'<div class="card-value">{_dec(score, 1)}%</div></div>\n'
            f'  <div class="card"><div class="card-label">Passed</div>'
            f'<div class="card-value">{passed}/{total}</div></div>\n'
            f'</div>\n'
            f'<div class="progress-bar">'
            f'<div class="progress-fill" style="width:{score}%;background:{bar_color};"></div>'
            f'</div>\n'
            f'<h3>Near-Term Criteria (C1-C28)</h3>\n'
            f'<table>\n'
            f'<tr><th>Criterion</th><th>Description</th><th>Status</th>'
            f'<th>Evidence</th><th>Notes</th></tr>\n'
            f'{nt_rows}</table>\n'
            f'<h3>Net-Zero Criteria (NZ-C1 to NZ-C14)</h3>\n'
            f'<table>\n'
            f'<tr><th>Criterion</th><th>Description</th><th>Status</th>'
            f'<th>Evidence</th><th>Notes</th></tr>\n'
            f'{nz_rows}</table>'
        )

    def _html_evidence_index(self, data: Dict[str, Any]) -> str:
        evidence = data.get("evidence_index", [])
        rows = ""
        for e in evidence:
            rows += (
                f'<tr><td>{e.get("ref", "-")}</td>'
                f'<td>{e.get("title", "-")}</td>'
                f'<td>{e.get("type", "-")}</td>'
                f'<td>{e.get("criteria_covered", "-")}</td>'
                f'<td>{e.get("date", "-")}</td>'
                f'<td>{e.get("version", "-")}</td>'
                f'<td>{self._html_status_badge(e.get("status", "N/A"))}</td></tr>\n'
            )
        return (
            f'<h2>4. Supporting Evidence Index</h2>\n'
            f'<table>\n'
            f'<tr><th>Ref</th><th>Document</th><th>Type</th>'
            f'<th>Criteria</th><th>Date</th><th>Version</th>'
            f'<th>Status</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_governance(self, data: Dict[str, Any]) -> str:
        gov = data.get("governance", {})
        roles = gov.get("roles", [])
        role_rows = ""
        for r in roles:
            role_rows += (
                f'<tr><td>{r.get("role", "-")}</td>'
                f'<td>{r.get("name", "-")}</td>'
                f'<td>{r.get("responsibility", "-")}</td>'
                f'<td>{r.get("authority", "-")}</td></tr>\n'
            )
        role_html = ""
        if roles:
            role_html = (
                f'<h3>Key Roles</h3>\n'
                f'<table><tr><th>Role</th><th>Name</th><th>Responsibility</th>'
                f'<th>Authority</th></tr>\n{role_rows}</table>\n'
            )

        return (
            f'<h2>5. Governance Documentation</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Board Approval</div>'
            f'<div class="card-value">{gov.get("board_approval", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">CEO Sign-Off</div>'
            f'<div class="card-value">{gov.get("ceo_signoff", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Review Frequency</div>'
            f'<div class="card-value">{gov.get("review_frequency", "Annual")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Verification</div>'
            f'<div class="card-value">{gov.get("external_verification", "N/A")}</div></div>\n'
            f'</div>\n'
            f'{role_html}'
        )

    def _html_emissions_inventory(self, data: Dict[str, Any]) -> str:
        inv = data.get("emissions_inventory", {})
        base = inv.get("base_year", {})
        current = inv.get("current_year", {})
        scopes = [
            ("Scope 1", "scope1"),
            ("Scope 2 (Location)", "scope2_location"),
            ("Scope 2 (Market)", "scope2_market"),
            ("Scope 3", "scope3"),
            ("Total", "total"),
        ]
        rows = ""
        for label, key in scopes:
            base_val = float(base.get(f"{key}_tco2e", 0))
            curr_val = float(current.get(f"{key}_tco2e", 0))
            change = (curr_val - base_val) / base_val * 100 if base_val > 0 else 0
            cls = "variance-negative" if change < 0 else "variance-positive" if change > 0 else ""
            rows += (
                f'<tr><td><strong>{label}</strong></td>'
                f'<td>{_dec_comma(base_val, 0)}</td>'
                f'<td>{_dec_comma(curr_val, 0)}</td>'
                f'<td class="{cls}">{"+" if change > 0 else ""}{_pct(change)}</td></tr>\n'
            )
        return (
            f'<h2>6. Emissions Inventory Summary</h2>\n'
            f'<table>\n'
            f'<tr><th>Scope</th><th>Base Year (tCO2e)</th>'
            f'<th>Current Year (tCO2e)</th><th>Change</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_pathway_methodology(self, data: Dict[str, Any]) -> str:
        pm = data.get("pathway_methodology", {})
        methods = pm.get("methods", [])
        rows = ""
        for m in methods:
            rows += (
                f'<tr><td>{m.get("scope", "-")}</td>'
                f'<td>{m.get("method", "-")}</td>'
                f'<td>{m.get("rationale", "-")}</td>'
                f'<td>{m.get("benchmark_source", "-")}</td>'
                f'<td>{_pct(m.get("annual_rate", 0))}</td>'
                f'<td>{m.get("validation", "-")}</td></tr>\n'
            )
        return (
            f'<h2>7. Pathway Methodology</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Primary Pathway</div>'
            f'<div class="card-value">{pm.get("primary_pathway", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Scenario</div>'
            f'<div class="card-value">{pm.get("scenario", "IEA NZE")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Temperature</div>'
            f'<div class="card-value">{pm.get("temperature", "1.5C")}</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Scope</th><th>Method</th><th>Rationale</th>'
            f'<th>Benchmark</th><th>Annual Rate</th><th>Validation</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_readiness_assessment(self, data: Dict[str, Any]) -> str:
        readiness = data.get("readiness", {})
        dimensions = readiness.get("dimensions", [])
        overall = float(readiness.get("overall_pct", 0))
        bar_color = "#43a047" if overall >= 80 else "#ff9800" if overall >= 50 else "#ef5350"

        dim_rows = ""
        for d in dimensions:
            score = float(d.get("score_pct", 0))
            weight = float(d.get("weight", 0))
            weighted = score * weight / 100
            dim_rows += (
                f'<tr><td>{d.get("name", "-")}</td>'
                f'<td>{_pct(score)}</td>'
                f'<td>{_dec(weight, 0)}</td>'
                f'<td>{_dec(weighted, 1)}</td>'
                f'<td>{self._html_status_badge(d.get("status", "N/A"))}</td></tr>\n'
            )

        blockers = readiness.get("blockers", [])
        blocker_rows = ""
        for i, b in enumerate(blockers, 1):
            sev = str(b.get("severity", "")).lower()
            sev_cls = (
                "severity-high" if sev == "high"
                else "severity-medium" if sev == "medium"
                else "severity-low"
            )
            blocker_rows += (
                f'<tr><td>{i}</td><td>{b.get("blocker", "-")}</td>'
                f'<td class="{sev_cls}">{b.get("severity", "-")}</td>'
                f'<td>{b.get("resolution", "-")}</td>'
                f'<td>{b.get("owner", "-")}</td>'
                f'<td>{b.get("eta", "-")}</td></tr>\n'
            )
        blocker_html = ""
        if blockers:
            blocker_html = (
                f'<h3>Submission Blockers</h3>\n'
                f'<table><tr><th>#</th><th>Blocker</th><th>Severity</th>'
                f'<th>Resolution</th><th>Owner</th><th>ETA</th></tr>\n'
                f'{blocker_rows}</table>\n'
            )

        return (
            f'<h2>8. Readiness Assessment</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Overall Readiness</div>'
            f'<div class="card-value">{_pct(overall)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Status</div>'
            f'<div class="card-value">'
            f'{self._html_status_badge(readiness.get("status", "N/A"))}</div></div>\n'
            f'</div>\n'
            f'<div class="progress-bar">'
            f'<div class="progress-fill" style="width:{overall}%;background:{bar_color};"></div>'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Dimension</th><th>Score</th><th>Weight</th>'
            f'<th>Weighted</th><th>Status</th></tr>\n'
            f'{dim_rows}</table>\n'
            f'{blocker_html}'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-023 SBTi '
            f'Alignment Pack on {ts}<br>'
            f'Submission package per SBTi Target Validation Protocol V5.3 '
            f'and Net-Zero Standard V1.3.<br>'
            f'<em>This document is for internal preparation. Official '
            f'submission must be made through the SBTi Target Validation '
            f'Platform.</em></div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
