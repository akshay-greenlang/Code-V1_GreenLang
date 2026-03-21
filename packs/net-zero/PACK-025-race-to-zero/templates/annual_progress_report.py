# -*- coding: utf-8 -*-
"""
AnnualProgressReportTemplate - Race to Zero annual progress for PACK-025.

Renders the Race to Zero annual progress report with year-over-year emissions
comparison, target pathway vs actual trajectory, on-track status dashboard,
credibility score evolution, action plan implementation status, verification
timeline, and recommendations.

Sections:
    1. Executive Summary
    2. Year-over-Year Emissions Comparison
    3. Target Pathway vs Actual Trajectory
    4. On-Track Status Dashboard
    5. Credibility Score Evolution
    6. Action Plan Implementation Status
    7. Key Achievements
    8. Verification Timeline
    9. Recommendations for Next Year

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

logger = logging.getLogger(__name__)

_MODULE_VERSION = "25.0.0"
_PACK_ID = "PACK-025"
_TEMPLATE_ID = "annual_progress_report"


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


def _safe_div(n: Any, d: Any) -> float:
    try:
        dv = float(d)
        return float(n) / dv if dv != 0 else 0.0
    except Exception:
        return 0.0


def _trend_arrow(current: float, previous: float) -> str:
    """Return a text indicator for trend direction."""
    if current < previous:
        return "DOWN (Favorable)"
    elif current > previous:
        return "UP (Unfavorable)"
    return "FLAT"


class AnnualProgressReportTemplate:
    """Race to Zero annual progress report template for PACK-025.

    Generates comprehensive annual progress reports with YoY comparisons,
    target pathway tracking, credibility score evolution, and implementation
    status across all emission scopes.
    """

    TEMPLATE_ID = _TEMPLATE_ID
    VERSION = _MODULE_VERSION
    PACK_ID = _PACK_ID

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    #  Public render methods                                               #
    # ------------------------------------------------------------------ #

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render the annual progress report as Markdown."""
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_executive_summary(data),
            self._md_yoy_comparison(data),
            self._md_trajectory(data),
            self._md_ontrack_status(data),
            self._md_credibility_score(data),
            self._md_implementation_status(data),
            self._md_key_achievements(data),
            self._md_verification_timeline(data),
            self._md_recommendations(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        return content + f"\n\n<!-- Provenance: {_compute_hash(content)} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render the annual progress report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_executive_summary(data),
            self._html_yoy_comparison(data),
            self._html_trajectory(data),
            self._html_ontrack_dashboard(data),
            self._html_credibility(data),
            self._html_implementation(data),
            self._html_achievements(data),
            self._html_verification(data),
            self._html_recommendations(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Race to Zero - Annual Progress Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render the annual progress report as structured JSON."""
        self.generated_at = _utcnow()
        current = data.get("current_year", {})
        previous = data.get("previous_year", {})
        baseline = data.get("baseline", {})
        target = data.get("interim_target", {})

        current_total = current.get("total_tco2e", 0)
        previous_total = previous.get("total_tco2e", 0)
        baseline_total = baseline.get("total_tco2e", 0)

        yoy_change_pct = _safe_div(current_total - previous_total, max(previous_total, 1)) * 100
        from_baseline_pct = _safe_div(baseline_total - current_total, max(baseline_total, 1)) * 100
        target_pct = target.get("reduction_pct", 50)
        progress_to_target_pct = _safe_div(from_baseline_pct, max(target_pct, 1)) * 100

        result: Dict[str, Any] = {
            "template": _TEMPLATE_ID,
            "version": _MODULE_VERSION,
            "pack_id": _PACK_ID,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "current_emissions": current,
            "previous_emissions": previous,
            "baseline_emissions": baseline,
            "yoy_change_pct": round(yoy_change_pct, 1),
            "from_baseline_reduction_pct": round(from_baseline_pct, 1),
            "progress_to_target_pct": round(progress_to_target_pct, 1),
            "is_on_track": from_baseline_pct >= self._expected_reduction(data),
            "credibility_score": data.get("credibility_score", {}),
            "implementation_status": data.get("implementation_status", []),
            "verification": data.get("verification", {}),
            "recommendations": data.get("recommendations", []),
        }
        result["provenance_hash"] = _compute_hash(result)
        return result

    def render_excel_data(self, data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Return structured data for Excel/openpyxl export."""
        self.generated_at = _utcnow()
        sheets: Dict[str, List[Dict[str, Any]]] = {}

        # Sheet 1: YoY Comparison
        current = data.get("current_year", {})
        previous = data.get("previous_year", {})
        baseline = data.get("baseline", {})
        sheets["YoY Comparison"] = [
            {"Metric": "Scope 1 (tCO2e)", "Baseline": baseline.get("scope1_tco2e", 0),
             "Previous Year": previous.get("scope1_tco2e", 0),
             "Current Year": current.get("scope1_tco2e", 0)},
            {"Metric": "Scope 2 (tCO2e)", "Baseline": baseline.get("scope2_tco2e", 0),
             "Previous Year": previous.get("scope2_tco2e", 0),
             "Current Year": current.get("scope2_tco2e", 0)},
            {"Metric": "Scope 3 (tCO2e)", "Baseline": baseline.get("scope3_tco2e", 0),
             "Previous Year": previous.get("scope3_tco2e", 0),
             "Current Year": current.get("scope3_tco2e", 0)},
            {"Metric": "Total (tCO2e)", "Baseline": baseline.get("total_tco2e", 0),
             "Previous Year": previous.get("total_tco2e", 0),
             "Current Year": current.get("total_tco2e", 0)},
        ]

        # Sheet 2: Target Trajectory
        trajectory = data.get("trajectory", [])
        traj_rows: List[Dict[str, Any]] = []
        for t in trajectory:
            traj_rows.append({
                "Year": t.get("year", ""),
                "Target (tCO2e)": t.get("target_tco2e", 0),
                "Actual (tCO2e)": t.get("actual_tco2e", ""),
                "Variance (tCO2e)": t.get("variance_tco2e", ""),
                "On Track": "Yes" if t.get("on_track", False) else "No",
            })
        sheets["Target Trajectory"] = traj_rows

        # Sheet 3: Credibility Score
        cs = data.get("credibility_score", {})
        dimensions = cs.get("dimensions", [])
        cs_rows: List[Dict[str, Any]] = []
        for dim in dimensions:
            cs_rows.append({
                "Dimension": dim.get("name", ""),
                "Current Score": dim.get("current_score", 0),
                "Previous Score": dim.get("previous_score", 0),
                "Max Score": dim.get("max_score", 100),
                "Change": dim.get("change", 0),
            })
        sheets["Credibility Score"] = cs_rows

        # Sheet 4: Implementation Status
        impl = data.get("implementation_status", [])
        impl_rows: List[Dict[str, Any]] = []
        for item in impl:
            impl_rows.append({
                "Action": item.get("action", ""),
                "Category": item.get("category", ""),
                "Status": item.get("status", ""),
                "Progress (%)": item.get("progress_pct", 0),
                "Expected Reduction (tCO2e)": item.get("expected_reduction_tco2e", 0),
                "Actual Reduction (tCO2e)": item.get("actual_reduction_tco2e", 0),
            })
        sheets["Implementation Status"] = impl_rows

        # Sheet 5: Recommendations
        recs = data.get("recommendations", [])
        rec_rows: List[Dict[str, Any]] = []
        for r in recs:
            rec_rows.append({
                "Priority": r.get("priority", ""),
                "Recommendation": r.get("recommendation", r) if isinstance(r, str) else r.get("recommendation", ""),
                "Impact": r.get("impact", ""),
                "Timeline": r.get("timeline", ""),
            })
        sheets["Recommendations"] = rec_rows

        return sheets

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _expected_reduction(self, data: Dict[str, Any]) -> float:
        """Calculate expected reduction % for the current year on a linear pathway."""
        baseline = data.get("baseline", {})
        target = data.get("interim_target", {})
        base_year = int(baseline.get("year", 2020))
        target_year = int(target.get("year", 2030))
        reporting_year = int(data.get("reporting_year", base_year + 1))
        target_pct = float(target.get("reduction_pct", 50))
        years_elapsed = reporting_year - base_year
        total_years = max(target_year - base_year, 1)
        return target_pct * _safe_div(years_elapsed, total_years)

    # ------------------------------------------------------------------ #
    #  Markdown sections                                                   #
    # ------------------------------------------------------------------ #

    def _md_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"# Race to Zero -- Annual Progress Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}\n\n---"
        )

    def _md_executive_summary(self, data: Dict[str, Any]) -> str:
        current = data.get("current_year", {})
        previous = data.get("previous_year", {})
        baseline = data.get("baseline", {})
        target = data.get("interim_target", {})

        curr_total = current.get("total_tco2e", 0)
        prev_total = previous.get("total_tco2e", 0)
        base_total = baseline.get("total_tco2e", 0)

        yoy_change = _safe_div(curr_total - prev_total, max(prev_total, 1)) * 100
        from_baseline = _safe_div(base_total - curr_total, max(base_total, 1)) * 100
        expected = self._expected_reduction(data)
        on_track = from_baseline >= expected

        cs = data.get("credibility_score", {})
        overall_score = cs.get("overall", 0)

        return (
            f"## 1. Executive Summary\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Current Year Emissions | {_dec_comma(curr_total)} tCO2e |\n"
            f"| Previous Year Emissions | {_dec_comma(prev_total)} tCO2e |\n"
            f"| YoY Change | {_pct(yoy_change)} ({_trend_arrow(curr_total, prev_total)}) |\n"
            f"| Baseline Emissions | {_dec_comma(base_total)} tCO2e ({baseline.get('year', '')}) |\n"
            f"| Reduction from Baseline | {_pct(from_baseline)} |\n"
            f"| Expected Reduction (Linear) | {_pct(expected)} |\n"
            f"| On-Track Status | **{'ON TRACK' if on_track else 'BEHIND TARGET'}** |\n"
            f"| Credibility Score | {_dec(overall_score, 1)}/100 |\n"
            f"| Interim Target | {_pct(target.get('reduction_pct', 50))} by {target.get('year', 2030)} |"
        )

    def _md_yoy_comparison(self, data: Dict[str, Any]) -> str:
        current = data.get("current_year", {})
        previous = data.get("previous_year", {})
        baseline = data.get("baseline", {})

        scopes = [
            ("Scope 1", "scope1_tco2e"),
            ("Scope 2", "scope2_tco2e"),
            ("Scope 3", "scope3_tco2e"),
            ("Total", "total_tco2e"),
        ]

        lines = [
            "## 2. Year-over-Year Emissions Comparison\n",
            "| Scope | Baseline | Previous | Current | YoY Change | From Baseline |",
            "|-------|--------:|--------:|--------:|:----------:|:-------------:|",
        ]

        for label, key in scopes:
            b_val = baseline.get(key, 0)
            p_val = previous.get(key, 0)
            c_val = current.get(key, 0)
            yoy = _safe_div(c_val - p_val, max(p_val, 1)) * 100
            from_b = _safe_div(b_val - c_val, max(b_val, 1)) * 100
            bold = "**" if label == "Total" else ""
            lines.append(
                f"| {bold}{label}{bold} | {bold}{_dec_comma(b_val)}{bold} "
                f"| {bold}{_dec_comma(p_val)}{bold} "
                f"| {bold}{_dec_comma(c_val)}{bold} "
                f"| {bold}{_pct(yoy)}{bold} "
                f"| {bold}{_pct(from_b)}{bold} |"
            )

        # Intensity metrics if available
        intensity = data.get("intensity_metrics", [])
        if intensity:
            lines.extend([
                "\n### Intensity Metrics\n",
                "| Metric | Previous | Current | Change |",
                "|--------|--------:|--------:|:------:|",
            ])
            for im in intensity:
                change = _safe_div(
                    im.get("current", 0) - im.get("previous", 0),
                    max(im.get("previous", 1), 1)
                ) * 100
                lines.append(
                    f"| {im.get('metric', '-')} "
                    f"| {_dec(im.get('previous', 0))} "
                    f"| {_dec(im.get('current', 0))} "
                    f"| {_pct(change)} |"
                )

        return "\n".join(lines)

    def _md_trajectory(self, data: Dict[str, Any]) -> str:
        trajectory = data.get("trajectory", [])
        lines = [
            "## 3. Target Pathway vs Actual Trajectory\n",
        ]

        if trajectory:
            lines.extend([
                "| Year | Target (tCO2e) | Actual (tCO2e) | Variance | On Track |",
                "|:----:|---------------:|--------------:|:--------:|:--------:|",
            ])
            for t in trajectory:
                target_val = t.get("target_tco2e", 0)
                actual_val = t.get("actual_tco2e", "")
                if actual_val != "":
                    variance = float(actual_val) - float(target_val)
                    on_track = "YES" if float(actual_val) <= float(target_val) else "NO"
                    lines.append(
                        f"| {t.get('year', '-')} | {_dec_comma(target_val)} "
                        f"| {_dec_comma(actual_val)} | {_dec_comma(variance)} "
                        f"| {on_track} |"
                    )
                else:
                    lines.append(
                        f"| {t.get('year', '-')} | {_dec_comma(target_val)} "
                        f"| -- | -- | -- |"
                    )

            # ASCII trajectory chart
            lines.append("\n### Trajectory Chart\n")
            lines.append("```")
            lines.append("Emissions (tCO2e)")
            lines.append("|")
            for t in trajectory:
                year = t.get("year", "")
                target = t.get("target_tco2e", 0)
                actual = t.get("actual_tco2e", "")
                bar_t = "T" * max(1, int(float(target) / max(float(trajectory[0].get("target_tco2e", 1)), 1) * 40))
                if actual != "":
                    bar_a = "A" * max(1, int(float(actual) / max(float(trajectory[0].get("target_tco2e", 1)), 1) * 40))
                    lines.append(f"| {year} T: {bar_t}")
                    lines.append(f"|      A: {bar_a}")
                else:
                    lines.append(f"| {year} T: {bar_t}")
            lines.append("|")
            lines.append("+------> Year")
            lines.append("T = Target pathway, A = Actual emissions")
            lines.append("```")
        else:
            lines.append("_Trajectory data not yet available._")

        return "\n".join(lines)

    def _md_ontrack_status(self, data: Dict[str, Any]) -> str:
        baseline = data.get("baseline", {})
        current = data.get("current_year", {})
        target = data.get("interim_target", {})

        base_total = baseline.get("total_tco2e", 0)
        curr_total = current.get("total_tco2e", 0)
        from_baseline = _safe_div(base_total - curr_total, max(base_total, 1)) * 100
        expected = self._expected_reduction(data)
        on_track = from_baseline >= expected

        # Scope-level on-track
        scope_status = data.get("scope_on_track", {})

        return (
            f"## 4. On-Track Status Dashboard\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Overall Status | **{'ON TRACK' if on_track else 'BEHIND TARGET'}** |\n"
            f"| Actual Reduction from Baseline | {_pct(from_baseline)} |\n"
            f"| Expected Reduction (Linear Pathway) | {_pct(expected)} |\n"
            f"| Gap to Expected | {_pct(from_baseline - expected)} |\n"
            f"| Scope 1 Status | {scope_status.get('scope1', 'N/A')} |\n"
            f"| Scope 2 Status | {scope_status.get('scope2', 'N/A')} |\n"
            f"| Scope 3 Status | {scope_status.get('scope3', 'N/A')} |\n"
            f"| Years to Interim Target | {int(target.get('year', 2030)) - int(data.get('reporting_year', 2025))} |\n"
            f"| Remaining Reduction Needed | {_pct(float(target.get('reduction_pct', 50)) - from_baseline)} |"
        )

    def _md_credibility_score(self, data: Dict[str, Any]) -> str:
        cs = data.get("credibility_score", {})
        overall = cs.get("overall", 0)
        previous_overall = cs.get("previous_overall", 0)
        dimensions = cs.get("dimensions", [])

        lines = [
            "## 5. Credibility Score Evolution\n",
            f"**Overall Score:** {_dec(overall, 1)}/100 "
            f"(Previous: {_dec(previous_overall, 1)}/100, "
            f"Change: {_dec(overall - previous_overall, 1)})\n",
        ]

        if dimensions:
            lines.extend([
                "### Score Breakdown by Dimension\n",
                "| # | Dimension | Current | Previous | Change | Max |",
                "|---|-----------|--------:|--------:|:------:|:---:|",
            ])
            for i, dim in enumerate(dimensions, 1):
                change = dim.get("change", dim.get("current_score", 0) - dim.get("previous_score", 0))
                lines.append(
                    f"| {i} | {dim.get('name', '-')} "
                    f"| {_dec(dim.get('current_score', 0), 1)} "
                    f"| {_dec(dim.get('previous_score', 0), 1)} "
                    f"| {'+' if change >= 0 else ''}{_dec(change, 1)} "
                    f"| {dim.get('max_score', 100)} |"
                )
        else:
            lines.extend([
                "### Default Credibility Dimensions\n",
                "| Dimension | Description |",
                "|-----------|-------------|",
                "| Ambition | Science-based target alignment and coverage |",
                "| Integrity | Offset usage, avoidance of greenwashing |",
                "| Transparency | Public disclosure and reporting quality |",
                "| Action | Concrete reduction actions and investment |",
                "| Governance | Board oversight and policy integration |",
                "| Verification | Third-party assurance level |",
                "| Engagement | Supply chain and stakeholder engagement |",
            ])

        return "\n".join(lines)

    def _md_implementation_status(self, data: Dict[str, Any]) -> str:
        items = data.get("implementation_status", [])
        lines = ["## 6. Action Plan Implementation Status\n"]

        if items:
            lines.extend([
                "| # | Action | Category | Progress | Expected (tCO2e) | Actual (tCO2e) | Status |",
                "|---|--------|----------|:--------:|-----------------:|---------------:|:------:|",
            ])
            total_expected = 0
            total_actual = 0
            for i, item in enumerate(items, 1):
                exp = item.get("expected_reduction_tco2e", 0)
                act = item.get("actual_reduction_tco2e", 0)
                total_expected += exp
                total_actual += act
                lines.append(
                    f"| {i} | {item.get('action', '-')} "
                    f"| {item.get('category', '-')} "
                    f"| {_pct(item.get('progress_pct', 0))} "
                    f"| {_dec_comma(exp)} "
                    f"| {_dec_comma(act)} "
                    f"| {item.get('status', '-')} |"
                )
            lines.append(
                f"| | **Total** | | | **{_dec_comma(total_expected)}** "
                f"| **{_dec_comma(total_actual)}** | |"
            )
            achievement = _safe_div(total_actual, max(total_expected, 1)) * 100
            lines.append(f"\n**Overall Achievement Rate:** {_pct(achievement)}")
        else:
            lines.append("_Implementation tracking data not available for this period._")

        return "\n".join(lines)

    def _md_key_achievements(self, data: Dict[str, Any]) -> str:
        achievements = data.get("achievements", [])
        lines = ["## 7. Key Achievements\n"]

        if achievements:
            for i, ach in enumerate(achievements, 1):
                if isinstance(ach, str):
                    lines.append(f"{i}. {ach}")
                else:
                    lines.append(
                        f"{i}. **{ach.get('title', '')}**: {ach.get('description', '')} "
                        f"({ach.get('impact', '')})"
                    )
        else:
            lines.append("_Key achievements to be documented._")

        return "\n".join(lines)

    def _md_verification_timeline(self, data: Dict[str, Any]) -> str:
        verification = data.get("verification", {})
        history = verification.get("history", [])

        lines = [
            "## 8. Verification Timeline\n",
            f"**Current Status:** {verification.get('current_status', 'Pending')}  \n"
            f"**Verifier:** {verification.get('verifier', 'N/A')}  \n"
            f"**Assurance Level:** {verification.get('assurance_level', 'Limited')}\n",
        ]

        if history:
            lines.extend([
                "### Verification History\n",
                "| Year | Type | Verifier | Level | Outcome |",
                "|:----:|------|----------|:-----:|:-------:|",
            ])
            for h in history:
                lines.append(
                    f"| {h.get('year', '-')} | {h.get('type', '-')} "
                    f"| {h.get('verifier', '-')} | {h.get('level', '-')} "
                    f"| {h.get('outcome', '-')} |"
                )

        return "\n".join(lines)

    def _md_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        lines = ["## 9. Recommendations for Next Year\n"]

        if recs:
            for i, rec in enumerate(recs, 1):
                if isinstance(rec, str):
                    lines.append(f"{i}. {rec}")
                else:
                    priority = rec.get("priority", "")
                    prefix = f"[{priority}] " if priority else ""
                    lines.append(
                        f"{i}. {prefix}**{rec.get('recommendation', '')}** "
                        f"-- {rec.get('rationale', '')}"
                    )
        else:
            lines.extend([
                "1. Accelerate Scope 1 reduction projects to close trajectory gap",
                "2. Expand Scope 3 supplier engagement to top 50 suppliers",
                "3. Improve data quality for material Scope 3 categories",
                "4. Engage third-party verifier for limited assurance",
                "5. Update transition plan with revised milestones",
            ])

        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n"
            f"*Generated by GreenLang PACK-025 Race to Zero Pack on {ts}.*  \n"
            f"*Progress assessed against Race to Zero campaign criteria.*"
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
            ".on-track{background:#c8e6c9;color:#1b5e20;padding:4px 12px;border-radius:4px;"
            "font-weight:700;font-size:1.1em;}"
            ".behind{background:#ffcdd2;color:#c62828;padding:4px 12px;border-radius:4px;"
            "font-weight:700;font-size:1.1em;}"
            ".progress-bar{background:#e0e0e0;border-radius:8px;height:20px;overflow:hidden;"
            "margin:4px 0;}"
            ".progress-fill{height:100%;border-radius:8px;background:linear-gradient(90deg,#43a047,#66bb6a);}"
            ".gauge{width:120px;height:120px;border-radius:50%;border:8px solid #e0e0e0;"
            "display:flex;align-items:center;justify-content:center;margin:0 auto;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Race to Zero -- Annual Progress Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts}</p>'
        )

    def _html_executive_summary(self, data: Dict[str, Any]) -> str:
        current = data.get("current_year", {})
        baseline = data.get("baseline", {})
        from_baseline = _safe_div(
            baseline.get("total_tco2e", 0) - current.get("total_tco2e", 0),
            max(baseline.get("total_tco2e", 1), 1)
        ) * 100
        on_track = from_baseline >= self._expected_reduction(data)
        cs = data.get("credibility_score", {})

        status_class = "on-track" if on_track else "behind"
        status_text = "ON TRACK" if on_track else "BEHIND TARGET"

        return (
            f'<h2>1. Executive Summary</h2>\n'
            f'<div class="cards">\n'
            f'  <div class="card"><div class="card-label">Current Emissions</div>'
            f'<div class="card-value">{_dec_comma(current.get("total_tco2e", 0))}</div>tCO2e</div>\n'
            f'  <div class="card"><div class="card-label">Reduction from Baseline</div>'
            f'<div class="card-value">{_pct(from_baseline)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Credibility Score</div>'
            f'<div class="card-value">{_dec(cs.get("overall", 0), 1)}</div>/100</div>\n'
            f'  <div class="card"><div class="card-label">Status</div>'
            f'<div class="card-value"><span class="{status_class}">{status_text}</span></div></div>\n'
            f'</div>'
        )

    def _html_yoy_comparison(self, data: Dict[str, Any]) -> str:
        current = data.get("current_year", {})
        previous = data.get("previous_year", {})
        baseline = data.get("baseline", {})
        scopes = [("Scope 1", "scope1_tco2e"), ("Scope 2", "scope2_tco2e"),
                  ("Scope 3", "scope3_tco2e"), ("Total", "total_tco2e")]
        rows = ""
        for label, key in scopes:
            b = baseline.get(key, 0)
            p = previous.get(key, 0)
            c = current.get(key, 0)
            yoy = _safe_div(c - p, max(p, 1)) * 100
            rows += (f'<tr><td>{"<strong>" + label + "</strong>" if label == "Total" else label}</td>'
                     f'<td>{_dec_comma(b)}</td><td>{_dec_comma(p)}</td>'
                     f'<td>{_dec_comma(c)}</td><td>{_pct(yoy)}</td></tr>\n')
        return (
            f'<h2>2. Year-over-Year Comparison</h2>\n'
            f'<table><tr><th>Scope</th><th>Baseline</th><th>Previous</th>'
            f'<th>Current</th><th>YoY Change</th></tr>\n{rows}</table>'
        )

    def _html_trajectory(self, data: Dict[str, Any]) -> str:
        trajectory = data.get("trajectory", [])
        rows = ""
        for t in trajectory:
            actual = t.get("actual_tco2e", "")
            on_track_class = ""
            if actual != "":
                on_track_class = "on-track" if float(actual) <= float(t.get("target_tco2e", 0)) else "behind"
                on_track_text = "YES" if float(actual) <= float(t.get("target_tco2e", 0)) else "NO"
                rows += (f'<tr><td>{t.get("year", "-")}</td><td>{_dec_comma(t.get("target_tco2e", 0))}</td>'
                         f'<td>{_dec_comma(actual)}</td>'
                         f'<td><span class="{on_track_class}">{on_track_text}</span></td></tr>\n')
            else:
                rows += (f'<tr><td>{t.get("year", "-")}</td><td>{_dec_comma(t.get("target_tco2e", 0))}</td>'
                         f'<td>--</td><td>--</td></tr>\n')
        if not rows:
            rows = '<tr><td colspan="4"><em>Trajectory data not available</em></td></tr>'
        return (
            f'<h2>3. Target Pathway vs Actual</h2>\n'
            f'<table><tr><th>Year</th><th>Target (tCO2e)</th><th>Actual (tCO2e)</th>'
            f'<th>On Track</th></tr>\n{rows}</table>'
        )

    def _html_ontrack_dashboard(self, data: Dict[str, Any]) -> str:
        baseline = data.get("baseline", {})
        current = data.get("current_year", {})
        from_baseline = _safe_div(
            baseline.get("total_tco2e", 0) - current.get("total_tco2e", 0),
            max(baseline.get("total_tco2e", 1), 1)
        ) * 100
        expected = self._expected_reduction(data)
        target = data.get("interim_target", {})
        target_pct = target.get("reduction_pct", 50)
        progress = _safe_div(from_baseline, max(target_pct, 1)) * 100

        return (
            f'<h2>4. On-Track Dashboard</h2>\n'
            f'<div class="cards">\n'
            f'  <div class="card"><div class="card-label">Actual Reduction</div>'
            f'<div class="card-value">{_pct(from_baseline)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Expected (Linear)</div>'
            f'<div class="card-value">{_pct(expected)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Progress to Target</div>'
            f'<div class="card-value">{_pct(progress)}</div></div>\n'
            f'</div>\n'
            f'<div class="progress-bar"><div class="progress-fill" '
            f'style="width:{min(progress, 100)}%"></div></div>'
        )

    def _html_credibility(self, data: Dict[str, Any]) -> str:
        cs = data.get("credibility_score", {})
        dims = cs.get("dimensions", [])
        rows = ""
        for dim in dims:
            score = dim.get("current_score", 0)
            rows += (f'<tr><td>{dim.get("name", "-")}</td><td>{_dec(score, 1)}</td>'
                     f'<td><div class="progress-bar"><div class="progress-fill" '
                     f'style="width:{score}%"></div></div></td></tr>\n')
        if not rows:
            rows = '<tr><td colspan="3"><em>Credibility dimensions not assessed</em></td></tr>'
        return (
            f'<h2>5. Credibility Score</h2>\n'
            f'<div class="cards"><div class="card"><div class="card-label">Overall Score</div>'
            f'<div class="card-value">{_dec(cs.get("overall", 0), 1)}/100</div></div></div>\n'
            f'<table><tr><th>Dimension</th><th>Score</th><th>Progress</th></tr>\n{rows}</table>'
        )

    def _html_implementation(self, data: Dict[str, Any]) -> str:
        items = data.get("implementation_status", [])
        rows = ""
        for item in items:
            prog = item.get("progress_pct", 0)
            rows += (f'<tr><td>{item.get("action", "-")}</td>'
                     f'<td><div class="progress-bar"><div class="progress-fill" '
                     f'style="width:{prog}%"></div></div> {_pct(prog)}</td>'
                     f'<td>{item.get("status", "-")}</td></tr>\n')
        if not rows:
            rows = '<tr><td colspan="3"><em>Implementation data not available</em></td></tr>'
        return (
            f'<h2>6. Implementation Status</h2>\n'
            f'<table><tr><th>Action</th><th>Progress</th><th>Status</th></tr>\n{rows}</table>'
        )

    def _html_achievements(self, data: Dict[str, Any]) -> str:
        achievements = data.get("achievements", [])
        items = ""
        for ach in achievements:
            if isinstance(ach, str):
                items += f'<li>{ach}</li>\n'
            else:
                items += f'<li><strong>{ach.get("title", "")}</strong>: {ach.get("description", "")}</li>\n'
        if not items:
            items = '<li><em>Achievements to be documented</em></li>'
        return f'<h2>7. Key Achievements</h2>\n<ul>\n{items}</ul>'

    def _html_verification(self, data: Dict[str, Any]) -> str:
        v = data.get("verification", {})
        return (
            f'<h2>8. Verification</h2>\n'
            f'<table><tr><th>Field</th><th>Value</th></tr>\n'
            f'<tr><td>Status</td><td>{v.get("current_status", "Pending")}</td></tr>\n'
            f'<tr><td>Verifier</td><td>{v.get("verifier", "N/A")}</td></tr>\n'
            f'<tr><td>Assurance Level</td><td>{v.get("assurance_level", "Limited")}</td></tr>\n'
            f'</table>'
        )

    def _html_recommendations(self, data: Dict[str, Any]) -> str:
        recs = data.get("recommendations", [])
        items = ""
        for rec in recs:
            if isinstance(rec, str):
                items += f'<li>{rec}</li>\n'
            else:
                items += f'<li><strong>[{rec.get("priority", "")}]</strong> {rec.get("recommendation", "")}</li>\n'
        if not items:
            items = '<li><em>Recommendations pending analysis</em></li>'
        return f'<h2>9. Recommendations</h2>\n<ol>\n{items}</ol>'

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">'
            f'Generated by GreenLang PACK-025 Race to Zero Pack on {ts}'
            f'</div>'
        )
