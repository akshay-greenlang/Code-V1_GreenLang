# -*- coding: utf-8 -*-
"""
FLAGAssessmentReportTemplate - Forest, Land & Agriculture assessment for PACK-023.

Renders a comprehensive FLAG (Forest, Land and Agriculture) commodity assessment
report covering the 11 FLAG commodities, 20% materiality trigger evaluation,
per-commodity pathway at 3.03%/yr, no-deforestation commitment checklist,
land use change (LUC) emissions quantification, and FLAG target readiness.

Sections:
    1. FLAG Overview (trigger evaluation, total FLAG emissions)
    2. Commodity Inventory (11 commodities with volumes/emissions)
    3. 20% Trigger Evaluation (FLAG as % of total, threshold check)
    4. Per-Commodity Pathway (3.03%/yr convergence table)
    5. No-Deforestation Commitment Checklist
    6. Land Use Change Emissions (LUC quantification)
    7. FLAG Target Summary & Readiness

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

# FLAG commodity definitions per SBTi FLAG Guidance V1.1
FLAG_COMMODITIES = [
    "Beef", "Dairy", "Pork", "Poultry", "Eggs",
    "Rice", "Wheat", "Maize", "Soy", "Palm Oil", "Timber & Pulp",
]

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

def _check_icon(passed: bool) -> str:
    """Return a text-based check indicator for markdown."""
    return "YES" if passed else "NO"

def _flag_trigger_status(pct_val: float) -> str:
    """Determine FLAG trigger status based on 20% threshold."""
    if pct_val >= 20.0:
        return "TRIGGERED - FLAG target required"
    elif pct_val >= 15.0:
        return "APPROACHING - FLAG target recommended"
    else:
        return "BELOW THRESHOLD - FLAG target optional"

class FLAGAssessmentReportTemplate:
    """
    FLAG commodity assessment report template for SBTi alignment.

    Renders the Forest, Land and Agriculture emissions assessment covering
    all 11 FLAG commodities, 20% materiality trigger evaluation, per-commodity
    pathway at 3.03%/yr, no-deforestation commitment, and land use change
    (LUC) emissions quantification per SBTi FLAG Guidance V1.1.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize FLAGAssessmentReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render FLAG assessment report as Markdown."""
        self.generated_at = utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_flag_overview(data),
            self._md_commodity_inventory(data),
            self._md_trigger_evaluation(data),
            self._md_commodity_pathway(data),
            self._md_deforestation_checklist(data),
            self._md_luc_emissions(data),
            self._md_flag_target_summary(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render FLAG assessment report as self-contained HTML."""
        self.generated_at = utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_flag_overview(data),
            self._html_commodity_inventory(data),
            self._html_trigger_evaluation(data),
            self._html_commodity_pathway(data),
            self._html_deforestation_checklist(data),
            self._html_luc_emissions(data),
            self._html_flag_target_summary(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>FLAG Assessment Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render FLAG assessment report as structured JSON."""
        self.generated_at = utcnow()
        commodities = data.get("commodities", [])
        trigger = data.get("trigger_evaluation", {})
        luc = data.get("luc_emissions", {})
        deforestation = data.get("deforestation_checklist", [])
        pathway = data.get("commodity_pathway", [])

        total_flag = sum(
            float(c.get("emissions_tco2e", 0)) for c in commodities
        )

        result: Dict[str, Any] = {
            "template": "flag_assessment_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "flag_overview": {
                "total_flag_emissions_tco2e": total_flag,
                "total_emissions_tco2e": data.get("total_emissions_tco2e", 0),
                "flag_pct_of_total": trigger.get("flag_pct_of_total", 0),
                "trigger_status": _flag_trigger_status(
                    float(trigger.get("flag_pct_of_total", 0))
                ),
                "commodities_assessed": len(commodities),
                "commodities_material": len([
                    c for c in commodities
                    if float(c.get("emissions_tco2e", 0)) > 0
                ]),
            },
            "commodities": commodities,
            "trigger_evaluation": trigger,
            "commodity_pathway": pathway,
            "deforestation_checklist": deforestation,
            "luc_emissions": luc,
            "flag_target_summary": data.get("flag_target_summary", {}),
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
            f"# FLAG Assessment Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standard:** SBTi FLAG Guidance V1.1\n\n---"
        )

    def _md_flag_overview(self, data: Dict[str, Any]) -> str:
        commodities = data.get("commodities", [])
        trigger = data.get("trigger_evaluation", {})
        total_flag = sum(
            float(c.get("emissions_tco2e", 0)) for c in commodities
        )
        total_all = float(data.get("total_emissions_tco2e", 0)) or 1
        flag_pct = float(trigger.get("flag_pct_of_total", total_flag / total_all * 100))
        material_count = len([
            c for c in commodities if float(c.get("emissions_tco2e", 0)) > 0
        ])
        status = _flag_trigger_status(flag_pct)

        return (
            f"## 1. FLAG Overview\n\n"
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total FLAG Emissions | {_dec_comma(total_flag, 0)} tCO2e |\n"
            f"| Total Company Emissions | {_dec_comma(total_all, 0)} tCO2e |\n"
            f"| FLAG as % of Total | {_pct(flag_pct)} |\n"
            f"| 20% Trigger Status | {status} |\n"
            f"| Commodities Assessed | {len(commodities)} / 11 |\n"
            f"| Material Commodities | {material_count} |\n"
            f"| Required Reduction Rate | 3.03%/yr (FLAG pathway) |\n"
            f"| Target Year (near-term) | 2030 |\n"
            f"| Net-Zero Year (FLAG) | 2050 |"
        )

    def _md_commodity_inventory(self, data: Dict[str, Any]) -> str:
        commodities = data.get("commodities", [])
        total_flag = sum(
            float(c.get("emissions_tco2e", 0)) for c in commodities
        ) or 1
        lines = [
            "## 2. Commodity Inventory\n",
            "Assessment of all 11 FLAG commodities per SBTi FLAG Guidance.\n",
            "| # | Commodity | Volume | Unit | Emissions (tCO2e) "
            "| % of FLAG | Source Type | Data Quality | Scope |",
            "|---|-----------|-------:|------|------------------:"
            "|:---------:|:----------:|:------------:|-------|",
        ]
        for i, c in enumerate(commodities, 1):
            emissions = float(c.get("emissions_tco2e", 0))
            pct = emissions / total_flag * 100 if total_flag > 0 else 0
            lines.append(
                f"| {i} | {c.get('commodity', '-')} "
                f"| {_dec_comma(c.get('volume', 0), 0)} "
                f"| {c.get('unit', '-')} "
                f"| {_dec_comma(emissions, 0)} "
                f"| {_pct(pct)} "
                f"| {c.get('source_type', '-')} "
                f"| {c.get('data_quality', '-')} "
                f"| {c.get('scope', '-')} |"
            )
        if not commodities:
            lines.append(
                "| - | _No commodities assessed_ | - | - | - | - | - | - | - |"
            )

        lines.append("")
        lines.append(
            f"**Total FLAG Emissions:** {_dec_comma(total_flag, 0)} tCO2e"
        )

        # Commodity not present
        assessed = {c.get("commodity", "").lower() for c in commodities}
        missing = [
            name for name in FLAG_COMMODITIES
            if name.lower() not in assessed
        ]
        if missing:
            lines.append(f"\n**Not Assessed:** {', '.join(missing)}")

        return "\n".join(lines)

    def _md_trigger_evaluation(self, data: Dict[str, Any]) -> str:
        trigger = data.get("trigger_evaluation", {})
        flag_pct = float(trigger.get("flag_pct_of_total", 0))
        status = _flag_trigger_status(flag_pct)
        lines = [
            "## 3. 20% Trigger Evaluation\n",
            "Per SBTi FLAG Guidance, companies with FLAG emissions >= 20% of total "
            "emissions must set a separate FLAG target.\n",
            f"| Metric | Value |\n|--------|-------|\n"
            f"| FLAG Emissions (tCO2e) | {_dec_comma(trigger.get('flag_emissions_tco2e', 0), 0)} |\n"
            f"| Total Emissions (tCO2e) | {_dec_comma(trigger.get('total_emissions_tco2e', 0), 0)} |\n"
            f"| FLAG as % of Total | {_pct(flag_pct)} |\n"
            f"| 20% Threshold | {'MET' if flag_pct >= 20 else 'NOT MET'} |\n"
            f"| Trigger Status | {status} |\n"
            f"| FLAG Scope 1 (tCO2e) | {_dec_comma(trigger.get('flag_scope1_tco2e', 0), 0)} |\n"
            f"| FLAG Scope 2 (tCO2e) | {_dec_comma(trigger.get('flag_scope2_tco2e', 0), 0)} |\n"
            f"| FLAG Scope 3 (tCO2e) | {_dec_comma(trigger.get('flag_scope3_tco2e', 0), 0)} |\n"
            f"| Land Use Change (tCO2e) | {_dec_comma(trigger.get('luc_emissions_tco2e', 0), 0)} |",
        ]

        # Breakdown by sector
        sector_breakdown = trigger.get("sector_breakdown", [])
        if sector_breakdown:
            lines.append("")
            lines.append("### Sector Contribution to FLAG\n")
            lines.append("| Sector | Emissions (tCO2e) | % of FLAG |")
            lines.append("|--------|------------------:|:---------:|")
            for s in sector_breakdown:
                lines.append(
                    f"| {s.get('sector', '-')} "
                    f"| {_dec_comma(s.get('emissions_tco2e', 0), 0)} "
                    f"| {_pct(s.get('pct_of_flag', 0))} |"
                )

        return "\n".join(lines)

    def _md_commodity_pathway(self, data: Dict[str, Any]) -> str:
        pathway = data.get("commodity_pathway", [])
        lines = [
            "## 4. Per-Commodity Pathway (3.03%/yr)\n",
            "FLAG commodity-level decarbonization pathway at the required "
            "3.03%/yr linear reduction rate.\n",
            "| Commodity | Base Year (tCO2e) | 2025 Target | 2027 Target "
            "| 2030 Target | 2035 Target | 2040 Target | 2050 Target "
            "| Annual Rate |",
            "|-----------|------------------:|:-----------:|:-----------:"
            "|:-----------:|:-----------:|:-----------:|:-----------:"
            "|:-----------:|",
        ]
        for p in pathway:
            lines.append(
                f"| {p.get('commodity', '-')} "
                f"| {_dec_comma(p.get('base_year_tco2e', 0), 0)} "
                f"| {_dec_comma(p.get('target_2025', 0), 0)} "
                f"| {_dec_comma(p.get('target_2027', 0), 0)} "
                f"| {_dec_comma(p.get('target_2030', 0), 0)} "
                f"| {_dec_comma(p.get('target_2035', 0), 0)} "
                f"| {_dec_comma(p.get('target_2040', 0), 0)} "
                f"| {_dec_comma(p.get('target_2050', 0), 0)} "
                f"| {_pct(p.get('annual_rate', 3.03))} |"
            )
        if not pathway:
            lines.append(
                "| - | _No pathway data_ | - | - | - | - | - | - | - |"
            )

        # Aggregate pathway
        agg = data.get("aggregate_pathway", {})
        if agg:
            lines.append("")
            lines.append("### Aggregate FLAG Pathway\n")
            lines.append(
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Base Year Total FLAG | {_dec_comma(agg.get('base_year_total', 0), 0)} tCO2e |\n"
                f"| 2030 Target | {_dec_comma(agg.get('target_2030', 0), 0)} tCO2e |\n"
                f"| 2050 Target (Net-Zero) | {_dec_comma(agg.get('target_2050', 0), 0)} tCO2e |\n"
                f"| Total Reduction Required | {_pct(agg.get('total_reduction_pct', 0))} |\n"
                f"| Pathway Method | Linear (3.03%/yr) |"
            )
        return "\n".join(lines)

    def _md_deforestation_checklist(self, data: Dict[str, Any]) -> str:
        checklist = data.get("deforestation_checklist", [])
        lines = [
            "## 5. No-Deforestation Commitment Checklist\n",
            "SBTi FLAG Guidance requires a no-deforestation commitment by "
            "2025 for all companies with FLAG targets.\n",
            "| # | Requirement | Status | Evidence | Notes |",
            "|---|-------------|:------:|----------|-------|",
        ]

        default_checks = [
            {
                "requirement": "Public no-deforestation commitment",
                "description": "Publicly available commitment to eliminate deforestation",
            },
            {
                "requirement": "2025 deadline for zero gross deforestation",
                "description": "Commitment includes 2025 cutoff date",
            },
            {
                "requirement": "Covers all commodities in scope",
                "description": "Commitment covers all FLAG commodities sourced",
            },
            {
                "requirement": "Covers direct and indirect suppliers",
                "description": "Extends beyond Tier 1 to upstream supply chain",
            },
            {
                "requirement": "Includes no land conversion commitment",
                "description": "Covers conversion of natural ecosystems, not just forests",
            },
            {
                "requirement": "Monitoring and verification mechanism",
                "description": "Satellite/ground-truth monitoring in place",
            },
            {
                "requirement": "Grievance and remediation mechanism",
                "description": "Process for addressing non-compliance by suppliers",
            },
            {
                "requirement": "Regular progress reporting",
                "description": "Annual or more frequent progress disclosure",
            },
            {
                "requirement": "Aligned with Accountability Framework Initiative",
                "description": "Commitment aligns with AFi core principles",
            },
            {
                "requirement": "Board/executive sign-off",
                "description": "Commitment has governance-level approval",
            },
        ]

        items = checklist if checklist else default_checks
        for i, item in enumerate(items, 1):
            status = _check_icon(
                str(item.get("status", "")).upper() in ("PASS", "YES", "MET", "TRUE")
            )
            lines.append(
                f"| {i} | {item.get('requirement', '-')} "
                f"| {status} "
                f"| {item.get('evidence', '-')} "
                f"| {item.get('notes', '-')} |"
            )

        passed = len([
            item for item in items
            if str(item.get("status", "")).upper() in ("PASS", "YES", "MET", "TRUE")
        ])
        total = len(items)
        lines.append("")
        lines.append(
            f"**Checklist Score:** {passed}/{total} requirements met  \n"
            f"**Status:** {'COMPLIANT' if passed == total else 'ACTION REQUIRED'}"
        )
        return "\n".join(lines)

    def _md_luc_emissions(self, data: Dict[str, Any]) -> str:
        luc = data.get("luc_emissions", {})
        sources = luc.get("sources", [])
        lines = [
            "## 6. Land Use Change Emissions\n",
            "Quantification of land use change (LUC) emissions associated "
            "with FLAG commodities.\n",
            f"| Metric | Value |\n|--------|-------|\n"
            f"| Total LUC Emissions | {_dec_comma(luc.get('total_luc_tco2e', 0), 0)} tCO2e |\n"
            f"| LUC as % of FLAG | {_pct(luc.get('luc_pct_of_flag', 0))} |\n"
            f"| Methodology | {luc.get('methodology', 'N/A')} |\n"
            f"| Data Source | {luc.get('data_source', 'N/A')} |\n"
            f"| Reference Period | {luc.get('reference_period', 'N/A')} |\n"
            f"| Deforestation-Free Verified | {luc.get('deforestation_free_verified', 'N/A')} |",
        ]

        if sources:
            lines.append("")
            lines.append("### LUC by Source\n")
            lines.append(
                "| Source | Commodity | Area (ha) | Emissions (tCO2e) "
                "| Method | Confidence |"
            )
            lines.append(
                "|--------|-----------|----------:|------------------:"
                "|--------|:----------:|"
            )
            for s in sources:
                lines.append(
                    f"| {s.get('source', '-')} "
                    f"| {s.get('commodity', '-')} "
                    f"| {_dec_comma(s.get('area_ha', 0), 0)} "
                    f"| {_dec_comma(s.get('emissions_tco2e', 0), 0)} "
                    f"| {s.get('method', '-')} "
                    f"| {s.get('confidence', '-')} |"
                )

        removals = luc.get("removals", {})
        if removals:
            lines.append("")
            lines.append("### Carbon Removals (Land Sector)\n")
            lines.append(
                f"| Metric | Value |\n|--------|-------|\n"
                f"| Total Removals | {_dec_comma(removals.get('total_removals_tco2e', 0), 0)} tCO2e |\n"
                f"| Reforestation | {_dec_comma(removals.get('reforestation_tco2e', 0), 0)} tCO2e |\n"
                f"| Soil Carbon | {_dec_comma(removals.get('soil_carbon_tco2e', 0), 0)} tCO2e |\n"
                f"| Other | {_dec_comma(removals.get('other_tco2e', 0), 0)} tCO2e |\n"
                f"| Net FLAG Emissions | {_dec_comma(removals.get('net_flag_tco2e', 0), 0)} tCO2e |"
            )

        return "\n".join(lines)

    def _md_flag_target_summary(self, data: Dict[str, Any]) -> str:
        summary = data.get("flag_target_summary", {})
        readiness = summary.get("readiness", {})
        dimensions = readiness.get("dimensions", [])
        lines = [
            "## 7. FLAG Target Summary & Readiness\n",
            f"| Metric | Value |\n|--------|-------|\n"
            f"| FLAG Target Type | {summary.get('target_type', 'Absolute reduction')} |\n"
            f"| Base Year | {summary.get('base_year', 'N/A')} |\n"
            f"| Near-Term Target Year | {summary.get('near_term_year', 2030)} |\n"
            f"| Long-Term / Net-Zero Year | {summary.get('net_zero_year', 2050)} |\n"
            f"| Required ARR (FLAG) | 3.03%/yr |\n"
            f"| Company FLAG ARR | {_pct(summary.get('company_flag_arr', 0))} |\n"
            f"| Meets FLAG Pathway | {summary.get('meets_flag_pathway', 'N/A')} |\n"
            f"| No-Deforestation Committed | {summary.get('no_deforestation', 'N/A')} |\n"
            f"| Overall Readiness | {_pct(readiness.get('overall_pct', 0))} |",
        ]

        if dimensions:
            lines.append("")
            lines.append("### Readiness Dimensions\n")
            lines.append("| Dimension | Score (%) | Status | Notes |")
            lines.append("|-----------|:---------:|--------|-------|")
            for d in dimensions:
                lines.append(
                    f"| {d.get('name', '-')} "
                    f"| {_pct(d.get('score_pct', 0))} "
                    f"| {d.get('status', '-')} "
                    f"| {d.get('notes', '-')} |"
                )

        actions = summary.get("priority_actions", [])
        if actions:
            lines.append("")
            lines.append("### Priority Actions\n")
            lines.append("| # | Action | Priority | Owner | Deadline |")
            lines.append("|---|--------|:--------:|-------|:--------:|")
            for i, a in enumerate(actions, 1):
                lines.append(
                    f"| {i} | {a.get('action', '-')} "
                    f"| {a.get('priority', '-')} "
                    f"| {a.get('owner', '-')} "
                    f"| {a.get('deadline', '-')} |"
                )

        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-023 SBTi Alignment Pack on {ts}*  \n"
            f"*FLAG assessment per SBTi FLAG Guidance V1.1 and Forest, Land & "
            f"Agriculture Science Based Target Setting Guidance.*"
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
            ".trigger-met{background:#ffcdd2;color:#c62828;font-weight:700;"
            "padding:8px 16px;border-radius:6px;text-align:center;margin:10px 0;}"
            ".trigger-not{background:#e8f5e9;color:#2e7d32;font-weight:600;"
            "padding:8px 16px;border-radius:6px;text-align:center;margin:10px 0;}"
            ".progress-bar{width:100%;height:20px;background:#e0e0e0;border-radius:10px;"
            "overflow:hidden;margin:4px 0;}"
            ".progress-fill{height:100%;border-radius:10px;}"
            ".checklist-yes{color:#2e7d32;font-weight:700;}"
            ".checklist-no{color:#c62828;font-weight:700;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>FLAG Assessment Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts} | '
            f'<strong>Standard:</strong> SBTi FLAG Guidance V1.1</p>'
        )

    def _html_flag_overview(self, data: Dict[str, Any]) -> str:
        commodities = data.get("commodities", [])
        trigger = data.get("trigger_evaluation", {})
        total_flag = sum(
            float(c.get("emissions_tco2e", 0)) for c in commodities
        )
        total_all = float(data.get("total_emissions_tco2e", 0)) or 1
        flag_pct = float(trigger.get("flag_pct_of_total", total_flag / total_all * 100))
        status = _flag_trigger_status(flag_pct)
        trigger_cls = "trigger-met" if flag_pct >= 20 else "trigger-not"

        return (
            f'<h2>1. FLAG Overview</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">FLAG Emissions</div>'
            f'<div class="card-value">{_dec_comma(total_flag, 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">% of Total</div>'
            f'<div class="card-value">{_pct(flag_pct)}</div></div>\n'
            f'  <div class="card"><div class="card-label">Commodities</div>'
            f'<div class="card-value">{len(commodities)}/11</div></div>\n'
            f'  <div class="card"><div class="card-label">Required ARR</div>'
            f'<div class="card-value">3.03%</div>'
            f'<div class="card-unit">per year</div></div>\n'
            f'</div>\n'
            f'<div class="{trigger_cls}">{status}</div>'
        )

    def _html_commodity_inventory(self, data: Dict[str, Any]) -> str:
        commodities = data.get("commodities", [])
        total_flag = sum(
            float(c.get("emissions_tco2e", 0)) for c in commodities
        ) or 1
        rows = ""
        for i, c in enumerate(commodities, 1):
            emissions = float(c.get("emissions_tco2e", 0))
            pct = emissions / total_flag * 100 if total_flag > 0 else 0
            rows += (
                f'<tr><td>{i}</td><td>{c.get("commodity", "-")}</td>'
                f'<td>{_dec_comma(c.get("volume", 0), 0)}</td>'
                f'<td>{c.get("unit", "-")}</td>'
                f'<td>{_dec_comma(emissions, 0)}</td>'
                f'<td>{_pct(pct)}</td>'
                f'<td>{c.get("source_type", "-")}</td>'
                f'<td>{c.get("data_quality", "-")}</td>'
                f'<td>{c.get("scope", "-")}</td></tr>\n'
            )
        return (
            f'<h2>2. Commodity Inventory</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Commodity</th><th>Volume</th><th>Unit</th>'
            f'<th>Emissions (tCO2e)</th><th>% of FLAG</th>'
            f'<th>Source</th><th>Quality</th><th>Scope</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_trigger_evaluation(self, data: Dict[str, Any]) -> str:
        trigger = data.get("trigger_evaluation", {})
        flag_pct = float(trigger.get("flag_pct_of_total", 0))
        bar_color = "#ef5350" if flag_pct >= 20 else "#ff9800" if flag_pct >= 15 else "#43a047"

        sector_breakdown = trigger.get("sector_breakdown", [])
        sector_rows = ""
        for s in sector_breakdown:
            sector_rows += (
                f'<tr><td>{s.get("sector", "-")}</td>'
                f'<td>{_dec_comma(s.get("emissions_tco2e", 0), 0)}</td>'
                f'<td>{_pct(s.get("pct_of_flag", 0))}</td></tr>\n'
            )
        sector_html = ""
        if sector_breakdown:
            sector_html = (
                f'<h3>Sector Contribution</h3>\n'
                f'<table><tr><th>Sector</th><th>Emissions (tCO2e)</th>'
                f'<th>% of FLAG</th></tr>\n{sector_rows}</table>\n'
            )

        return (
            f'<h2>3. 20% Trigger Evaluation</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">FLAG %</div>'
            f'<div class="card-value">{_pct(flag_pct)}</div>'
            f'<div class="progress-bar"><div class="progress-fill" '
            f'style="width:{min(flag_pct * 5, 100)}%;background:{bar_color};"></div></div></div>\n'
            f'  <div class="card"><div class="card-label">Threshold</div>'
            f'<div class="card-value">20%</div></div>\n'
            f'  <div class="card"><div class="card-label">Status</div>'
            f'<div class="card-value">{"TRIGGERED" if flag_pct >= 20 else "NOT TRIGGERED"}'
            f'</div></div>\n'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Metric</th><th>Value</th></tr>\n'
            f'<tr><td>FLAG Scope 1</td>'
            f'<td>{_dec_comma(trigger.get("flag_scope1_tco2e", 0), 0)} tCO2e</td></tr>\n'
            f'<tr><td>FLAG Scope 2</td>'
            f'<td>{_dec_comma(trigger.get("flag_scope2_tco2e", 0), 0)} tCO2e</td></tr>\n'
            f'<tr><td>FLAG Scope 3</td>'
            f'<td>{_dec_comma(trigger.get("flag_scope3_tco2e", 0), 0)} tCO2e</td></tr>\n'
            f'<tr><td>Land Use Change</td>'
            f'<td>{_dec_comma(trigger.get("luc_emissions_tco2e", 0), 0)} tCO2e</td></tr>\n'
            f'</table>\n'
            f'{sector_html}'
        )

    def _html_commodity_pathway(self, data: Dict[str, Any]) -> str:
        pathway = data.get("commodity_pathway", [])
        rows = ""
        for p in pathway:
            rows += (
                f'<tr><td>{p.get("commodity", "-")}</td>'
                f'<td>{_dec_comma(p.get("base_year_tco2e", 0), 0)}</td>'
                f'<td>{_dec_comma(p.get("target_2025", 0), 0)}</td>'
                f'<td>{_dec_comma(p.get("target_2027", 0), 0)}</td>'
                f'<td>{_dec_comma(p.get("target_2030", 0), 0)}</td>'
                f'<td>{_dec_comma(p.get("target_2035", 0), 0)}</td>'
                f'<td>{_dec_comma(p.get("target_2040", 0), 0)}</td>'
                f'<td>{_dec_comma(p.get("target_2050", 0), 0)}</td>'
                f'<td>{_pct(p.get("annual_rate", 3.03))}</td></tr>\n'
            )
        return (
            f'<h2>4. Per-Commodity Pathway (3.03%/yr)</h2>\n'
            f'<table>\n'
            f'<tr><th>Commodity</th><th>Base Year</th><th>2025</th>'
            f'<th>2027</th><th>2030</th><th>2035</th><th>2040</th>'
            f'<th>2050</th><th>Annual Rate</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_deforestation_checklist(self, data: Dict[str, Any]) -> str:
        checklist = data.get("deforestation_checklist", [])
        default_checks = [
            {"requirement": "Public no-deforestation commitment"},
            {"requirement": "2025 deadline for zero gross deforestation"},
            {"requirement": "Covers all commodities in scope"},
            {"requirement": "Covers direct and indirect suppliers"},
            {"requirement": "Includes no land conversion commitment"},
            {"requirement": "Monitoring and verification mechanism"},
            {"requirement": "Grievance and remediation mechanism"},
            {"requirement": "Regular progress reporting"},
            {"requirement": "Aligned with Accountability Framework Initiative"},
            {"requirement": "Board/executive sign-off"},
        ]
        items = checklist if checklist else default_checks
        rows = ""
        passed_count = 0
        for i, item in enumerate(items, 1):
            is_pass = str(item.get("status", "")).upper() in ("PASS", "YES", "MET", "TRUE")
            if is_pass:
                passed_count += 1
            cls = "checklist-yes" if is_pass else "checklist-no"
            label = "YES" if is_pass else "NO"
            rows += (
                f'<tr><td>{i}</td><td>{item.get("requirement", "-")}</td>'
                f'<td class="{cls}">{label}</td>'
                f'<td>{item.get("evidence", "-")}</td>'
                f'<td>{item.get("notes", "-")}</td></tr>\n'
            )
        total = len(items)
        score_pct = passed_count / total * 100 if total > 0 else 0
        bar_color = "#43a047" if passed_count == total else "#ff9800" if score_pct >= 50 else "#ef5350"

        return (
            f'<h2>5. No-Deforestation Commitment Checklist</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Score</div>'
            f'<div class="card-value">{passed_count}/{total}</div></div>\n'
            f'  <div class="card"><div class="card-label">Status</div>'
            f'<div class="card-value">'
            f'{"COMPLIANT" if passed_count == total else "ACTION REQUIRED"}'
            f'</div></div>\n'
            f'</div>\n'
            f'<div class="progress-bar">'
            f'<div class="progress-fill" style="width:{score_pct}%;background:{bar_color};"></div>'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Requirement</th><th>Status</th>'
            f'<th>Evidence</th><th>Notes</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_luc_emissions(self, data: Dict[str, Any]) -> str:
        luc = data.get("luc_emissions", {})
        sources = luc.get("sources", [])
        source_rows = ""
        for s in sources:
            source_rows += (
                f'<tr><td>{s.get("source", "-")}</td>'
                f'<td>{s.get("commodity", "-")}</td>'
                f'<td>{_dec_comma(s.get("area_ha", 0), 0)}</td>'
                f'<td>{_dec_comma(s.get("emissions_tco2e", 0), 0)}</td>'
                f'<td>{s.get("method", "-")}</td>'
                f'<td>{s.get("confidence", "-")}</td></tr>\n'
            )
        source_html = ""
        if sources:
            source_html = (
                f'<h3>LUC by Source</h3>\n'
                f'<table><tr><th>Source</th><th>Commodity</th><th>Area (ha)</th>'
                f'<th>Emissions (tCO2e)</th><th>Method</th><th>Confidence</th></tr>\n'
                f'{source_rows}</table>\n'
            )

        removals = luc.get("removals", {})
        removals_html = ""
        if removals:
            removals_html = (
                f'<h3>Carbon Removals (Land Sector)</h3>\n'
                f'<div class="summary-cards">\n'
                f'  <div class="card"><div class="card-label">Total Removals</div>'
                f'<div class="card-value">{_dec_comma(removals.get("total_removals_tco2e", 0), 0)}</div>'
                f'<div class="card-unit">tCO2e</div></div>\n'
                f'  <div class="card"><div class="card-label">Net FLAG</div>'
                f'<div class="card-value">{_dec_comma(removals.get("net_flag_tco2e", 0), 0)}</div>'
                f'<div class="card-unit">tCO2e</div></div>\n'
                f'</div>\n'
            )

        return (
            f'<h2>6. Land Use Change Emissions</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total LUC</div>'
            f'<div class="card-value">{_dec_comma(luc.get("total_luc_tco2e", 0), 0)}</div>'
            f'<div class="card-unit">tCO2e</div></div>\n'
            f'  <div class="card"><div class="card-label">% of FLAG</div>'
            f'<div class="card-value">{_pct(luc.get("luc_pct_of_flag", 0))}</div></div>\n'
            f'  <div class="card"><div class="card-label">Methodology</div>'
            f'<div class="card-value">{luc.get("methodology", "N/A")}</div></div>\n'
            f'</div>\n'
            f'{source_html}'
            f'{removals_html}'
        )

    def _html_flag_target_summary(self, data: Dict[str, Any]) -> str:
        summary = data.get("flag_target_summary", {})
        readiness = summary.get("readiness", {})
        overall_pct = float(readiness.get("overall_pct", 0))
        bar_color = "#43a047" if overall_pct >= 80 else "#ff9800" if overall_pct >= 50 else "#ef5350"
        dimensions = readiness.get("dimensions", [])
        dim_rows = ""
        for d in dimensions:
            dim_rows += (
                f'<tr><td>{d.get("name", "-")}</td>'
                f'<td>{_pct(d.get("score_pct", 0))}</td>'
                f'<td>{d.get("status", "-")}</td>'
                f'<td>{d.get("notes", "-")}</td></tr>\n'
            )
        dim_html = ""
        if dimensions:
            dim_html = (
                f'<h3>Readiness Dimensions</h3>\n'
                f'<table><tr><th>Dimension</th><th>Score</th>'
                f'<th>Status</th><th>Notes</th></tr>\n{dim_rows}</table>\n'
            )

        actions = summary.get("priority_actions", [])
        action_rows = ""
        for i, a in enumerate(actions, 1):
            action_rows += (
                f'<tr><td>{i}</td><td>{a.get("action", "-")}</td>'
                f'<td>{a.get("priority", "-")}</td>'
                f'<td>{a.get("owner", "-")}</td>'
                f'<td>{a.get("deadline", "-")}</td></tr>\n'
            )
        action_html = ""
        if actions:
            action_html = (
                f'<h3>Priority Actions</h3>\n'
                f'<table><tr><th>#</th><th>Action</th><th>Priority</th>'
                f'<th>Owner</th><th>Deadline</th></tr>\n{action_rows}</table>\n'
            )

        return (
            f'<h2>7. FLAG Target Summary & Readiness</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Overall Readiness</div>'
            f'<div class="card-value">{_pct(overall_pct)}</div></div>\n'
            f'  <div class="card"><div class="card-label">FLAG ARR</div>'
            f'<div class="card-value">{_pct(summary.get("company_flag_arr", 0))}</div></div>\n'
            f'  <div class="card"><div class="card-label">Meets Pathway</div>'
            f'<div class="card-value">{summary.get("meets_flag_pathway", "N/A")}</div></div>\n'
            f'  <div class="card"><div class="card-label">Deforestation</div>'
            f'<div class="card-value">{summary.get("no_deforestation", "N/A")}</div></div>\n'
            f'</div>\n'
            f'<div class="progress-bar">'
            f'<div class="progress-fill" style="width:{overall_pct}%;background:{bar_color};"></div>'
            f'</div>\n'
            f'{dim_html}'
            f'{action_html}'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-023 SBTi '
            f'Alignment Pack on {ts}<br>'
            f'FLAG assessment per SBTi FLAG Guidance V1.1 and Forest, Land & '
            f'Agriculture Science Based Target Setting Guidance.</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
