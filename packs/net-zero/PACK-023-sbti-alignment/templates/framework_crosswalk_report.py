# -*- coding: utf-8 -*-
"""
FrameworkCrosswalkReportTemplate - Multi-framework alignment mapping for PACK-023.

Renders a comprehensive framework crosswalk report mapping SBTi targets and
requirements to CDP Climate Change (C4), TCFD metrics and recommended
disclosures, CSRD/ESRS E1-4 requirements, GHG Protocol standards, and
ISO 14064 clauses. Includes requirement-by-requirement crosswalk, coverage
status tracking, gap identification, and multi-framework reporting guide.

Sections:
    1. Crosswalk Overview (frameworks covered, alignment summary)
    2. SBTi to CDP Mapping (C4 climate targets, C6 emissions)
    3. SBTi to TCFD Mapping (metrics, targets, transition plan)
    4. SBTi to CSRD/ESRS Mapping (E1-4 climate disclosures)
    5. SBTi to GHG Protocol Mapping (inventory, target methods)
    6. SBTi to ISO 14064 Mapping (Part 1 quantification)
    7. Coverage Status & Gap Analysis
    8. Multi-Framework Reporting Guide

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

logger = logging.getLogger(__name__)

_MODULE_VERSION = "23.0.0"

# Framework identifiers
FRAMEWORKS = {
    "SBTi": "Science Based Targets initiative",
    "CDP": "Carbon Disclosure Project (Climate Change)",
    "TCFD": "Task Force on Climate-Related Financial Disclosures",
    "CSRD": "Corporate Sustainability Reporting Directive (ESRS E1)",
    "GHG": "GHG Protocol (Corporate Standard + Value Chain)",
    "ISO14064": "ISO 14064-1:2018 (GHG Quantification)",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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


def _coverage_label(status: str) -> str:
    """Normalize coverage status for display."""
    s = str(status).upper()
    if s in ("FULL", "FULLY_COVERED", "COVERED"):
        return "Fully Covered"
    elif s in ("PARTIAL", "PARTIALLY_COVERED"):
        return "Partial"
    elif s in ("NONE", "NOT_COVERED", "GAP"):
        return "Gap"
    elif s in ("NA", "N/A", "NOT_APPLICABLE"):
        return "N/A"
    return status


class FrameworkCrosswalkReportTemplate:
    """
    Multi-framework crosswalk report template for SBTi alignment.

    Renders the alignment mapping between SBTi targets and CDP Climate
    Change, TCFD, CSRD/ESRS E1, GHG Protocol, and ISO 14064 requirements
    with requirement-by-requirement crosswalk, coverage tracking, gap
    analysis, and multi-framework reporting guidance.

    Attributes:
        config: Optional configuration overrides.
        generated_at: Timestamp of last render.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize FrameworkCrosswalkReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render framework crosswalk report as Markdown."""
        self.generated_at = _utcnow()
        sections: List[str] = [
            self._md_header(data),
            self._md_crosswalk_overview(data),
            self._md_cdp_mapping(data),
            self._md_tcfd_mapping(data),
            self._md_csrd_mapping(data),
            self._md_ghg_mapping(data),
            self._md_iso14064_mapping(data),
            self._md_coverage_gaps(data),
            self._md_reporting_guide(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render framework crosswalk report as self-contained HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_crosswalk_overview(data),
            self._html_cdp_mapping(data),
            self._html_tcfd_mapping(data),
            self._html_csrd_mapping(data),
            self._html_ghg_mapping(data),
            self._html_iso14064_mapping(data),
            self._html_coverage_gaps(data),
            self._html_reporting_guide(data),
            self._html_footer(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Framework Crosswalk Report</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render framework crosswalk report as structured JSON."""
        self.generated_at = _utcnow()
        cdp = data.get("cdp_mapping", [])
        tcfd = data.get("tcfd_mapping", [])
        csrd = data.get("csrd_mapping", [])
        ghg = data.get("ghg_mapping", [])
        iso = data.get("iso14064_mapping", [])
        coverage = data.get("coverage_summary", {})

        all_mappings = cdp + tcfd + csrd + ghg + iso
        fully_covered = len([
            m for m in all_mappings
            if _coverage_label(m.get("coverage", "")) == "Fully Covered"
        ])
        total = len(all_mappings)

        result: Dict[str, Any] = {
            "template": "framework_crosswalk_report",
            "version": _MODULE_VERSION,
            "generated_at": self.generated_at.isoformat(),
            "report_id": _new_uuid(),
            "org_name": data.get("org_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "overview": {
                "frameworks_mapped": len(FRAMEWORKS),
                "total_requirements": total,
                "fully_covered": fully_covered,
                "coverage_pct": round(fully_covered / total * 100, 1) if total > 0 else 0,
            },
            "cdp_mapping": cdp,
            "tcfd_mapping": tcfd,
            "csrd_mapping": csrd,
            "ghg_mapping": ghg,
            "iso14064_mapping": iso,
            "coverage_summary": coverage,
            "reporting_guide": data.get("reporting_guide", []),
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
            f"# Framework Crosswalk Report\n\n"
            f"**Organization:** {org}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Frameworks Mapped:** CDP, TCFD, CSRD/ESRS, GHG Protocol, ISO 14064\n\n---"
        )

    def _md_crosswalk_overview(self, data: Dict[str, Any]) -> str:
        fw_summary = data.get("framework_summary", [])
        lines = [
            "## 1. Crosswalk Overview\n",
            "Alignment summary across all mapped frameworks.\n",
            "| Framework | Full Name | Requirements Mapped "
            "| Fully Covered | Partial | Gaps | Coverage (%) |",
            "|-----------|-----------|:-------------------:"
            "|:-------------:|:-------:|:----:|:------------:|",
        ]
        if fw_summary:
            for fw in fw_summary:
                lines.append(
                    f"| {fw.get('framework', '-')} "
                    f"| {fw.get('full_name', '-')} "
                    f"| {fw.get('total_requirements', 0)} "
                    f"| {fw.get('fully_covered', 0)} "
                    f"| {fw.get('partial', 0)} "
                    f"| {fw.get('gaps', 0)} "
                    f"| {_pct(fw.get('coverage_pct', 0))} |"
                )
        else:
            # Generate from individual mappings
            for code, name in FRAMEWORKS.items():
                if code == "SBTi":
                    continue
                mapping_key = f"{code.lower()}_mapping"
                mappings = data.get(mapping_key, [])
                full = len([m for m in mappings if _coverage_label(m.get("coverage", "")) == "Fully Covered"])
                partial = len([m for m in mappings if _coverage_label(m.get("coverage", "")) == "Partial"])
                gaps = len([m for m in mappings if _coverage_label(m.get("coverage", "")) == "Gap"])
                total = len(mappings)
                cov_pct = full / total * 100 if total > 0 else 0
                lines.append(
                    f"| {code} | {name} | {total} "
                    f"| {full} | {partial} | {gaps} | {_pct(cov_pct)} |"
                )

        # Overall
        all_keys = ["cdp_mapping", "tcfd_mapping", "csrd_mapping", "ghg_mapping", "iso14064_mapping"]
        all_maps = []
        for k in all_keys:
            all_maps.extend(data.get(k, []))
        total_all = len(all_maps)
        full_all = len([m for m in all_maps if _coverage_label(m.get("coverage", "")) == "Fully Covered"])
        overall_pct = full_all / total_all * 100 if total_all > 0 else 0
        lines.append("")
        lines.append(
            f"**Overall Cross-Framework Coverage:** {_pct(overall_pct)} "
            f"({full_all}/{total_all} requirements fully covered)"
        )

        return "\n".join(lines)

    def _md_cdp_mapping(self, data: Dict[str, Any]) -> str:
        mappings = data.get("cdp_mapping", [])
        lines = [
            "## 2. SBTi to CDP Mapping\n",
            "Alignment of SBTi targets and data to CDP Climate Change questionnaire.\n",
            "| CDP Question | CDP Section | SBTi Requirement "
            "| SBTi Criterion | Coverage | Data Source | Notes |",
            "|-------------|:-----------:|:-----------------:"
            "|:--------------:|:--------:|:----------:|-------|",
        ]
        for m in mappings:
            lines.append(
                f"| {m.get('cdp_question', '-')} "
                f"| {m.get('cdp_section', '-')} "
                f"| {m.get('sbti_requirement', '-')} "
                f"| {m.get('sbti_criterion', '-')} "
                f"| {_coverage_label(m.get('coverage', ''))} "
                f"| {m.get('data_source', '-')} "
                f"| {m.get('notes', '-')} |"
            )
        if not mappings:
            lines.append(
                "| - | _No CDP mappings_ | - | - | - | - | - |"
            )
        return "\n".join(lines)

    def _md_tcfd_mapping(self, data: Dict[str, Any]) -> str:
        mappings = data.get("tcfd_mapping", [])
        lines = [
            "## 3. SBTi to TCFD Mapping\n",
            "Alignment with TCFD recommended disclosures and metrics.\n",
            "| TCFD Recommendation | Pillar | SBTi Alignment "
            "| SBTi Data Point | Coverage | Disclosure Location | Notes |",
            "|---------------------|:------:|:---------------:"
            "|:---------------:|:--------:|:-------------------:|-------|",
        ]
        for m in mappings:
            lines.append(
                f"| {m.get('tcfd_recommendation', '-')} "
                f"| {m.get('pillar', '-')} "
                f"| {m.get('sbti_alignment', '-')} "
                f"| {m.get('sbti_data_point', '-')} "
                f"| {_coverage_label(m.get('coverage', ''))} "
                f"| {m.get('disclosure_location', '-')} "
                f"| {m.get('notes', '-')} |"
            )
        if not mappings:
            lines.append(
                "| - | _No TCFD mappings_ | - | - | - | - | - |"
            )
        return "\n".join(lines)

    def _md_csrd_mapping(self, data: Dict[str, Any]) -> str:
        mappings = data.get("csrd_mapping", [])
        lines = [
            "## 4. SBTi to CSRD/ESRS Mapping\n",
            "Alignment with ESRS E1 Climate Change disclosure requirements.\n",
            "| ESRS Requirement | ESRS Reference | SBTi Alignment "
            "| SBTi Data Point | Coverage | Mandatory | Notes |",
            "|------------------|:--------------:|:---------------:"
            "|:---------------:|:--------:|:---------:|-------|",
        ]
        for m in mappings:
            lines.append(
                f"| {m.get('esrs_requirement', '-')} "
                f"| {m.get('esrs_reference', '-')} "
                f"| {m.get('sbti_alignment', '-')} "
                f"| {m.get('sbti_data_point', '-')} "
                f"| {_coverage_label(m.get('coverage', ''))} "
                f"| {m.get('mandatory', '-')} "
                f"| {m.get('notes', '-')} |"
            )
        if not mappings:
            lines.append(
                "| - | _No CSRD/ESRS mappings_ | - | - | - | - | - |"
            )
        return "\n".join(lines)

    def _md_ghg_mapping(self, data: Dict[str, Any]) -> str:
        mappings = data.get("ghg_mapping", [])
        lines = [
            "## 5. SBTi to GHG Protocol Mapping\n",
            "Alignment with GHG Protocol Corporate Standard and Scope 3 Standard.\n",
            "| GHG Protocol Requirement | Standard | SBTi Alignment "
            "| SBTi Criterion | Coverage | Notes |",
            "|--------------------------|:--------:|:---------------:"
            "|:--------------:|:--------:|-------|",
        ]
        for m in mappings:
            lines.append(
                f"| {m.get('ghg_requirement', '-')} "
                f"| {m.get('standard', '-')} "
                f"| {m.get('sbti_alignment', '-')} "
                f"| {m.get('sbti_criterion', '-')} "
                f"| {_coverage_label(m.get('coverage', ''))} "
                f"| {m.get('notes', '-')} |"
            )
        if not mappings:
            lines.append(
                "| - | _No GHG Protocol mappings_ | - | - | - | - |"
            )
        return "\n".join(lines)

    def _md_iso14064_mapping(self, data: Dict[str, Any]) -> str:
        mappings = data.get("iso14064_mapping", [])
        lines = [
            "## 6. SBTi to ISO 14064 Mapping\n",
            "Alignment with ISO 14064-1:2018 GHG quantification requirements.\n",
            "| ISO 14064 Clause | Description | SBTi Alignment "
            "| SBTi Data Point | Coverage | Notes |",
            "|------------------|-------------|:---------------:"
            "|:---------------:|:--------:|-------|",
        ]
        for m in mappings:
            lines.append(
                f"| {m.get('iso_clause', '-')} "
                f"| {m.get('description', '-')} "
                f"| {m.get('sbti_alignment', '-')} "
                f"| {m.get('sbti_data_point', '-')} "
                f"| {_coverage_label(m.get('coverage', ''))} "
                f"| {m.get('notes', '-')} |"
            )
        if not mappings:
            lines.append(
                "| - | _No ISO 14064 mappings_ | - | - | - | - |"
            )
        return "\n".join(lines)

    def _md_coverage_gaps(self, data: Dict[str, Any]) -> str:
        gaps = data.get("coverage_gaps", [])
        lines = [
            "## 7. Coverage Status & Gap Analysis\n",
            f"**Total Gaps Identified:** {len(gaps)}\n",
            "| # | Framework | Requirement | Gap Description "
            "| Impact | Remediation | Priority | Effort |",
            "|---|-----------|-------------|:-----------------:"
            "|:------:|-------------|:--------:|:------:|",
        ]
        for i, g in enumerate(gaps, 1):
            lines.append(
                f"| {i} | {g.get('framework', '-')} "
                f"| {g.get('requirement', '-')} "
                f"| {g.get('gap_description', '-')} "
                f"| {g.get('impact', '-')} "
                f"| {g.get('remediation', '-')} "
                f"| {g.get('priority', '-')} "
                f"| {g.get('effort', '-')} |"
            )
        if not gaps:
            lines.append(
                "| - | _No gaps identified_ | - | - | - | - | - | - |"
            )

        # Coverage matrix
        matrix = data.get("coverage_matrix", [])
        if matrix:
            lines.append("")
            lines.append("### Cross-Framework Coverage Matrix\n")
            lines.append(
                "| SBTi Requirement | CDP | TCFD | CSRD/ESRS "
                "| GHG Protocol | ISO 14064 |"
            )
            lines.append(
                "|------------------|:---:|:----:|:---------:"
                "|:------------:|:---------:|"
            )
            for m in matrix:
                lines.append(
                    f"| {m.get('sbti_requirement', '-')} "
                    f"| {_coverage_label(m.get('cdp', ''))} "
                    f"| {_coverage_label(m.get('tcfd', ''))} "
                    f"| {_coverage_label(m.get('csrd', ''))} "
                    f"| {_coverage_label(m.get('ghg', ''))} "
                    f"| {_coverage_label(m.get('iso14064', ''))} |"
                )

        return "\n".join(lines)

    def _md_reporting_guide(self, data: Dict[str, Any]) -> str:
        guide = data.get("reporting_guide", [])
        lines = [
            "## 8. Multi-Framework Reporting Guide\n",
            "Practical guidance for using SBTi data across framework disclosures.\n",
            "| # | Framework | Disclosure Area | SBTi Data Available "
            "| How to Report | Template/Format | Timing |",
            "|---|-----------|-----------------|:--------------------:"
            "|:-------------:|:---------------:|:------:|",
        ]
        for i, g in enumerate(guide, 1):
            lines.append(
                f"| {i} | {g.get('framework', '-')} "
                f"| {g.get('disclosure_area', '-')} "
                f"| {g.get('sbti_data', '-')} "
                f"| {g.get('how_to_report', '-')} "
                f"| {g.get('template_format', '-')} "
                f"| {g.get('timing', '-')} |"
            )
        if not guide:
            lines.append(
                "| - | _No reporting guide_ | - | - | - | - | - |"
            )

        # Synergies
        synergies = data.get("synergies", [])
        if synergies:
            lines.append("")
            lines.append("### Cross-Framework Synergies\n")
            lines.append("| # | Synergy | Frameworks | Benefit | Action |")
            lines.append("|---|---------|:----------:|--------|--------|")
            for i, s in enumerate(synergies, 1):
                lines.append(
                    f"| {i} | {s.get('synergy', '-')} "
                    f"| {s.get('frameworks', '-')} "
                    f"| {s.get('benefit', '-')} "
                    f"| {s.get('action', '-')} |"
                )

        # Efficiency recommendations
        efficiency = data.get("efficiency_recommendations", [])
        if efficiency:
            lines.append("")
            lines.append("### Reporting Efficiency Recommendations\n")
            for i, e in enumerate(efficiency, 1):
                lines.append(
                    f"{i}. **{e.get('title', '-')}:** {e.get('description', '-')}"
                )

        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f"---\n\n*Generated by GreenLang PACK-023 SBTi Alignment Pack on {ts}*  \n"
            f"*Framework crosswalk covering CDP Climate Change 2024, TCFD "
            f"Recommendations (2017), CSRD/ESRS E1 (2024), GHG Protocol "
            f"Corporate Standard, and ISO 14064-1:2018.*"
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
            ".cov-full{display:inline-block;background:#43a047;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".cov-partial{display:inline-block;background:#ff9800;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".cov-gap{display:inline-block;background:#ef5350;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".cov-na{display:inline-block;background:#9e9e9e;color:#fff;"
            "border-radius:4px;padding:2px 8px;font-size:0.85em;}"
            ".fw-badge{display:inline-block;border-radius:4px;padding:2px 10px;"
            "font-size:0.85em;font-weight:600;margin:2px;}"
            ".fw-cdp{background:#0053a0;color:#fff;}"
            ".fw-tcfd{background:#004d40;color:#fff;}"
            ".fw-csrd{background:#1565c0;color:#fff;}"
            ".fw-ghg{background:#2e7d32;color:#fff;}"
            ".fw-iso{background:#6a1b9a;color:#fff;}"
            ".progress-bar{width:100%;height:20px;background:#e0e0e0;border-radius:10px;"
            "overflow:hidden;margin:4px 0;}"
            ".progress-fill{height:100%;border-radius:10px;}"
            ".footer{margin-top:40px;padding-top:20px;border-top:2px solid #c8e6c9;"
            "color:#689f38;font-size:0.85em;text-align:center;}"
        )

    def _html_coverage_badge(self, status: str) -> str:
        """Return an HTML badge for coverage status."""
        label = _coverage_label(status)
        if label == "Fully Covered":
            return '<span class="cov-full">Full</span>'
        elif label == "Partial":
            return '<span class="cov-partial">Partial</span>'
        elif label == "Gap":
            return '<span class="cov-gap">Gap</span>'
        elif label == "N/A":
            return '<span class="cov-na">N/A</span>'
        return label

    def _html_fw_badge(self, framework: str) -> str:
        """Return a framework-specific badge."""
        fw = str(framework).upper()
        if fw == "CDP":
            return '<span class="fw-badge fw-cdp">CDP</span>'
        elif fw == "TCFD":
            return '<span class="fw-badge fw-tcfd">TCFD</span>'
        elif fw in ("CSRD", "ESRS"):
            return '<span class="fw-badge fw-csrd">CSRD</span>'
        elif fw in ("GHG", "GHG PROTOCOL"):
            return '<span class="fw-badge fw-ghg">GHG</span>'
        elif fw in ("ISO14064", "ISO"):
            return '<span class="fw-badge fw-iso">ISO</span>'
        return framework

    def _html_header(self, data: Dict[str, Any]) -> str:
        org = data.get("org_name", "Organization")
        year = data.get("reporting_year", "")
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<h1>Framework Crosswalk Report</h1>\n'
            f'<p><strong>Organization:</strong> {org} | '
            f'<strong>Year:</strong> {year} | '
            f'<strong>Generated:</strong> {ts}</p>\n'
            f'<p>Frameworks: '
            f'{self._html_fw_badge("CDP")} '
            f'{self._html_fw_badge("TCFD")} '
            f'{self._html_fw_badge("CSRD")} '
            f'{self._html_fw_badge("GHG")} '
            f'{self._html_fw_badge("ISO14064")}</p>'
        )

    def _html_crosswalk_overview(self, data: Dict[str, Any]) -> str:
        all_keys = ["cdp_mapping", "tcfd_mapping", "csrd_mapping", "ghg_mapping", "iso14064_mapping"]
        all_maps = []
        for k in all_keys:
            all_maps.extend(data.get(k, []))
        total_all = len(all_maps)
        full_all = len([m for m in all_maps if _coverage_label(m.get("coverage", "")) == "Fully Covered"])
        partial_all = len([m for m in all_maps if _coverage_label(m.get("coverage", "")) == "Partial"])
        gap_all = len([m for m in all_maps if _coverage_label(m.get("coverage", "")) == "Gap"])
        overall_pct = full_all / total_all * 100 if total_all > 0 else 0
        bar_color = "#43a047" if overall_pct >= 80 else "#ff9800" if overall_pct >= 50 else "#ef5350"

        fw_rows = ""
        fw_codes = [("CDP", "cdp_mapping"), ("TCFD", "tcfd_mapping"),
                     ("CSRD", "csrd_mapping"), ("GHG", "ghg_mapping"),
                     ("ISO14064", "iso14064_mapping")]
        for code, key in fw_codes:
            mappings = data.get(key, [])
            full = len([m for m in mappings if _coverage_label(m.get("coverage", "")) == "Fully Covered"])
            partial = len([m for m in mappings if _coverage_label(m.get("coverage", "")) == "Partial"])
            gaps = len([m for m in mappings if _coverage_label(m.get("coverage", "")) == "Gap"])
            total = len(mappings)
            cov_pct = full / total * 100 if total > 0 else 0
            fw_rows += (
                f'<tr><td>{self._html_fw_badge(code)}</td>'
                f'<td>{FRAMEWORKS.get(code, code)}</td>'
                f'<td>{total}</td>'
                f'<td>{full}</td><td>{partial}</td><td>{gaps}</td>'
                f'<td>{_pct(cov_pct)}</td></tr>\n'
            )

        return (
            f'<h2>1. Crosswalk Overview</h2>\n'
            f'<div class="summary-cards">\n'
            f'  <div class="card"><div class="card-label">Total Requirements</div>'
            f'<div class="card-value">{total_all}</div></div>\n'
            f'  <div class="card"><div class="card-label">Fully Covered</div>'
            f'<div class="card-value">{full_all}</div></div>\n'
            f'  <div class="card"><div class="card-label">Partial</div>'
            f'<div class="card-value">{partial_all}</div></div>\n'
            f'  <div class="card"><div class="card-label">Gaps</div>'
            f'<div class="card-value">{gap_all}</div></div>\n'
            f'  <div class="card"><div class="card-label">Overall Coverage</div>'
            f'<div class="card-value">{_pct(overall_pct)}</div></div>\n'
            f'</div>\n'
            f'<div class="progress-bar">'
            f'<div class="progress-fill" style="width:{overall_pct}%;background:{bar_color};"></div>'
            f'</div>\n'
            f'<table>\n'
            f'<tr><th>Framework</th><th>Full Name</th><th>Total</th>'
            f'<th>Full</th><th>Partial</th><th>Gaps</th><th>Coverage</th></tr>\n'
            f'{fw_rows}</table>'
        )

    def _html_cdp_mapping(self, data: Dict[str, Any]) -> str:
        mappings = data.get("cdp_mapping", [])
        rows = ""
        for m in mappings:
            rows += (
                f'<tr><td>{m.get("cdp_question", "-")}</td>'
                f'<td>{m.get("cdp_section", "-")}</td>'
                f'<td>{m.get("sbti_requirement", "-")}</td>'
                f'<td>{m.get("sbti_criterion", "-")}</td>'
                f'<td>{self._html_coverage_badge(m.get("coverage", ""))}</td>'
                f'<td>{m.get("data_source", "-")}</td>'
                f'<td>{m.get("notes", "-")}</td></tr>\n'
            )
        return (
            f'<h2>2. SBTi to CDP Mapping</h2>\n'
            f'<table>\n'
            f'<tr><th>CDP Question</th><th>Section</th><th>SBTi Req.</th>'
            f'<th>Criterion</th><th>Coverage</th><th>Source</th>'
            f'<th>Notes</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_tcfd_mapping(self, data: Dict[str, Any]) -> str:
        mappings = data.get("tcfd_mapping", [])
        rows = ""
        for m in mappings:
            rows += (
                f'<tr><td>{m.get("tcfd_recommendation", "-")}</td>'
                f'<td>{m.get("pillar", "-")}</td>'
                f'<td>{m.get("sbti_alignment", "-")}</td>'
                f'<td>{m.get("sbti_data_point", "-")}</td>'
                f'<td>{self._html_coverage_badge(m.get("coverage", ""))}</td>'
                f'<td>{m.get("disclosure_location", "-")}</td>'
                f'<td>{m.get("notes", "-")}</td></tr>\n'
            )
        return (
            f'<h2>3. SBTi to TCFD Mapping</h2>\n'
            f'<table>\n'
            f'<tr><th>TCFD Recommendation</th><th>Pillar</th>'
            f'<th>SBTi Alignment</th><th>Data Point</th><th>Coverage</th>'
            f'<th>Location</th><th>Notes</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_csrd_mapping(self, data: Dict[str, Any]) -> str:
        mappings = data.get("csrd_mapping", [])
        rows = ""
        for m in mappings:
            rows += (
                f'<tr><td>{m.get("esrs_requirement", "-")}</td>'
                f'<td>{m.get("esrs_reference", "-")}</td>'
                f'<td>{m.get("sbti_alignment", "-")}</td>'
                f'<td>{m.get("sbti_data_point", "-")}</td>'
                f'<td>{self._html_coverage_badge(m.get("coverage", ""))}</td>'
                f'<td>{m.get("mandatory", "-")}</td>'
                f'<td>{m.get("notes", "-")}</td></tr>\n'
            )
        return (
            f'<h2>4. SBTi to CSRD/ESRS Mapping</h2>\n'
            f'<table>\n'
            f'<tr><th>ESRS Requirement</th><th>Reference</th>'
            f'<th>SBTi Alignment</th><th>Data Point</th><th>Coverage</th>'
            f'<th>Mandatory</th><th>Notes</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_ghg_mapping(self, data: Dict[str, Any]) -> str:
        mappings = data.get("ghg_mapping", [])
        rows = ""
        for m in mappings:
            rows += (
                f'<tr><td>{m.get("ghg_requirement", "-")}</td>'
                f'<td>{m.get("standard", "-")}</td>'
                f'<td>{m.get("sbti_alignment", "-")}</td>'
                f'<td>{m.get("sbti_criterion", "-")}</td>'
                f'<td>{self._html_coverage_badge(m.get("coverage", ""))}</td>'
                f'<td>{m.get("notes", "-")}</td></tr>\n'
            )
        return (
            f'<h2>5. SBTi to GHG Protocol Mapping</h2>\n'
            f'<table>\n'
            f'<tr><th>GHG Requirement</th><th>Standard</th>'
            f'<th>SBTi Alignment</th><th>Criterion</th><th>Coverage</th>'
            f'<th>Notes</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_iso14064_mapping(self, data: Dict[str, Any]) -> str:
        mappings = data.get("iso14064_mapping", [])
        rows = ""
        for m in mappings:
            rows += (
                f'<tr><td>{m.get("iso_clause", "-")}</td>'
                f'<td>{m.get("description", "-")}</td>'
                f'<td>{m.get("sbti_alignment", "-")}</td>'
                f'<td>{m.get("sbti_data_point", "-")}</td>'
                f'<td>{self._html_coverage_badge(m.get("coverage", ""))}</td>'
                f'<td>{m.get("notes", "-")}</td></tr>\n'
            )
        return (
            f'<h2>6. SBTi to ISO 14064 Mapping</h2>\n'
            f'<table>\n'
            f'<tr><th>ISO Clause</th><th>Description</th>'
            f'<th>SBTi Alignment</th><th>Data Point</th><th>Coverage</th>'
            f'<th>Notes</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_coverage_gaps(self, data: Dict[str, Any]) -> str:
        gaps = data.get("coverage_gaps", [])
        rows = ""
        for i, g in enumerate(gaps, 1):
            rows += (
                f'<tr><td>{i}</td>'
                f'<td>{self._html_fw_badge(g.get("framework", "-"))}</td>'
                f'<td>{g.get("requirement", "-")}</td>'
                f'<td>{g.get("gap_description", "-")}</td>'
                f'<td>{g.get("impact", "-")}</td>'
                f'<td>{g.get("remediation", "-")}</td>'
                f'<td>{g.get("priority", "-")}</td>'
                f'<td>{g.get("effort", "-")}</td></tr>\n'
            )

        matrix = data.get("coverage_matrix", [])
        matrix_rows = ""
        for m in matrix:
            matrix_rows += (
                f'<tr><td>{m.get("sbti_requirement", "-")}</td>'
                f'<td>{self._html_coverage_badge(m.get("cdp", ""))}</td>'
                f'<td>{self._html_coverage_badge(m.get("tcfd", ""))}</td>'
                f'<td>{self._html_coverage_badge(m.get("csrd", ""))}</td>'
                f'<td>{self._html_coverage_badge(m.get("ghg", ""))}</td>'
                f'<td>{self._html_coverage_badge(m.get("iso14064", ""))}</td></tr>\n'
            )
        matrix_html = ""
        if matrix:
            matrix_html = (
                f'<h3>Cross-Framework Coverage Matrix</h3>\n'
                f'<table><tr><th>SBTi Requirement</th><th>CDP</th><th>TCFD</th>'
                f'<th>CSRD</th><th>GHG</th><th>ISO</th></tr>\n'
                f'{matrix_rows}</table>\n'
            )

        return (
            f'<h2>7. Coverage Status & Gap Analysis</h2>\n'
            f'<p><strong>Total Gaps:</strong> {len(gaps)}</p>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Framework</th><th>Requirement</th>'
            f'<th>Gap</th><th>Impact</th><th>Remediation</th>'
            f'<th>Priority</th><th>Effort</th></tr>\n'
            f'{rows}</table>\n'
            f'{matrix_html}'
        )

    def _html_reporting_guide(self, data: Dict[str, Any]) -> str:
        guide = data.get("reporting_guide", [])
        rows = ""
        for i, g in enumerate(guide, 1):
            rows += (
                f'<tr><td>{i}</td>'
                f'<td>{self._html_fw_badge(g.get("framework", "-"))}</td>'
                f'<td>{g.get("disclosure_area", "-")}</td>'
                f'<td>{g.get("sbti_data", "-")}</td>'
                f'<td>{g.get("how_to_report", "-")}</td>'
                f'<td>{g.get("template_format", "-")}</td>'
                f'<td>{g.get("timing", "-")}</td></tr>\n'
            )
        return (
            f'<h2>8. Multi-Framework Reporting Guide</h2>\n'
            f'<table>\n'
            f'<tr><th>#</th><th>Framework</th><th>Area</th>'
            f'<th>SBTi Data</th><th>How to Report</th>'
            f'<th>Format</th><th>Timing</th></tr>\n'
            f'{rows}</table>'
        )

    def _html_footer(self, data: Dict[str, Any]) -> str:
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return (
            f'<div class="footer">Generated by GreenLang PACK-023 SBTi '
            f'Alignment Pack on {ts}<br>'
            f'Framework crosswalk covering CDP Climate Change 2024, TCFD '
            f'Recommendations (2017), CSRD/ESRS E1 (2024), GHG Protocol '
            f'Corporate Standard, and ISO 14064-1:2018.</div>'
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _provenance(self, content: str) -> str:
        """Compute SHA-256 provenance hash."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
