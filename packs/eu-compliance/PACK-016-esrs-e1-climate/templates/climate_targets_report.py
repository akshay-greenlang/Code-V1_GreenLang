# -*- coding: utf-8 -*-
"""
ClimateTargetsReportTemplate - ESRS E1-4 Climate Targets Disclosure Report

Renders target overview, SBTi alignment status, progress tracker,
base year information, and interim targets per ESRS E1-4.

Sections:
    1. Target Overview
    2. SBTi Alignment
    3. Progress Tracker
    4. Base Year Info
    5. Interim Targets

Author: GreenLang Team
Version: 16.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SECTIONS: List[str] = [
    "target_overview",
    "sbti_alignment",
    "progress_tracker",
    "base_year_info",
    "interim_targets",
]


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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


class ClimateTargetsReportTemplate:
    """
    Climate targets disclosure report template per ESRS E1-4.

    Renders GHG emission reduction targets with SBTi validation status,
    base year recalculation details, interim milestones, and progress
    against absolute and intensity-based targets.

    Example:
        >>> tpl = ClimateTargetsReportTemplate()
        >>> md = tpl.render_markdown(data)
        >>> html = tpl.render_html(data)
        >>> js = tpl.render_json(data)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ClimateTargetsReportTemplate."""
        self.config = config or {}
        self.generated_at: Optional[datetime] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render full report as structured dict."""
        self.generated_at = _utcnow()
        result: Dict[str, Any] = {}
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
        if not data.get("reporting_year"):
            errors.append("reporting_year is required")
        if not data.get("entity_name"):
            errors.append("entity_name is required")
        if "climate_targets" not in data:
            warnings.append("climate_targets missing; will default to empty")
        if "base_year" not in data:
            warnings.append("base_year missing; will default to empty")
        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def render_markdown(self, data: Dict[str, Any]) -> str:
        """Render climate targets report as Markdown."""
        self.generated_at = _utcnow()
        sections = [
            self._md_header(data),
            self._md_target_overview(data),
            self._md_sbti(data),
            self._md_progress(data),
            self._md_base_year(data),
            self._md_interim(data),
            self._md_footer(data),
        ]
        content = "\n\n".join(sections)
        prov = _compute_hash(content)
        return content + f"\n\n<!-- Provenance: {prov} -->"

    def render_html(self, data: Dict[str, Any]) -> str:
        """Render climate targets report as HTML."""
        self.generated_at = _utcnow()
        css = self._css()
        body = "\n".join([
            self._html_header(data),
            self._html_target_overview(data),
            self._html_sbti(data),
            self._html_progress(data),
        ])
        prov = _compute_hash(body)
        return (
            f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="UTF-8">\n'
            f'<title>Climate Targets Report - ESRS E1-4</title>\n'
            f'<style>\n{css}\n</style>\n</head>\n<body>\n'
            f'<div class="report">\n{body}\n</div>\n'
            f'<!-- Provenance: {prov} -->\n</body>\n</html>'
        )

    def render_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Render climate targets report as JSON."""
        self.generated_at = _utcnow()
        targets = data.get("climate_targets", [])
        result = {
            "template": "climate_targets_report",
            "esrs_reference": "E1-4",
            "version": "16.0.0",
            "generated_at": self.generated_at.isoformat(),
            "entity_name": data.get("entity_name", ""),
            "reporting_year": data.get("reporting_year", ""),
            "total_targets": len(targets),
            "sbti_validated": data.get("sbti_validated", False),
            "absolute_targets": sum(1 for t in targets if t.get("type") == "absolute"),
            "intensity_targets": sum(1 for t in targets if t.get("type") == "intensity"),
            "base_year": data.get("base_year", {}),
        }
        prov = _compute_hash(result)
        result["provenance_hash"] = prov
        return result

    # ------------------------------------------------------------------
    # Section renderers (dict)
    # ------------------------------------------------------------------

    def _section_target_overview(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build target overview section."""
        targets = data.get("climate_targets", [])
        return {
            "title": "Climate Target Overview",
            "total_targets": len(targets),
            "absolute_targets": sum(1 for t in targets if t.get("type") == "absolute"),
            "intensity_targets": sum(1 for t in targets if t.get("type") == "intensity"),
            "targets": [
                {
                    "name": t.get("name", ""),
                    "type": t.get("type", ""),
                    "scope_coverage": t.get("scope_coverage", []),
                    "target_year": t.get("target_year", ""),
                    "reduction_pct": round(t.get("reduction_pct", 0.0), 1),
                    "base_year": t.get("base_year", ""),
                    "status": t.get("status", ""),
                }
                for t in targets
            ],
        }

    def _section_sbti_alignment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build SBTi alignment section."""
        sbti = data.get("sbti_alignment", {})
        return {
            "title": "Science Based Targets Initiative Alignment",
            "committed": sbti.get("committed", False),
            "validated": sbti.get("validated", False),
            "validation_date": sbti.get("validation_date", ""),
            "near_term_target": sbti.get("near_term_target", ""),
            "long_term_target": sbti.get("long_term_target", ""),
            "sector_pathway": sbti.get("sector_pathway", ""),
            "temperature_alignment": sbti.get("temperature_alignment", ""),
            "net_zero_commitment_year": sbti.get("net_zero_commitment_year", ""),
        }

    def _section_progress_tracker(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build progress tracker section."""
        targets = data.get("climate_targets", [])
        progress_entries = []
        for t in targets:
            base_emissions = t.get("base_year_tco2e", 0.0)
            current_emissions = t.get("current_tco2e", 0.0)
            target_reduction = t.get("reduction_pct", 0.0)
            actual_reduction = (
                round((base_emissions - current_emissions) / base_emissions * 100, 1)
                if base_emissions > 0 else 0.0
            )
            progress_entries.append({
                "target_name": t.get("name", ""),
                "target_reduction_pct": round(target_reduction, 1),
                "actual_reduction_pct": actual_reduction,
                "on_track": actual_reduction >= (target_reduction * 0.8),
                "gap_pct": round(target_reduction - actual_reduction, 1),
            })
        return {
            "title": "Target Progress Tracker",
            "targets": progress_entries,
            "targets_on_track": sum(1 for p in progress_entries if p["on_track"]),
            "total_targets": len(progress_entries),
        }

    def _section_base_year_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build base year information section."""
        base = data.get("base_year", {})
        return {
            "title": "Base Year Information",
            "year": base.get("year", ""),
            "scope1_tco2e": round(base.get("scope1_tco2e", 0.0), 2),
            "scope2_tco2e": round(base.get("scope2_tco2e", 0.0), 2),
            "scope3_tco2e": round(base.get("scope3_tco2e", 0.0), 2),
            "total_tco2e": round(base.get("total_tco2e", 0.0), 2),
            "recalculation_policy": base.get("recalculation_policy", ""),
            "recalculation_threshold_pct": base.get("recalculation_threshold_pct", 5.0),
            "last_recalculation": base.get("last_recalculation", ""),
        }

    def _section_interim_targets(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Build interim targets section."""
        interim = data.get("interim_targets", [])
        return {
            "title": "Interim Targets and Milestones",
            "milestones": [
                {
                    "year": m.get("year", ""),
                    "target_tco2e": round(m.get("target_tco2e", 0.0), 2),
                    "reduction_pct": round(m.get("reduction_pct", 0.0), 1),
                    "scope_coverage": m.get("scope_coverage", ""),
                    "status": m.get("status", ""),
                }
                for m in interim
            ],
            "total_milestones": len(interim),
        }

    # ------------------------------------------------------------------
    # Markdown helpers
    # ------------------------------------------------------------------

    def _md_header(self, data: Dict[str, Any]) -> str:
        """Render markdown header."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        entity = data.get("entity_name", "")
        year = data.get("reporting_year", "")
        return (
            f"# Climate Targets Report - ESRS E1-4\n\n"
            f"**Entity:** {entity}  \n"
            f"**Reporting Year:** {year}  \n"
            f"**Generated:** {ts}  \n"
            f"**Standard:** ESRS E1-4 Targets Related to Climate Change Mitigation "
            f"and Adaptation"
        )

    def _md_target_overview(self, data: Dict[str, Any]) -> str:
        """Render target overview markdown."""
        sec = self._section_target_overview(data)
        lines = [
            f"## {sec['title']}\n",
            f"**Total Targets:** {sec['total_targets']} "
            f"(Absolute: {sec['absolute_targets']}, "
            f"Intensity: {sec['intensity_targets']})\n",
            "| Target | Type | Scope | Year | Reduction |",
            "|--------|------|-------|-----:|----------:|",
        ]
        for t in sec["targets"]:
            scopes = ", ".join(t["scope_coverage"]) if t["scope_coverage"] else "N/A"
            lines.append(
                f"| {t['name']} | {t['type']} | {scopes} "
                f"| {t['target_year']} | {t['reduction_pct']:.1f}% |"
            )
        return "\n".join(lines)

    def _md_sbti(self, data: Dict[str, Any]) -> str:
        """Render SBTi alignment markdown."""
        sec = self._section_sbti_alignment(data)
        committed = "Yes" if sec["committed"] else "No"
        validated = "Yes" if sec["validated"] else "No"
        return (
            f"## {sec['title']}\n\n"
            f"**SBTi Committed:** {committed}  \n"
            f"**Targets Validated:** {validated}  \n"
            f"**Validation Date:** {sec['validation_date']}  \n"
            f"**Temperature Alignment:** {sec['temperature_alignment']}  \n"
            f"**Near-Term Target:** {sec['near_term_target']}  \n"
            f"**Long-Term Target:** {sec['long_term_target']}"
        )

    def _md_progress(self, data: Dict[str, Any]) -> str:
        """Render progress tracker markdown."""
        sec = self._section_progress_tracker(data)
        lines = [
            f"## {sec['title']}\n",
            f"**On Track:** {sec['targets_on_track']}/{sec['total_targets']}\n",
            "| Target | Target % | Actual % | Gap | On Track |",
            "|--------|--------:|--------:|----:|:--------:|",
        ]
        for t in sec["targets"]:
            track = "Yes" if t["on_track"] else "No"
            lines.append(
                f"| {t['target_name']} | {t['target_reduction_pct']:.1f}% "
                f"| {t['actual_reduction_pct']:.1f}% | {t['gap_pct']:.1f}pp "
                f"| {track} |"
            )
        return "\n".join(lines)

    def _md_base_year(self, data: Dict[str, Any]) -> str:
        """Render base year info markdown."""
        sec = self._section_base_year_info(data)
        return (
            f"## {sec['title']}\n\n"
            f"**Base Year:** {sec['year']}  \n"
            f"**Scope 1:** {sec['scope1_tco2e']:,.2f} tCO2e  \n"
            f"**Scope 2:** {sec['scope2_tco2e']:,.2f} tCO2e  \n"
            f"**Scope 3:** {sec['scope3_tco2e']:,.2f} tCO2e  \n"
            f"**Total:** {sec['total_tco2e']:,.2f} tCO2e  \n"
            f"**Recalculation Threshold:** {sec['recalculation_threshold_pct']:.1f}%"
        )

    def _md_interim(self, data: Dict[str, Any]) -> str:
        """Render interim targets markdown."""
        sec = self._section_interim_targets(data)
        lines = [
            f"## {sec['title']}\n",
            "| Year | Target (tCO2e) | Reduction | Status |",
            "|-----:|--------------:|----------:|--------|",
        ]
        for m in sec["milestones"]:
            lines.append(
                f"| {m['year']} | {m['target_tco2e']:,.2f} "
                f"| {m['reduction_pct']:.1f}% | {m['status']} |"
            )
        return "\n".join(lines)

    def _md_footer(self, data: Dict[str, Any]) -> str:
        """Render markdown footer."""
        ts = self.generated_at.strftime("%Y-%m-%d %H:%M UTC") if self.generated_at else ""
        return f"---\n*Report generated by PACK-016 ESRS E1 Climate Pack on {ts}*"

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _css(self) -> str:
        """Return CSS for HTML rendering."""
        return (
            "body{font-family:Arial,sans-serif;margin:2em;color:#333}"
            ".report{max-width:900px;margin:auto}"
            "h1{color:#1a5632;border-bottom:2px solid #1a5632;padding-bottom:.3em}"
            "h2{color:#2d7a4f;margin-top:1.5em}"
            "table{border-collapse:collapse;width:100%;margin:1em 0}"
            "th,td{border:1px solid #ccc;padding:8px;text-align:left}"
            "th{background:#f0f7f3}"
        )

    def _html_header(self, data: Dict[str, Any]) -> str:
        """Render HTML header."""
        entity = data.get("entity_name", "")
        year = data.get("reporting_year", "")
        return (
            f"<h1>Climate Targets Report - ESRS E1-4</h1>\n"
            f"<p><strong>{entity}</strong> | {year}</p>"
        )

    def _html_target_overview(self, data: Dict[str, Any]) -> str:
        """Render target overview HTML."""
        sec = self._section_target_overview(data)
        rows = "".join(
            f"<tr><td>{t['name']}</td><td>{t['type']}</td>"
            f"<td>{t['target_year']}</td><td>{t['reduction_pct']:.1f}%</td></tr>"
            for t in sec["targets"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Target</th><th>Type</th><th>Year</th>"
            f"<th>Reduction</th></tr>{rows}</table>"
        )

    def _html_sbti(self, data: Dict[str, Any]) -> str:
        """Render SBTi alignment HTML."""
        sec = self._section_sbti_alignment(data)
        validated = "Validated" if sec["validated"] else "Not Validated"
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<p>Status: <strong>{validated}</strong> | "
            f"Alignment: {sec['temperature_alignment']}</p>"
        )

    def _html_progress(self, data: Dict[str, Any]) -> str:
        """Render progress tracker HTML."""
        sec = self._section_progress_tracker(data)
        rows = "".join(
            f"<tr><td>{t['target_name']}</td>"
            f"<td>{t['target_reduction_pct']:.1f}%</td>"
            f"<td>{t['actual_reduction_pct']:.1f}%</td>"
            f"<td>{'Yes' if t['on_track'] else 'No'}</td></tr>"
            for t in sec["targets"]
        )
        return (
            f"<h2>{sec['title']}</h2>\n"
            f"<table><tr><th>Target</th><th>Target %</th><th>Actual %</th>"
            f"<th>On Track</th></tr>{rows}</table>"
        )
